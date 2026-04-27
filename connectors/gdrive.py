"""
Google Drive connector for authentication and file operations.
Uses OAuth2 with credentials.json and persists token to token.json.
"""

import asyncio
import io
import json
import os
import ssl
import shutil
from functools import partial
from typing import Dict, List, Optional

import aiofiles
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


class GoogleDriveConnector:
    """Handle Google Drive authentication and file operations."""

    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    CREDENTIALS_FILE = "credentials.json"
    TOKEN_FILE = "token.json"
    DATA_FOLDER = "./data"
    SYNC_STATE_FILE = "synced_files.json"
    GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
    TEXT_MIME_TYPE = "text/plain"
    PDF_MIME_TYPE = "application/pdf"

    def __init__(self):
        """Initialize the Google Drive connector without blocking I/O."""
        self.credentials = None
        self.service = None
        self._sync_state_lock = asyncio.Lock()

    @classmethod
    async def create(cls) -> "GoogleDriveConnector":
        """Construct and asynchronously initialize the connector."""
        connector = cls()
        try:
            await connector._ensure_data_folder()
            await connector._authenticate()
            return connector
        except Exception as e:
            print(f"❌ Error initializing GoogleDriveConnector: {str(e)}")
            raise

    async def _run_in_executor(self, func, *args, **kwargs):
        """Run blocking work in the default executor."""
        loop = asyncio.get_running_loop()
        bound_func = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound_func)

    async def _ensure_data_folder(self) -> None:
        """Create data folder if it doesn't exist."""
        try:
            exists = await asyncio.to_thread(os.path.exists, self.DATA_FOLDER)
            if not exists:
                await asyncio.to_thread(os.makedirs, self.DATA_FOLDER, exist_ok=True)
                print(f"✅ Created data folder at {self.DATA_FOLDER}")
        except Exception as e:
            print(f"❌ Error creating data folder: {str(e)}")
            raise

    async def _authenticate(self) -> None:
        """Authenticate with Google Drive using OAuth2."""
        try:
            token_exists = await asyncio.to_thread(os.path.exists, self.TOKEN_FILE)
            if token_exists:
                print(f"📂 Loading token from {self.TOKEN_FILE}")
                async with aiofiles.open(self.TOKEN_FILE, "r", encoding="utf-8") as file_handle:
                    token_content = await file_handle.read()
                token_data = json.loads(token_content)
                self.credentials = Credentials.from_authorized_user_info(token_data, scopes=self.SCOPES)
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    print("🔄 Token expired, refreshing...")
                    await self._run_in_executor(self.credentials.refresh, Request())
                    await self._save_token()

            if not self.credentials or not self.credentials.valid:
                print(f"📝 Starting OAuth2 flow with {self.CREDENTIALS_FILE}")
                async with aiofiles.open(self.CREDENTIALS_FILE, "r", encoding="utf-8") as file_handle:
                    credentials_content = await file_handle.read()
                client_config = json.loads(credentials_content)
                flow = InstalledAppFlow.from_client_config(client_config, self.SCOPES)
                self.credentials = await self._run_in_executor(flow.run_local_server, port=0)
                await self._save_token()

            self.service = await self._run_in_executor(build, "drive", "v3", credentials=self.credentials)
            print("✅ Google Drive authentication successful")
        except GoogleAuthError as e:
            print(f"❌ Google authentication error: {str(e)}")
            raise
        except Exception as e:
            print(f"❌ Error during authentication: {str(e)}")
            raise

    async def _save_token(self) -> None:
        """Save credentials to token.json for future use."""
        try:
            async with aiofiles.open(self.TOKEN_FILE, "w", encoding="utf-8") as file_handle:
                await file_handle.write(self.credentials.to_json())
            print(f"✅ Token saved to {self.TOKEN_FILE}")
        except Exception as e:
            print(f"❌ Error saving token: {str(e)}")
            raise

    async def _load_sync_state(self) -> Dict[str, Dict[str, str]]:
        """Load synced file metadata from disk."""
        try:
            exists = await asyncio.to_thread(os.path.exists, self.SYNC_STATE_FILE)
            if not exists:
                return {}

            async with aiofiles.open(self.SYNC_STATE_FILE, "r", encoding="utf-8") as file_handle:
                content = await file_handle.read()
            try:
                data = json.loads(content) if content else {}
            except json.JSONDecodeError:
                backup_path = f"{self.SYNC_STATE_FILE}.corrupt"
                await asyncio.to_thread(shutil.copyfile, self.SYNC_STATE_FILE, backup_path)
                print(
                    f"⚠️  Corrupted sync state detected. Backed up to {backup_path} "
                    f"and starting with a clean state."
                )
                return {}
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"❌ Error loading sync state: {str(e)}")
            raise

    async def _save_sync_state(self, sync_state: Dict[str, Dict[str, str]]) -> None:
        """Persist synced file metadata to disk."""
        try:
            async with aiofiles.open(self.SYNC_STATE_FILE, "w", encoding="utf-8") as file_handle:
                await file_handle.write(json.dumps(sync_state, indent=2))
            print(f"✅ Sync state saved to {self.SYNC_STATE_FILE}")
        except Exception as e:
            print(f"❌ Error saving sync state: {str(e)}")
            raise

    async def update_synced_file(self, file_id: str, file_name: str, modified_time: str) -> None:
        """Update sync state for one successfully synced file."""
        await self.update_synced_files([
            {
                "file_id": file_id,
                "file_name": file_name,
                "modified_time": modified_time,
            }
        ])

    async def update_synced_files(self, files: List[Dict[str, str]]) -> None:
        """Update sync state for multiple successfully synced files in one write."""
        if not files:
            return

        async with self._sync_state_lock:
            sync_state = await self._load_sync_state()
            for file_info in files:
                sync_state[file_info["file_id"]] = {
                    "fileName": file_info["file_name"],
                    "modifiedTime": file_info["modified_time"],
                }
            await self._save_sync_state(sync_state)

    async def _execute_request(self, request) -> dict:
        """Execute a Google API request without blocking the event loop."""
        return await self._run_in_executor(request.execute)

    async def list_files(self, query: str = "trashed=false", max_results: int = 10) -> List[dict]:
        """List files from Google Drive."""
        try:
            request = self.service.files().list(
                q=query,
                spaces="drive",
                fields="files(id, name, mimeType, size, createdTime, modifiedTime)",
                pageSize=max_results,
            )
            results = await self._execute_request(request)
            files = results.get("files", [])
            print(f"✅ Retrieved {len(files)} files from Google Drive")
            return files
        except Exception as e:
            print(f"❌ Error listing files: {str(e)}")
            raise

    async def list_syncable_files(self, max_results: int = 100) -> List[dict]:
        """List supported Google Drive files for syncing."""
        supported_mime_types = [
            self.PDF_MIME_TYPE,
            self.TEXT_MIME_TYPE,
            self.GOOGLE_DOC_MIME_TYPE,
        ]
        mime_query = " or ".join([f"mimeType='{mime_type}'" for mime_type in supported_mime_types])
        query = f"trashed=false and ({mime_query})"
        return await self.list_files(query=query, max_results=max_results)

    def _resolve_download_path(self, file_name: str, mime_type: str) -> str:
        """Build a local file path for the downloaded file."""
        _, ext = os.path.splitext(file_name)
        if mime_type == self.GOOGLE_DOC_MIME_TYPE and ext.lower() != ".txt":
            file_name = f"{file_name}.txt"
        return os.path.join(self.DATA_FOLDER, file_name)

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes into a human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.2f} MB"

    def _download_request_bytes(self, request) -> bytes:
        """Download a Google API media request into memory."""
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                print(f"📥 Download progress: {progress}%")
        return buffer.getvalue()

    def _is_ssl_error(self, error: Exception) -> bool:
        """Detect SSL-related download failures."""
        if isinstance(error, ssl.SSLError):
            return True
        return "ssl" in str(error).lower()

    async def download_file(
        self,
        file_id: str,
        file_name: str,
        mime_type: Optional[str] = None,
        force_download: bool = False,
        retry_limit: int = 1,
    ) -> str:
        """Download a supported file from Google Drive to the data folder."""
        attempts = max(1, retry_limit)
        last_error = None

        for attempt in range(1, attempts + 1):
            try:
                await self._ensure_data_folder()

                metadata_request = self.service.files().get(
                    fileId=file_id,
                    fields="id, name, mimeType, size",
                )
                file_metadata = await self._execute_request(metadata_request)
                resolved_name = file_metadata.get("name", file_name)
                resolved_mime_type = mime_type or file_metadata.get("mimeType", "")
                remote_size = int(file_metadata.get("size", 0) or 0)
                file_path = self._resolve_download_path(resolved_name, resolved_mime_type)

                print(
                    f"📥 Downloading {resolved_name} "
                    f"(mimeType={resolved_mime_type}, expected_size={self._format_size(remote_size)})"
                )

                exists = await asyncio.to_thread(os.path.exists, file_path)
                if exists and not force_download:
                    existing_size = await asyncio.to_thread(os.path.getsize, file_path)
                    print(
                        f"✅ File already exists: {file_path} "
                        f"(size={self._format_size(existing_size)})"
                    )
                    return file_path

                if exists and force_download:
                    await asyncio.to_thread(os.remove, file_path)
                    print(f"🔄 Re-downloading updated file: {file_path}")

                if resolved_mime_type == self.GOOGLE_DOC_MIME_TYPE:
                    request = self.service.files().export_media(
                        fileId=file_id,
                        mimeType=self.TEXT_MIME_TYPE,
                    )
                elif resolved_mime_type in {self.PDF_MIME_TYPE, self.TEXT_MIME_TYPE}:
                    request = self.service.files().get_media(fileId=file_id)
                else:
                    raise ValueError(
                        f"Unsupported Google Drive mime type for {resolved_name}: {resolved_mime_type}"
                    )

                file_bytes = await self._run_in_executor(self._download_request_bytes, request)
                async with aiofiles.open(file_path, "wb") as file_handle:
                    await file_handle.write(file_bytes)

                downloaded_size = await asyncio.to_thread(os.path.getsize, file_path)
                print(
                    f"✅ Download succeeded: {resolved_name} -> {file_path} "
                    f"(size={self._format_size(downloaded_size)})"
                )
                return file_path
            except Exception as e:
                last_error = e
                file_path = locals().get("file_path")
                if file_path and await asyncio.to_thread(os.path.exists, file_path):
                    try:
                        await asyncio.to_thread(os.remove, file_path)
                        print(f"🧹 Removed partial download: {file_path}")
                    except OSError:
                        pass

                if self._is_ssl_error(e):
                    print(f"⚠️  SSL error downloading {file_name}; skipping without retry")
                    break

                if attempt < attempts:
                    print(f"🔁 Retrying download for {file_name} ({attempt}/{attempts})")
                else:
                    print(f"❌ Download failed for {file_name}: {str(e)}")

        raise last_error

    async def download_files_by_name(self, file_names: List[str]) -> List[str]:
        """Download multiple files by name from Google Drive."""
        try:
            all_files = await self.list_files(max_results=100)
            matching_files = []

            for file_name in file_names:
                matching_file = next((f for f in all_files if f["name"] == file_name), None)
                if matching_file:
                    matching_files.append(matching_file)
                else:
                    print(f"⚠️  File not found in Google Drive: {file_name}")

            download_tasks = [
                self.download_file(
                    matching_file["id"],
                    matching_file["name"],
                    matching_file.get("mimeType"),
                )
                for matching_file in matching_files
            ]
            if not download_tasks:
                return []

            return await asyncio.gather(*download_tasks)
        except Exception as e:
            print(f"❌ Error downloading files by name: {str(e)}")
            raise

    async def get_file_by_id(self, file_id: str) -> Optional[str]:
        """Get file path by downloading a file from Google Drive by ID."""
        try:
            request = self.service.files().get(
                fileId=file_id,
                fields="name, mimeType",
            )
            file = await self._execute_request(request)
            file_name = file["name"]
            return await self.download_file(file_id, file_name, file.get("mimeType"))
        except Exception as e:
            print(f"❌ Error getting file by ID: {str(e)}")
            raise

    async def _sync_candidate(self, file_metadata: Dict[str, str], sync_state: Dict[str, Dict[str, str]]) -> Dict[str, object]:
        """Sync a single candidate file and return its result."""
        file_id = file_metadata["id"]
        file_name = file_metadata["name"]
        mime_type = file_metadata.get("mimeType", "")
        modified_time = file_metadata.get("modifiedTime", "")
        previous_state = sync_state.get(file_id)
        is_new = previous_state is None
        is_modified = bool(previous_state and previous_state.get("modifiedTime") != modified_time)

        if not is_new and not is_modified:
            print(f"⏭️ Skipping unchanged file: {file_name}")
            return {"status": "skipped", "file_id": file_id, "file_name": file_name}

        try:
            local_path = await self.download_file(
                file_id=file_id,
                file_name=file_name,
                mime_type=mime_type,
                force_download=is_modified,
            )
            return {
                "status": "new" if is_new else "modified",
                "file_id": file_id,
                "file_name": file_name,
                "file_path": local_path,
                "modified_time": modified_time,
            }
        except Exception as e:
            return {
                "status": "failed",
                "file_id": file_id,
                "file_name": file_name,
                "error": str(e),
            }

    async def fetch_files(
        self,
        file_metadatas: List[Dict[str, str]],
        sync_state: Dict[str, Dict[str, str]],
        max_files: int = 20,
    ) -> List[Dict[str, object]]:
        """Fetch at most max_files successfully downloaded items."""
        results: List[Dict[str, object]] = []
        successful_downloads = 0

        for file_metadata in file_metadatas:
            if successful_downloads >= max_files:
                print(f"🛑 Reached max successful download limit: {max_files}")
                break

            result = await self._sync_candidate(file_metadata, sync_state)
            results.append(result)
            if result.get("status") in {"new", "modified"}:
                successful_downloads += 1

        return results

    async def sync_files(self, max_results: int = 100, max_files: int = 20) -> Dict[str, object]:
        """Sync only new or modified supported files from Google Drive."""
        try:
            all_files = await self.list_syncable_files(max_results=max_results)
            sync_state = await self._load_sync_state()
            summary = {
                "new": 0,
                "modified": 0,
                "skipped": 0,
                "failed": 0,
                "downloaded_files": [],
                "failed_files": [],
            }

            results = await self.fetch_files(all_files, sync_state, max_files=max_files)

            for result in results:
                status = result.get("status")
                if status == "new":
                    summary["new"] += 1
                    summary["downloaded_files"].append(result)
                elif status == "modified":
                    summary["modified"] += 1
                    summary["downloaded_files"].append(result)
                elif status == "skipped":
                    summary["skipped"] += 1
                else:
                    summary["failed"] += 1
                    summary["failed_files"].append(result)

            return summary
        except Exception as e:
            print(f"❌ Error syncing files: {str(e)}")
            raise
