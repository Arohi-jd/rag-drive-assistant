"""
FastAPI routes for the RAG system.
Handles document ingestion, search, and augmentation.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Type
from fastapi import APIRouter, BackgroundTasks, Body, status
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, ValidationError
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from connectors.gdrive import GoogleDriveConnector
from processing.chunker import DocumentChunker
from search.faiss_store import FAISSStore

# Initialize router
router = APIRouter()

SAMPLE_IO_FILE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_io.json")
)

# Initialize components
chunker = DocumentChunker()
faiss_store = FAISSStore()

# Load Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None


class SearchQuery(BaseModel):
    """Search query model."""
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class AskRequest(BaseModel):
    """Ask request model with optional metadata filters."""
    query: str = Field(min_length=1)
    file_name: Optional[str] = None
    source: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class UploadDocumentRequest(BaseModel):
    """Document upload request model."""
    file_path: str = Field(min_length=1)


class DownloadFromDriveRequest(BaseModel):
    """Download from Google Drive request model."""
    file_names: List[str] = Field(default_factory=list)


class SyncDriveRequest(BaseModel):
    """Sync Google Drive request model."""
    max_results: int = Field(default=100, ge=1, le=1000)


class APIResponse(BaseModel):
    """Consistent API response envelope."""
    success: bool
    message: str
    data: Any = None


def success_response(message: str, data: Any = None, status_code: int = status.HTTP_200_OK) -> JSONResponse:
    """Build a consistent success response."""
    return JSONResponse(
        status_code=status_code,
        content=APIResponse(success=True, message=message, data=data).model_dump()
    )


def error_response(message: str, status_code: int, data: Any = None) -> JSONResponse:
    """Build a consistent error response."""
    return JSONResponse(
        status_code=status_code,
        content=APIResponse(success=False, message=message, data=data).model_dump()
    )


def validate_request(payload: Optional[Dict[str, Any]], model_class: Type[BaseModel]) -> BaseModel:
    """Validate a request body with a Pydantic model."""
    return model_class.model_validate(payload or {})


def get_index_overview() -> Dict[str, Any]:
    """Return document and chunk counts from the in-memory FAISS metadata."""
    metadata = faiss_store.metadata or []
    doc_ids = {
        item.get('doc_id') or item.get('file_name') or item.get('source')
        for item in metadata
        if item.get('doc_id') or item.get('file_name') or item.get('source')
    }
    chunk_count = len(metadata)
    return {
        "documents_indexed": len(doc_ids),
        "chunks_indexed": chunk_count
    }


def filter_metadata_chunks(file_name: Optional[str] = None, source: Optional[str] = None) -> List[Dict[str, Any]]:
    """Filter indexed chunks by optional metadata fields."""
    filtered_chunks = []
    for item in faiss_store.metadata or []:
        item_file_name = item.get("file_name")
        item_source = item.get("source")

        if file_name and item_file_name != file_name:
            continue
        if source and item_source != source:
            continue
        filtered_chunks.append(item)

    return filtered_chunks


def search_filtered_chunks(query: str, filtered_chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Search only within a filtered subset of indexed chunks."""
    if not filtered_chunks:
        return []

    query_embedding = faiss_store.embedder.embed_text(query)
    scored_results = []
    for chunk in filtered_chunks:
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            continue
        chunk_embedding = faiss_store.embedder.embed_text(chunk_text)
        similarity = faiss_store.embedder.similarity_score(query_embedding, chunk_embedding)
        scored_results.append({
            "text": chunk_text,
            "similarity": similarity,
            "metadata": chunk,
        })

    scored_results.sort(key=lambda item: item["similarity"], reverse=True)
    return scored_results[:top_k]


async def search_filtered_chunks_async(
    query: str,
    filtered_chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run filtered chunk search off the event loop."""
    return await asyncio.to_thread(search_filtered_chunks, query, filtered_chunks, top_k)


def build_context_blocks(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build context blocks and source lists from ranked chunk results."""
    context_blocks = []
    context_texts = []
    source_file_names = []

    for index, result in enumerate(results[:5], start=1):
        metadata = result.get("metadata", {})
        file_name = metadata.get("file_name") or metadata.get("source") or "Unknown source"
        chunk_text = result.get("text", "")
        context_texts.append(chunk_text)
        source_file_names.append(file_name)
        context_blocks.append(
            f"{index}. File: {file_name}\n"
            f"Content:\n{chunk_text}"
        )

    return {
        "context_texts": context_texts,
        "context": "\n\n".join(context_blocks) if context_blocks else "No relevant context found.",
        "sources": list(dict.fromkeys(source_file_names)),
    }


async def create_groq_completion(messages: List[Dict[str, str]]) -> str:
    """Run Groq completion off the event loop."""
    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=groq_model,
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content


@router.post("/search", status_code=status.HTTP_200_OK)
async def search_documents(payload: Optional[Dict[str, Any]] = Body(default=None)) -> JSONResponse:
    """
    Search for similar documents in the FAISS index.
    
    Args:
        request: Search query request
        
    Returns:
        Search results with similarity scores
    """
    try:
        request = validate_request(payload, SearchQuery)
        print(f"🔍 Searching for: {request.query}")
        results = await faiss_store.search_with_scores(request.query, k=request.top_k)

        return success_response(
            message="Search completed successfully",
            data={
                "query": request.query,
                "results": results,
                "total_results": len(results)
            }
        )
    except ValidationError as e:
        return error_response(
            message="Invalid search request",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={"errors": e.errors()}
        )
    except Exception as e:
        print(f"❌ Error searching documents: {str(e)}")
        return error_response(
            message="Failed to search documents",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"error": str(e)}
        )


@router.post("/ingest", status_code=status.HTTP_200_OK)
async def ingest_document(payload: Optional[Dict[str, Any]] = Body(default=None)) -> JSONResponse:
    """
    Ingest a document into the FAISS index.
    
    Args:
        request: Document path request
        
    Returns:
        Ingestion result with chunk count
    """
    try:
        request = validate_request(payload, UploadDocumentRequest)
        if not await asyncio.to_thread(os.path.exists, request.file_path):
            return error_response(
                message=f"File not found: {request.file_path}",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        print(f"📖 Ingesting document: {request.file_path}")
        
        # Process document
        chunks = await asyncio.to_thread(chunker.process_file, request.file_path)
        
        # Extract text and metadata
        texts = [chunk['text'] for chunk in chunks]
        metadata = [dict(chunk) for chunk in chunks]
        
        # Add to FAISS
        await faiss_store.add_documents(texts, metadata)

        return success_response(
            message=f"Successfully ingested {len(chunks)} chunks",
            data={
                "file_path": request.file_path,
                "chunks_created": len(chunks),
                "index_overview": get_index_overview()
            }
        )
    except ValidationError as e:
        return error_response(
            message="Invalid ingest request",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={"errors": e.errors()}
        )
    except Exception as e:
        print(f"❌ Error ingesting document: {str(e)}")
        return error_response(
            message="Failed to ingest document",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"error": str(e)}
        )


@router.post("/download-from-drive", status_code=status.HTTP_200_OK)
async def download_from_drive(
    background_tasks: BackgroundTasks,
    payload: Optional[Dict[str, Any]] = Body(default=None)
) -> JSONResponse:
    """
    Download files from Google Drive and ingest them.
    
    Args:
        request: Download request with file names
        background_tasks: Background task manager
        
    Returns:
        Download and ingestion status
    """
    try:
        request = validate_request(payload, DownloadFromDriveRequest)
        try:
            gdrive = await GoogleDriveConnector.create()
        except Exception as e:
            print(f"⚠️  Google Drive not configured: {str(e)}")
            return error_response(
                message="Google Drive is not configured. Ensure credentials.json exists.",
                status_code=status.HTTP_400_BAD_REQUEST,
                data={"error": str(e)}
            )
        
        if not request.file_names:
            print("⚠️  No file names provided, listing available files")
            files = await gdrive.list_files(max_results=10)
            return success_response(
                message="No files specified. Here are available files.",
                data={"available_files": files}
            )
        
        print(f"📥 Downloading {len(request.file_names)} files from Google Drive")
        
        # Download files
        downloaded_paths = await gdrive.download_files_by_name(request.file_names)

        if not downloaded_paths:
            return error_response(
                message="None of the requested files were found in Google Drive",
                status_code=status.HTTP_404_NOT_FOUND,
                data={"requested_files": request.file_names}
            )
        
        # Ingest files in background
        for file_path in downloaded_paths:
            background_tasks.add_task(
                _ingest_file_task,
                file_path
            )

        return success_response(
            message=f"Downloaded {len(downloaded_paths)} files. Ingestion in progress...",
            data={
                "downloaded_files": downloaded_paths,
                "requested_files": request.file_names
            }
        )
    except ValidationError as e:
        return error_response(
            message="Invalid Google Drive download request",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={"errors": e.errors()}
        )
    except Exception as e:
        print(f"❌ Error downloading from drive: {str(e)}")
        return error_response(
            message="Failed to download files from Google Drive",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"error": str(e)}
        )


@router.post("/sync-drive", status_code=status.HTTP_200_OK)
async def sync_drive(payload: Optional[Dict[str, Any]] = Body(default=None)) -> JSONResponse:
    """Incrementally sync supported files from Google Drive."""
    try:
        request = validate_request(payload, SyncDriveRequest)
        try:
            gdrive = await GoogleDriveConnector.create()
        except Exception as e:
            print(f"⚠️  Google Drive not configured: {str(e)}")
            return error_response(
                message="Google Drive is not configured. Ensure credentials.json exists.",
                status_code=status.HTTP_400_BAD_REQUEST,
                data={"error": str(e)}
            )

        sync_summary = await gdrive.sync_files(max_results=request.max_results)
        synced_files = sync_summary.get("downloaded_files", [])
        
        async def process_synced_file(file_info: Dict[str, Any]) -> Dict[str, Any]:
            try:
                file_path = file_info["file_path"]
                chunks = await asyncio.to_thread(chunker.process_file, file_path)
                texts = [chunk["text"] for chunk in chunks]
                metadata = [dict(chunk) for chunk in chunks]
                doc_id = metadata[0]["doc_id"] if metadata else None
                if doc_id:
                    await faiss_store.replace_document(doc_id, texts, metadata)
                return {"status": "success", "file": file_info}
            except Exception as e:
                return {
                    "status": "failed",
                    "file_id": file_info.get("file_id"),
                    "file_name": file_info.get("file_name"),
                    "error": str(e),
                }

        processed_results = await asyncio.gather(
            *(process_synced_file(file_info) for file_info in synced_files)
        )
        successful_files = [
            result["file"] for result in processed_results
            if result.get("status") == "success"
        ]
        await gdrive.update_synced_files(successful_files)
        processing_failures = [
            result for result in processed_results
            if result.get("status") == "failed"
        ]

        all_failures = sync_summary.get("failed_files", []) + processing_failures
        newly_synced = sum(1 for item in successful_files if item.get("status") == "new")
        modified_synced = sum(1 for item in successful_files if item.get("status") == "modified")

        return success_response(
            message="Google Drive sync completed",
            data={
                "new_files": newly_synced,
                "modified_files": modified_synced,
                "skipped_files": sync_summary.get("skipped", 0),
                "failed_files": len(all_failures),
                "synced_files": successful_files,
                "failed_details": all_failures,
            }
        )
    except ValidationError as e:
        return error_response(
            message="Invalid sync request",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={"errors": e.errors()}
        )
    except Exception as e:
        print(f"❌ Error syncing Google Drive: {str(e)}")
        return error_response(
            message="Failed to sync Google Drive files",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"error": str(e)}
        )


@router.post("/augment", status_code=status.HTTP_200_OK)
async def augment_with_context(payload: Optional[Dict[str, Any]] = Body(default=None)) -> JSONResponse:
    """
    Search for context and augment response using Groq.
    
    Args:
        request: Search query request
        
    Returns:
        Augmented response with context
    """
    try:
        if not groq_client:
            return error_response(
                message="Groq API key is not configured in .env file",
                status_code=status.HTTP_400_BAD_REQUEST
            )

        request = validate_request(payload, SearchQuery)
        
        print(f"🤖 Augmenting response for: {request.query}")
        
        # Search for context
        results = await faiss_store.search_with_scores(request.query, k=5)
        context_payload = build_context_blocks(results)

        # Prepare prompt
        prompt = (
            f"Context blocks:\n{context_payload['context']}\n\n"
            f"User question: {request.query}"
        )
        
        # Call Groq
        answer = await create_groq_completion([
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using ONLY "
                    "the context provided below. If the answer is not in the context, say "
                    "I could not find this information in the documents. Always mention "
                    "which document your answer came from."
                )
            },
            {"role": "user", "content": prompt}
        ])

        return success_response(
            message="Augmented response generated successfully",
            data={
                "query": request.query,
                "context": context_payload["context_texts"],
                "answer": answer,
                "sources": context_payload["sources"]
            }
        )
    except ValidationError as e:
        return error_response(
            message="Invalid augmentation request",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={"errors": e.errors()}
        )
    except Exception as e:
        print(f"❌ Error augmenting response: {str(e)}")
        return error_response(
            message="Failed to generate augmented response",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"error": str(e)}
        )


@router.post("/ask", status_code=status.HTTP_200_OK)
async def ask_with_filters(payload: Optional[Dict[str, Any]] = Body(default=None)) -> JSONResponse:
    """Answer a question using only chunks that match optional metadata filters."""
    try:
        if not groq_client:
            return error_response(
                message="Groq API key is not configured in .env file",
                status_code=status.HTTP_400_BAD_REQUEST
            )

        request = validate_request(payload, AskRequest)
        print(f"🤖 Asking with filters for: {request.query}")

        filtered_chunks = filter_metadata_chunks(
            file_name=request.file_name,
            source=request.source,
        )
        if not filtered_chunks:
            return error_response(
                message="No chunks matched the provided metadata filters.",
                status_code=status.HTTP_404_NOT_FOUND,
                data={
                    "file_name": request.file_name,
                    "source": request.source,
                }
            )

        results = await search_filtered_chunks_async(
            query=request.query,
            filtered_chunks=filtered_chunks,
            top_k=request.top_k,
        )
        if not results:
            return error_response(
                message="No searchable chunks were found after applying the filters.",
                status_code=status.HTTP_404_NOT_FOUND,
                data={
                    "file_name": request.file_name,
                    "source": request.source,
                }
            )

        context_payload = build_context_blocks(results)
        prompt = (
            f"Context blocks:\n{context_payload['context']}\n\n"
            f"User question: {request.query}"
        )

        answer = await create_groq_completion([
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using ONLY "
                    "the context provided below. If the answer is not in the context, say "
                    "I could not find this information in the documents. Always mention "
                    "which document your answer came from."
                )
            },
            {"role": "user", "content": prompt}
        ])

        return success_response(
            message="Answer generated successfully",
            data={
                "query": request.query,
                "filters": {
                    "file_name": request.file_name,
                    "source": request.source,
                    "top_k": request.top_k,
                },
                "matched_chunks": len(filtered_chunks),
                "context": context_payload["context_texts"],
                "answer": answer,
                "sources": context_payload["sources"],
            }
        )
    except ValidationError as e:
        return error_response(
            message="Invalid ask request",
            status_code=status.HTTP_400_BAD_REQUEST,
            data={"errors": e.errors()}
        )
    except Exception as e:
        print(f"❌ Error answering with filters: {str(e)}")
        return error_response(
            message="Failed to generate answer",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"error": str(e)}
        )


@router.get("/index-stats", status_code=status.HTTP_200_OK)
async def get_index_stats() -> JSONResponse:
    """Get FAISS index statistics."""
    try:
        stats = await faiss_store.get_statistics()
        return success_response(
            message="Index statistics retrieved successfully",
            data=stats
        )
    except Exception as e:
        print(f"❌ Error getting index stats: {str(e)}")
        return error_response(
            message="Failed to get index statistics",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"error": str(e)}
        )


@router.post("/clear-index", status_code=status.HTTP_200_OK)
async def clear_faiss_index() -> JSONResponse:
    """Clear the FAISS index."""
    try:
        await faiss_store.clear_index()
        return success_response(
            message="FAISS index cleared successfully",
            data=get_index_overview()
        )
    except Exception as e:
        print(f"❌ Error clearing index: {str(e)}")
        return error_response(
            message="Failed to clear FAISS index",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"error": str(e)}
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    try:
        index_overview = get_index_overview()
        return success_response(
            message="Server is healthy",
            data={
                "server_status": "healthy",
                "documents_indexed": index_overview["documents_indexed"],
                "chunks_indexed": index_overview["chunks_indexed"]
            }
        )
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return error_response(
            message="Health check failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"error": str(e)}
        )


@router.get("/sample-io", status_code=status.HTTP_200_OK)
async def download_sample_io() -> FileResponse:
    """Download sample input/output payloads."""
    if not os.path.exists(SAMPLE_IO_FILE):
        return error_response(
            message="Sample IO file not found",
            status_code=status.HTTP_404_NOT_FOUND
        )

    return FileResponse(
        SAMPLE_IO_FILE,
        media_type="application/json",
        filename="sample_io.json"
    )


async def _ingest_file_task(file_path: str) -> None:
    """Background task to ingest a file."""
    try:
        print(f"🔄 Ingesting file in background: {file_path}")
        chunks = await asyncio.to_thread(chunker.process_file, file_path)
        texts = [chunk['text'] for chunk in chunks]
        metadata = [dict(chunk) for chunk in chunks]
        await faiss_store.add_documents(texts, metadata)
        print(f"✅ Background ingestion completed for {file_path}")
    except Exception as e:
        print(f"❌ Error in background ingestion task: {str(e)}")
