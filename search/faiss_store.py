"""
FAISS (Facebook AI Similarity Search) store for efficient similarity search.
Persists index and metadata to disk.
"""

import asyncio
import json
import os
from functools import partial
from typing import Dict, List, Tuple

import aiofiles
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

USE_FAISS = os.environ.get("USE_FAISS", "true").lower() == "true"

from embedding.embedder import Embedder


class FAISSStore:
    """Manage FAISS index for similarity search."""

    INDEX_FILE = "faiss_index.bin"
    METADATA_FILE = "metadata.json"

    def __init__(self, index_file: str = INDEX_FILE, metadata_file: str = METADATA_FILE):
        """
        Initialize FAISS store.

        Args:
            index_file: Path to save/load the FAISS index
            metadata_file: Path to save/load metadata
        """
        try:
            self.index_file = index_file
            self.metadata_file = metadata_file
            self.index = None
            self.metadata = []
            self.embedder = Embedder()
            self.embedding_dim = 384
            self.use_faiss = USE_FAISS
            self.embeddings_list = []
            self.metadata_list = []
            self._lock = asyncio.Lock()
            print(f"✅ FAISSStore initialized with embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"❌ Error initializing FAISSStore: {str(e)}")
            raise

    async def _run_in_executor(self, func, *args, **kwargs):
        """Run blocking work in the default executor."""
        loop = asyncio.get_running_loop()
        bound_func = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound_func)

    def _create_index_sync(self):
        """Create a new FAISS index synchronously."""
        return faiss.IndexFlatL2(self.embedding_dim)

    def _load_numpy_embeddings_sync(self, file_path: str) -> np.ndarray:
        """Load numpy embeddings from disk without extension changes."""
        with open(file_path, "rb") as file_handle:
            return np.load(file_handle)

    def _save_numpy_embeddings_sync(self, file_path: str, embeddings: np.ndarray) -> None:
        """Save numpy embeddings to disk without extension changes."""
        with open(file_path, "wb") as file_handle:
            np.save(file_handle, embeddings)

    async def create_index(self) -> None:
        """Create a new FAISS index."""
        try:
            async with self._lock:
                if self.use_faiss and FAISS_AVAILABLE:
                    self.index = await self._run_in_executor(self._create_index_sync)
                    self.metadata = []
                    print(f"✅ Created new FAISS index with dimension {self.embedding_dim}")
                else:
                    self.index = None
                    self.metadata = []
                    self.embeddings_list = []
                    self.metadata_list = []
                    print("✅ Initialized numpy fallback index")
        except Exception as e:
            print(f"❌ Error creating FAISS index: {str(e)}")
            raise

    async def load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            index_exists = await asyncio.to_thread(os.path.exists, self.index_file)
            metadata_exists = await asyncio.to_thread(os.path.exists, self.metadata_file)

            if self.use_faiss and FAISS_AVAILABLE:
                if index_exists and metadata_exists:
                    loaded_index = await self._run_in_executor(faiss.read_index, self.index_file)
                    async with aiofiles.open(self.metadata_file, "r", encoding="utf-8") as file_handle:
                        metadata_content = await file_handle.read()
                    loaded_metadata = json.loads(metadata_content) if metadata_content else []

                    async with self._lock:
                        self.index = loaded_index
                        self.metadata = loaded_metadata
                    print(f"✅ Loaded FAISS index with {len(self.metadata)} documents")
                else:
                    print("📝 Index files not found, creating new index")
                    await self.create_index()
            else:
                if index_exists and metadata_exists:
                    embeddings = await asyncio.to_thread(self._load_numpy_embeddings_sync, self.index_file)
                    async with aiofiles.open(self.metadata_file, "r", encoding="utf-8") as file_handle:
                        metadata_content = await file_handle.read()
                    loaded_metadata = json.loads(metadata_content) if metadata_content else []

                    async with self._lock:
                        self.embeddings_list = embeddings.astype(np.float32).tolist()
                        self.metadata_list = loaded_metadata
                        self.metadata = list(loaded_metadata)
                    print(f"✅ Loaded numpy index with {len(self.metadata_list)} documents")
                else:
                    print("📝 Index files not found, creating new index")
                    await self.create_index()
        except Exception as e:
            print(f"❌ Error loading FAISS index: {str(e)}")
            raise

    async def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            if self.use_faiss and FAISS_AVAILABLE:
                async with self._lock:
                    if self.index is None:
                        print("⚠️  Index is empty, nothing to save")
                        return
                    index_to_write = self.index
                    metadata_to_write = list(self.metadata)

                await self._run_in_executor(faiss.write_index, index_to_write, self.index_file)
                async with aiofiles.open(self.metadata_file, "w", encoding="utf-8") as file_handle:
                    await file_handle.write(json.dumps(metadata_to_write, indent=2))
                print(f"✅ Saved FAISS index to {self.index_file}")
                print(f"✅ Saved metadata to {self.metadata_file}")
            else:
                async with self._lock:
                    if not self.embeddings_list:
                        print("⚠️  Index is empty, nothing to save")
                        return
                    embeddings_to_write = np.array(self.embeddings_list, dtype=np.float32)
                    metadata_to_write = list(self.metadata_list)

                await asyncio.to_thread(self._save_numpy_embeddings_sync, self.index_file, embeddings_to_write)
                async with aiofiles.open(self.metadata_file, "w", encoding="utf-8") as file_handle:
                    await file_handle.write(json.dumps(metadata_to_write, indent=2))
                print(f"✅ Saved numpy index to {self.index_file}")
                print(f"✅ Saved metadata to {self.metadata_file}")
        except Exception as e:
            print(f"❌ Error saving FAISS index: {str(e)}")
            raise

    async def add_documents(self, texts: List[str], metadata: List[Dict] = None) -> None:
        """
        Add documents to the FAISS index.

        Args:
            texts: List of text documents to add
            metadata: List of metadata dictionaries for each text
        """
        try:
            if not texts:
                print("⚠️  No texts provided")
                return

            async with self._lock:
                if self.use_faiss and FAISS_AVAILABLE:
                    if self.index is None:
                        self.index = await self._run_in_executor(self._create_index_sync)
                        self.metadata = []

                    print(f"🔄 Generating embeddings for {len(texts)} documents...")
                    embeddings = await self._run_in_executor(self.embedder.embed_texts, texts)
                    embeddings = np.array(embeddings).astype(np.float32)
                    await self._run_in_executor(self.index.add, embeddings)

                    if metadata is None:
                        metadata = [{"text": text} for text in texts]

                    self.metadata.extend(metadata)
                else:
                    print(f"🔄 Generating embeddings for {len(texts)} documents...")
                    embeddings = await self._run_in_executor(self.embedder.embed_texts, texts)
                    embeddings = np.array(embeddings).astype(np.float32)

                    if metadata is None:
                        metadata = [{"text": text} for text in texts]

                    self.embeddings_list.extend(embeddings.tolist())
                    self.metadata_list.extend(metadata)
                    self.metadata.extend(metadata)

            print(f"✅ Added {len(texts)} documents to FAISS index")
            await self.save_index()
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise

    async def replace_document(self, doc_id: str, texts: List[str], metadata: List[Dict]) -> None:
        """Replace all chunks for a document and rebuild the index."""
        try:
            if not doc_id:
                raise ValueError("doc_id is required to replace a document")

            async with self._lock:
                if self.use_faiss and FAISS_AVAILABLE:
                    remaining_metadata = [
                        item for item in self.metadata
                        if item.get("doc_id") != doc_id
                    ]

                    rebuilt_index = await self._run_in_executor(self._create_index_sync)

                    if remaining_metadata:
                        remaining_texts = [item.get("text", "") for item in remaining_metadata]
                        remaining_embeddings = await self._run_in_executor(self.embedder.embed_texts, remaining_texts)
                        remaining_embeddings = np.array(remaining_embeddings).astype(np.float32)
                        await self._run_in_executor(rebuilt_index.add, remaining_embeddings)

                    if texts:
                        new_embeddings = await self._run_in_executor(self.embedder.embed_texts, texts)
                        new_embeddings = np.array(new_embeddings).astype(np.float32)
                        await self._run_in_executor(rebuilt_index.add, new_embeddings)
                        remaining_metadata.extend(metadata)

                    self.index = rebuilt_index
                    self.metadata = remaining_metadata
                else:
                    remaining_metadata = [
                        item for item in self.metadata_list
                        if item.get("doc_id") != doc_id
                    ]

                    remaining_embeddings = []
                    if remaining_metadata:
                        remaining_texts = [item.get("text", "") for item in remaining_metadata]
                        remaining_embeddings = await self._run_in_executor(self.embedder.embed_texts, remaining_texts)
                        remaining_embeddings = np.array(remaining_embeddings).astype(np.float32).tolist()

                    new_embeddings = []
                    if texts:
                        new_embeddings = await self._run_in_executor(self.embedder.embed_texts, texts)
                        new_embeddings = np.array(new_embeddings).astype(np.float32).tolist()
                        remaining_metadata.extend(metadata)

                    self.embeddings_list = list(remaining_embeddings) + list(new_embeddings)
                    self.metadata_list = remaining_metadata
                    self.metadata = list(remaining_metadata)

            await self.save_index()
            print(f"✅ Replaced indexed chunks for document {doc_id}")
        except Exception as e:
            print(f"❌ Error replacing document {doc_id}: {str(e)}")
            raise

    async def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (text, distance, metadata) tuples
        """
        try:
            query_embedding = await self._run_in_executor(self.embedder.embed_text, query)
            query_embedding = np.array(query_embedding, dtype=np.float32)

            async with self._lock:
                if self.use_faiss and FAISS_AVAILABLE:
                    if self.index is None or len(self.metadata) == 0:
                        print("⚠️  Index is empty")
                        return []
                    metadata_snapshot = list(self.metadata)
                    distances, indices = await self._run_in_executor(
                        self.index.search,
                        np.array([query_embedding]).astype(np.float32),
                        k,
                    )

                    results = []
                    for i, idx in enumerate(indices[0]):
                        if idx < len(metadata_snapshot):
                            distance = float(distances[0][i])
                            similarity = 1 / (1 + distance)
                            results.append((
                                metadata_snapshot[idx].get("text", ""),
                                similarity,
                                metadata_snapshot[idx],
                            ))
                else:
                    if not self.embeddings_list or len(self.metadata_list) == 0:
                        print("⚠️  Index is empty")
                        return []

                    metadata_snapshot = list(self.metadata_list)
                    embeddings_matrix = np.array(self.embeddings_list, dtype=np.float32)
                    query_norm = np.linalg.norm(query_embedding)
                    if query_norm == 0.0:
                        return []
                    query_vector = query_embedding / query_norm

                    emb_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
                    emb_norms[emb_norms == 0.0] = 1.0
                    normalized_embeddings = embeddings_matrix / emb_norms

                    scores = np.dot(normalized_embeddings, query_vector)
                    top_k = min(k, len(scores))
                    top_indices = np.argsort(scores)[-top_k:][::-1]

                    results = []
                    for idx in top_indices:
                        similarity = float(scores[idx])
                        results.append((
                            metadata_snapshot[idx].get("text", ""),
                            similarity,
                            metadata_snapshot[idx],
                        ))

            print(f"✅ Found {len(results)} similar documents")
            return results
        except Exception as e:
            print(f"❌ Error searching index: {str(e)}")
            raise

    async def search_with_scores(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search and return results as dictionaries.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of result dictionaries with text, similarity, and metadata
        """
        try:
            results = await self.search(query, k)
            return [
                {
                    "text": text,
                    "similarity": score,
                    "metadata": meta,
                }
                for text, score, meta in results
            ]
        except Exception as e:
            print(f"❌ Error searching with scores: {str(e)}")
            raise

    async def get_statistics(self) -> Dict:
        """Get statistics about the index."""
        try:
            async with self._lock:
                index = self.index
                metadata_snapshot = list(self.metadata)
                embeddings_count = len(self.embeddings_list)

            if self.use_faiss and FAISS_AVAILABLE:
                if index is None:
                    return {
                        "total_documents": 0,
                        "embedding_dimension": self.embedding_dim,
                        "index_size": 0,
                    }
                index_size = index.ntotal
            else:
                if embeddings_count == 0:
                    return {
                        "total_documents": 0,
                        "embedding_dimension": self.embedding_dim,
                        "index_size": 0,
                    }
                index_size = embeddings_count

            files_on_disk = 0
            index_exists = await asyncio.to_thread(os.path.exists, self.index_file)
            if index_exists:
                files_on_disk = await asyncio.to_thread(os.path.getsize, self.index_file)

            return {
                "total_documents": len(metadata_snapshot),
                "embedding_dimension": self.embedding_dim,
                "index_size": index_size,
                "files_on_disk": files_on_disk,
            }
        except Exception as e:
            print(f"❌ Error getting statistics: {str(e)}")
            raise

    async def clear_index(self) -> None:
        """Clear the index."""
        try:
            await self.create_index()
            await self.save_index()
            print("✅ Index cleared")
        except Exception as e:
            print(f"❌ Error clearing index: {str(e)}")
            raise
