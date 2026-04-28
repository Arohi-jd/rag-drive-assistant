"""
Embedder using the Hugging Face Inference API for feature extraction.
"""

import os
from typing import List

import numpy as np
import requests
from dotenv import load_dotenv


load_dotenv()


class Embedder:
    """Generate embeddings for text using the Hugging Face Inference API."""

    ENDPOINT = (
        "https://api-inference.huggingface.co/pipeline/feature-extraction/"
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY is not configured in .env")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._embedding_dim = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts and return a numpy array."""
        if not texts:
            return np.array([], dtype=np.float32)

        response = requests.post(
            self.ENDPOINT,
            headers=self.headers,
            json={"inputs": texts},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Hugging Face API error {response.status_code}: {response.text}"
            )

        payload = response.json()
        embeddings = np.array(payload, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if self._embedding_dim is None and embeddings.size:
            self._embedding_dim = int(embeddings.shape[-1])
        return embeddings

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text and return the first result."""
        embeddings = self.embed([text])
        return embeddings[0]

    def embed_text(self, text: str) -> np.ndarray:
        """Backwards-compatible single-text embed wrapper."""
        return self.embed_one(text)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Backwards-compatible multi-text embed wrapper."""
        return self.embed(texts)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self._embedding_dim is None:
            raise RuntimeError("Embedding dimension is unknown until embed() is called")
        return self._embedding_dim

    def similarity_score(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
