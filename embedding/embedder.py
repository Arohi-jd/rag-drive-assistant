"""
Embedder using sentence-transformers for generating embeddings.
Uses the all-MiniLM-L6-v2 model.
"""

import os
import json
import hashlib
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class Embedder:
    """Generate embeddings for text using sentence-transformers."""
    
    MODEL_NAME = "all-MiniLM-L6-v2"
    CACHE_FILE = "embedding_cache.json"
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the embedder with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        try:
            print(f"🔄 Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.cache = self._load_cache()
            print(f"✅ Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

    def _load_cache(self) -> dict:
        """Load embedding cache from disk."""
        try:
            if not os.path.exists(self.CACHE_FILE):
                return {}

            with open(self.CACHE_FILE, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"❌ Error loading embedding cache: {str(e)}")
            raise

    def _save_cache(self) -> None:
        """Persist embedding cache to disk."""
        try:
            with open(self.CACHE_FILE, "w", encoding="utf-8") as file_handle:
                json.dump(self.cache, file_handle)
        except Exception as e:
            print(f"❌ Error saving embedding cache: {str(e)}")
            raise

    def _text_hash(self, text: str) -> str:
        """Create a stable hash for exact chunk text cache lookups."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
        """
        try:
            text_hash = self._text_hash(text)
            cached_embedding = self.cache.get(text_hash)
            if cached_embedding is not None:
                return np.array(cached_embedding, dtype=np.float32)

            embedding = self.model.encode(text, convert_to_numpy=True)
            self.cache[text_hash] = embedding.tolist()
            self._save_cache()
            return embedding
        except Exception as e:
            print(f"❌ Error embedding text: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Embeddings as numpy array of shape (len(texts), embedding_dim)
        """
        try:
            if not texts:
                print("⚠️  Empty text list provided")
                return np.array([])

            cached_count = 0
            fresh_count = 0
            uncached_indices = []
            embeddings: List[Union[np.ndarray, None]] = [None] * len(texts)

            for index, text in enumerate(texts):
                text_hash = self._text_hash(text)
                cached_embedding = self.cache.get(text_hash)
                if cached_embedding is not None:
                    embeddings[index] = np.array(cached_embedding, dtype=np.float32)
                    cached_count += 1
                else:
                    uncached_indices.append(index)

            if uncached_indices:
                uncached_texts = [texts[index] for index in uncached_indices]
                new_embeddings = self.model.encode(
                    uncached_texts,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    show_progress_bar=True
                )

                for index, embedding in zip(uncached_indices, new_embeddings):
                    text_hash = self._text_hash(texts[index])
                    embeddings[index] = embedding
                    self.cache[text_hash] = embedding.tolist()
                    fresh_count += 1

                self._save_cache()

            final_embeddings = np.array(embeddings, dtype=np.float32)
            print(
                f"✅ Generated embeddings for {len(texts)} texts "
                f"({cached_count} from cache, {fresh_count} freshly embedded)"
            )
            return final_embeddings
        except Exception as e:
            print(f"❌ Error embedding texts: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        try:
            return self.embedding_dim
        except Exception as e:
            print(f"❌ Error getting embedding dimension: {str(e)}")
            raise
    
    def similarity_score(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between -1 and 1
        """
        try:
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            print(f"❌ Error calculating similarity: {str(e)}")
            raise
