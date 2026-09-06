import logging
import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingService(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = settings.EMBEDDING_DIMENSION

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            elif hasattr(self._model, "get_sentence_embedding_dimension"):
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # BGE models benefit from instruction prefix on query if applicable,
        # but bge-small-en works directly with standard encoding or "Represent this sentence for searching relevant passages:"
        prefix = "Represent this sentence for searching relevant passages: " if "bge" in self.model_name.lower() else ""
        query_text = f"{prefix}{text}"
        embedding = self.model.encode(
            query_text,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()


class OpenAIEmbeddingService(BaseEmbeddingService):
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or settings.OPENAI_API_KEY
        self._client = None
        self._dimension = 1536 if "3-small" in model_name or "ada" in model_name else 3072

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name,
        )
        return response.data[0].embedding


class MockEmbeddingService(BaseEmbeddingService):
    """
    Deterministic pseudo-embedding for testing and offline environments.
    Uses SHA-256 seed to produce normalized unit vector.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _generate_vector(self, text: str) -> List[float]:
        # Generate deterministic vector based on text tokens and hash
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)


def get_embedding_service() -> BaseEmbeddingService:
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "openai" and settings.OPENAI_API_KEY:
        return OpenAIEmbeddingService()
    elif provider == "sentence_transformers":
        try:
            return SentenceTransformerEmbeddingService(
                model_name=settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize SentenceTransformer ({e}). Falling back to MockEmbeddingService.")
            return MockEmbeddingService(dimension=settings.EMBEDDING_DIMENSION)
    else:
        return MockEmbeddingService(dimension=settings.EMBEDDING_DIMENSION)
