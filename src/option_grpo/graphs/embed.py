"""Embedders for function identifiers (the map ``phi`` of Algorithm 1)."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from option_grpo.sdk.introspect import FunctionInfo
from option_grpo.utils import get_logger, split_identifier

log = get_logger(__name__)


def identifier_text(fn: FunctionInfo) -> str:
    """Natural-language rendering of a function identifier, e.g. ``"data frame where"``."""
    owner = fn.owner.rsplit(".", 1)[-1] if fn.owner and fn.kind != "constructor" else ""
    return " ".join(split_identifier(f"{owner} {fn.name}"))


class Embedder(ABC):
    """Maps texts to L2-normalised vectors."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape ``(len(texts), dim)`` with unit-norm rows."""

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (matrix / norms).astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    """Sentence-transformer embeddings (the configuration used in the paper)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is required for this embedder: pip install 'option-grpo[embed]' "
                "(or set graph.embedder=hashing)"
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self.model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=False)
        return self._normalise(np.asarray(vectors, dtype=np.float32))


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free embedder based on hashed word and character n-grams.

    Words contribute most of the mass, character trigrams provide soft matching between
    morphological variants (``read``/``reader``).  Intended for tests and offline use.
    """

    def __init__(self, dim: int = 384, word_weight: float = 1.0, ngram_weight: float = 0.35) -> None:
        self.dim = dim
        self.word_weight = word_weight
        self.ngram_weight = ngram_weight

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                index, sign = self._bucket(f"w:{word}")
                matrix[row, index] += sign * self.word_weight
                padded = f"^{word}$"
                for start in range(len(padded) - 2):
                    index, sign = self._bucket(f"g:{padded[start:start + 3]}")
                    matrix[row, index] += sign * self.ngram_weight
        return self._normalise(matrix)


def build_embedder(spec: str) -> Embedder:
    """``"hashing"`` or the name of a sentence-transformers model."""
    if spec in {"hashing", "hash"}:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(spec)
