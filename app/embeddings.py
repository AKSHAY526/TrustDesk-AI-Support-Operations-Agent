"""Embeddings for the semantic half of hybrid retrieval."""

from __future__ import annotations

import re
import zlib
from typing import Protocol, Sequence

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that turns text into L2-normalised vectors."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Stable hashing vectoriser over words and character trigrams."""

    name = "hashing-ngram"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        lowered = text.lower()
        words = _WORD_RE.findall(lowered)
        features: list[str] = [f"w:{w}" for w in words]
        # Word bigrams capture short phrases such as "final sale".
        features += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        # Character trigrams give tolerance to word-form differences.
        joined = " ".join(words)
        features += [f"c:{joined[i:i + 3]}" for i in range(max(0, len(joined) - 2))]
        return features

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for feature in self._features(text):
                # crc32, not hash(): str hashes are randomised per process.
                bucket = zlib.crc32(feature.encode("utf-8")) % self.dim
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
            for bucket, count in counts.items():
                # Sublinear term frequency dampens very repetitive text.
                out[row, bucket] = 1.0 + np.log(count)
        return _l2_normalise(out)


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers."""

    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # heavy, imported lazily

        try:
            # Cache first: loading online retries for a long time behind a proxy.
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet
            self._model = SentenceTransformer(model_name)

        self.name = model_name
        try:
            self.dim = int(self._model.get_embedding_dimension())
        except AttributeError:  # older sentence-transformers releases
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


_cached: Embedder | None = None


def get_embedder(prefer_real: bool = True) -> Embedder:
    """Return the best available embedder, caching the choice."""
    global _cached
    if _cached is not None:
        return _cached
    if prefer_real:
        try:
            _cached = SentenceTransformerEmbedder()
            return _cached
        except Exception:  # noqa: BLE001 - any import/download failure falls back
            pass
    _cached = HashingEmbedder()
    return _cached


def set_embedder(embedder: Embedder | None) -> None:
    """Override the cached embedder. Used by tests and the loader."""
    global _cached
    _cached = embedder


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes | None, dim: int) -> np.ndarray:
    if not blob:
        return np.zeros(dim, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)
