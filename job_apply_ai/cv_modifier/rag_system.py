"""RAG system for indexing and retrieving CV content chunks."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class CVRAGSystem:
    """In-memory RAG index for a single CV document."""

    def __init__(
        self,
        chunking_strategy: str = "document_aware",
        chunk_size: int = 300,
    ):
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.embedding_dim = 384
        self.documents: list[dict[str, Any]] = []
        self.embeddings: list[np.ndarray] = []

        if HAS_SENTENCE_TRANSFORMERS:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("RAG using sentence-transformers (all-MiniLM-L6-v2)")
        else:
            self.embedder = None
            logger.warning("sentence-transformers unavailable; using normalized hash embeddings")

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.use_faiss = True
        else:
            self.index = None
            self.use_faiss = False
            logger.warning("FAISS unavailable; using pure Python similarity search")

    def clear(self) -> None:
        """Remove all indexed chunks."""
        self.documents = []
        self.embeddings = []
        if self.use_faiss and self.index is not None:
            self.index.reset()

    def index_cv(self, cv_text: str, source_name: str = "cv") -> int:
        """Chunk and index CV text. Returns number of chunks indexed."""
        self.clear()
        if not cv_text or not cv_text.strip():
            return 0

        chunks = self._split_text_into_chunks(cv_text)
        for index, chunk in enumerate(chunks, start=1):
            chunk_text = f"CV: {source_name} SECTION: {index} CONTENT: {chunk}"
            self._add_document(chunk_text, chunk, index)

        logger.info("Indexed %s CV chunks using %s chunking", len(chunks), self.chunking_strategy)
        return len(chunks)

    def search(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        """Return top-k CV chunks relevant to the query."""
        if not self.documents:
            return []

        query_embedding = self._encode(query)
        if self.use_faiss and self.index is not None:
            distances, indices = self.index.search(
                query_embedding.astype("float32"),
                min(k * 3, len(self.documents)),
            )
            results: list[dict[str, Any]] = []
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                if score <= 0.05:
                    continue
                results.append(
                    {
                        **self.documents[idx],
                        "score": float(score),
                    }
                )
            results.sort(key=lambda item: item["score"], reverse=True)
            return results[:k]

        from sklearn.metrics.pairwise import cosine_similarity

        matrix = np.vstack(self.embeddings)
        similarities = cosine_similarity(query_embedding, matrix)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        for idx in top_indices:
            if similarities[idx] <= 0.05:
                continue
            results.append(
                {
                    **self.documents[idx],
                    "score": float(similarities[idx]),
                }
            )
        return results

    def _add_document(self, indexed_text: str, original_chunk: str, chunk_index: int) -> None:
        embedding = self._encode(indexed_text)
        if self.use_faiss and self.index is not None:
            self.index.add(embedding.astype("float32"))
        else:
            self.embeddings.append(embedding[0])

        self.documents.append(
            {
                "text": indexed_text,
                "original_text": original_chunk,
                "chunk_index": chunk_index,
            }
        )

    def _encode(self, text: str) -> np.ndarray:
        if self.embedder:
            embedding = self.embedder.encode([text])
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            return (embedding / norm).astype("float32")

        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal((1, self.embedding_dim)).astype("float32")
        vector /= np.linalg.norm(vector)
        return vector

    def _split_text_into_chunks(self, text: str) -> list[str]:
        if self.chunking_strategy == "fixed":
            return self._fixed_size_chunking(text)
        if self.chunking_strategy == "sentence":
            return self._sentence_chunking(text)
        if self.chunking_strategy == "semantic":
            return self._semantic_chunking(text)
        if self.chunking_strategy == "document_aware":
            return self._document_aware_chunking(text)
        return self._recursive_chunking(text)

    def _fixed_size_chunking(self, text: str) -> list[str]:
        words = text.split()
        overlap = min(self.chunk_size // 10, 50)
        chunks: list[str] = []
        step = max(self.chunk_size - overlap, 1)
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + self.chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks or [text.strip()]

    def _sentence_chunking(self, text: str) -> list[str]:
        sentences = re.split(r"([.!?]+\s*)", text)
        reconstructed: list[str] = []
        for index in range(0, len(sentences) - 1, 2):
            sentence = (sentences[index] + sentences[index + 1]).strip()
            if sentence:
                reconstructed.append(sentence)
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            reconstructed.append(sentences[-1].strip())

        chunks: list[str] = []
        current = ""
        current_words = 0
        for sentence in reconstructed:
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= self.chunk_size or not current:
                current = f"{current} {sentence}".strip()
                current_words += sentence_words
            else:
                chunks.append(current)
                current = sentence
                current_words = sentence_words
        if current:
            chunks.append(current)
        return chunks or [text.strip()]

    def _recursive_chunking(self, text: str) -> list[str]:
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""]

        def split_chunk(value: str) -> list[str]:
            words = value.split()
            if len(words) <= self.chunk_size:
                return [value.strip()] if value.strip() else []

            for separator in separators:
                if separator == "":
                    return [
                        " ".join(words[index : index + self.chunk_size]).strip()
                        for index in range(0, len(words), self.chunk_size)
                        if " ".join(words[index : index + self.chunk_size]).strip()
                    ]

                parts = value.split(separator)
                if len(parts) <= 1:
                    continue

                combined: list[str] = []
                current = ""
                current_words = 0
                for part in parts:
                    part_words = len(part.split())
                    if current_words + part_words > self.chunk_size and current.strip():
                        combined.append(current.strip())
                        current = part
                        current_words = part_words
                    else:
                        current = separator.join(filter(None, [current, part]))
                        current_words += part_words
                if current.strip():
                    combined.append(current.strip())
                if len(combined) > 1:
                    return combined
            return [value.strip()]

        return split_chunk(text)

    def _semantic_chunking(self, text: str) -> list[str]:
        if not self.embedder:
            return self._recursive_chunking(text)

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return sentences or [text.strip()]

        sentence_embeddings = self.embedder.encode(sentences)
        chunks: list[str] = []
        current_chunk = sentences[0]
        current_embedding = sentence_embeddings[0]
        current_words = len(current_chunk.split())
        min_chunk_size = int(self.chunk_size * 0.4)

        for index in range(1, len(sentences)):
            next_embedding = sentence_embeddings[index]
            similarity = float(
                np.dot(current_embedding, next_embedding)
                / (np.linalg.norm(current_embedding) * np.linalg.norm(next_embedding) + 1e-8)
            )
            sentence_words = len(sentences[index].split())
            potential_words = current_words + sentence_words
            should_split = (
                potential_words > self.chunk_size
                or (similarity < 0.15 and current_words >= min_chunk_size)
            )
            if should_split:
                chunks.append(current_chunk.strip())
                current_chunk = sentences[index]
                current_words = sentence_words
                current_embedding = next_embedding
            else:
                current_chunk = f"{current_chunk}. {sentences[index]}".strip()
                current_words = potential_words
                current_embedding = (current_embedding + next_embedding) / 2

        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def _document_aware_chunking(self, text: str) -> list[str]:
        if not self.embedder:
            return self._recursive_chunking(text)

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return sentences or [text.strip()]

        sentence_embeddings = self.embedder.encode(sentences)
        chunks: list[str] = []
        current_chunk = sentences[0]
        current_embedding = sentence_embeddings[0]
        current_words = len(current_chunk.split())
        min_chunk_size = int(self.chunk_size * 0.4)
        preferred_min_size = int(self.chunk_size * 0.5)

        for index in range(1, len(sentences)):
            next_embedding = sentence_embeddings[index]
            similarity = float(
                np.dot(current_embedding, next_embedding)
                / (np.linalg.norm(current_embedding) * np.linalg.norm(next_embedding) + 1e-8)
            )
            sentence_words = len(sentences[index].split())
            potential_words = current_words + sentence_words
            should_split = (
                potential_words > self.chunk_size
                or (similarity < 0.18 and current_words >= preferred_min_size)
                or (similarity < 0.12 and current_words >= min_chunk_size)
            )
            if should_split:
                chunks.append(current_chunk.strip())
                current_chunk = sentences[index]
                current_words = sentence_words
                current_embedding = next_embedding
            else:
                separator = ". " if not current_chunk.rstrip().endswith((".", "!", "?")) else " "
                current_chunk = f"{current_chunk}{separator}{sentences[index]}".strip()
                current_words = potential_words
                current_embedding = (current_embedding + next_embedding) / 2

        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        protected = text
        placeholders: dict[str, str] = {}
        url_pattern = r"\b\w+\.(com|org|net|edu|gov|ir|io|ai|co)\b"
        for index, match in enumerate(re.finditer(url_pattern, text, re.IGNORECASE)):
            placeholder = f"URLPLACEHOLDER{index}"
            placeholders[placeholder] = match.group(0)
            protected = protected.replace(match.group(0), placeholder)

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
        restored: list[str] = []
        for sentence in sentences:
            for placeholder, url in placeholders.items():
                sentence = sentence.replace(placeholder, url)
            if sentence.strip():
                restored.append(sentence.strip())
        return restored
