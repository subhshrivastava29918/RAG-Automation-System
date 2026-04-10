"""
RAG Core Engine - Handles chunking, embedding, and vector store operations.
"""

import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the RAG system."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    doc_type: str = "text"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Chunk:
    """Represents a text chunk derived from a document."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    token_count: int = 0


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """
    Advanced text chunker with multiple strategies:
    - Fixed-size with overlap
    - Sentence-aware chunking
    - Semantic paragraph chunking
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: str = "sentence_aware",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        return len(text) // 4

    def chunk_fixed(self, text: str) -> List[str]:
        """Fixed character-size chunks with overlap."""
        char_size = self.chunk_size * 4
        overlap = self.chunk_overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            end = start + char_size
            chunks.append(text[start:end].strip())
            start += char_size - overlap
        return [c for c in chunks if c]

    def chunk_sentence_aware(self, text: str) -> List[str]:
        """Chunks that respect sentence boundaries."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = self.estimate_tokens(sentence)
            if current_tokens + s_tokens > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Overlap: keep last N sentences
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    t = self.estimate_tokens(s)
                    if overlap_tokens + t <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += t
                    else:
                        break
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
            current_chunk.append(sentence)
            current_tokens += s_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    def chunk_paragraph(self, text: str) -> List[str]:
        """Split by paragraphs, merge small ones."""
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        chunks = []
        current = []
        current_tokens = 0

        for para in paragraphs:
            p_tokens = self.estimate_tokens(para)
            if current_tokens + p_tokens > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            current.append(para)
            current_tokens += p_tokens

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        """Main chunking method - returns Chunk objects."""
        text = text.strip()
        if not text:
            return []

        if self.strategy == "fixed":
            raw_chunks = self.chunk_fixed(text)
        elif self.strategy == "paragraph":
            raw_chunks = self.chunk_paragraph(text)
        else:  # sentence_aware (default)
            raw_chunks = self.chunk_sentence_aware(text)

        chunks = []
        for i, content in enumerate(raw_chunks):
            chunk_id = hashlib.md5(f"{doc_id}_{i}_{content[:50]}".encode()).hexdigest()
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                chunk_index=i,
                token_count=self.estimate_tokens(content),
            ))
        return chunks


class TFIDFEmbedder:
    """
    Fallback TF-IDF based embedder when sentence-transformers model is unavailable.
    Uses a fixed vocabulary + hashing trick for fixed-size vectors.
    """
    DIM = 384

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.corpus: List[List[str]] = []
        import math
        self._math = math

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def _fit(self, texts: List[str]):
        import math
        N = len(texts)
        df: Dict[str, int] = {}
        tokenized = [self._tokenize(t) for t in texts]
        for tokens in tokenized:
            for w in set(tokens):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}
        all_words = sorted(self.idf.keys())
        self.vocab = {w: i % self.DIM for i, w in enumerate(all_words)}

    def _vectorize(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = [0.0] * self.DIM
        for word, count in tf.items():
            if word in self.vocab:
                idx = self.vocab[word]
                idf_val = self.idf.get(word, 1.0)
                vec[idx] += (count / max(len(tokens), 1)) * idf_val
        # L2 normalize
        norm = sum(x**2 for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.vocab:
            self._fit(texts)
        return [self._vectorize(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._vectorize(query)


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for local embeddings.
    Falls back to TF-IDF if the model cannot be downloaded.
    """

    MODELS = {
        "fast": "all-MiniLM-L6-v2",          # 22M params, fast
        "balanced": "all-mpnet-base-v2",       # 110M params, better quality
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
    }

    def __init__(self, model_name: str = "fast"):
        model_path = self.MODELS.get(model_name, model_name)
        logger.info(f"Loading embedding model: {model_path}")
        self._backend = None
        self.model_name = model_path
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_path)
            self._backend = "sentence_transformers"
            logger.info(f"Using sentence-transformers backend: {model_path}")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformers model ({e}). Falling back to TF-IDF.")
            self._tfidf = TFIDFEmbedder()
            self._backend = "tfidf"
            self.model_name = "tfidf-fallback"

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        if not texts:
            return []
        if self._backend == "sentence_transformers":
            embeddings = self._st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            return self._tfidf.embed(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        if self._backend == "sentence_transformers":
            return self.embed([query])[0]
        else:
            return self._tfidf.embed_query(query)


class VectorStore:
    """
    ChromaDB-backed vector store with full CRUD operations,
    metadata filtering, and hybrid search support.
    """

    def __init__(self, persist_dir: str = "./rag_db", collection_name: str = "rag_collection"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"VectorStore initialized at {persist_dir} | collection={collection_name}")

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """Add chunks with their embeddings to the store."""
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {**c.metadata}
            meta["doc_id"] = c.doc_id
            meta["chunk_index"] = c.chunk_index
            meta["token_count"] = c.token_count
            metadatas.append(meta)

        # Upsert to avoid duplicates
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Added/updated {len(chunks)} chunks in vector store.")
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Semantic search against the vector store."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                score = 1.0 - results["distances"][0][i]  # cosine similarity
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    content=results["documents"][0][i],
                    score=score,
                    metadata=meta,
                ))
        return search_results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document."""
        results = self.collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for doc_id={doc_id}")
            return len(results["ids"])
        return 0

    def get_all_doc_ids(self) -> List[str]:
        """Return unique document IDs in the store."""
        results = self.collection.get(include=["metadatas"])
        doc_ids = set()
        for meta in results.get("metadatas", []):
            if meta and "doc_id" in meta:
                doc_ids.add(meta["doc_id"])
        return list(doc_ids)

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vector store reset.")
