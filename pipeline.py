"""
RAG Pipeline Orchestrator
Ties together: ingestion → chunking → embedding → storage → retrieval → generation
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from .engine import Document, Chunk, SearchResult, TextChunker, EmbeddingModel, VectorStore
from .ingestion import DocumentIngester

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_strategy: str = "sentence_aware"  # fixed | sentence_aware | paragraph

    # Embedding
    embedding_model: str = "fast"  # fast | balanced | multilingual

    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.3
    rerank: bool = True

    # Generation (Anthropic Claude)
    llm_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0
    system_prompt: str = (
        "You are a helpful, precise assistant. Answer questions based ONLY on the "
        "provided context. If the answer is not in the context, say so clearly. "
        "Always cite the source document when relevant."
    )

    # Storage
    persist_dir: str = "./rag_db"
    collection_name: str = "rag_main"


@dataclass
class RAGResponse:
    """Full response from the RAG pipeline."""
    query: str
    answer: str
    sources: List[SearchResult] = field(default_factory=list)
    context_used: str = ""
    tokens_used: int = 0
    model: str = ""

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [
                {
                    "doc_id": s.doc_id,
                    "score": round(s.score, 4),
                    "snippet": s.content[:200] + "...",
                    "metadata": s.metadata,
                }
                for s in self.sources
            ],
            "tokens_used": self.tokens_used,
            "model": self.model,
        }


class ContextBuilder:
    """Builds the context string from retrieved chunks."""

    def build(
        self,
        results: List[SearchResult],
        max_tokens: int = 3000,
        include_scores: bool = True,
    ) -> str:
        """Build a structured context block from search results."""
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # rough estimate

        for i, result in enumerate(results):
            header = f"[Source {i+1}"
            if result.metadata.get("filename"):
                header += f" | {result.metadata['filename']}"
            elif result.metadata.get("url"):
                header += f" | {result.metadata['url']}"
            if include_scores:
                header += f" | Relevance: {result.score:.2f}"
            header += "]"

            chunk_text = f"{header}\n{result.content}"
            if total_chars + len(chunk_text) > max_chars:
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)


class SimpleReranker:
    """
    Lightweight reranker using keyword overlap + MMR (Maximal Marginal Relevance).
    For production use, replace with cross-encoder model.
    """

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        query_words = set(query.lower().split())

        def score(r: SearchResult) -> float:
            doc_words = set(r.content.lower().split())
            keyword_overlap = len(query_words & doc_words) / max(len(query_words), 1)
            return r.score * 0.7 + keyword_overlap * 0.3

        return sorted(results, key=score, reverse=True)


class RAGPipeline:
    """
    Full RAG pipeline with:
    - Multi-format document ingestion
    - Intelligent chunking
    - Local embeddings
    - Persistent vector store (ChromaDB)
    - Retrieval with optional reranking
    - Generation via Claude API
    - Conversation memory
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()

        # Initialize components
        self.ingester = DocumentIngester()
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy=self.config.chunk_strategy,
        )
        self.embedder = EmbeddingModel(model_name=self.config.embedding_model)
        self.vector_store = VectorStore(
            persist_dir=self.config.persist_dir,
            collection_name=self.config.collection_name,
        )
        self.context_builder = ContextBuilder()
        self.reranker = SimpleReranker() if self.config.rerank else None
        self._conversation_history: List[Dict] = []

        logger.info("RAG Pipeline initialized successfully.")

    # ──────────────────────────────────────────────
    # Ingestion Methods
    # ──────────────────────────────────────────────

    def add_text(self, text: str, source: str = "manual", metadata: Optional[Dict] = None) -> int:
        """Add raw text to the knowledge base."""
        doc = self.ingester.ingest_text(text, source=source, metadata=metadata)
        if not doc:
            return 0
        return self._process_document(doc)

    def add_file(self, path: str, metadata: Optional[Dict] = None) -> int:
        """Add a file (txt, md, pdf, docx) to the knowledge base."""
        doc = self.ingester.ingest_file(path, metadata=metadata)
        if not doc:
            return 0
        return self._process_document(doc)

    def add_directory(self, dir_path: str, recursive: bool = True, extensions: Optional[List[str]] = None) -> int:
        """Add all supported files from a directory."""
        docs = self.ingester.ingest_directory(dir_path, recursive=recursive, extensions=extensions)
        total = 0
        for doc in docs:
            total += self._process_document(doc)
        logger.info(f"Added {total} chunks from {len(docs)} files in {dir_path}")
        return total

    def add_url(self, url: str, metadata: Optional[Dict] = None) -> int:
        """Scrape a URL and add its content."""
        doc = self.ingester.ingest_url(url, metadata=metadata)
        if not doc:
            return 0
        return self._process_document(doc)

    def add_json_list(self, data: List[Dict], text_field: str = "text", metadata_fields: Optional[List[str]] = None) -> int:
        """Add documents from a list of dicts."""
        docs = self.ingester.ingest_json_list(data, text_field=text_field, metadata_fields=metadata_fields)
        total = sum(self._process_document(doc) for doc in docs)
        return total

    def _process_document(self, doc: Document) -> int:
        """Internal: chunk → embed → store a document."""
        chunks = self.chunker.chunk(doc.content, doc_id=doc.doc_id)
        if not chunks:
            return 0

        # Attach document metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
            chunk.metadata["source"] = doc.source
            chunk.metadata["doc_type"] = doc.doc_type

        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts)
        added = self.vector_store.add_chunks(chunks, embeddings)
        logger.info(f"Processed doc_id={doc.doc_id}: {added} chunks stored")
        return added

    # ──────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        where: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Retrieve relevant chunks for a query."""
        k = top_k or self.config.top_k
        threshold = score_threshold or self.config.score_threshold

        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=k * 2, where=where)  # Over-fetch for rerank

        # Filter by score threshold
        results = [r for r in results if r.score >= threshold]

        # Rerank
        if self.reranker and results:
            results = self.reranker.rerank(query, results)

        return results[:k]

    # ──────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────

    def _call_llm(self, messages: List[Dict], system: str) -> Tuple[str, int]:
        """Call Anthropic Claude API."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                system=system,
                messages=messages,
            )
            answer = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return answer, tokens
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[Error calling LLM: {e}]", 0

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        where: Optional[Dict] = None,
        use_history: bool = False,
    ) -> RAGResponse:
        """
        Full RAG query: retrieve context → build prompt → generate answer.
        """
        # Retrieve
        sources = self.retrieve(question, top_k=top_k, score_threshold=score_threshold, where=where)
        context = self.context_builder.build(sources, max_tokens=3000)

        # Build messages
        user_content = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"Please answer based on the context above."
        ) if context else (
            f"QUESTION: {question}\n\n"
            f"(No relevant context was found in the knowledge base.)"
        )

        messages = []
        if use_history:
            messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": user_content})

        answer, tokens = self._call_llm(messages, self.config.system_prompt)

        # Update conversation history
        if use_history:
            self._conversation_history.append({"role": "user", "content": user_content})
            self._conversation_history.append({"role": "assistant", "content": answer})
            # Trim history to last 10 turns
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            context_used=context,
            tokens_used=tokens,
            model=self.config.llm_model,
        )

    def chat(self, question: str, **kwargs) -> RAGResponse:
        """Query with conversation history enabled."""
        return self.query(question, use_history=True, **kwargs)

    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history = []

    # ──────────────────────────────────────────────
    # Knowledge Base Management
    # ──────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> int:
        """Remove a document from the knowledge base."""
        return self.vector_store.delete_by_doc_id(doc_id)

    def list_documents(self) -> List[str]:
        """List all document IDs in the knowledge base."""
        return self.vector_store.get_all_doc_ids()

    def stats(self) -> Dict[str, Any]:
        """Return knowledge base statistics."""
        return {
            "total_chunks": self.vector_store.count(),
            "total_documents": len(self.list_documents()),
            "embedding_model": self.embedder.model_name,
            "llm_model": self.config.llm_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "chunk_strategy": self.config.chunk_strategy,
            "persist_dir": self.config.persist_dir,
        }

    def reset_knowledge_base(self):
        """⚠️ Permanently delete all documents and chunks."""
        self.vector_store.reset()
        self.ingester._seen_hashes.clear()
        logger.warning("Knowledge base has been reset.")
