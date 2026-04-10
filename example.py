#!/usr/bin/env python3
"""
RAG System - Example Usage & Demo
Run this to see the full pipeline in action without the CLI.
"""

import os
import sys
import json

# Add parent dir to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import RAGPipeline, RAGConfig, BatchProcessor, FileWatcher

def demo_basic():
    """Basic RAG pipeline demo."""
    print("\n" + "="*60)
    print("  DEMO: Basic RAG Pipeline")
    print("="*60)

    # Configure the pipeline
    config = RAGConfig(
        chunk_size=256,
        chunk_overlap=32,
        chunk_strategy="sentence_aware",
        embedding_model="fast",
        top_k=3,
        score_threshold=0.2,
        persist_dir="./rag_demo_db",
    )

    pipeline = RAGPipeline(config)

    # ── Add documents ──
    print("\n[1] Adding documents to knowledge base...")

    pipeline.add_text(
        """Artificial Intelligence (AI) is the simulation of human intelligence processes by 
        computer systems. Key applications include natural language processing, image recognition, 
        autonomous vehicles, and medical diagnosis. Machine learning is a subset of AI that allows 
        systems to learn from data without being explicitly programmed.""",
        source="ai_overview",
        metadata={"topic": "AI", "category": "technology"},
    )

    pipeline.add_text(
        """Python is a high-level, interpreted programming language known for its simplicity and 
        readability. It was created by Guido van Rossum and first released in 1991. Python supports 
        multiple programming paradigms including procedural, object-oriented, and functional. 
        It is widely used in data science, web development, automation, and AI research.""",
        source="python_overview",
        metadata={"topic": "Python", "category": "programming"},
    )

    pipeline.add_text(
        """Retrieval-Augmented Generation (RAG) is an AI technique that combines information 
        retrieval with text generation. Instead of relying solely on a model's training data, 
        RAG retrieves relevant documents from an external knowledge base and uses them as context 
        for generating answers. This reduces hallucinations and enables up-to-date responses.""",
        source="rag_overview",
        metadata={"topic": "RAG", "category": "AI"},
    )

    pipeline.add_text(
        """ChromaDB is an open-source vector database designed for AI applications. It stores 
        embeddings alongside metadata and enables fast similarity search. ChromaDB supports 
        persistent storage, filtering by metadata, and integration with popular ML frameworks. 
        It can run locally or as a distributed service.""",
        source="chromadb_overview",
        metadata={"topic": "ChromaDB", "category": "database"},
    )

    stats = pipeline.stats()
    print(f"   ✓ Total chunks in KB: {stats['total_chunks']}")
    print(f"   ✓ Total documents: {stats['total_documents']}")

    # ── Single query ──
    print("\n[2] Single Query Example:")
    resp = pipeline.query("What is RAG and how does it work?")
    print(f"   Q: {resp.query}")
    print(f"   A: {resp.answer[:300]}...")
    print(f"   Sources: {len(resp.sources)} | Tokens: {resp.tokens_used}")

    # ── Filtered query ──
    print("\n[3] Filtered Query (category=AI):")
    resp2 = pipeline.query(
        "Tell me about machine learning",
        where={"category": "AI"},
    )
    print(f"   Q: {resp2.query}")
    print(f"   A: {resp2.answer[:300]}...")

    # ── Multi-turn chat ──
    print("\n[4] Multi-turn Conversation:")
    q1 = pipeline.chat("What programming language is popular for AI?")
    print(f"   Q1: What programming language is popular for AI?")
    print(f"   A1: {q1.answer[:200]}...")

    q2 = pipeline.chat("What was the first version released and when?")
    print(f"   Q2: What was the first version released and when?")
    print(f"   A2: {q2.answer[:200]}...")
    pipeline.clear_history()

    # ── Retrieve only (no generation) ──
    print("\n[5] Retrieve-Only (no LLM):")
    chunks = pipeline.retrieve("vector database embeddings", top_k=2)
    for i, c in enumerate(chunks):
        print(f"   [{i+1}] score={c.score:.3f} | {c.content[:100]}...")

    # ── Stats ──
    print("\n[6] Knowledge Base Stats:")
    for k, v in pipeline.stats().items():
        print(f"   {k}: {v}")

    # Cleanup
    pipeline.reset_knowledge_base()
    print("\n✓ Demo complete. Knowledge base cleaned up.")


def demo_batch():
    """Batch processing demo."""
    print("\n" + "="*60)
    print("  DEMO: Batch Processing")
    print("="*60)

    config = RAGConfig(persist_dir="./rag_batch_db", score_threshold=0.1)
    pipeline = RAGPipeline(config)

    # Add data
    for item in [
        ("The Eiffel Tower is located in Paris, France. It was built in 1889.", "eiffel"),
        ("Mount Everest is the highest mountain on Earth at 8,849 meters.", "everest"),
        ("The Amazon River is the largest river by discharge in the world.", "amazon"),
    ]:
        pipeline.add_text(item[0], source=item[1])

    # Batch queries
    queries = [
        "Where is the Eiffel Tower?",
        "What is the height of Mount Everest?",
        "Which is the largest river?",
    ]

    processor = BatchProcessor(pipeline)
    results = processor.run_queries(queries, output_file="batch_output.json")

    print(f"\n   Processed {len(results)} queries:")
    for r in results:
        ans = r.get("answer", r.get("error", ""))[:100]
        print(f"   Q: {r['query'][:50]}")
        print(f"   A: {ans}...")
        print()

    pipeline.reset_knowledge_base()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set.")
        print("   Set it with: export ANTHROPIC_API_KEY=your_key_here")
        print("   Running in retrieve-only mode (no LLM generation)\n")

    try:
        demo_basic()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
