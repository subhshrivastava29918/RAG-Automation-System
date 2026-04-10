#!/usr/bin/env python3
"""
RAG Automation System - Interactive CLI
Usage: python cli.py [command] [options]
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

# Try to import Rich for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich import print as rprint
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

def print_header():
    if RICH:
        console.print(Panel.fit(
            "[bold cyan]🔍 RAG Automation System[/bold cyan]\n"
            "[dim]Retrieval-Augmented Generation | Powered by Claude + ChromaDB[/dim]",
            border_style="cyan"
        ))
    else:
        print("=" * 60)
        print("  RAG Automation System")
        print("  Retrieval-Augmented Generation")
        print("=" * 60)

def print_response(resp):
    if RICH:
        console.print()
        console.print(Panel(
            Markdown(resp.answer),
            title="[bold green]Answer[/bold green]",
            border_style="green"
        ))
        if resp.sources:
            table = Table(title="Sources", show_header=True)
            table.add_column("Rank", style="dim", width=5)
            table.add_column("Score", width=8)
            table.add_column("Source")
            table.add_column("Snippet")
            for i, src in enumerate(resp.sources):
                src_name = src.metadata.get("filename") or src.metadata.get("url") or src.doc_id[:12]
                snippet = src.content[:80].replace("\n", " ") + "..."
                table.add_row(
                    str(i+1),
                    f"{src.score:.3f}",
                    src_name,
                    snippet
                )
            console.print(table)
        console.print(f"[dim]Tokens used: {resp.tokens_used} | Model: {resp.model}[/dim]")
    else:
        print("\n--- ANSWER ---")
        print(resp.answer)
        print(f"\n--- SOURCES ({len(resp.sources)}) ---")
        for i, src in enumerate(resp.sources):
            src_name = src.metadata.get("filename") or src.metadata.get("url") or src.doc_id[:12]
            print(f"  [{i+1}] {src_name} (score: {src.score:.3f})")

def print_stats(stats):
    if RICH:
        table = Table(title="Knowledge Base Stats", show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")
        for k, v in stats.items():
            table.add_row(str(k).replace("_", " ").title(), str(v))
        console.print(table)
    else:
        print("\n--- Knowledge Base Stats ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")


# ─────────────────────────────────────────────────
# CLI Commands
# ─────────────────────────────────────────────────

def cmd_interactive(args):
    """Interactive chat REPL."""
    from core import RAGPipeline, RAGConfig
    config = RAGConfig(
        persist_dir=args.db,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        embedding_model=args.embedding,
    )
    pipeline = RAGPipeline(config)

    print_header()
    stats = pipeline.stats()
    if RICH:
        console.print(f"[dim]Knowledge base: {stats['total_chunks']} chunks from {stats['total_documents']} documents[/dim]\n")
    else:
        print(f"Chunks: {stats['total_chunks']} | Documents: {stats['total_documents']}\n")

    if stats["total_chunks"] == 0:
        msg = "⚠️  Knowledge base is empty. Add documents first with: python cli.py add <path>"
        if RICH:
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            print(msg)

    use_history = not args.no_history
    if RICH:
        console.print(f"[dim]Conversation history: {'ON' if use_history else 'OFF'} | type 'exit' to quit, 'clear' to reset history[/dim]\n")

    while True:
        try:
            if RICH:
                question = Prompt.ask("[bold blue]You[/bold blue]")
            else:
                question = input("\nYou: ").strip()

            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                break
            if question.lower() == "clear":
                pipeline.clear_history()
                print("History cleared.")
                continue
            if question.lower() == "stats":
                print_stats(pipeline.stats())
                continue

            resp = pipeline.chat(question) if use_history else pipeline.query(question)
            print_response(resp)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def cmd_add(args):
    """Add documents to the knowledge base."""
    from core import RAGPipeline, RAGConfig
    config = RAGConfig(
        persist_dir=args.db,
        chunk_size=args.chunk_size,
        embedding_model=args.embedding,
    )
    pipeline = RAGPipeline(config)

    total = 0
    for source in args.sources:
        p = Path(source)
        if p.is_dir():
            n = pipeline.add_directory(str(p))
        elif p.is_file():
            n = pipeline.add_file(str(p))
        elif source.startswith("http"):
            n = pipeline.add_url(source)
        else:
            n = pipeline.add_text(source, source="cli_input")

        msg = f"Added {n} chunks from: {source}"
        if RICH:
            console.print(f"[green]✓[/green] {msg}")
        else:
            print(f"✓ {msg}")
        total += n

    print(f"\nTotal: {total} chunks added.")
    print_stats(pipeline.stats())


def cmd_query(args):
    """Single query (non-interactive)."""
    from core import RAGPipeline, RAGConfig
    config = RAGConfig(
        persist_dir=args.db,
        top_k=args.top_k,
        embedding_model=args.embedding,
    )
    pipeline = RAGPipeline(config)
    resp = pipeline.query(args.question)

    if args.json:
        print(json.dumps(resp.to_dict(), indent=2))
    else:
        print_response(resp)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(resp.to_dict(), f, indent=2)
        print(f"\nResult saved to {args.output}")


def cmd_batch(args):
    """Run batch queries from a file."""
    from core import RAGPipeline, RAGConfig, BatchProcessor
    config = RAGConfig(persist_dir=args.db, top_k=args.top_k, embedding_model=args.embedding)
    pipeline = RAGPipeline(config)
    processor = BatchProcessor(pipeline)

    results = processor.run_from_file(args.queries_file, output_file=args.output)
    if RICH:
        console.print(f"[green]✓ Processed {len(results)} queries[/green]")
        if args.output:
            console.print(f"[dim]Results saved to {args.output}[/dim]")
    else:
        print(f"Processed {len(results)} queries.")


def cmd_watch(args):
    """Watch a directory and auto-ingest new files."""
    from core import RAGPipeline, RAGConfig, FileWatcher
    config = RAGConfig(persist_dir=args.db, chunk_size=args.chunk_size, embedding_model=args.embedding)
    pipeline = RAGPipeline(config)

    watcher = FileWatcher(
        watch_dir=args.watch_dir,
        pipeline=pipeline,
        poll_interval=args.interval,
    )
    watcher.start()

    if RICH:
        console.print(f"[green]👁  Watching {args.watch_dir}[/green]  [dim](Ctrl+C to stop)[/dim]")
    else:
        print(f"Watching {args.watch_dir} ... (Ctrl+C to stop)")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        print("\nWatcher stopped.")


def cmd_stats(args):
    """Show knowledge base statistics."""
    from core import RAGPipeline, RAGConfig
    config = RAGConfig(persist_dir=args.db)
    pipeline = RAGPipeline(config)
    print_stats(pipeline.stats())


def cmd_reset(args):
    """Reset the knowledge base."""
    from core import RAGPipeline, RAGConfig
    if RICH:
        if not Confirm.ask("[red]⚠️  This will permanently delete all data. Continue?[/red]"):
            return
    else:
        confirm = input("⚠️  This will permanently delete all data. Type YES to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            return

    config = RAGConfig(persist_dir=args.db)
    pipeline = RAGPipeline(config)
    pipeline.reset_knowledge_base()
    print("✓ Knowledge base reset.")


def cmd_list_docs(args):
    """List all documents in the knowledge base."""
    from core import RAGPipeline, RAGConfig
    config = RAGConfig(persist_dir=args.db)
    pipeline = RAGPipeline(config)
    docs = pipeline.list_documents()
    if not docs:
        print("No documents found.")
        return
    if RICH:
        table = Table(title=f"Documents ({len(docs)})")
        table.add_column("#", style="dim")
        table.add_column("Document ID", style="cyan")
        for i, doc_id in enumerate(docs):
            table.add_row(str(i+1), doc_id)
        console.print(table)
    else:
        for i, doc_id in enumerate(docs):
            print(f"  [{i+1}] {doc_id}")


# ─────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG Automation System - Retrieval-Augmented Generation",
    )
    parser.add_argument("--db", default="./rag_db", help="Vector store directory")
    parser.add_argument("--verbose", "-v", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # chat (default interactive)
    p_chat = sub.add_parser("chat", help="Interactive Q&A chat")
    p_chat.add_argument("--top-k", type=int, default=5)
    p_chat.add_argument("--chunk-size", type=int, default=512)
    p_chat.add_argument("--embedding", default="fast", choices=["fast", "balanced", "multilingual"])
    p_chat.add_argument("--no-history", action="store_true", help="Disable conversation history")

    # add
    p_add = sub.add_parser("add", help="Add documents to knowledge base")
    p_add.add_argument("sources", nargs="+", help="Files, directories, URLs, or text strings")
    p_add.add_argument("--chunk-size", type=int, default=512)
    p_add.add_argument("--embedding", default="fast", choices=["fast", "balanced", "multilingual"])

    # query
    p_query = sub.add_parser("query", help="Single query (non-interactive)")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--embedding", default="fast")
    p_query.add_argument("--json", action="store_true", help="Output as JSON")
    p_query.add_argument("--output", "-o", help="Save result to file")

    # batch
    p_batch = sub.add_parser("batch", help="Run queries from a file")
    p_batch.add_argument("queries_file", help=".txt or .json file with queries")
    p_batch.add_argument("--top-k", type=int, default=5)
    p_batch.add_argument("--output", "-o", default="batch_results.json")
    p_batch.add_argument("--embedding", default="fast")

    # watch
    p_watch = sub.add_parser("watch", help="Auto-ingest files dropped in a folder")
    p_watch.add_argument("watch_dir", help="Directory to watch")
    p_watch.add_argument("--interval", type=float, default=2.0, help="Poll interval in seconds")
    p_watch.add_argument("--chunk-size", type=int, default=512)
    p_watch.add_argument("--embedding", default="fast")

    # stats
    sub.add_parser("stats", help="Show knowledge base statistics")

    # reset
    sub.add_parser("reset", help="⚠️  Reset the knowledge base")

    # list
    sub.add_parser("list", help="List all documents")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=getattr(args, "verbose", False))

    handlers = {
        "chat": cmd_interactive,
        "add": cmd_add,
        "query": cmd_query,
        "batch": cmd_batch,
        "watch": cmd_watch,
        "stats": cmd_stats,
        "reset": cmd_reset,
        "list": cmd_list_docs,
    }

    if args.command is None:
        # Default to interactive mode
        args.command = "chat"
        args.top_k = 5
        args.chunk_size = 512
        args.embedding = "fast"
        args.no_history = False

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
