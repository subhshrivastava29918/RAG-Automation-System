"""
RAG Automation Module
- File watcher (auto-ingest new files dropped in a folder)
- Batch processing
- Export/import knowledge base
- Query automation (run queries from file)
"""

import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class FileWatcher:
    """
    Watches a directory and automatically ingests new files into the RAG pipeline.
    """

    def __init__(
        self,
        watch_dir: str,
        pipeline,
        extensions: Optional[List[str]] = None,
        poll_interval: float = 2.0,
        recursive: bool = False,
    ):
        self.watch_dir = Path(watch_dir)
        self.pipeline = pipeline
        self.extensions = {e.lower() for e in (extensions or [".txt", ".md", ".pdf", ".docx"])}
        self.poll_interval = poll_interval
        self.recursive = recursive
        self._stop_event = threading.Event()
        self._seen_files: set = set()
        self._thread: Optional[threading.Thread] = None

        self.watch_dir.mkdir(parents=True, exist_ok=True)
        # Seed known files so we don't re-ingest existing ones
        self._seed_existing()

    def _seed_existing(self):
        pattern = "**/*" if self.recursive else "*"
        for p in self.watch_dir.glob(pattern):
            if p.is_file() and p.suffix.lower() in self.extensions:
                self._seen_files.add(str(p.resolve()))

    def _scan(self):
        pattern = "**/*" if self.recursive else "*"
        for p in self.watch_dir.glob(pattern):
            if not p.is_file():
                continue
            if p.suffix.lower() not in self.extensions:
                continue
            abs_path = str(p.resolve())
            if abs_path not in self._seen_files:
                self._seen_files.add(abs_path)
                logger.info(f"[FileWatcher] New file detected: {p.name}")
                try:
                    n = self.pipeline.add_file(abs_path)
                    logger.info(f"[FileWatcher] Ingested {n} chunks from {p.name}")
                except Exception as e:
                    logger.error(f"[FileWatcher] Failed to ingest {p.name}: {e}")

    def _run(self):
        logger.info(f"[FileWatcher] Watching {self.watch_dir} every {self.poll_interval}s")
        while not self._stop_event.is_set():
            self._scan()
            time.sleep(self.poll_interval)

    def start(self):
        """Start watching in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[FileWatcher] Started.")

    def stop(self):
        """Stop the watcher."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[FileWatcher] Stopped.")


class BatchProcessor:
    """Process multiple queries in bulk and save results."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run_queries(
        self,
        queries: List[str],
        output_file: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        results = []
        for i, query in enumerate(queries):
            logger.info(f"[Batch] Query {i+1}/{len(queries)}: {query[:60]}...")
            try:
                resp = self.pipeline.query(query, top_k=top_k)
                results.append(resp.to_dict())
            except Exception as e:
                results.append({"query": query, "error": str(e)})

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"[Batch] Saved {len(results)} results to {output_file}")

        return results

    def run_from_file(self, queries_file: str, output_file: Optional[str] = None) -> List[Dict]:
        """Load queries from a .txt (one per line) or .json file."""
        p = Path(queries_file)
        if not p.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_file}")

        if p.suffix.lower() == ".json":
            with open(p) as f:
                data = json.load(f)
            queries = data if isinstance(data, list) else [d.get("query", "") for d in data]
        else:
            with open(p, encoding="utf-8") as f:
                queries = [l.strip() for l in f if l.strip()]

        return self.run_queries(queries, output_file=output_file)


class KnowledgeBaseExporter:
    """Export and import knowledge base documents."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def export_stats(self, output_file: str):
        """Export knowledge base stats to JSON."""
        stats = self.pipeline.stats()
        stats["exported_at"] = datetime.now().isoformat()
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Stats exported to {output_file}")
        return stats
