 RAG Automation System
A fully functional, advanced Retrieval-Augmented Generation (RAG) system in Python.
Combines local embeddings, persistent vector storage, intelligent chunking, and Claude LLM generation.

✨ Features
FeatureDetailMulti-format ingestion.txt, .md, .pdf, .docx, URLs, raw text, JSON listsIntelligent chunkingFixed, sentence-aware, paragraph strategies with overlapLocal embeddingssentence-transformers — no external embedding API neededPersistent vector storeChromaDB with cosine similarity, metadata filteringRerankingKeyword + semantic score fusion for better resultsClaude generationClaude Sonnet via Anthropic APIConversation memoryMulti-turn chat with historyFile watcherAuto-ingest new files dropped in a folderBatch processingRun hundreds of queries from a fileRich CLIBeautiful terminal interface

🚀 Quick Start
1. Install dependencies
bashpip install -r requirements.txt
2. Set your Anthropic API key
bashexport ANTHROPIC_API_KEY=sk-ant-...
3. Add documents
bash# Add a file
python cli.py add my_document.pdf

# Add a directory
python cli.py add ./my_docs/

# Add a URL
python cli.py add https://example.com/article

# Add multiple sources at once
python cli.py add report.pdf notes.txt https://docs.example.com
4. Start chatting
bashpython cli.py chat

📁 Project Structure
rag_system/
├── core/
│   ├── __init__.py       # Package exports
│   ├── engine.py         # TextChunker, EmbeddingModel, VectorStore
│   ├── ingestion.py      # DocumentIngester (multi-format)
│   ├── pipeline.py       # RAGPipeline orchestrator
│   └── automation.py     # FileWatcher, BatchProcessor
├── cli.py                # Command-line interface
├── example.py            # Usage demo
└── requirements.txt

🖥️ CLI Reference
bashpython cli.py [--db ./rag_db] <command> [options]
CommandDescriptionchatInteractive multi-turn chatadd <sources...>Add files, directories, or URLsquery "question"Single query (non-interactive)batch queries.txtRun batch queries from filewatch ./inbox/Auto-ingest new files from folderstatsShow knowledge base statisticslistList all document IDsreset⚠️ Delete all data
Options
--db ./my_db          Vector store directory (default: ./rag_db)
--top-k 5             Number of chunks to retrieve
--chunk-size 512      Tokens per chunk
--embedding fast      Model: fast | balanced | multilingual
--no-history          Disable conversation memory (chat command)
--json                JSON output (query command)
-o output.json        Save results to file

🐍 Python API
pythonfrom core import RAGPipeline, RAGConfig

# Configure
config = RAGConfig(
    chunk_size=512,
    chunk_overlap=64,
    chunk_strategy="sentence_aware",  # fixed | sentence_aware | paragraph
    embedding_model="fast",           # fast | balanced | multilingual
    top_k=5,
    score_threshold=0.3,
    llm_model="claude-sonnet-4-20250514",
    persist_dir="./my_rag_db",
)

# Initialize
pipeline = RAGPipeline(config)

# Add documents
pipeline.add_file("report.pdf")
pipeline.add_directory("./docs/")
pipeline.add_url("https://example.com/page")
pipeline.add_text("Custom text...", source="manual")

# Query
response = pipeline.query("What does the report say about Q3?")
print(response.answer)
print(f"Sources: {[s.metadata.get('filename') for s in response.sources]}")

# Multi-turn chat
r1 = pipeline.chat("Summarize the main findings")
r2 = pipeline.chat("What were the risks mentioned?")  # Uses history
pipeline.clear_history()

# Retrieve without generation
chunks = pipeline.retrieve("machine learning", top_k=3)

# Metadata filtering
resp = pipeline.query("revenue", where={"category": "finance"})

# Stats
print(pipeline.stats())
Automation
pythonfrom core import FileWatcher, BatchProcessor

# Auto-ingest new files
watcher = FileWatcher(watch_dir="./inbox", pipeline=pipeline)
watcher.start()
# ... drop files into ./inbox, they get ingested automatically
watcher.stop()

# Batch queries
processor = BatchProcessor(pipeline)
results = processor.run_from_file("questions.txt", output_file="answers.json")

⚙️ Configuration Reference
pythonRAGConfig(
    # Chunking
    chunk_size=512,           # Target tokens per chunk
    chunk_overlap=64,         # Overlap tokens between chunks
    chunk_strategy="sentence_aware",  # fixed | sentence_aware | paragraph
    
    # Embedding (local, no API needed)
    embedding_model="fast",   # fast=MiniLM | balanced=MPNet | multilingual
    
    # Retrieval
    top_k=5,                  # Number of chunks to retrieve
    score_threshold=0.3,      # Min cosine similarity (0-1)
    rerank=True,              # Enable keyword+semantic reranking
    
    # Generation
    llm_model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    system_prompt="...",      # Custom system prompt
    
    # Storage
    persist_dir="./rag_db",   # ChromaDB storage location
    collection_name="rag_main",
)

🔄 How It Works
Input Documents
      │
      ▼
 DocumentIngester ──→ Extract text from PDF/DOCX/URL/text
      │
      ▼
  TextChunker ──→ Split into overlapping chunks (sentence-aware)
      │
      ▼
 EmbeddingModel ──→ sentence-transformers (local, offline)
      │
      ▼
  VectorStore ──→ ChromaDB (persistent cosine similarity index)
      
Query Flow:
  User Question
      │
      ▼
  Embed Query ──→ same embedding model
      │
      ▼
  Vector Search ──→ top-k most similar chunks
      │
      ▼
  Reranker ──→ fuse semantic + keyword scores
      │
      ▼
  Context Builder ──→ structured prompt with sources
      │
      ▼
  Claude API ──→ grounded answer with citations

🗂️ Batch Query File Formats
Text file (questions.txt):
What is the revenue for Q3?
Who are the main competitors?
What risks are mentioned?
JSON file (questions.json):
json[
  {"query": "What is the main topic?"},
  {"query": "Summarize the conclusions"}
]

📝 License
MIT — use freely in any project.





   
