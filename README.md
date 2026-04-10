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






   
