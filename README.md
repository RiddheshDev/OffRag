# OffRAG: Production-Grade Offline Multi-Tenant RAG Pipeline & Chatbot

OffRAG is a secure, private, layout-aware Retrieval-Augmented Generation (RAG) system. It ingests unstructured documents, parses complex elements (tables and code blocks) accurately, indexes them under secure tenant shards in a local vector database, and offers a conversational chatbot featuring dynamic intent routing, parent-context reconstruction, and Model Context Protocol (MCP) real-time web search fallback.

---

## 🛠️ System Architecture & Workflows

### 1. Document Ingestion Pipeline
```text
[Input PDF]
     │
     ▼
[Document Classifier] ──► Scanned OCR vs Table-Heavy vs Text-Heavy
     │
     ▼
[Intelligent Extractor] ──► Isolates Code Blocks & runs Hybrid Camelot/pdfplumber table parser
     │
     ▼
[Adaptive Chunker] ──► Sentence Window / Hierarchical / Recursive Strategies
     │
     ▼
[Weaviate Uploader] ──► Generates local BGE embeddings & saves to secure Tenant Shards
```

### 2. Conversational Agent & Retrieval Pipeline
```text
               [User Query]
                    │
                    ▼
          [Dynamic Intent Router]
           (Uses Conversational Memory)
            /                        \
    (Needs Search)               (Greetings/Filler)
          /                            \
         ▼                              ▼
[Weaviate Tenant Shard]         [Generate Direct Response]
  - Hybrid query (alpha)
  - Department filtering
         │
         ▼
[Cross-Encoder Reranker]
         │
         ├─── (Score >= 0.1) ──► [LLM Generator] ──► [Assistant Answer (Citations)]
         │
         └─── (Score < 0.1)  ──► [MCP Search Fallback] (Brave Search / DDG)
                                       │
                                       ▼
                                [LLM Generator] ──► [Assistant Answer (Web References)]
```

---

## 💻 Tech Stack & Dependencies

OffRAG is constructed using state-of-the-art Python AI and database technologies:

* **PDF Layout Parsing & OCR**:
  * `pdfplumber`: Page-layout geometry, vertical text slicing, and fallback text table crop extraction.
  * `Camelot-py` (with OpenCV & `Ghostscript` dependencies): Dual-flavor (Lattice & Stream) table extraction with parsing quality verification.
  * `PaddleOCR` & `Pillow`: Image pre-processing and optical character recognition for scanned PDFs.
* **Chunking & Node Modeling**:
  * `llama-index-core`: TextNode data models and dependency-free relationship hierarchies.
* **Local Embeddings**:
  * `sentence-transformers` / `llama-index-embeddings-huggingface`: Generates local dense embeddings using `BAAI/bge-base-en-v1.5`.
* **Database & Indexing**:
  * `Weaviate` v4 Client: Connected to local instance over gRPC, utilizing **Native Multi-Tenancy** and metadata filter schemas.
* **Reranking Engine**:
  * `sentence-transformers`: Local Cross-Encoder reranker using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **Prompt LLM Generation**:
  * `LLMGenerator`: Cloud `openai` client (GPT models) and local HuggingFace `transformers` text generation pipelines (`Qwen/Qwen2.5-0.5B-Instruct` or `Phi-3-mini`).
* **Agentic Web Search Fallback**:
  * Anthropic official stdio-based `mcp` client: Calls Brave Search MCP servers (`npx -y @modelcontextprotocol/server-brave-search`) with a `duckduckgo-search` library fallback.

---

## 📂 Project Directory Structure

```text
data_chunking_pipeline/
│
├── config.yaml               # Pipeline, LLM, retrieval parameters, and fallbacks
├── main.py                   # Ingestion pipeline orchestration entry point
├── query_pipeline.py         # Retrieval, reranking, and cloud/local LLM answer pipeline
├── chatbot.py                # Conversational chatbot with memory & dynamic intent routing
│
├── core/
│   ├── classifier.py         # Documents classifier (Scanned, Text-Heavy, Table-Heavy)
│   ├── extractor.py          # Programmatic layout parser and vertical text slicer
│   ├── table_extractor.py    # Hybrid Camelot + pdfplumber parser with quality evaluations
│   ├── chunker.py            # Sentence Window, Hierarchical, and Recursive text splitters
│   ├── ocr_pipeline.py       # Scanned document image preprocessing and PaddleOCR extraction
│   ├── uploader.py           # Embeddings generation and batch multi-tenant Weaviate uploads
│   ├── retriever.py          # Hybrid query, tenant bounding, and Cross-Encoder reranking
│   ├── generator.py          # Unified OpenAI API and local HuggingFace generation pipeline
│   └── mcp_client.py         # Stdio transport Model Context Protocol Brave Search client
│
├── utils/
│   └── helpers.py            # Configuration loaders, table formatters, and path validators
│
└── requirements.txt          # Python library dependencies
```

---

## 🚀 Usage Guide

### 1. Ingestion Pipeline
Parse a PDF using the sentence window strategy, isolate table and code blocks, and upload it to a specific client tenant shard and department:
```powershell
& python data_chunking_pipeline/main.py `
  --pdf "data/API-Documentation-1.0v.pdf" `
  --strategy sentence_window `
  --upload `
  --tenant "client_a" `
  --department "engineering"
```

### 2. Retrieval & Generation Pipeline
Execute a single query with citations, scoped to a specific tenant shard and department filter:
```powershell
& python data_chunking_pipeline/query_pipeline.py `
  --query "what is the URL and method for Language Detection?" `
  --llm local `
  --tenant "client_a" `
  --department "engineering"
```

### 3. Interactive Conversational Chatbot
Launch the terminal chatbot session. Casually converse, ask technical details from your files, or trigger real-time web searches automatically when retrieval scores fall below threshold:
```powershell
& python data_chunking_pipeline/chatbot.py `
  --llm local `
  --tenant "client_a" `
  --department "engineering"
```

---

## 🛡️ Enterprise Security: Multi-Client Tenancy & Hierarchy

OffRAG provides multi-tenancy at two levels:

1. **Client Tenant Shards (Hard Isolation)**:
   Weaviate's native multi-tenancy splits data into physically separated indices for each client. Querying `client_a` will never scan or leak data from `client_b`.
2. **Department Filter (Soft/Hierarchical Scoping)**:
   Data objects are tagged with a `department_id` property. Queries within a client's shard can be restricted to a specific department (e.g., `hr`, `finance`, `engineering`) or searched across the entire tenant when no filter is supplied.
