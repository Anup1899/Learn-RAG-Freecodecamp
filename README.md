# Learn RAG - FreeCodeCamp

A hands-on project for learning **Retrieval-Augmented Generation (RAG)** using LangChain and Claude (Anthropic).

---

## What is RAG?

RAG is a technique that enhances LLM responses by retrieving relevant documents from a knowledge base before generating an answer — grounding the model in real, up-to-date information rather than relying solely on its training data.

**RAG Pipeline Overview:**

```
INDEXING   : Document Loading -> Chunking -> Embedding -> Store in Vector DB
QUERYING   : Embed Query -> Search Vector DB -> Retrieve Docs -> LLM generates Answer
```

> Both phases must use the **same embedding model** so query and document vectors are in the same space.

---

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- An Anthropic API key

---

## Setup

**1. Install dependencies**
```bash
uv sync
```

**2. Configure environment variables**

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_api_key_here
```

---

## Run

```bash
uv run python main.py
```

---

## Project Structure

```
.
├── main.py                  # Entry point — LLM setup and test
├── documentsloader1.py      # Document loaders (PDF, Text)
├── chunkingEmbedding2.py    # Chunking strategies and embedding concepts
├── docs/                    # Sample PDF documents
├── .env                     # API keys (never commit this)
└── pyproject.toml           # Dependencies
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Chains, retrievers, document loaders |
| `langchain-core` | Base abstractions (Runnable, BaseMessage) |
| `langchain-anthropic` | Claude integration for LangChain |
| `langchain-community` | Community integrations (PDF loaders, etc.) |
| `anthropic` | Anthropic SDK |
| `langgraph` | Agent and graph-based workflows |
| `python-dotenv` | Load `.env` variables |

> **Note:** `langchain-community` is being sunset. Prefer standalone packages (e.g. `langchain-pypdf`) when available.

---

## Model

Uses **Claude Haiku 4.5** (`claude-haiku-4-5`) — fast and cost-effective, ideal for learning and experimentation.

```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
```

`temperature=0` makes responses deterministic — important for RAG where consistency matters.

---

## Document Loaders

Document loaders ingest raw data and convert it into LangChain `Document` objects (with `.page_content` and `.metadata`).

### Core Loader Types

| Loader | Description |
|---|---|
| **PDF Loaders** | Loads PDF documents and extracts text content. |
| **Text Loaders** | Loads plain text files. |
| **Directory Loaders** | Loads all files from a directory (batch processing). |
| **WebBase Loaders** | Fetches and processes content from web pages. |
| **Unstructured Loaders** | Handles emails, chat logs, and other non-standard formats. |

### PDF Loader Options (by speed & metadata quality)

| Loader | Speed | Metadata | Notes |
|---|---|---|---|
| `PyPDFLoader` | Fast | Basic | Good for simple text extraction; may struggle with complex layouts. |
| `PyMuPDFLoader` | Fast | Good | Balanced option; may need extra config for best results. |
| `UnstructuredPDFLoader` | Slower | Rich | Best for complex layouts and preserving metadata. |

---

## Chunking

Large documents are split into smaller chunks before embedding. Chunk size affects retrieval quality.

### Chunking Strategies

| Strategy | When to Use |
|---|---|
| **Fixed-size** | Simple, fast — splits by character/token count. May break context unnaturally. |
| **Recursive** (default) | Splits by natural boundaries (paragraphs, sentences). Best general-purpose choice. |
| **Semantic** | Splits by meaning/topic. Best quality — use when accuracy matters more than speed. |
| **Late chunking** | Embeds full doc first, then chunks by embedding similarity. Most context-aware. |

### Chunking Decision Guide

| Content Type | Strategy | Chunk Size |
|---|---|---|
| General documents | Recursive (default) | 500-1000 tokens |
| Technical documents | Semantic | Auto (by meaning) |
| Code | Code-Splitter | Auto (by function/class) |
| Markdown | Markdown-Splitter | Auto (by heading/section) |

---

## Embeddings

Embeddings convert text into vectors (lists of numbers) that represent semantic meaning in a high-dimensional space. Similar text produces similar vectors.

### Chat Models vs Embedding Models

| | Chat Models | Embedding Models |
|---|---|---|
| **Output** | Human-like text response | Vector (list of numbers) |
| **Use** | Generate answers | Represent meaning for similarity search |

### Embedding Dimensions

| Model | Dimensions |
|---|---|
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| Gemini Embedding | 768 |
| BGE-Small | 384 |

More dimensions = more nuance captured, but more compute and storage required.

---

## Vector Database

A vector database stores embeddings and enables fast similarity search to find the most relevant chunks for a query.

### How It Works

1. User asks a query
2. Query is embedded using the **same embedding model** used during indexing
3. The embedded query is sent to the vector database
4. The DB returns the most similar vectors along with their metadata (document ID, chunk ID, etc.)
5. Retrieved chunks are passed to the LLM to generate a grounded, accurate answer

---

## Production-Ready RAG: 3 Key Rules

| Rule | Description |
|---|---|
| **1. Same embedding model** | Use the same model for both indexing and querying — ensures vectors are in the same space for accurate retrieval. |
| **2. Quality over quantity** | Prioritize high-quality, relevant vectors over a large number of low-quality ones — reduces noise and improves results. |
| **3. Test retrieval separately** | Validate retrieval performance independent of generation — isolates issues and makes debugging easier. |



<!-- 
# The 5 RAG Failure Modes
1. Bad Chunking -- Wrong Cuts, WRORS, ILLOGICAL
2. Embedding Mismatch -- Query Vector and Document Vector mismatch. When user queries use different words than the words found in the document, then the semantic seach fails 
3. Retrival Noise -- Signal Noise and Irrelevant Results
4. Context Overflow - Data Splitting Out, Overflow and Exceed Limits
5. Hallucination - Imagination, False Information


-->
