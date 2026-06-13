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



---

## The 5 RAG Failure Modes

Even a well-architected RAG pipeline can silently fail. These are the five most common failure modes and how to recognize and fix them.

### 1. Bad Chunking — Wrong Cuts, Illogical Splits

**What happens:** The document is split at the wrong boundaries — mid-sentence, mid-table, or mid-thought — so each chunk is missing the context needed to answer a question correctly.

**Example:** A paragraph explaining that "the returns policy applies only to items purchased after Jan 1, 2024" gets split into two chunks. One chunk has the policy; the other has the date condition. Neither chunk alone answers "Can I return an item I bought in December 2023?"

**Symptoms:**
- Retrieved chunks look relevant but the LLM gives wrong or incomplete answers
- Answers that contradict the source document

**Fix:**
- Use recursive or semantic chunking instead of fixed-size character splits
- Add chunk overlap (e.g. 100–200 tokens) so boundary context is preserved in both adjacent chunks
- Use a code-aware splitter for code, a markdown-aware splitter for markdown

---

### 2. Embedding Mismatch — Query and Document in Different Vocabularies

**What happens:** The user's query uses different words or phrasing than the document, so the query vector and the document vector end up far apart in embedding space even though they mean the same thing.

**Example:** The document says "myocardial infarction treatment protocol". The user asks "what do I do if someone has a heart attack?". Semantically identical — but if the embedding model wasn't trained on medical language, the vectors won't be close enough to retrieve the right chunk.

**Symptoms:**
- Retrieval returns documents that are thematically related but not the right answer
- Changing the wording of the query dramatically changes which documents are retrieved

**Fix:**
- Use a domain-specific embedding model (e.g. a biomedical model for medical documents)
- Apply query expansion or HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer first, embed that, and use it as the query vector
- Add synonyms and alternate phrasings to your index via metadata

---

### 3. Retrieval Noise — Irrelevant Results Drown Out the Signal

**What happens:** The similarity search returns chunks that score well on the embedding metric but are not actually relevant to the question. These noisy chunks consume the LLM's context window and mislead the answer.

**Example:** A query about "employee vacation policy" retrieves chunks about "vacation rentals", "holiday pay", and an unrelated HR blog post — all of which scored high in cosine similarity but none of which contain the actual policy.

**Symptoms:**
- Increasing `k` (number of retrieved chunks) makes answers worse, not better
- The LLM hedges, contradicts itself, or cites information from the wrong document

**Fix:**
- Add a **re-ranker** (cross-encoder) after retrieval to re-score chunks based on the actual query-document pair, not just vector distance
- Use **metadata filtering** to restrict retrieval to relevant document types, date ranges, or categories before running similarity search
- Tune `k` — more retrieved chunks is not always better; start with 3–5

---

### 4. Context Overflow — Too Much Data, Truncated Context

**What happens:** The retrieved chunks plus the system prompt plus the conversation history exceed the LLM's context window limit. The model either refuses, truncates silently, or loses important information that was pushed out.

**Example:** You retrieve 20 chunks at 500 tokens each (10,000 tokens), add a 2,000-token system prompt, and send it to a model with an 8,192-token context limit. The last several chunks are silently dropped, and the LLM answers as if that content doesn't exist.

**Symptoms:**
- The answer ignores information that was clearly in your documents
- The model says "I don't have enough information" even when the documents contain the answer
- API errors about exceeding token limits

**Fix:**
- Calculate your budget: `context_limit - system_prompt - response_reserve = max_retrieved_tokens`
- Reduce `k` or reduce chunk size to fit within budget
- Use **map-reduce** or **refine** chains to process chunks in batches instead of all at once
- Use a model with a larger context window for document-heavy use cases

---

### 5. Hallucination — The Model Invents Confident Answers

**What happens:** The LLM generates a plausible-sounding answer that is not grounded in the retrieved documents. This happens when the retrieved context is insufficient, ambiguous, or the model over-relies on its parametric (training-time) knowledge.

**Example:** Retrieved chunks discuss a product's general features. The user asks about a specific version number that doesn't appear in any chunk. Instead of saying "I don't know," the model confidently states an incorrect version number it learned during training.

**Symptoms:**
- Answers contain details not present in any retrieved chunk
- Citations point to the right document but the quoted text doesn't appear in it
- Answers are confident even when the question is outside the knowledge base

**Fix:**
- Use strict system prompt instructions: "Answer only from the provided context. If the answer is not in the context, say 'I don't have enough information.'"
- Implement **faithfulness evaluation** — after generating, verify each claim against the retrieved chunks
- Use **citations**: force the LLM to quote the source chunk it used, making hallucinations easier to detect

---

## When Vector Search Fails — and Why Hybrid Search is the Fix

Pure vector (semantic) search works well for natural language queries but breaks down when the query contains **exact, non-semantic identifiers** that have no meaning in embedding space.

### Failure Cases

| Query Type | Example | Why Vector Search Fails |
|---|---|---|
| **Product / SKU codes** | `SKU-7742X` | No semantic meaning — a random string of characters. All SKUs are equidistant in embedding space. |
| **Acronyms** | `WCAG compliance` | The embedding model may not know the expansion ("Web Content Accessibility Guidelines"). The acronym and its full form end up as different vectors. |
| **Exact names with partial overlap** | `John Smith accounting` | The document contains "Smith Family Tree", "Accounting Principles", and "John's Bakery" as separate chunks. Semantic search blends them; none ranks at the top. |
| **Error codes** | `E_CONN_REFUSED` | Just a string of characters. The model has no notion of what a connection refusal is in this context. |
| **Version numbers / dates** | `v2.3.1 release notes` | Numeric identifiers are nearly identical in embedding space regardless of their semantic difference. |

### The Fix: Hybrid Search

Hybrid search combines two retrieval methods and merges their results using **Reciprocal Rank Fusion (RRF)** or a weighted score:

```
Hybrid Score = α × BM25 Score + (1 - α) × Vector Score
```

| Method | Good At | Bad At |
|---|---|---|
| **BM25 (keyword / sparse)** | Exact matches — codes, names, IDs, acronyms | Synonyms, paraphrasing, semantic meaning |
| **Vector (dense)** | Semantic similarity, paraphrasing, concept matching | Exact strings, rare identifiers, out-of-vocabulary tokens |
| **Hybrid** | Both — combines the strengths of each | Adds pipeline complexity; requires tuning `α` |

**When to use hybrid search:**
- Your knowledge base contains product catalogs, technical documentation, or any data with structured identifiers
- Users are likely to search with both natural language AND exact codes/names in the same session
- Pure vector retrieval is missing obvious keyword matches



<!--
# Hybrid Search Pipeline
- [QueryBox] SKU-7742 specifications
- [Vector Search Result] ---- Doc 3[Rank 1], Doc 7[Rank 2], Doc 1[Rank 5]
- [BM25 Search Result] ---- Doc 1[Rank 1], Doc 3[Rank 2], Doc 5[Rank 53]
- [Reciprocal Rank Fusion (RRF)** or a weighted score] -- Document that appears good in both rise on the top
- [Final Result] --- Doc 1 (won appreas in both)


# Hybrid Search adds the Latency of 20-25ms

# Vector Retriver and BM25 Retriver are passed from the Ensemble Retriver, which combines both the result and provide the results on the top which are in found in both the retriver


# Production consideration for Hybrid Search
# BM25 REBUILD 
- BM25 doesn't support incremental updates
- Rebuild when adding documents

# k Value
- Retrive more let RRF sort
- k = 4 r higher recommended

# Latency
- Hybrid add 20-25ms
- Two seached instead of one

# Too much of context


# TOKEN BUDEGETING


-->