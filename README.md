# Learn RAG - FreeCodeCamp

A hands-on project for learning **Retrieval-Augmented Generation (RAG)** using LangChain, OpenAI, and modern production techniques.

This README is organized **file by file** — each lesson script (`A1` → `A18`) has its own section documenting every concept the code uses, plus additional background where it helps. Cross-cutting deep dives (HNSW internals, failure modes, self-hosted vs managed databases) live in the [Appendix](#appendix--cross-cutting-deep-dives).

---

## What is RAG?

RAG enhances LLM responses by retrieving relevant documents from a knowledge base before generating an answer — grounding the model in real, up-to-date information rather than relying solely on its training data.

**RAG Pipeline Overview:**

```
INDEXING   : Document Loading -> Chunking -> Embedding -> Store in Vector DB
QUERYING   : Embed Query -> Search Vector DB -> Retrieve Docs -> LLM generates Answer
```

> Both phases must use the **same embedding model** so query and document vectors live in the same space.

---

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- API keys for the providers you use (OpenAI, and optionally Qdrant Cloud, LangSmith, Anthropic)

---

## Setup

**1. Install dependencies**
```bash
uv sync
```

**2. Configure environment variables**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
QDRANT_URL=your_qdrant_url          # only for A2
QDRANT_API_KEY=your_qdrant_key      # only for A2
LANGCHAIN_TRACING_V2=true           # only for A8/A12 tracing
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=Freecodecamp-RAG
```

---

## Run

Each lesson is a standalone script. Run any of them with:

```bash
uv run python A3_ragpipeline3.py
```

Most files gate their demos behind `if __name__ == "__main__":` and comment out all but the active demo — uncomment the function you want to run.

---

## Project Structure

```
.
├── main.py                   # Entry point — LLM setup and test
├── A1_documentsloader1.py    # Document loaders (PDF, Text)
├── A2_chunkingEmbedding2.py  # Chunking + embedding theory; Qdrant indexing & filtered search
├── A3_ragpipeline3.py        # Basic RAG pipeline (LCEL) + source attribution
├── A4_textSplitter4.py       # Text splitter strategies + chunk overlap
├── A5_embeddings_deep5.py    # Embeddings deep dive: similarity, normalization, caching
├── A6_hybridSearch.py        # Hybrid search: BM25 + vector via EnsembleRetriever
├── A7_costOptimization.py    # Token budgeting + semantic caching
├── A8_langsmith_setup.py     # LangSmith observability and tracing
├── A9_semanticChunking.py    # Semantic chunking vs recursive
├── A10_prodReady.py          # Production chunking: semantic with recursive fallback
├── A11_advanced_rag.py       # Multi-Query, Compression, Parent-Child retrievers
├── A12_monitoring.py         # Structured logging + metrics (three pillars)
├── A13_LongContextVsRAG.py   # Long context vs RAG: cost, latency, hybrid
├── A14_ContextualRetrival.py # Contextual Retrieval (Anthropic technique)
├── A15_lateChunking.py       # Late chunking — embed full doc, then split
├── A16_agenticRAGpy          # Agentic RAG with LangGraph (retrieve→grade→retry→generate)
├── A17_graphRAGIntro.py      # GraphRAG — knowledge graphs for multi-hop reasoning
├── A18_multiModelRAG.py      # Multimodal RAG with ColPali (vision-based)
├── Supabase/                 # Supabase / pgvector connection example
├── docs/                     # Sample PDF documents
├── vector_store/             # Persisted Chroma vector database
├── .env                      # API keys (never commit this)
└── pyproject.toml            # Dependencies
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Chains, retrievers, document loaders |
| `langchain-core` | Base abstractions (Runnable, BaseMessage, Document) |
| `langchain-openai` | OpenAI LLM and embedding integration |
| `langchain-chroma` | Chroma vector store integration |
| `langchain-community` | Community integrations (PDF loaders, BM25Retriever) |
| `langchain-classic` | Legacy retrievers: ParentDocumentRetriever, ContextualCompressionRetriever, MultiQueryRetriever, EnsembleRetriever, CacheBackedEmbeddings |
| `langchain-experimental` | `SemanticChunker` (A9, A10) |
| `langchain-text-splitters` | Recursive/Character/Token/Markdown splitters |
| `qdrant-client` | Qdrant vector database client with server-side inference (A2) |
| `langsmith` | Tracing and evaluation platform |
| `openai` | OpenAI SDK |
| `python-dotenv` | Load `.env` variables |
| `langgraph` | Cyclic, stateful agent workflows (A16) |
| `tiktoken` | Token counting for OpenAI models (A13) |
| `networkx` | Knowledge graph construction (A17) |
| `numpy` | Vector math — cosine similarity, norms (A5, A7, A15) |

> **Missing dependency check:** `A17_graphRAGIntro.py` imports `networkx`. If it isn't in `pyproject.toml` / `uv.lock`, run `uv add networkx` before running that script.

> **Note:** `langchain-community` is being sunset. Prefer standalone packages (e.g. `langchain-pypdf`) when available. `langchain-classic` bundles retrievers that were moved out of `langchain` core.

---

## Model

Most lessons use **GPT-4o Mini** (`gpt-4o-mini`) via `init_chat_model` or `ChatOpenAI` — fast and cost-effective, ideal for learning and experimentation. (A13 uses a newer nano model string for its cost demo.)

```python
from langchain.chat_models import init_chat_model
llm = init_chat_model(model="gpt-4o-mini", temperature=0.2)
```

`temperature=0` makes responses deterministic — important for RAG where consistency matters. `init_chat_model` is a provider-agnostic factory that works with OpenAI, Anthropic, and others by swapping the model string.

---

# Lessons

---

## A1 — Document Loaders (`A1_documentsloader1.py`)

Document loaders ingest raw data and convert it into LangChain `Document` objects. Every `Document` has two attributes:

- **`page_content`** — the extracted text
- **`metadata`** — a dict of provenance info (source path, page number, etc.)

### What the code does

The script demonstrates two loaders:

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# TextLoader — writes a temp .txt file, loads it, prints content + metadata, cleans up
text_loader = TextLoader(temp_file_path)
documents = text_loader.load()          # -> list[Document]

# PyPDFLoader — one Document per page; metadata inspected via json.dumps(indent=2)
pdf_loader = PyPDFLoader("./docs/langchain.pdf")
documents = pdf_loader.load()
```

Key details from the code:

- `tempfile.NamedTemporaryFile(delete=False, suffix=".txt")` creates a throwaway file for the demo; it is removed in a `finally` block so cleanup always runs.
- `PyPDFLoader.load()` returns **one `Document` per page** — `len(documents)` equals the page count.
- Metadata is printed with `json.dumps(doc.metadata, indent=2)` to make the structure (source, page) readable.

### Core Loader Types

| Loader | Description |
|---|---|
| **PDF Loaders** | Load PDF documents and extract text content. |
| **Text Loaders** | Load plain text files. |
| **Directory Loaders** | Load all files from a directory (batch processing). |
| **WebBase Loaders** | Fetch and process content from web pages. |
| **Unstructured Loaders** | Handle emails, chat logs, and other non-standard formats. |

### PDF Loader Options (by speed & metadata quality)

| Loader | Speed | Metadata | Notes |
|---|---|---|---|
| `PyPDFLoader` | Fast | Basic | Good for simple text extraction; may struggle with complex layouts. |
| `PyMuPDFLoader` | Fast | Good | Balanced option; may need extra config for best results. |
| `UnstructuredPDFLoader` | Slower | Rich | Best for complex layouts and preserving metadata. |

> **Why metadata matters:** the `source` field set here is what later powers **source attribution** (A3) — the ability to tell the user which document an answer came from.

---

## A2 — Chunking, Embeddings & Vector DB with Qdrant (`A2_chunkingEmbedding2.py`)

This file is the theory backbone of the course (a long comment block) **plus** a working Qdrant indexing-and-search pipeline over a restaurant menu.

### Chunking strategies

Large documents are split into smaller chunks before embedding. Chunk size affects retrieval quality.

| Strategy | When to Use |
|---|---|
| **Fixed-size** | Simple, fast — splits by character/token count. May break context unnaturally. |
| **Recursive** (default) | Splits by natural boundaries (paragraphs, sentences). Best general-purpose choice. |
| **Semantic** | Splits by meaning/topic. Best quality — use when accuracy matters more than speed. |
| **Late chunking** | Embeds full doc first, then chunks by embedding similarity. Most context-aware. |

**Chunking decision guide:**

| Content Type | Strategy | Chunk Size |
|---|---|---|
| General documents | Recursive (default) | 500–1000 tokens |
| Technical documents | Semantic | Auto (by meaning) |
| Code | Code-Splitter | Auto (by function/class) |
| Markdown | Markdown-Splitter | Auto (by heading/section) |

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
| BGE-Small / `all-MiniLM-L6-v2` | 384 |

More dimensions = more nuance captured, but more compute and storage required. Choose based on the accuracy-vs-efficiency trade-off for your corpus size.

### The two-phase RAG pipeline

```
INDEXING  (Phase 1): Document Loading -> Chunking -> Embedding -> Store in Vector DB
QUERYING  (Phase 2): Embed Query -> Search Vector DB -> Retrieve -> LLM generates Answer
```

Both phases **must** use the same embedding model so document and query vectors share one space.

### Three rules for production-ready RAG

| Rule | Description |
|---|---|
| **1. Same embedding model** | Use the same model for indexing and querying — ensures vectors are comparable. |
| **2. Quality over quantity** | Prioritize high-quality, relevant vectors over many low-quality ones — reduces noise. |
| **3. Test retrieval separately** | Validate retrieval independent of generation — isolates issues and eases debugging. |

### The Qdrant code (what's actually running)

The script embeds and searches 30 menu items using **Qdrant Cloud with server-side inference** (the embedding model runs on Qdrant's side, so you send text and Qdrant embeds it):

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Document,
    Filter, FieldCondition, MatchValue, PayloadSchemaType,
)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, cloud_inference=True)

# --- INDEXING ---
# recreate collection so vector config is always correct
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # 384 = all-MiniLM-L6-v2
)

# payload index enables fast *filtered* search on a field
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD,
)

# each point = server-side embedded text + structured payload (metadata)
point = PointStruct(
    id=i,
    vector=Document(text=f"{name} {description}", model="sentence-transformers/all-MiniLM-L6-v2"),
    payload={"item_name": ..., "description": ..., "price": ..., "category": ...},
)
client.upsert(collection_name=COLLECTION_NAME, points=points)
```

**Concepts introduced here:**

- **`VectorParams(size=384, distance=Distance.COSINE)`** — declares the vector dimension and similarity metric. Size **must** match the embedding model (384 for `all-MiniLM-L6-v2`). Cosine distance measures the angle between vectors, ignoring magnitude.
- **`cloud_inference=True` + `Document(text=..., model=...)`** — Qdrant embeds the text server-side. You never call an embedding API yourself; you hand Qdrant raw text.
- **`PointStruct`** — the unit stored in Qdrant: an `id`, a `vector`, and a `payload` (arbitrary metadata returned with search hits).
- **Payload index (`create_payload_index`)** — required to filter efficiently on a field. Without it, filtered search would scan every point.

### Two search modes

```python
# 1. Basic similarity search
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=Document(text=query_text, model=EMBEDDING_MODEL),
    with_payload=True, limit=5,
)

# 2. Filtered similarity search — semantic match *within* a category
category_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value="Noodles"))])
filtered = client.query_points(..., query_filter=category_filter, ...)
```

**Filtered search** combines a hard metadata constraint (`category == "Noodles"`) with semantic ranking — the vector search only considers points that pass the filter. Each hit exposes `result.score` (similarity) and `result.payload` (the metadata). Setting a `score_threshold` (commented out in the code) would drop weak matches entirely.

> **Deeper reading:** how the vector DB actually finds neighbors quickly is covered in the [HNSW deep dive](#hnsw--how-vector-databases-find-similar-vectors-quickly). Whether to run the DB yourself is covered in [Self-Hosted vs Managed](#self-hosted-vs-managed-vector-database).

---

## A3 — Basic RAG Pipeline & Source Attribution (`A3_ragpipeline3.py`)

The first end-to-end RAG pipeline, built with **LangChain Expression Language (LCEL)** and a persisted Chroma store.

### Building the knowledge base

```python
loader = PyPDFLoader("./docs/langchain.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(loader.load())

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./vector_store",   # persists to disk, survives restarts
)
```

- `split_documents` (vs `split_text`) preserves each source `Document`'s metadata onto its chunks.
- `persist_directory` writes the index to `./vector_store` so it doesn't need rebuilding on every run.

### The LCEL RAG chain

```python
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the following context:
    {context}
    Question : {question}
    Answer :
    Make sure to answer in concise manner, and if you don't know the answer, say you don't know.
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = rag_chain.invoke("What is langchain?")
```

**Concepts:**

- **LCEL `|` piping** — each component (retriever, prompt, llm, parser) is a `Runnable`; the `|` operator composes them into a pipeline.
- **The parallel dict** `{"context": ..., "question": ...}` runs both branches; `RunnablePassthrough()` forwards the raw input question unchanged.
- **`format_docs`** flattens retrieved `Document`s into one context string.
- **`StrOutputParser`** extracts the plain string from the model's message object.
- **`as_retriever(search_kwargs={"k": 2})`** — retrieve the top 2 most similar chunks. The instruction *"if you don't know, say you don't know"* is a first line of defense against [hallucination](#the-5-rag-failure-modes).

### RAG with source attribution

A common production requirement is showing **which document** each part of the answer came from. Instead of concatenating raw chunk text, prefix each chunk with its source:

```python
def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
    return "\n\n".join(formatted)
```

The prompt then instructs the LLM to cite the sources it used (`"Answer (include sources):"`).

| Without attribution | With attribution |
|---|---|
| LLM answer is a black box | User can verify claims against source documents |
| Hallucinations are invisible | Hallucinated facts have no matching source citation |
| Debugging requires re-running retrieval | Source label instantly shows which chunk drove the answer |

Source attribution is the first step toward **faithfulness evaluation** — checking that every claim in the answer traces back to a retrieved chunk.

---

## A4 — Text Splitters Deep Dive (`A4_textSplitter4.py`)

Where A2 covered *when* to use each chunking strategy, A4 covers the actual **splitter classes** and demonstrates why **chunk overlap** matters.

### The splitter family

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # splits on a priority list of separators
    CharacterTextSplitter,           # splits on a single separator
    TokenTextSplitter,               # splits by token count (not characters)
    MarkdownTextSplitter,            # respects markdown structure (headings, lists)
    Language,                        # language-aware code splitting (Python, JS, etc.)
)
```

| Splitter | Splits by | Best for |
|---|---|---|
| `RecursiveCharacterTextSplitter` | Ordered separators `["\n\n", "\n", " ", ""]` | General text — the default choice |
| `CharacterTextSplitter` | A single separator | Simple, uniform text |
| `TokenTextSplitter` | Token count | Staying under model token limits precisely |
| `MarkdownTextSplitter` | Markdown structure | Docs, READMEs, wikis |
| `Language`-configured splitter | Code syntax | Splitting source code at function/class boundaries |

The recursive splitter tries separators **in order** — paragraphs first, then lines, then spaces — only moving to a finer separator when a chunk is still too big. This keeps splits at the most natural boundary possible.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_text(SAMPLE_TEXT)
```

### Why chunk overlap matters

The `overlap_importance()` demo splits the same text with `chunk_overlap=0` and `chunk_overlap=20` and prints the boundaries:

```
Without overlap: Chunk 1 ends "...lazy dog. The quick"   Chunk 2 starts "brown fox jumps..."
With overlap:    Chunk 1 ends "...lazy dog. The quick"   Chunk 2 starts "The quick brown fox..."
```

**Overlap** repeats the last N characters of one chunk at the start of the next. This prevents a fact that straddles a boundary from being lost — a sentence cut in half still appears whole in one of the two chunks. A typical overlap is 10–20% of chunk size. The trade-off: more overlap means more chunks, more storage, and more embedding cost.

---

## A5 — Embeddings Deep Dive (`A5_embeddings_deep5.py`)

Embeddings convert text into vectors that represent semantic meaning. Similar text → similar vectors. A5 explores the mechanics using OpenAI's `text-embedding-3-small`.

### Single vs batch embedding

```python
single = embeddings_model.embed_query("What is Machine Learning?")     # one vector
batch  = embeddings_model.embed_documents([text1, text2, text3])       # list of vectors
```

- `embed_query` — embeds one string (used for the incoming query).
- `embed_documents` — embeds a list in one call (used for indexing). Batching is cheaper than looping one call at a time (see [cost optimization](#appendix--cross-cutting-deep-dives)).

### Vector normalization

```python
import numpy as np
print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")   # ≈ 1.0000
```

OpenAI embeddings are **L2-normalized** — their length (norm) is ~1.0. This matters because when vectors are unit-length, the **dot product equals cosine similarity**, so similarity search can use the cheaper dot product directly.

### Cosine similarity and ranking

```python
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarities = [cosine_similarity(query_vector, d) for d in doc_vectors]
ranked = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
```

**Cosine similarity** measures the angle between two vectors: `1.0` = identical direction, `0` = unrelated, `-1` = opposite. The demo embeds four documents (three about ML/DL/NLP, one about New Delhi) and shows the "Deep learning" query ranks the deep-learning doc highest and the unrelated New Delhi doc lowest — this is *exactly* what a vector database does internally at scale.

### Caching embeddings

```python
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings_model,
    document_embedding_cache=LocalFileStore(root_path=tempdir),
    namespace="exercise",
)
cached_embeddings.embed_documents([text])   # 1st call hits the API
cached_embeddings.embed_documents([text])   # 2nd call served from cache — identical vectors
```

`CacheBackedEmbeddings` wraps any embedding model and stores results in a byte store (here a local file store). Re-embedding identical text returns the cached vector instead of paying for another API call — `np.allclose(v1, v2)` confirms they're identical. The `namespace` keeps different models' caches from colliding.

---

## A6 — Hybrid Search (`A6_hybridSearch.py`)

Pure vector (semantic) search works well for natural language but breaks down on **exact, non-semantic identifiers** — SKU codes, error codes, version numbers — that have no meaning in embedding space.

### When vector search fails

| Query Type | Example | Why Vector Search Fails |
|---|---|---|
| Product / SKU codes | `SKU-7742X` | No semantic meaning — a random string. All SKUs are equidistant in embedding space. |
| Acronyms | `WCAG compliance` | Model may not know the expansion; acronym and full form become different vectors. |
| Error codes | `E_CONN_REFUSED` | Just characters — no notion of a connection refusal. |
| Version numbers / dates | `v2.3.1 release notes` | Numeric identifiers are nearly identical in embedding space. |

### The fix: combine BM25 + vector

```python
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

vector_retriever = Chroma.from_documents(documents, embeddings).as_retriever(
    search_type="similarity", search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(documents, k=3)

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],   # 50% semantic + 50% keyword
)
```

| Method | Good At | Bad At |
|---|---|---|
| **BM25 (keyword / sparse)** | Exact matches — codes, names, IDs, acronyms | Synonyms, paraphrasing, meaning |
| **Vector (dense)** | Semantic similarity, paraphrasing | Exact strings, rare identifiers |
| **Hybrid** | Both — combines their strengths | Adds complexity; requires tuning weights |

The A6 demo runs four queries (`SKU-7742X specifications`, `E_CONN_REFUSED error`, `How do I check network connectivity?`, `router configuration`) against VECTOR, BM25, and HYBRID retrievers side by side — the keyword-heavy queries expose where pure vector search misses and hybrid recovers.

### How results are merged: Reciprocal Rank Fusion (RRF)

```
Hybrid Score = α × BM25 Score + (1 - α) × Vector Score
```

```
User Query: "SKU-7742X specifications"
        ├──► Vector Search  → Doc3[Rank 1], Doc7[Rank 2], Doc1[Rank 5]
        └──► BM25 Search    → Doc1[Rank 1], Doc3[Rank 2]
                                    │
                          Reciprocal Rank Fusion (RRF)
                                    │
                          1. Doc3 (top in both)
                          2. Doc1 (strong in BM25, present in vector)
```

**RRF formula:**
```
RRF_score(doc) = Σ  1 / (k + rank_in_retriever)      # k = 60 (standard constant)
                retrievers
```

`k = 60` dampens the advantage of very top ranks, so documents appearing *consistently* across retrievers beat those dominating a *single* retriever. A doc ranking well in both semantic and keyword search is almost certainly relevant.

### `EnsembleRetriever` weights

| Weight config | Effect |
|---|---|
| `[0.5, 0.5]` | Equal hybrid — good general default |
| `[0.7, 0.3]` | BM25-heavy — queries are ID/code-heavy |
| `[0.3, 0.7]` | Vector-heavy — queries are conversational |

### Production considerations

- **BM25 is a static index** — built once over a fixed corpus with no incremental updates. New documents are invisible until you rebuild: `BM25Retriever.from_documents(all_documents, k=4)`. For real-time freshness, use Elasticsearch/OpenSearch instead.
- **Tune `k` higher than you need** — RRF needs enough candidates to re-rank. `k=4–6` is a good default; `k=1–2` starves the fusion pool.
- **Latency** — the two retrievers run in parallel, so hybrid adds only ~20–25ms over vector-only (`max(bm25, vector) + RRF`, not the sum).
- **Context overflow** — merging two retrievers means more chunks. Budget tokens before setting `k`: `max_chunks = (context_limit − system_prompt − response_reserve) // avg_chunk_tokens`.

---

## A7 — Cost Optimization: Token Budgeting & Semantic Caching (`A7_costOptimization.py`)

Two levers for cutting LLM cost: **budgeting** tokens per request, and **caching** responses so repeated questions skip the LLM entirely.

### Token budgeting

```python
class TokenBudget:
    def estimate_tokens(self, text):        # rough: words × 1.3 (real code uses tiktoken)
        return int(len(text.split()) * 1.3)
    def check_budget(self, text):           # (within_budget, token_count)
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_per_request, tokens
```

`BudgetedLLM` wraps `ChatOpenAI`, rejecting any query whose estimated tokens exceed the budget **before** paying for the call, and tracking cumulative usage (`total_input`, `total_output`, `requests`, `avg_per_request`). This prevents a single runaway prompt from blowing the budget and gives you per-run cost visibility.

### Semantic caching — two approaches

**1. Hash-based cache (exact match after normalization):**

```python
def hash_query(self, query):
    normalized = query.lower().strip()             # normalize
    return hashlib.md5(normalized.encode()).hexdigest()   # fixed-length key
```

Two layers:
1. **Normalize** — lowercase, strip whitespace/punctuation, so `"What is AI?"` and `"what is ai?"` collapse to the same key.
2. **Hash** — turn the normalized string into a fixed-length key so lookups stay cheap regardless of query length.

**Drawback:** it only catches queries identical *after normalization*. Semantically similar but differently-worded queries still miss:

```
"What is AI?" vs "Explain artificial intelligence"
→ different strings → different hashes → cache miss → unnecessary LLM call
```

**2. Embedding-based semantic cache (matches by meaning):**

```python
class EmbeddingSemanticCache:
    def get(self, query):
        query_vector = self.embedder.embed_query(query)
        # find the closest cached query by cosine similarity
        best_entry, best_score = max(
            ((e, cosine_similarity(query_vector, e["embedding"])) for e in self.cache),
            key=lambda x: x[1], default=(None, 0.0))
        if best_entry and best_score >= self.threshold:   # e.g. 0.85
            return best_entry["response"], best_score
        return None, best_score
```

Steps: **embed** the query → **search** the cache by vector similarity → **threshold check** (return cached response if similarity ≥ threshold, else call the LLM and store the new pair). This catches paraphrases (`"What is AI?"` ≈ `"Explain artificial intelligence"`), at the cost of an embedding call and a similarity scan per lookup instead of an O(1) dict lookup.

Both wrappers track `cache_hits`, `cache_misses`, and `hit_rate`.

> **When caching pays off:** query distributions with a "long tail" — a small set of popular questions asked by many users. See the [vector-DB cost optimization appendix](#vector-database-cost-optimization) for storage-side levers (dimension reduction, quantization).

---

## A8 — Observability with LangSmith (`A8_langsmith_setup.py`)

Before observability, understand *why LLM pipelines are uniquely hard to debug* — LangSmith exists to solve exactly these:

| Problem | What happens |
|---|---|
| **Non-determinism** | Same input, different output (`temperature > 0`) — bugs don't reproduce reliably. Use `temperature=0` for tests. |
| **Cascading errors** | One bad step (e.g. wrong chunk retrieved) silently poisons everything downstream — no exception, just a confident wrong answer. |
| **Silent failure** | The pipeline returns HTTP 200 with a hallucinated answer; no alert fires. |
| **Cost surprise** | An agent loop meant to run twice runs ten times — works correctly but costs 5× more, unlogged. |

LangSmith makes every step visible.

### What LangSmith captures per run

| Signal | What you see |
|---|---|
| Inputs / Outputs | Exact prompt sent and response received at every step |
| Latency | Time taken by each node (retriever, LLM, parser) |
| Token usage | Prompt tokens, completion tokens, and cost per step |
| Errors | Which step threw, with the full input that caused it |
| Trace tree | Parent → child relationship of every chain, tool, and agent call |

### Setup — environment variables only

```python
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")     # enables tracing globally
os.environ.setdefault("LANGCHAIN_PROJECT", "Freecodecamp-RAG")  # groups traces
# LANGCHAIN_API_KEY comes from .env
```

Once `LANGCHAIN_TRACING_V2=true`, **every** `chain.invoke()` is captured automatically — no code changes to the chain. The A8 RAG chain (retriever | prompt | llm | parser over Chroma) traces each node's inputs, outputs, latency, and tokens with zero extra instrumentation.

### `@traceable` — trace plain Python functions

For logic outside LangChain components (validation, pre/post-processing), wrap it to appear as a named span:

```python
from langsmith import traceable

@traceable(name="validate_query")
def validate_query(query):
    if not query.strip():
        raise ValueError("Query must not be empty.")
    return query.strip()

@traceable(name="rag_pipeline")
def run_rag(query):
    clean = validate_query(query)      # child span
    answer = rag_chain.invoke(clean)   # LangChain chain — also a child span
    return {"query": clean, "answer": answer}
```

Resulting trace tree:
```
rag_pipeline
  ├── validate_query          (your custom logic)
  └── RunnableSequence        (LangChain chain)
        ├── retriever
        ├── ChatPromptTemplate
        ├── ChatOpenAI        (tokens, cost, latency)
        └── StrOutputParser
```

### Manual feedback — log human signals

```python
from langsmith import Client

def log_feedback(run_id, score, comment=""):
    Client().create_feedback(run_id=run_id, key="user_feedback",
                             score=score, comment=comment)   # score 1=correct, 0=wrong
```

After a user rates an answer (thumbs up/down), log it back to the run. Over time these accumulate into a labeled dataset to measure whether pipeline changes actually improve quality.

---

## A9 — Semantic Chunking (`A9_semanticChunking.py`)

Recursive chunking splits on *characters*; **semantic chunking** splits on *meaning* — it embeds sentences and starts a new chunk wherever the topic shifts (a large embedding-distance "breakpoint"). A9 runs both on the same Authentication Guide and compares them.

```python
from langchain_experimental.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",   # how breakpoints are chosen
    breakpoint_threshold_amount=90,           # split where distance is in the top 10%
)
semantic_chunks = semantic_splitter.split_text(document)
```

**Key parameters:**

- **`breakpoint_threshold_type`** — the method for deciding what counts as a topic shift: `percentile` (default), `standard_deviation`, `interquartile`, or `gradient`.
- **`breakpoint_threshold_amount=90`** — with `percentile`, split at points where the sentence-to-sentence embedding distance is in the top 10%. Higher = fewer, larger chunks.

### Comparing the two strategies

The `compare_chunking_strategies()` function reports **total chunks, average/min/max length, and size variance** for each approach. The typical finding: recursive produces uniform-size chunks (low variance, predictable), while semantic produces variable-size chunks that better respect topic boundaries.

`test_retrieval()` then runs real queries (rate limiting, OAuth2 security, webhook retries) against both vector stores and reports overlap — showing that the "better" strategy depends on whether your priority is consistency (recursive) or topical coherence (semantic).

> Semantic chunking costs more at indexing time (it embeds every sentence to find breakpoints) but can improve retrieval on documents with clear topic structure.

---

## A10 — Production Chunking: Semantic with Fallback (`A10_prodReady.py`)

Semantic chunking is higher quality but can misbehave — producing too few chunks or one giant chunk on oddly structured input. A production system needs a **fallback**. A10 wraps semantic chunking with automatic degradation to recursive.

```python
USE_SEMANTIC_CHUNKING   = True
MIN_SEMANTIC_CHUNK_COUNT = 2      # fallback if semantic yields too few chunks
MAX_SEMANTIC_CHUNK_SIZE  = 3000   # fallback if any chunk is unreasonably large

def chunk_document(text, use_semantic_chunking=True) -> tuple[list[str], str]:
    if not use_semantic_chunking:
        return get_recursive_chunks(text), "recursive"
    try:
        chunks = get_semantic_chunks(text)
        if len(chunks) < MIN_SEMANTIC_CHUNK_COUNT:
            return get_recursive_chunks(text), "recursive (fallback: too few chunks)"
        if any(len(c) > MAX_SEMANTIC_CHUNK_SIZE for c in chunks):
            return get_recursive_chunks(text), "recursive (fallback: oversized chunks)"
        return chunks, "semantic"
    except Exception as e:
        return get_recursive_chunks(text), "recursive (fallback: exception)"
```

**The pattern — primary strategy with guarded fallback — triggers recursive chunking when:**

1. Semantic chunking is disabled by config, **or**
2. `SemanticChunker` raises an exception, **or**
3. The result has fewer than `MIN_SEMANTIC_CHUNK_COUNT` chunks, **or**
4. Any single chunk exceeds `MAX_SEMANTIC_CHUNK_SIZE` characters.

The function returns **`(chunks, strategy_used)`** so downstream code and logs know *which* strategy actually ran. This "try the good thing, fall back to the safe thing, and record what happened" shape is the core lesson — the same defensive pattern applies to model routing, retrieval, and generation in production RAG.

---

## A11 — Advanced Retrieval (`A11_advanced_rag.py`)

Standard similarity search often returns irrelevant chunks or misses relevant ones when query phrasing doesn't match document phrasing. A11 demonstrates three fixes on a tech-docs knowledge base.

### 1. Multi-Query Retriever

**Problem:** a single query vector may miss documents that use different vocabulary for the same idea.
**Solution:** use an LLM to generate several rephrasings, retrieve for each, and deduplicate.

```python
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), llm=llm)
docs = retriever.invoke("What tools can I use to build AI applications?")
```

Enable logging to watch the generated variations:
```python
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```
**Cost:** one extra LLM call per query. **Use for:** open-ended, conceptual questions.

### 2. Contextual Compression Retriever

**Problem:** retrieved chunks bury relevant facts inside mostly-irrelevant text; the LLM gets noise.
**Solution:** after retrieval, pass each chunk through an LLM extractor that keeps only query-relevant sentences.

```python
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
)
```

| | Without Compression | With Compression |
|---|---|---|
| Chunk size sent to LLM | 300–500 tokens each | 20–80 tokens each |
| LLM calls | 1 (generation) | 1 per retrieved chunk + 1 generation |
| Answer quality when docs are noisy | Degrades | Stays high |

### 3. Parent Document Retriever

**Problem:** small chunks embed precisely but strip context; large chunks give context but retrieve noisily.
**Solution:** index small **child** chunks for search precision, return the large **parent** chunk for context.

```python
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,   # searches small child chunks
    docstore=InMemoryStore(),  # fetches large parent chunks by ID
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents([long_doc])
```

```
Indexing:  Document → parent_splitter → Parent Chunks (docstore, keyed by ID)
                    → child_splitter  → Child Chunks (vectorstore, tagged with parent ID)
Retrieval: Query → search vectorstore → matching Child → look up parent ID → return Parent Chunk
```

### Which advanced retriever to use?

| Strategy | Fixes | Cost | When to Use |
|---|---|---|---|
| **Multi-Query** | Vocabulary mismatch, low recall | +1 LLM call/query | Open-ended, conceptual queries |
| **Contextual Compression** | Noisy chunks | +1 LLM call/chunk | Relevant info buried in noise |
| **Parent Document** | Small-chunk context loss | No extra LLM calls | Long docs needing surrounding context |

These can be **combined**: Multi-Query → Parent Document → Contextual Compression for maximum quality at higher cost. See also [The 5 RAG Failure Modes](#the-5-rag-failure-modes) in the appendix.

---

## A12 — Monitoring: The Three Pillars (`A12_monitoring.py`)

Once a RAG pipeline is live, monitoring tells you it's actually working. A12 builds all three pillars around an instrumented LLM.

1. **Structured Logging** — a human-readable story in a machine-parseable format (JSON) so logs can be searched, filtered, aggregated.
2. **Metrics Collection** — numbers on a dashboard (latency, tokens, error rate, cache hit rate) revealing trends.
3. **Instrumented LLM Calls** — wrapping every LLM call to capture inputs, outputs, tokens, latency, and cost automatically.

> **Monitoring is the outermost layer** — it observes everything but changes nothing. Its place in the stack: `Security → Cost Optimization → Error Handling → Monitoring`.

### Structured JSON logging

```python
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        })
```

A custom `logging.Formatter` emits each log line as JSON — trivially ingested by log aggregators (Datadog, CloudWatch, ELK) for querying and alerting.

### Metrics collection

`MetricsCollector` aggregates counters (`requests_total`, `errors_total`, `latency_sum`, `tokens_input/output`, `cache_hits/misses`) and derives a summary: **average latency, error rate, and cache hit rate**. `record_request()` is called on every request; `get_summary()` produces the dashboard view.

### Instrumented LLM

`InstrumentedLLM.invoke()` ties it together: times each call, estimates tokens, records the metric, logs a structured line on success, logs an error and re-raises on failure — and is wrapped with `@traceable` so it also shows up in LangSmith. This is the production shape: **every** LLM call is measured and logged, so failures and cost spikes surface on a dashboard instead of in a user complaint.

---

## A13 — Long Context vs RAG (`A13_LongContextVsRAG.py`)

Modern LLMs support 1M+ token context windows, raising the question: **is RAG dead?** No — the choice between stuffing a whole corpus into the prompt ("long context") and retrieving only relevant pieces ("RAG") is a trade-off of **cost, latency, and use case**.

> **The real answer isn't "RAG vs Long Context" — it's "RAG AND Long Context."**

### Cost comparison

**Scenario:** query against 100,000 tokens of docs; query 100 tokens; output 500 tokens (pricing $2.50/1M input, $10/1M output):

| Approach | Input tokens | Cost per query |
|---|---|---|
| **Long Context** (stuff everything) | 100,100 | $0.25525 |
| **RAG** (4 chunks × 500 tokens) | 2,100 | $0.01025 |

RAG is **~25× cheaper per query**. At 10,000 queries/day: Long Context ≈ $2,552/day vs RAG ≈ $102/day — **~$73,500/month saved**. The gap only grows with corpus size, since RAG cost depends on `k` and chunk size, *not* corpus size.

### Latency comparison

More input tokens = more to process before generating. At small scale the difference is noisy (network/queueing variance dominates — the demo even shows the small context occasionally slower). At 100K+ tokens the gap becomes dramatic.

| Context size | Effect on latency |
|---|---|
| Small (~50–500 tokens) | Minimal difference — variance dominates |
| Medium (~2,500 tokens) | Measurable |
| Large (100K+ tokens) | Long context grows substantially; RAG stays ~constant |

### Why long context isn't always better

- **"Lost in the middle"** — models retrieve facts near the *start*/*end* of a prompt better than those buried in the middle.
- **No citations by default** — no structural signal of which passage an answer came from.
- **Reprocessing cost on every change** — a changed doc means re-sending (and re-paying for) the whole corpus; RAG only re-embeds the changed chunk.
- **Prompt caching changes the math (partially)** — caching can cut repeated-query cost against a *static* large context by up to ~90%, narrowing the gap — but not when the corpus is large *and* changes often *and* volume is high.

### Decision framework

| Factor | Favors Long Context | Favors RAG |
|---|---|---|
| Corpus size | Small (< 50K tokens) | Large (> 100K tokens) |
| Query volume | Low (< 100/day) | High (1000s/day) |
| Query type | Whole-document analysis | Specific, targeted questions |
| Document volatility | Changes often | Relatively stable |
| Cost sensitivity | Low priority | High priority |
| Need for citations | Not required | Required |

### Hybrid approach (what the demo builds)

```python
# 1. RAG finds the right document (cheap, fast)
relevant_docs = vectorstore.as_retriever(search_kwargs={"k": 1}).invoke(query)
# 2. Load the FULL document into context (not just the matched chunk)
full_doc = relevant_docs[0].page_content
# 3. Generate a detailed answer over the complete document
response = (prompt | llm).invoke({"document": full_doc, "query": query})
```

Use RAG to **find** the right document(s) cheaply, then load the **full** document for a complete answer. Best for "find the relevant doc, then analyze it thoroughly" — HR policy lookups, contract analysis, single-report Q&A. You pay for one document's tokens, not the whole corpus.

---

## A14 — Contextual Retrieval (`A14_ContextualRetrival.py`)

Anthropic's technique (introduced late 2024). Chunking strips the context each chunk depends on; Contextual Retrieval has an LLM prepend a short, disambiguating context to every chunk **before** embedding. Anthropic reports **67% fewer retrieval failures**.

### The problem

```
Chunk 1: "The company specializes in..." — Which company?
Chunk 2: "Revenue for fiscal year 2025 reached $4.2 billion" — For what company?
Chunk 3: "The company plans to expand..."  — Plans of what company?
```

A query *"What is ACME's revenue?"* may fail to match Chunk 2 because the literal string "ACME" never appears in it.

### The solution

Give the LLM the **chunk** and its **full document**; ask for a 1–2 sentence prefix that situates it (entities, title, surrounding facts); prepend that prefix before embedding:

```python
def add_contextual_prefix(chunk, full_document, document_title, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a SHORT context (1-2 sentences) that situates the chunk "
                   "within the document. Output ONLY the prefix."),
        ("human", "Document Title: {title}\n\nFull Document:\n{document}\n\n"
                  "Chunk to contextualize:\n{chunk}"),
    ])
    return (prompt | llm).invoke(
        {"title": document_title, "document": full_document, "chunk": chunk}).content

contextualized_chunk = f"{add_contextual_prefix(chunk, doc, title, llm)} {chunk}"
```

The demo embeds raw vs contextualized chunks into separate Chroma collections and compares `similarity_search_with_score` — showing contextual chunks match ACME queries better (lower distance in Chroma). `create_contextual_chunks()` is the production-ready version that splits, contextualizes, and stores `original_chunk` + `context_prefix` in metadata.

### Production considerations

| Factor | Impact |
|---|---|
| **Cost** | ~$0.01–$0.05 per document — one-time indexing cost, far cheaper than retrieval failures |
| **Latency** | +1–2s per chunk during indexing; no query-time impact (batch offline) |
| **Storage** | Chunks grow ~20–30% larger — minimal DB-cost effect |
| **Result** | Anthropic: **67% fewer retrieval failures** |

**When to use:** context lives in headers/titles, entities referenced with pronouns ("the company", "they"), or corpora from multiple sources. Combine with BM25 hybrid search (A6) for best results.

---

## A15 — Late Chunking (`A15_lateChunking.py`)

Contextual Retrieval (A14) fixes context loss by *describing* it with an LLM. **Late chunking** fixes it *architecturally*: embed the full document **first** (so every token attends to the whole document), then split the token-level embeddings into chunk vectors.

### Early vs late chunking

```
EARLY CHUNKING:                          LATE CHUNKING:
Document → Split → Chunk 1,2,3           Document → Embed FULL doc (token embeddings)
              ↓        ↓       ↓                      → Split embeddings by position
           Embed    Embed   Embed                     → Vector 1,2,3 (pooled)
              ↓        ↓       ↓
          Vector1  Vector2 Vector3

Each chunk embedded INDEPENDENTLY.       Each chunk vector "knows" the whole document —
No cross-chunk context.                  "he" is embedded knowing "he" = "Steve Jobs."
```

**Failure mode it fixes:** a biography chunked into sentences yields *"He co-founded Apple Computer in 1976"* with no "Steve Jobs" — because that name appeared only in an earlier chunk. Under early chunking, a query for "companies Steve Jobs founded" may not match at all. The A15 demo counts how many chunks contain only the pronoun "he" and shows the problem concretely.

> **Note:** True late chunking needs token-level embedding access (e.g. Jina Embeddings v3 with `late_chunking=True`). OpenAI's API doesn't expose this, so A15 *simulates* the concept via context-prepending rather than true token-level pooling.

### Implementation options

| Approach | Context Quality | Implementation Cost |
|---|---|---|
| Early chunking (traditional) | Poor — pronouns orphaned | Free — standard approach |
| Overlapping chunks | Better — some context kept | Free — config change |
| **Contextual Retrieval** (A14) | Excellent — LLM adds context | ~$0.01/doc (LLM cost) |
| **Late chunking (native)** | Excellent — full doc context | Needs token-level model (Jina) |
| Parent-Child Retriever (A11) | Excellent — returns parent doc | Extra storage (2× vector store) |

**Accuracy vs traditional chunking:** overlap +3–5%, Contextual Retrieval +15–20%, Late Chunking +10–12%, combined +25–30%.

**Recommendation:** start with Contextual Retrieval (works with any embedding model); move to native late chunking (Jina) if you control the embedding model; combine with a Parent Document Retriever for the best of both.

---

## A16 — Agentic RAG with LangGraph (`A16_agenticRAGpy`)

Traditional RAG is **one-shot**: retrieve, then generate — no recovery if retrieval returns garbage. **Agentic RAG** makes it a loop that grades its own retrieval and self-corrects: `retrieve → grade → [rewrite & retry] → generate`. This is the production RAG pattern for 2026.

### State schema (LangGraph 1.x uses `TypedDict`, not Pydantic)

```python
class RAGState(TypedDict):
    query: str
    rewritten_query: str
    documents: list[Document]
    generation: str
    relevance_score: float
    retry_count: int
    max_retries: int
```

### Graph structure

```
START → RETRIEVE → GRADE ──┬── (good relevance) ───────────────→ GENERATE → END
                            ├── (low relevance, retries left) → REWRITE → back to RETRIEVE
                            └── (no docs, out of retries) ────→ FALLBACK → END
```

```python
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate_answer)
workflow.add_node("fallback", generate_fallback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", should_retry_or_generate,
    {"rewrite": "rewrite", "generate": "generate", "fallback": "fallback"})
workflow.add_edge("rewrite", "retrieve")   # the self-correcting loop
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)
app = workflow.compile()
```

### The nodes

- **`retrieve_documents`** — searches the vectorstore using `rewritten_query` if present, else `query`.
- **`grade_documents`** — the key difference from traditional RAG: an LLM scores each doc's relevance 0–1; docs scoring ≥ 0.5 are kept, and the average becomes `relevance_score`. Evaluation happens *before* generation.
- **`rewrite_query`** — reformulates the query (synonyms, specificity) and increments `retry_count`.
- **`generate_answer`** — produces the answer with source citations from the kept docs.
- **`generate_fallback`** — a graceful "I couldn't find relevant information" response.

### The router: `should_retry_or_generate`

| Condition | Route |
|---|---|
| `relevance_score >= 0.5` and documents exist | → **generate** |
| Low relevance, `retry_count < max_retries` | → **rewrite** (reformulate, loop back) |
| Out of retries but some documents exist | → **generate** (best effort) |
| Out of retries and zero documents | → **fallback** (graceful "I don't know") |

The demo runs three queries — two that succeed on first retrieval, and *"How do I make pizza?"* which exhausts retries and hits the fallback path.

**When to use:** complex queries needing reformulation, high-stakes answer quality, diverse document types, user-facing (not batch) apps. Cost is extra LLM calls per grade/rewrite — worth it when a wrong or "I don't know" answer is expensive.

> **Note:** this file is named `A16_agenticRAGpy` (no `.py` extension) — likely a typo. It runs fine as a Python script; rename to `A16_agenticRAG.py` for consistency if you like.

---

## A17 — GraphRAG: Multi-Hop Reasoning (`A17_graphRAGIntro.py`)

Vector search answers "what documents are similar to this query" — but can't answer questions that require **traversing relationships** between entities across documents.

### The multi-hop problem

```
Query: "Who works in the same department as the CEO's assistant?"
Requires chaining: CEO → CEO's assistant → their department → other employees there
```

Vector search for "CEO's assistant" won't match *"Sarah Johnson works in the Executive department"* — Sarah's name isn't in the query, so embeddings don't align. The answer requires following a chain of relationships, not matching one similar passage.

### The solution: knowledge graphs

GraphRAG extracts **entities** (nodes) and **relationships** (edges) into a traversable graph, then walks the graph to answer:

```python
import networkx as nx
G = nx.DiGraph()
G.add_node("Sarah Johnson", type="Person", role="Executive Assistant")
G.add_edge("Sarah Johnson", "John Smith", relation="ASSISTANT_TO")
G.add_edge("Sarah Johnson", "Executive Department", relation="WORKS_IN")
```

The demo builds this graph by hand and traverses it step by step (CEO → assistant → department → coworkers) to answer the multi-hop query deterministically. In production, entity/relationship extraction is automated with an LLM:

```python
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract ENTITIES (people, orgs, places, concepts) and "
               "RELATIONSHIPS (WORKS_FOR, MANAGES, LOCATED_IN, ...) from the text."),
    ("human", "Extract entities and relationships from this text:\n\n{text}"),
])
```

### Architecture — two phases

```
INDEXING:  Documents → LLM extracts entities/relations → Knowledge Graph
                                                             → Community Detection (group related entities)
                                                             → LLM generates community summaries
QUERYING:  LOCAL SEARCH  (multi-hop): identify query entities, traverse relationships
           GLOBAL SEARCH (holistic):  aggregate knowledge via community summaries
```

### Implementation options

| Option | Pros | Cons |
|---|---|---|
| **Microsoft GraphRAG** | Full-featured, production-ready | Heavy — significant indexing compute |
| **LangGraph + Neo4j** (`GraphCypherQAChain`) | Full control, existing Neo4j | Must manage Neo4j, manual extraction |
| **LlamaIndex `KnowledgeGraphIndex`** | Easy to use | Less powerful than full GraphRAG |
| **Hybrid (Vector + Graph)** | Vector narrows candidates, graph handles multi-hop | More complex |

### When to use GraphRAG

| Use it | Skip it |
|---|---|
| Documents describe relationships (org charts, papers) | Simple fact retrieval — standard RAG suffices |
| Queries need multi-hop reasoning | Small document sets (< 100 docs) |
| Need global summarization across documents | Real-time indexing requirements |
| | Cost-sensitive apps — indexing is expensive |

> Requires `networkx` (see the dependency note above).

---

## A18 — Multimodal RAG with ColPali (`A18_multiModelRAG.py`)

Standard RAG extracts text from PDFs before chunking — but **text extraction is lossy** for anything visual: tables, charts, diagrams, and layout degrade or disappear.

### What text extraction destroys

A table like:
```
Region  | Q1 Target | Q1 Actual | Variance
North   | $2.5M     | $2.8M     | +12% ✓
South   | $1.8M     | $1.5M     | -17% ✗
```
commonly becomes:
```
Q1 2025 Sales by Region Region Q1 Target Q1 Actual Variance
North $2.5M $2.8M +12% South $1.8M $1.5M -17% East $3.2M $3.4M +6%
```
Row/column alignment, the ✓/✗ indicators, and the visual "South is the outlier" signal are all gone.

### The solution: vision-based document RAG (ColPali)

Instead of extracting text, **convert each PDF page to an image and embed the image directly**:

```
PDF → Convert to images → Embed images (ColPali) → Store in Vector DB
Query → Embed query (ColPali) → Find similar page images → Return page IMAGES
                                                                ↓
                                                        Vision LLM (GPT-4o / Claude)
                                                        "sees" the table/chart/diagram
```

ColPali (built on Google's PaliGemma) produces **one embedding per page image** capturing both text and visual layout — no OCR or extraction step. The retrieved page image goes to a vision-capable LLM that answers directly about tables, charts, and diagrams.

```python
# Core pipeline shape (A18 has the full reference implementation)
images     = pdf_to_images(pdf_path)                     # pdf2image
embeddings = embed_document_images(images)               # ColPali (needs GPU)
results    = search_documents(query, embeddings, images) # similarity over image embeddings
answer     = answer_with_vision(query, results)          # GPT-4o/Claude vision call
```

A18's runnable `demo_vision_llm_analysis()` sends a described sales table to a vision-capable model (`gpt-4o`) and answers "which region is underperforming and by how much?" — illustrating what vision preserves that extraction loses.

### Cost trade-off

| | Text RAG | Multimodal RAG (ColPali) |
|---|---|---|
| Embedding | ~$0.0001/page | ~$0.001/page (GPU) |
| Query | ~$0.01/query (GPT-4o-mini) | ~$0.10/query (GPT-4o w/ images) |
| **Overall** | Baseline | **~10× more expensive** — but preserves visual info |

### When to use it

| Good fit | Not necessary |
|---|---|
| Financial reports (nested tables, charts) | Plain text (novels, articles) |
| Technical docs (architecture diagrams, flowcharts) | Simple structured/CSV data |
| Scientific papers (figures, equations) | Text extraction already works well |
| Legal documents (formatted contracts) | Real-time apps (vision is slower) |
| Medical records (diagnostic images, lab tables) | Cost-sensitive apps |

---

# Appendix — Cross-Cutting Deep Dives

These topics span multiple lessons rather than a single file.

---

## The 5 RAG Failure Modes

Even a well-architected pipeline can silently fail. (Relevant to A2–A6, A11.)

### 1. Bad Chunking — Wrong Cuts
Splits at the wrong boundary (mid-sentence, mid-table) leave each chunk missing needed context.
**Fix:** recursive/semantic chunking (A4, A9); add overlap; use code/markdown-aware splitters.

### 2. Embedding Mismatch — Different Vocabularies
Query wording differs from document wording, so vectors land far apart despite equivalent meaning ("heart attack" vs "myocardial infarction").
**Fix:** domain-specific embedding model; query expansion / HyDE; hybrid search (A6); Multi-Query (A11).

### 3. Retrieval Noise — Irrelevant Results
High-scoring but irrelevant chunks drown out the signal.
**Fix:** add a cross-encoder re-ranker; metadata filtering (A2); tune `k` down to 3–5.

### 4. Context Overflow — Truncated Context
Retrieved chunks + prompt + history exceed the context window; content is silently dropped.
**Fix:** budget tokens (`context_limit − system_prompt − response_reserve`); reduce `k`/chunk size; map-reduce/refine; larger-context model.

### 5. Hallucination — Confident Invented Answers
The LLM answers from parametric knowledge when retrieved context is insufficient.
**Fix:** strict "answer only from context, else say 'I don't know'" prompts (A3); faithfulness evaluation; forced citations (A3).

---

## HNSW — How Vector Databases Find Similar Vectors Quickly

Relevant to every lesson that stores embeddings (A2, A3, A6, A9, A11...). A vector DB doesn't compare your query against every vector — that's too slow at millions of vectors. Almost all (Pinecone, Weaviate, Qdrant, Milvus, FAISS) use **HNSW — Hierarchical Navigable Small World**.

> **One-liner:** HNSW is an approximate nearest neighbor (ANN) algorithm that builds a multi-layer graph so search skips most data and navigates only through well-connected "shortcut" nodes.

**Mental model — a highway system:** top layers are highways (few nodes, long-range jumps); bottom layers are local streets (all nodes, short hops). Search enters at the top, greedily jumps toward the query, and drops down layer by layer — skipping ~99% of the data.

### The 3 core parameters

**M — max connections per node**

| M | Memory | Build | Recall | Use |
|---|---|---|---|---|
| Low (8–12) | Low | Fast | Lower | Resource-constrained |
| Medium (16–32) | Moderate | Moderate | Good | **Production sweet spot** |
| High (48–64) | High | Slow | Best | Accuracy-critical |

Changing M requires **rebuilding the index**.

**efSearch — effort at query time** (the one you tune most)

| efSearch | Speed | Recall | Notes |
|---|---|---|---|
| Low (10–30) | Fast | Lower | May miss the true best match |
| Medium (50–100) | Balanced | Good | **Recommended start** |
| High (200+) | Slow | Best | Accuracy > latency |

Must be ≥ `k`; **changeable at runtime** without rebuilding.

**efConstruction — effort at build time** — higher builds a smarter graph (100–200 is a good production default). Like M, changing it means rebuilding.

### The tradeoff triangle

```
         Speed
        /     \
   Memory ——— Accuracy (Recall)
```
You can't optimize all three: more accuracy (↑M, ↑efSearch, ↑efConstruction) costs memory and speed; more speed (↓efSearch) costs recall; memory savings (↓M) cost recall.

### ANN vs exact search

| | Exact NN | ANN (HNSW) |
|---|---|---|
| Guarantee | Always the best match | Best match ~95–99% of the time |
| Speed at 1M vectors | O(n) — slow | O(log n) — milliseconds |
| Production use | Rare | Industry standard |

For RAG, 99% recall is plenty — a slightly sub-optimal chunk still yields a correct answer almost always.

---

## Self-Hosted vs Managed Vector Database

Relevant once A2/A3's stores go to production.

- **Self-hosted** — you run it (Qdrant, Milvus, Weaviate, pgvector on your Postgres).
- **Managed** — a vendor runs it (Pinecone, Weaviate Cloud, Qdrant Cloud, MongoDB Atlas).

| | Self-Hosted | Managed |
|---|---|---|
| Cost at low scale | Low (just infra) | Can be expensive (per-vector) |
| Cost at high scale | Cheaper | Expensive — grows with vector count |
| Operational burden | High — upgrades, backups, scaling | Low — vendor handles it |
| Data privacy | Full control | Data leaves your network |
| Scaling | Manual | Automatic |
| Time to first query | Days | Minutes |
| Customization | Full HNSW/hardware control | Limited to vendor's exposure |

**Choose managed** for prototypes, small teams, no ops burden. **Choose self-hosted** for compliance-heavy industries, large scale (50M+ vectors), or fine-grained HNSW tuning.

> The decision is rarely purely technical — it's driven by compliance, team size, and cost at scale.

---

## Vector Database Cost Optimization

Storage-side levers (complements A7's LLM-side token budgeting and caching).

| Strategy | Savings | Effort | When |
|---|---|---|---|
| **Reduce dimensions** (e.g. 1536 → 512 via OpenAI `dimensions` param) | 30–60% storage | Low | Model supports native reduction |
| **Quantization** (float32 → int8/binary) | 50–75% memory | Medium | Large indexes, memory-bound |
| **Batch queries** (one API call, many queries) | 10–30% | Low | Pipelines processing queries one at a time |
| **Semantic caching** (A7) | 10–40% | Medium | High rate of repeated queries |
| **Right-size the index** (don't over-provision) | 20–50% | Low | Managed deployments provisioned "safely" |

Combining dimension reduction + quantization can store ~2.5× more vectors for the same price. Cost optimization is operational discipline, not premature optimization.

---

## Reference: File → Concept Map

| File | Core concepts |
|---|---|
| A1 | Document/metadata, PyPDFLoader, TextLoader, loader types |
| A2 | Chunking strategies, embedding dimensions, RAG phases, 3 rules; Qdrant collection/payload index/filtered search |
| A3 | LCEL chain, Chroma persist, retriever, source attribution |
| A4 | Recursive/Character/Token/Markdown/Language splitters, chunk overlap |
| A5 | embed_query/documents, normalization, cosine similarity, `CacheBackedEmbeddings` |
| A6 | BM25, `EnsembleRetriever`, RRF, hybrid weights |
| A7 | Token budgeting, hash cache, embedding-based semantic cache |
| A8 | LangSmith tracing, `@traceable`, feedback logging; why LLM debugging is hard |
| A9 | `SemanticChunker`, breakpoint thresholds, strategy comparison |
| A10 | Semantic-with-recursive-fallback pattern |
| A11 | Multi-Query, Contextual Compression, Parent-Child retrievers |
| A12 | Structured JSON logging, metrics, instrumented LLM (three pillars) |
| A13 | Long context vs RAG cost/latency, decision framework, hybrid |
| A14 | Contextual Retrieval (LLM context prefixes before embedding) |
| A15 | Late chunking (embed full doc, then split) |
| A16 | Agentic RAG, LangGraph `StateGraph`, grade/rewrite/fallback |
| A17 | GraphRAG, knowledge graphs, multi-hop, `networkx` |
| A18 | Multimodal RAG, ColPali, vision LLMs |

---

*The previous thematic version of this README is preserved as `README.backup.md`.*
