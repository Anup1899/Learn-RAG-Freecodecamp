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
├── A1_documentsloader1.py   # Document loaders (PDF, Text, Directory, Web)
├── A2_chunkingEmbedding2.py # Chunking strategies and embedding concepts
├── A3_ragpipeline3.py       # Basic RAG pipeline + RAG with source attribution
├── A4_textSplitter4.py      # Advanced text splitter strategies
├── A5_embeddings_deep5.py   # Deep dive into embeddings
├── A6_hybridSearch.py       # Hybrid search with BM25 + EnsembleRetriever
├── A7_costOptimization.py   # Token budgeting and cost optimization
├── A8_langsmith_setup.py    # LangSmith observability and tracing
├── A9_semanticChunking.py   # Semantic chunking
├── A10_prodReady.py         # Production-ready RAG patterns
├── A11_advanced_rag.py      # Advanced retrieval: Multi-Query, Compression, Parent-Child
├── A12_monitoring.py        # Structured logging and production monitoring
├── A13_LongContextVsRAG.py  # Long context vs RAG: cost, latency, decision framework, hybrid demo
├── A14_ContextualRetrival.py # Contextual Retrieval — LLM-generated context prefixes before embedding
├── A15_lateChunking.py      # Late chunking — embed full doc, then split embeddings
├── A16_agenticRAGpy         # Agentic RAG with LangGraph — retrieve, grade, retry, generate
├── A17_graphRAGIntro.py     # GraphRAG — knowledge graphs for multi-hop reasoning
├── A18_multiModelRAG.py     # Multimodal RAG with ColPali — vision-based document understanding
├── docs/                    # Sample PDF documents
├── vector_store/            # Persisted Chroma vector database
├── .env                     # API keys (never commit this)
└── pyproject.toml           # Dependencies
```

---

## Dependencies

| Package | Purpose |
|---|---|
| Package | Purpose |
|---|---|
| `langchain` | Chains, retrievers, document loaders |
| `langchain-core` | Base abstractions (Runnable, BaseMessage) |
| `langchain-openai` | OpenAI LLM and embedding integration |
| `langchain-chroma` | Chroma vector store integration |
| `langchain-community` | Community integrations (PDF loaders, etc.) |
| `langchain-classic` | Legacy retrievers: ParentDocumentRetriever, ContextualCompressionRetriever, MultiQueryRetriever |
| `langsmith` | Tracing and evaluation platform |
| `openai` | OpenAI SDK |
| `python-dotenv` | Load `.env` variables |
| `langgraph` | Cyclic, stateful agent workflows (used for Agentic RAG) |
| `tiktoken` | Token counting for OpenAI models (cost estimation, budget checks) |

> **Missing dependency:** `A17_graphRAGIntro.py` imports `networkx` to build the knowledge graph, but it is not yet declared in `pyproject.toml` / `uv.lock`. Run `uv add networkx` before running that script.

> **Note:** `langchain-community` is being sunset. Prefer standalone packages (e.g. `langchain-pypdf`) when available. `langchain-classic` bundles retrievers that were moved out of `langchain` core.

---

## Model

Uses **GPT-4o Mini** (`gpt-4o-mini`) via `init_chat_model` — fast and cost-effective, ideal for learning and experimentation.

```python
from langchain.chat_models import init_chat_model
llm = init_chat_model(model="gpt-4o-mini", temperature=0.2)
```

`temperature=0` makes responses deterministic — important for RAG where consistency matters. `init_chat_model` is a provider-agnostic factory that works with OpenAI, Anthropic, and others by swapping the model string.

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

## RAG with Source Attribution

A common production requirement is showing the user **which document** each part of the answer came from. This is done by including source metadata in the formatted context before it reaches the LLM.

### How It Works

Instead of concatenating raw chunk text, each chunk is prefixed with its source path so the LLM can reference it in the answer:

```python
def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
    return "\n\n".join(formatted)
```

The prompt then instructs the LLM to include which sources it used:

```python
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below. Include which sources you used.

Context:
{context}

Question: {question}

Answer (include sources):""")
```

### Why Source Attribution Matters

| Without attribution | With attribution |
|---|---|
| LLM answer is a black box | User can verify claims against source documents |
| Hallucinations are invisible | Hallucinated facts have no matching source citation |
| Debugging requires re-running retrieval | Source label instantly shows which chunk drove the answer |

Source attribution is the first step toward **faithfulness evaluation** — checking that every claim in the answer is traceable to a retrieved chunk.

---

## Advanced RAG Retrieval

Standard similarity search (`k` nearest neighbors) often returns irrelevant chunks or misses relevant ones when the query phrasing doesn't match the document phrasing. Advanced retrieval strategies fix this.

### 1. Multi-Query Retriever

**Problem:** A single query vector may miss documents that use different vocabulary to express the same idea.

**Solution:** Generate multiple rephrased versions of the query using an LLM, run each through the retriever, and deduplicate the results. More query perspectives = better recall.

```python
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=llm
)

# The retriever internally generates ~3 query variations and retrieves for each
docs = retriever.invoke("What tools can I use to build AI applications?")
```

**When to use:** When users ask open-ended, conceptual questions where the exact wording varies. Adds one extra LLM call per query.

**Enable logging to see generated queries:**
```python
import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```

---

### 2. Contextual Compression Retriever

**Problem:** Retrieved chunks contain relevant information *buried inside* mostly irrelevant content. The LLM receives noise alongside the signal, degrading answer quality.

**Solution:** After retrieval, pass each chunk through an LLM compressor that extracts only the query-relevant sentences. The LLM receives compressed, high-signal content.

```python
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
)

docs = compression_retriever.invoke("What frameworks exist for building LLM applications?")
# Returns short, extracted snippets instead of full 500-token chunks
```

**Trade-off:**

| | Without Compression | With Compression |
|---|---|---|
| Chunk size sent to LLM | 300–500 tokens each | 20–80 tokens each |
| LLM calls | 1 (generation only) | 1 per retrieved chunk + 1 for generation |
| Answer quality when docs have noise | Degrades | Stays high |

**When to use:** When your documents contain relevant facts embedded in large blocks of irrelevant surrounding text (e.g., a company history document that also mentions LangChain in passing).

---

### 3. Parent Document Retriever

**Problem:** There is a tension between chunk size for retrieval and chunk size for context:
- **Small chunks** → precise embedding matches, but strip surrounding context
- **Large chunks** → richer context for the LLM, but noisy embeddings that retrieve off-topic content

**Solution:** Index small child chunks for retrieval precision, but return the large parent chunk they came from so the LLM gets full context.

```python
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

vectorstore = Chroma(collection_name="parent_child_demo", embedding_function=embeddings)
store = InMemoryStore()  # stores parent chunks by ID

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,   # searches small child chunks
    docstore=store,            # fetches large parent chunks
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents([long_doc])
parent_docs = retriever.invoke("What is LangGraph used for?")
# Returns the 800-token parent chunk, not the 200-token child that matched
```

**How the two-store architecture works:**

```
Indexing:
  Document → parent_splitter → Parent Chunks (stored in docstore with ID)
                              → child_splitter → Child Chunks (stored in vectorstore, tagged with parent ID)

Retrieval:
  Query → embed → search vectorstore → find matching Child Chunk
       → look up parent ID → fetch Parent Chunk from docstore → return to LLM
```

**When to use:** Long documents (guides, manuals, reports) where precise retrieval matters but the answer requires surrounding context to be complete.

---

### Comparison: Which Advanced Retriever to Use?

| Strategy | Fixes | Cost | When to Use |
|---|---|---|---|
| **Multi-Query** | Vocabulary mismatch, low recall | +1 LLM call per query | Open-ended, conceptual queries |
| **Contextual Compression** | Noisy chunks, irrelevant context | +1 LLM call per chunk | Docs with relevant info buried in noise |
| **Parent Document** | Small-chunk context loss | No extra LLM calls | Long documents needing surrounding context |

These strategies can be **combined**: e.g., Multi-Query → Parent Document → Contextual Compression for maximum retrieval quality at higher cost.

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



---

## How the Hybrid Search Pipeline Works (Step by Step)

For a query like `"SKU-7742X specifications"`, here is what happens inside the pipeline:

```
User Query: "SKU-7742X specifications"
        │
        ├──► Vector Search (semantic)    → Doc3[Rank 1], Doc7[Rank 2], Doc1[Rank 5]
        │
        └──► BM25 Search (keyword)       → Doc1[Rank 1], Doc3[Rank 2], Doc5[Rank 53]
                                                    │
                                        Reciprocal Rank Fusion (RRF)
                                                    │
                                        Doc1 appears Rank 5 (vector) + Rank 1 (BM25)
                                        Doc3 appears Rank 1 (vector) + Rank 2 (BM25)
                                                    │
                                           Final Ranked Results
                                        1. Doc3  (top in both)
                                        2. Doc1  (strong in BM25, present in vector)
```

**Why RRF works:** A document that ranks highly in both retrieval methods is almost certainly relevant — it matches both the semantic meaning and the exact keywords of the query. RRF rewards consistent cross-method presence rather than a single very high score in one method.

**RRF formula:**
```
RRF_score(doc) = Σ  1 / (k + rank_in_retriever)
                retrievers
```
`k = 60` is the standard constant — it dampens the advantage of very top ranks and prevents a single Rank 1 result from dominating. Documents ranked Rank 1 and Rank 5 score closer together than you'd expect, so results that appear *consistently* across retrievers beat results that dominate *one* retriever.

---

## How `EnsembleRetriever` Works

`EnsembleRetriever` is LangChain's built-in implementation of the hybrid pipeline above. You pass it a list of retrievers and a weight list — it runs them in parallel, collects ranked results, applies weighted RRF, and returns a single merged list.

```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]   # equal weight to keyword and semantic
)
results = ensemble_retriever.invoke("SKU-7742X specifications")
```

The `weights` parameter scales each retriever's RRF contribution before summing:

| Weight config | Effect |
|---|---|
| `[0.5, 0.5]` | Equal hybrid — good general default |
| `[0.7, 0.3]` | BM25-heavy — use when queries are ID/code-heavy |
| `[0.3, 0.7]` | Vector-heavy — use when queries are conversational |

---

## Production Considerations for Hybrid Search

### 1. BM25 Rebuild Strategy

BM25 is a **static index** — it is built once over a fixed corpus and has no concept of incremental updates. Adding new documents without rebuilding means those documents are invisible to BM25 retrieval.

| Scenario | Recommended approach |
|---|---|
| Small corpus, infrequent updates | Rebuild the full BM25 index on every document addition |
| Large corpus, frequent updates | Rebuild on a schedule (e.g. nightly) and accept a staleness window |
| Real-time freshness required | Use a keyword search engine (Elasticsearch, OpenSearch) instead of BM25 — they support incremental indexing |

```python
# Full rebuild — call this whenever documents change
bm25_retriever = BM25Retriever.from_documents(all_documents, k=4)
```

---

### 2. Tuning `k` — Retrieve More, Let RRF Sort

Set `k` (number of results each retriever returns) **higher than you actually need**. RRF needs enough candidates from each retriever to do meaningful re-ranking. If `k` is too low, a relevant document might not even enter the fusion pool.

| k value | Effect |
|---|---|
| `k = 1–2` | Too narrow — RRF has almost nothing to re-rank |
| `k = 3` | Minimum viable for most use cases |
| `k = 4–6` | Recommended default — good recall without context overflow |
| `k = 10+` | Use only if feeding into a re-ranker that compresses the list before the LLM |

```python
bm25_retriever = BM25Retriever.from_documents(documents, k=4)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
```

After RRF, the merged list is de-duplicated and re-ranked — you won't necessarily send all `k × n_retrievers` chunks to the LLM, just the top results from the fused list.

---

### 3. Latency — Two Searches Instead of One

Hybrid search adds approximately **20–25ms** of latency compared to a single vector search. The two retrievers run in parallel (LangChain's `EnsembleRetriever` does this by default), so the total latency is roughly `max(BM25_latency, vector_latency) + RRF_overhead`, not the sum.

| Component | Typical latency |
|---|---|
| Vector search (Chroma/Pinecone) | 10–50ms |
| BM25 (in-memory, small corpus) | 1–5ms |
| RRF merge | < 1ms |
| **Total hybrid overhead vs vector-only** | **~20–25ms** |

This overhead is almost always worth it when retrieval accuracy matters. If latency is critical, cache BM25 results for frequent queries or pre-filter the corpus with metadata before running either retriever.

---

### 4. Context Overflow — Too Many Retrieved Chunks

More retrieved chunks = more tokens sent to the LLM. Hybrid search can make this worse because you are merging results from two retrievers.

**Calculate your token budget before setting `k`:**

```
available_tokens  = model_context_limit - system_prompt_tokens - response_reserve
max_chunks        = available_tokens // avg_chunk_size_in_tokens

# Example: 8192 limit, 500-token system prompt, 500-token response reserve, 300-token chunks
max_chunks = (8192 - 500 - 500) // 300 = 23 chunks max
```

**Practical rules:**
- Start with `k = 4` per retriever; the RRF-merged list will have ≤ 4 unique top results if there is strong overlap
- Add a re-ranker (cross-encoder) after RRF to compress the list to the top 2–3 before sending to the LLM
- Use `max_tokens_limit` in LangChain's `ContextualCompressionRetriever` to enforce a hard token cap on retrieved content

---

### 5. Token Budgeting

Token budgeting is the practice of explicitly allocating the model's context window across all inputs before making an API call. Without it, retrieved chunks silently crowd out other important content.

**Context window allocation:**

```
┌─────────────────────────────────────────────┐
│  Model Context Window (e.g. 8,192 tokens)   │
├─────────────────┬───────────────────────────┤
│ System Prompt   │  ~500 tokens (fixed)       │
├─────────────────┼───────────────────────────┤
│ Chat History    │  ~1,000 tokens (variable)  │
├─────────────────┼───────────────────────────┤
│ Retrieved Chunks│  ~4,000 tokens (RAG budget)│
├─────────────────┼───────────────────────────┤
│ User Query      │  ~200 tokens               │
├─────────────────┼───────────────────────────┤
│ Response Reserve│  ~2,492 tokens             │
└─────────────────┴───────────────────────────┘
```

**Key practices:**
- Define the RAG budget explicitly and enforce it — do not let retrieval consume the full remaining window
- Count tokens *before* the API call using the model's tokenizer (e.g. `tiktoken` for OpenAI, `anthropic.count_tokens()` for Claude)
- If retrieved content exceeds the budget, truncate or compress the lowest-ranked chunks first — never truncate the top-ranked ones
- For long conversations, summarize or evict old turns to protect the RAG budget



---

## Why Debugging LLM Pipelines is Hard

Traditional software fails loudly — an exception is raised, a stack trace points to the line, and the fix is clear. LLM pipelines fail quietly in ways that are much harder to catch and reproduce.

### 1. Non-Determinism — Same Input, Different Output

LLMs are probabilistic. With `temperature > 0`, the same query can produce a different answer on every call. This means a bug you saw once might not reproduce on the next run, and a test that passes today might fail tomorrow with identical inputs.

**Why it matters for debugging:**
- You cannot rely on "run it again and see if it breaks"
- A regression may only appear 1 in 10 runs — easy to miss in CI
- Comparing outputs across runs requires semantic evaluation, not string equality

**Mitigation:** Use `temperature=0` for deterministic testing. Log every input/output pair so you can replay the exact run that failed.

---

### 2. Cascading Errors — One Bad Step Poisons Everything Downstream

In a multi-step pipeline, an error in an early stage silently propagates and corrupts every stage after it. No exception is raised — the pipeline just produces a wrong answer confidently.

```
Query: "What is our refund policy for SKU-7742X?"

Step 1 — Retrieval:  Returns docs about general returns (wrong chunk retrieved)
                              ↓
Step 2 — Prompt:     "Based on the context, answer the question..."
                              ↓
Step 3 — LLM:        Generates a plausible-sounding but incorrect refund policy
                              ↓
Step 4 — Output:     "Your refund window is 30 days." ← confident wrong answer
```

The LLM had nothing wrong with it. The retriever failed silently, and the error cascaded through. Without tracing each step, you would only see the final wrong answer and have no idea where the pipeline broke.

**Mitigation:** Test and evaluate each stage in isolation — retrieval quality separately from generation quality.

---

### 3. Silent Failure — No Crash, Just a Confident Wrong Answer

Traditional bugs crash. LLM bugs smile and lie. The pipeline completes successfully with status 200, the user gets a response, and no alert fires — but the answer is wrong, outdated, or hallucinated.

**Common silent failure patterns:**

| Failure | What the user sees | What actually happened |
|---|---|---|
| Wrong chunk retrieved | A confident answer | Retriever returned an unrelated doc |
| Hallucination | A specific, detailed answer | LLM invented a fact not in any chunk |
| Prompt injection | An unexpected response | User input modified the prompt behavior |
| Truncated context | A partial or vague answer | Context window overflowed silently |

**Mitigation:** Add faithfulness checks — after generating, verify each claim is grounded in the retrieved chunks. Alert on low-confidence or out-of-scope responses rather than returning them silently.

---

### 4. Cost Surprise — 10 LLM Calls Instead of 2

LLM API costs scale with token usage and the number of calls. A pipeline that seems simple can silently make far more calls than expected — due to retries, agent loops, sub-chains, or unbounded context windows.

**Example: an agent loop that should run twice runs ten times**
```
Expected:  Query → Retrieve → Generate  (2 LLM calls, ~2,000 tokens, ~$0.002)
Actual:    Query → Retrieve → Generate → Retry → Retrieve → Generate → ...
           (10 LLM calls, ~20,000 tokens, ~$0.02 per query × 10,000 users = $200/day)
```

The pipeline works correctly — it just costs 5× more than budgeted, and no error is logged.

**Mitigation:** Set explicit `max_iterations` on agents. Log token usage per run. Set cost alerts in your LLM provider dashboard. Track cost per query over time, not just total monthly spend.

---

## Observability with LangSmith

LangSmith is LangChain's tracing and evaluation platform. It solves all four debugging problems above by making every step of the pipeline visible.

### What LangSmith Captures Per Run

| Signal | What you see |
|---|---|
| **Inputs / Outputs** | Exact prompt sent and response received at every step |
| **Latency** | Time taken by each node (retriever, LLM, parser) |
| **Token usage** | Prompt tokens, completion tokens, and cost per step |
| **Errors** | Which step threw, with the full input that caused it |
| **Trace tree** | Parent → child relationship of every chain, tool, and agent call |

### Setup — Three Environment Variables

```bash
# .env
LANGCHAIN_TRACING_V2=true          # enables tracing globally
LANGCHAIN_API_KEY=ls__your_key     # LangSmith API key
LANGCHAIN_PROJECT=rag-freecodecamp # groups traces under a named project
```

No other code changes are needed. Every `chain.invoke()` call is automatically captured as a full trace.

### `@traceable` — Trace Plain Python Functions

For logic that lives outside LangChain components (validation, pre-processing, post-processing), wrap it with `@traceable` to include it as a named span in the trace tree:

```python
from langsmith import traceable

@traceable(name="validate_query")
def validate_query(query: str) -> str:
    if not query.strip():
        raise ValueError("Query must not be empty.")
    return query.strip()

@traceable(name="rag_pipeline")
def run_rag(query: str) -> dict:
    clean_query = validate_query(query)   # appears as a child span
    answer = rag_chain.invoke(clean_query) # LangChain chain appears as a child span
    return {"query": clean_query, "answer": answer}
```

The resulting trace tree in LangSmith looks like:
```
rag_pipeline
  ├── validate_query          (your custom logic)
  └── RunnableSequence        (LangChain chain)
        ├── retriever         (vector search)
        ├── ChatPromptTemplate
        ├── ChatOpenAI        (LLM call — tokens, cost, latency)
        └── StrOutputParser
```

### Manual Feedback — Log Human Signals on Runs

After a user rates an answer (thumbs up / thumbs down), log that signal back to the run so it feeds your evaluation dataset:

```python
from langsmith import Client

def log_feedback(run_id: str, score: int, comment: str = ""):
    client = Client()
    client.create_feedback(
        run_id=run_id,
        key="user_feedback",
        score=score,        # 1 = correct, 0 = wrong
        comment=comment,
    )
```

Over time, these signals accumulate into a labeled dataset you can use to measure whether pipeline changes actually improve answer quality.

---

## HNSW — How Vector Databases Find Similar Vectors Quickly

When you store embeddings in a vector database and run a similarity search, the database does not compare your query against every single vector. That would be too slow at scale (millions of vectors). Instead, almost every major vector database (Pinecone, Weaviate, Qdrant, Milvus, FAISS) uses an algorithm called **HNSW — Hierarchical Navigable Small World** to find approximate nearest neighbors in milliseconds.

> **Interview one-liner:** HNSW is an approximate nearest neighbor (ANN) algorithm that builds a multi-layer graph of vectors so search skips most of the data and navigates only through well-connected "shortcut" nodes.

### How HNSW Works (Simple Mental Model)

Think of it like a highway system:
- **Top layers** = highways with few nodes but long-range connections (fast, coarse navigation)
- **Bottom layers** = local roads with many nodes and short-range connections (slow, precise navigation)

At query time, the algorithm enters at the top layer and greedily jumps toward the query vector, dropping down layer by layer until it reaches the bottom where it finds the actual nearest neighbors. This is why it is fast: it skips 99% of the data.

```
Layer 2  (few nodes, wide connections — "highway")
Layer 1  (more nodes — "city roads")
Layer 0  (all nodes, dense connections — "local streets")
```

### The 3 Core Parameters to Know

#### M — Max Connections per Node

Controls how many neighbor-links each node keeps in the graph.

| M value | Memory | Build speed | Recall (accuracy) | Typical use |
|---|---|---|---|---|
| Low (8–12) | Low | Fast | Lower | Resource-constrained environments |
| Medium (16–32) | Moderate | Moderate | Good | **Production sweet spot** |
| High (48–64) | High | Slow | Best | When accuracy is critical and memory is not a concern |

> Changing M requires **rebuilding the entire index** — it is not a runtime setting.

#### efSearch — How Hard the Algorithm Looks at Query Time

At query time, this controls how many candidate nodes the algorithm explores before returning results. This is the parameter you will tune most in production.

| efSearch | Query speed | Recall | Notes |
|---|---|---|---|
| Low (10–30) | Fast | Lower | May miss the true best match |
| Medium (50–100) | Balanced | Good | **Recommended starting point** |
| High (200+) | Slow | Best | Use when accuracy is more important than latency |

- Must be ≥ `k` (the number of results you want back)
- Can be changed at **runtime without rebuilding the index** — tune it live

#### efConstruction — How Hard the Algorithm Looks at Build Time

When the index is being built, this controls how thoroughly the algorithm searches for good neighbors to connect each new node to. A higher value builds a "smarter" graph.

| efConstruction | Build time | Graph quality | Typical use |
|---|---|---|---|
| 100 | Fast | Acceptable | Quick prototyping |
| 100–200 | Moderate | Good | **Production default** |
| 400+ | Slow | Best | Offline batch indexing where quality > speed |

> Like M, this is a **build-time setting** — changing it means rebuilding the index.

### The HNSW Tradeoff Triangle

Every HNSW configuration is a tradeoff between three competing goals. You cannot optimize all three at the same time — improving one always costs another.

```
         Speed
        /     \
       /       \
      /         \
  Memory ——— Accuracy (Recall)
```

| If you want more... | Increase | Cost |
|---|---|---|
| **Accuracy (Recall)** | M, efSearch, efConstruction | More memory + slower build/search |
| **Speed** | Lower efSearch | Lower recall — might miss best matches |
| **Memory savings** | Lower M | Worse graph quality, lower recall |

### ANN vs Exact Search — Why "Approximate" is OK

HNSW is **approximate** nearest neighbor — it does not guarantee finding the mathematically closest vector. It finds *very close* neighbors very quickly.

| | Exact Nearest Neighbor | ANN (HNSW) |
|---|---|---|
| Guarantee | Always returns the best match | Returns the best match ~95–99% of the time |
| Speed at 1M vectors | O(n) — slow | O(log n) — milliseconds |
| Production use | Rarely used | Industry standard |

For RAG, a 99% recall is more than sufficient — a slightly sub-optimal chunk still leads to a correct answer in almost all cases.

---

## Self-Hosted vs Managed Vector Database

Once you move beyond prototyping, you need to decide how to run your vector database in production. There are two paths.

**Self-hosted** means you run the vector database yourself — on your own VMs, Kubernetes cluster, or bare metal (e.g. Qdrant, Milvus, Weaviate, or pgvector on your own Postgres).

**Managed** means a vendor runs it for you and exposes an API (e.g. Pinecone, Weaviate Cloud, Qdrant Cloud, MongoDB Atlas Vector Search).

### Comparison Table

| | Self-Hosted | Managed |
|---|---|---|
| **Cost at low scale** | Low (just infra) | Can be expensive (per-vector pricing) |
| **Cost at high scale** | Cheaper | Expensive — pricing grows with vector count |
| **Operational burden** | High — you handle upgrades, backups, scaling, monitoring | Low — vendor handles everything |
| **Data privacy / compliance** | Full control — data never leaves your infra | Data leaves your network; check vendor's compliance certs |
| **Latency** | Low if deployed close to your app | Depends on vendor region |
| **Scaling** | Manual (you provision more VMs/pods) | Automatic (vendor scales for you) |
| **Time to first query** | Days (setup, config, ops) | Minutes (sign up and get an API key) |
| **Customization** | Full control over HNSW params, hardware, schema | Limited to what the vendor exposes |

### When to Choose Each

| Scenario | Recommendation |
|---|---|
| Prototype / learning | **Managed** — get up and running in minutes |
| Compliance-heavy industry (healthcare, finance) | **Self-hosted** — data must not leave your network |
| Early startup, small team | **Managed** — no ops burden |
| Large scale (50M+ vectors, cost-sensitive) | **Self-hosted** — per-vector managed pricing becomes expensive |
| Need fine-grained HNSW tuning | **Self-hosted** — full parameter control |
| No dedicated ML/infra team | **Managed** — vendor handles reliability and upgrades |

> **Interview tip:** The managed vs self-hosted decision is rarely purely technical — it is usually driven by data compliance requirements, team size, and cost at scale. Know all three angles.

### Popular Options

| Type | Options |
|---|---|
| **Self-hosted** | Qdrant, Milvus, Weaviate (open source), pgvector (Postgres extension), FAISS (in-memory, no persistence) |
| **Managed** | Pinecone, Weaviate Cloud, Qdrant Cloud, MongoDB Atlas Vector Search, Azure AI Search |

---

## Cost Optimization Strategies for Vector Databases

Vector database costs grow with the number of vectors stored and the number of queries per second. These are the highest-impact levers you can pull to reduce costs without sacrificing meaningful quality.

### 1. Reduce Dimensions

Shrink the size of each embedding vector. For example, going from 1,536 dimensions (OpenAI `text-embedding-3-small`) down to 512 dimensions using the model's built-in dimension reduction.

| | Before | After |
|---|---|---|
| Dimensions | 1,536 | 512 |
| Storage per vector | 6.1 KB | 2.0 KB |
| Savings | — | **30–60% storage reduction** |
| Quality impact | Baseline | Slightly lower recall — test before committing |

**How:** OpenAI's `text-embedding-3-*` models support native dimension reduction via the `dimensions` parameter. Re-embed and re-index after changing.

**Effort:** Low

---

### 2. Quantization — Shrink Each Number

Each dimension in an embedding is stored as a 32-bit float by default. Quantization compresses it to a smaller format (e.g. 8-bit integer), reducing memory with minimal accuracy loss.

| Format | Bytes per dimension | Savings vs float32 | Quality |
|---|---|---|---|
| `float32` (default) | 4 bytes | — | Baseline |
| `int8` (scalar quantization) | 1 byte | **75% reduction** | ~1-2% recall drop |
| Binary (1 bit) | 0.125 bytes | **97% reduction** | Noticeable recall drop — only for extreme scale |

**When to use:** Qdrant, Weaviate, Milvus all support built-in quantization. Enable it on large indexes where memory is the bottleneck.

**Effort:** Medium (requires re-indexing and recall testing)

---

### 3. Batch Queries

Instead of sending one embedding request per user query, batch multiple queries into a single API call. The embedding model processes them together, reducing per-request overhead.

```python
# Instead of this (N separate API calls):
for query in user_queries:
    embedding = embed(query)

# Do this (1 API call):
embeddings = embed_batch(user_queries)
```

**Savings:** 10–30% on embedding API costs
**Effort:** Low — typically a one-line change in your embedding call

---

### 4. Cache Frequent Queries (Semantic Caching)

Many users ask the same questions (e.g. "What is your return policy?"). Cache the retrieved chunks and/or the final answer so identical queries skip the embedding and vector search entirely.

```
First request:   Query → Embed → VectorDB Search → LLM → Answer  (full cost)
Cached request:  Query → Cache Hit → Return cached Answer          (near-zero cost)
```

**What to cache:** The final answer (cheapest — skips everything), or the retrieved chunks (skips vector search but still calls the LLM).

**Two-layer cache key strategy** — a naive cache keyed on the raw query string misses obvious duplicates like `"What is AI?"` vs `"what is ai?"`. Two layers fix this:

1. **Normalize** — lowercase, strip whitespace, remove punctuation, etc. `"What is AI?"` and `"what is ai?"` both normalize to `"what is ai"`, so they're treated as the same query.
2. **Hash** — run the normalized query through a hash function (e.g. SHA-256) and use the hash as the cache key. This keeps the key a fixed length regardless of query length, so lookups stay cheap even for long or verbose queries.

```python
import hashlib
import re

def cache_key(query: str) -> str:
    normalized = re.sub(r"[^\w\s]", "", query.lower().strip())
    return hashlib.sha256(normalized.encode()).hexdigest()

cache_key("What is AI?") == cache_key("what is ai?")  # True — same cache entry
```

**Drawback of hash-based caching:** it only catches queries that are identical after normalization. Semantically similar but differently-worded queries still miss:

```
"What is AI?" vs "Explain artificial intelligence"
→ Different strings → different hashes → cache miss → unnecessary LLM call,
  even though the two queries mean the same thing.
```

**Solution — embedding-based semantic caching:** compare queries by meaning instead of exact string match.

1. **Embed** the incoming query into a vector.
2. **Search** the cache by vector similarity (e.g. cosine similarity) against previously cached query embeddings.
3. **Threshold check** — if the best match's similarity exceeds a threshold (e.g. `0.9`), return its cached response; otherwise call the LLM and store the new query-response pair (with its embedding) in the cache.

This catches paraphrases and reduces LLM calls further than hash-based caching alone, at the cost of an embedding call per lookup and a vector-similarity search instead of an O(1) dict lookup.

**Savings:** 10–40% depending on query distribution
**Effort:** Medium — requires a cache layer (Redis is common) and a cache key strategy (hash of the query string)

> **Interview tip:** Caching works best when your query distribution has a "long tail" — a small set of popular questions asked by many users.

---

### 5. Right-Size Your Index (Don't Over-Provision)

Vector databases, especially managed ones, charge for provisioned capacity — not just what you use. Spinning up a pod sized for 10M vectors when you only have 500K vectors wastes 95% of your budget.

**Steps:**
1. Measure your actual vector count and query throughput
2. Pick the smallest tier that meets your latency SLA
3. Set up auto-scaling if the vendor supports it
4. Re-evaluate every quarter as your data grows

**Savings:** 20–50% for teams that started with a "safe" large instance and never right-sized
**Effort:** Low — mostly configuration, not code

---

### Cost Optimization Summary

| Strategy | Savings | Effort | When to Use |
|---|---|---|---|
| **Reduce dimensions** | 30–60% | Low | When using a model that supports native reduction (e.g. OpenAI `text-embedding-3-*`) |
| **Quantization** | 50–75% | Medium | Large indexes where memory cost dominates |
| **Batch queries** | 10–30% | Low | Any pipeline that processes queries one at a time |
| **Caching** | 10–40% | Medium | Knowledge bases with a high rate of repeated queries |
| **Right-size** | 20–50% | Low | Any managed deployment that was provisioned without benchmarking |

> **Interview tip:** Cost optimization is not premature optimization — it is operational discipline. In production, a 60% storage reduction from dimension reduction + quantization together means you can store 2.5× more vectors for the same price, or cut your bill by 60% on the same dataset.

---

## Monitoring: The Three Pillars of Production Visibility

Once a RAG pipeline is live, monitoring is what tells you it's actually working — without it, failures like the ones above (bad chunking, hallucination, cost surprises) go unnoticed until a user complains.

1. **Structured Logging** — A human-readable story of what happened, in a machine-parseable format (e.g. JSON) so logs can be searched, filtered, and aggregated.
2. **Metrics Collection** — Numbers on a dashboard (latency, token usage, error rate, cache hit rate) that reveal trends over time.
3. **Instrumented LLM Calls** — Wrapping every LLM call so its inputs, outputs, tokens, latency, and cost are captured automatically.

> **Monitoring is the outermost layer** — it observes everything happening in the system but does not change the system's behavior.

**Where monitoring fits in the production stack:**

```
Security -> Cost Optimization -> Error Handling -> Monitoring
```

Security and cost/error handling shape *how* the system behaves; monitoring only watches and reports on that behavior after the fact.

---

## Long Context vs RAG — When to Use Each

Modern LLMs support huge context windows (1M+ tokens for some models), which raises an obvious question: **is RAG dead?** The answer is no — the choice between stuffing an entire corpus into the prompt ("long context") and retrieving only the relevant pieces ("RAG") is a tradeoff between **cost, latency, and use case**, not a strict upgrade path.

> **The real answer isn't "RAG vs Long Context" — it's "RAG AND Long Context."** Use both strategically.

### Cost Comparison

Cost scales directly with input tokens, so stuffing 100K tokens of docs into every query is dramatically more expensive than retrieving a handful of relevant chunks.

**Scenario:** Query against 100,000 tokens of documentation. Query: 100 tokens, expected output: 500 tokens.

| Approach | Input tokens | Cost per query |
|---|---|---|
| **Long Context** (stuff everything) | 100,100 | $0.25525 |
| **RAG** (4 chunks × 500 tokens) | 2,100 | $0.01025 |

RAG is **~25x cheaper per query** in this scenario.

**At scale (10,000 queries/day):**

| Approach | Daily cost | Monthly cost |
|---|---|---|
| Long Context | $2,552.50 | ~$76,575 |
| RAG | $102.50 | ~$3,075 |
| **Savings with RAG** | — | **~$73,500/month** |

> Pricing example uses $2.50/1M input tokens and $10.00/1M output tokens. The multiplier only grows as the document corpus grows — at 1M tokens of docs, long context cost scales roughly 10x further while RAG cost stays flat (it only depends on `k` and chunk size, not corpus size).

### Latency Comparison

More input tokens means more tokens for the model to process before it can start generating a response. RAG's smaller, targeted context is usually faster to process than a massive stuffed context — though at small scales (a few thousand tokens) the difference can be negligible or even noisy, since network/queueing variance dominates. The gap becomes dramatic once you're in the 100K+ token range, where the model has to attend over far more tokens per request.

| Context size | Effect on latency |
|---|---|
| Small (~50–500 tokens) | Minimal difference vs long context — variance dominates |
| Medium (~2,500 tokens) | Difference becomes measurable |
| Large (100K+ tokens) | Long context latency grows substantially; RAG latency stays roughly constant |

### Why Long Context Isn't Always Better

Beyond raw cost and latency, stuffing huge amounts of text into a single prompt has quality risks that don't show up in a token-count comparison:

- **"Lost in the middle" effect** — research on long-context models shows retrieval accuracy is highest for information near the *start* or *end* of the prompt, and measurably worse for facts buried in the middle. A 100K-token stuffed prompt doesn't guarantee the model actually "sees" every fact equally well.
- **No citations by default** — long context gives the model everything at once with no structural signal of *which passage* an answer came from, making source attribution harder to bolt on than in RAG (where each chunk already carries `metadata.source`).
- **Reprocessing cost on every change** — if a document changes, long context means re-sending (and re-paying for) the entire corpus on the next query. RAG only needs to re-embed and re-index the changed chunk.
- **Prompt caching changes the math (partially)** — providers like Anthropic and OpenAI support prompt/context caching, which can cut the cost of *repeated* queries against the *same* large context by up to ~90%. This narrows the cost gap for a static corpus queried repeatedly without changes, but doesn't help if the corpus is large *and* changes frequently *and* query volume is high — RAG still wins there.

### Decision Framework

**Use Long Context when:**
- Document corpus is small (< 50K tokens)
- Query volume is low (< 100 queries/day)
- You need to analyze the **entire** document holistically (e.g. summarization, cross-referencing distant sections)
- Documents change frequently (avoids embedding/re-indexing overhead)
- Simplicity is worth more than cost optimization

**Use RAG when:**
- Document corpus is large (> 100K tokens)
- Query volume is high (thousands of queries/day)
- Users ask about specific topics rather than requesting whole-document analysis
- Cost and latency matter
- You need citations / source tracking
- Documents are relatively stable

| Factor | Favors Long Context | Favors RAG |
|---|---|---|
| Corpus size | Small (< 50K tokens) | Large (> 100K tokens) |
| Query volume | Low (< 100/day) | High (1000s/day) |
| Query type | Whole-document analysis | Specific, targeted questions |
| Document volatility | Changes often | Relatively stable |
| Cost sensitivity | Low priority | High priority |
| Need for citations | Not required | Required |

### Hybrid Approach: RAG + Long Context Together

The two techniques compose well: use RAG to **find** the right document(s) cheaply, then load the **full** document into context for a detailed, complete answer — instead of relying on a small chunk that might be missing surrounding detail.

```
1. RAG retrieves the candidate document(s) via vector similarity search
2. The full document (not just the matched chunk) is loaded into the LLM's context
3. The LLM generates a comprehensive answer using the complete document
```

```python
# Step 1: RAG finds the right document
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
relevant_docs = retriever.invoke(query)

# Step 2: Load the FULL document into context (not just the matched chunk)
full_doc = relevant_docs[0].page_content

# Step 3: Generate a detailed answer with the complete document as context
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the full document below to give a comprehensive answer.\n\n{document}"),
    ("human", "{query}"),
])
response = (prompt | llm).invoke({"document": full_doc, "query": query})
```

**Why this works well:**

| Benefit | Explanation |
|---|---|
| Fast document discovery | RAG's vector search narrows thousands of documents down to the 1–2 relevant ones in milliseconds |
| No missing context | The LLM sees the *entire* matched document, not a 500-token chunk that might cut off a relevant detail |
| Still far cheaper than full corpus stuffing | You pay for one full document's tokens, not the entire knowledge base's tokens |
| Best for | "Find the relevant doc, then analyze it thoroughly" — e.g. HR policy lookups, contract analysis, single-report Q&A |

### Key Takeaways

1. RAG is **not dead** — the tradeoff is about cost and latency at scale, not capability
2. Long context wins for small corpora, low query volume, and whole-document analysis
3. RAG wins for large corpora, high query volume, and targeted queries
4. The hybrid approach — RAG to find the doc, long context to analyze it — combines the strengths of both
5. The production answer is rarely "one or the other" — it's using both strategically depending on the query shape

---

## Contextual Retrieval (Anthropic's Technique)

Chunking a document strips away the surrounding context each chunk depends on. A chunk that says *"The company plans to expand into renewable energy"* means nothing to an embedding model without knowing which company. Anthropic's **Contextual Retrieval** technique fixes this by having an LLM prepend a short, disambiguating context to every chunk **before** it gets embedded.

### The Problem

```
Chunk 1: "The company" - Which company?
Chunk 2: "fiscal year 2025" - For what company?
Chunk 3: "The company plans" - Plans of what company?
```

A query like *"What is ACME's revenue?"* may fail to match Chunk 2 because the literal string "ACME" never appears in it — only the pronoun-like phrase "the company."

### The Solution

An LLM is given the **chunk** and the **full document** it came from, and asked to write a 1–2 sentence prefix that situates the chunk (entities, document title, relevant surrounding facts). That prefix is prepended to the chunk before embedding.

```python
def add_contextual_prefix(chunk, full_document, document_title, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a SHORT context (1-2 sentences) that helps "
                   "situate the chunk within the document. Output ONLY the prefix."),
        ("human", "Document Title: {title}\n\nFull Document:\n{document}\n\n"
                  "Chunk to contextualize:\n{chunk}"),
    ])
    return (prompt | llm).invoke(
        {"title": document_title, "document": full_document, "chunk": chunk}
    ).content

contextualized_chunk = f"{add_contextual_prefix(chunk, doc, title, llm)} {chunk}"
```

### Production Considerations

| Factor | Impact |
|---|---|
| **Cost** | ~$0.01–$0.05 per document — a one-time indexing cost, far cheaper than retrieval failures |
| **Latency** | +1–2s per chunk during indexing; no impact on query latency (batch process offline) |
| **Storage** | Chunks grow ~20–30% larger — minimal effect on vector DB cost |
| **Result** | Anthropic reports **67% fewer retrieval failures** |

**When to use:** Documents with important context in headers/titles, entities referenced with pronouns ("the company", "they"), or corpora pulled from multiple sources. Combine with BM25 hybrid search for best results.

*See [A14_ContextualRetrival.py](A14_ContextualRetrival.py) for the full runnable demo, including a side-by-side retrieval-score comparison between raw and contextualized chunks.*

---

## Late Chunking

Contextual Retrieval fixes context loss by asking an LLM to describe it explicitly. **Late chunking** fixes the same problem architecturally: instead of splitting text and then embedding each piece independently, it **embeds the full document first**, then splits the resulting token-level embeddings into chunk vectors.

### Early (Traditional) Chunking vs. Late Chunking

```
EARLY CHUNKING:                          LATE CHUNKING:
Document → Split → Chunk 1, 2, 3         Document → Embed FULL doc (token embeddings)
              ↓         ↓        ↓                        → Split embeddings by position
          Embed     Embed    Embed                        → Vector 1, 2, 3 (pooled)
              ↓         ↓        ↓
         Vector 1  Vector 2  Vector 3

Each chunk embedded INDEPENDENTLY.       Each chunk vector "knows" the whole document —
No cross-chunk context.                  a pronoun like "he" is embedded with the
                                          knowledge that "he" = "Steve Jobs."
```

**Example failure mode this fixes:** A biography chunked into sentences produces chunks like *"He co-founded Apple Computer in 1976"* with no mention of "Steve Jobs" — because that name appeared only in an earlier chunk. Under early chunking, a query for "companies Steve Jobs founded" may not match this chunk at all.

### Implementation Options

| Approach | Context Quality | Implementation Cost |
|---|---|---|
| Early chunking (traditional) | Poor — pronouns orphaned | Free — standard approach |
| Overlapping chunks | Better — some context kept | Free — just a config change |
| **Contextual Retrieval** | Excellent — LLM adds context | ~$0.01/doc (LLM cost) |
| **Late chunking (native)** | Excellent — full doc context | Needs a model with token-level output (e.g. Jina Embeddings v3 with `late_chunking=True`) |
| Parent-Child Retriever | Excellent — returns parent doc | Extra storage (2x vector store) |

**Accuracy improvements vs. traditional chunking:** overlapping chunks +3–5%, Contextual Retrieval +15–20%, Late Chunking +10–12%, combined approaches +25–30%.

> **Note:** True late chunking requires token-level embedding access (e.g. Jina models). OpenAI's standard embedding API doesn't expose this, so [A15_lateChunking.py](A15_lateChunking.py) demonstrates the *concept* via context-prepending rather than true token-level pooling.

**Recommendation:** Start with Contextual Retrieval, since it works with any embedding model. Move to native late chunking (Jina) if you control the embedding model. Combine with a Parent Document Retriever for the best of both.

---

## Agentic RAG with LangGraph

Traditional RAG is a **one-shot** pipeline: retrieve, then generate — with no way to recover if retrieval returns garbage. **Agentic RAG** turns this into a loop that can evaluate its own retrieval quality and self-correct: `retrieve → grade → [rewrite query and retry if needed] → generate`.

### Why This Matters

A single vector search can miss the answer simply because the user's phrasing doesn't match the document's vocabulary. A one-shot pipeline generates a confident (possibly wrong) answer anyway. Agentic RAG adds a **grading step** that catches this before generation ever happens.

### Graph Structure

```
START → RETRIEVE → GRADE ──┬── (good relevance) ──────────────→ GENERATE → END
                            ├── (low relevance, retries left) → REWRITE → back to RETRIEVE
                            └── (no docs, out of retries) ────→ FALLBACK → END
```

Built with LangGraph's `StateGraph`, using a `TypedDict` state schema (LangGraph 1.x convention — not Pydantic):

```python
class RAGState(TypedDict):
    query: str
    rewritten_query: str
    documents: list[Document]
    generation: str
    relevance_score: float
    retry_count: int
    max_retries: int

workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate_answer)
workflow.add_node("fallback", generate_fallback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade", should_retry_or_generate,
    {"rewrite": "rewrite", "generate": "generate", "fallback": "fallback"},
)
workflow.add_edge("rewrite", "retrieve")   # the self-correcting loop
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

app = workflow.compile()
```

### The Router: `should_retry_or_generate`

This function is the decision-making core of the workflow — it inspects the average LLM-assigned relevance score (0–1 per document) and the retry budget to decide the next step:

| Condition | Route |
|---|---|
| `relevance_score >= 0.5` and documents exist | → **generate** |
| Low relevance, `retry_count < max_retries` | → **rewrite** (reformulate query, loop back to retrieve) |
| Out of retries but some documents exist | → **generate** (best effort with what's available) |
| Out of retries and zero documents | → **fallback** (graceful "I don't know" response) |

**When to use Agentic RAG:** complex queries that may need reformulation, high-stakes applications where answer quality matters, diverse document types, user-facing (not batch) applications. The cost is extra LLM calls per grading/rewrite step — worth it when a wrong or "I don't know" answer is expensive.

*See [A16_agenticRAGpy](A16_agenticRAGpy) for the full implementation, including a demo that runs three test queries — two that succeed on the first retrieval and one designed to exhaust all retries and hit the fallback path.*

> **Note:** This file is currently named `A16_agenticRAGpy` (no `.py` extension) rather than `A16_agenticRAG.py` — likely a typo from when it was created. It is a valid Python script and runs fine as-is, but you may want to rename it for consistency with the other lesson files.

---

## GraphRAG — Multi-Hop Reasoning with Knowledge Graphs

Vector similarity search answers "what documents are semantically similar to this query" — but it can't answer questions that require **traversing relationships** between entities across multiple documents.

### The Multi-Hop Problem

```
Query: "Who works in the same department as the CEO's assistant?"

Requires chaining:  CEO → CEO's assistant → their department → other employees in that department
```

Vector search for "CEO's assistant" won't semantically match a chunk like *"Sarah Johnson works in the Executive department"* — Sarah's name never appears in the query, so the embeddings don't align. This is a **multi-hop** problem: the answer requires following a chain of relationships, not matching a single semantically-similar passage.

### The Solution: Knowledge Graphs

GraphRAG extracts **entities** (nodes) and **relationships** (edges) from documents into a traversable graph, then answers multi-hop queries by walking the graph instead of (or in addition to) running vector search:

```python
G = nx.DiGraph()
G.add_node("Sarah Johnson", type="Person", role="Executive Assistant")
G.add_edge("Sarah Johnson", "John Smith", relation="ASSISTANT_TO")
G.add_edge("Sarah Johnson", "Executive Department", relation="WORKS_IN")
```

In production, entity/relationship extraction is automated with an LLM rather than hand-built:

```python
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract ENTITIES (people, organizations, places, concepts) and "
               "RELATIONSHIPS (WORKS_FOR, MANAGES, LOCATED_IN, etc.) from the text."),
    ("human", "Extract entities and relationships from this text:\n\n{text}"),
])
```

### GraphRAG Architecture — Two Phases

```
INDEXING:  Documents → LLM extracts entities/relations → Knowledge Graph
                                                              ↓
                                                     Community Detection (group related entities)
                                                              ↓
                                                     LLM generates community summaries

QUERYING:  Two modes —
  LOCAL SEARCH  (multi-hop reasoning): identify entities in query, traverse relationships
  GLOBAL SEARCH (holistic themes):    aggregate knowledge via community summaries
```

### Implementation Options

| Option | Pros | Cons |
|---|---|---|
| **Microsoft GraphRAG** | Full-featured, production-ready, well-documented | Heavy — significant compute for indexing |
| **LangGraph + Neo4j** (`GraphCypherQAChain`) | Full control, integrates with existing Neo4j | Must manage Neo4j, manual entity extraction |
| **LlamaIndex `KnowledgeGraphIndex`** | Easy to use, good LlamaIndex integration | Less powerful than full GraphRAG |
| **Hybrid (Vector + Graph)** | Vector search narrows candidates, graph traversal handles multi-hop | More complex to implement |

### When to Use GraphRAG

| Use it | Skip it |
|---|---|
| Documents describe relationships (org charts, research papers) | Simple fact retrieval — standard RAG is enough |
| Queries need multi-hop reasoning ("who is connected to X?") | Small document sets (< 100 docs) |
| Need global summarization across documents | Real-time indexing requirements |
| | Cost-sensitive applications — indexing is expensive |

*See [A17_graphRAGIntro.py](A17_graphRAGIntro.py) for a hand-built graph walkthrough (CEO → assistant → department → coworkers) plus an LLM-based entity extraction demo. Requires `networkx` — see the dependency note above.*

---

## Multimodal RAG with ColPali

Standard RAG pipelines extract text from PDFs before chunking and embedding — but **text extraction is a lossy operation** for anything visual: tables, charts, diagrams, and layout all degrade or disappear entirely.

### What Text Extraction Destroys

A table like:

```
Region  | Q1 Target | Q1 Actual | Variance
North   | $2.5M     | $2.8M     | +12% ✓
South   | $1.8M     | $1.5M     | -17% ✗
```

commonly becomes, after extraction:

```
Q1 2025 Sales by Region Region Q1 Target Q1 Actual Variance
North $2.5M $2.8M +12% South $1.8M $1.5M -17% East $3.2M $3.4M +6%
```

Row/column alignment, checkmarks/X's, and the visual "South is the outlier" signal are all gone — the LLM has to infer structure from a jumbled string.

### The Solution: Vision-Based Document RAG (ColPali)

Instead of extracting text, **convert each PDF page to an image and embed the image directly**:

```
PDF → Convert to images → Embed images (ColPali) → Store in Vector DB
Query → Embed query (ColPali) → Find similar page images → Return page IMAGES
                                                                  ↓
                                                          Vision LLM (GPT-4o / Claude)
                                                          "sees" the table/chart/diagram
```

ColPali (built on Google's PaliGemma) produces one embedding per page image that captures **both text and visual layout** — no OCR or text extraction step required. The retrieved page image is then sent to a vision-capable LLM (GPT-4o, Claude) that can directly answer questions about tables, charts, and diagrams.

```python
# Core pipeline shape (see file for full implementation)
images = pdf_to_images(pdf_path)                       # pdf2image
embeddings = embed_document_images(images)              # ColPali
results = search_documents(query, embeddings, images)   # similarity search over image embeddings
answer = answer_with_vision(query, results)              # GPT-4o/Claude vision call with page images
```

### Cost Trade-off

| | Text RAG | Multimodal RAG (ColPali) |
|---|---|---|
| Embedding | ~$0.0001/page | ~$0.001/page (requires GPU) |
| Query | ~$0.01/query (GPT-4o-mini) | ~$0.10/query (GPT-4o with images) |
| **Overall** | Baseline | **~10x more expensive** — but preserves visual information |

### When to Use It

| Good fit | Not necessary |
|---|---|
| Financial reports (nested tables, charts, footnotes) | Plain text documents (novels, articles) |
| Technical docs (architecture diagrams, flowcharts) | Simple structured/CSV-like data |
| Scientific papers (figures, equations, graphs) | Documents where text extraction already works well |
| Legal documents (formatted contracts, term tables) | Real-time applications (vision models are slower) |
| Medical records (diagnostic images, lab tables) | Cost-sensitive applications |

*See [A18_multiModelRAG.py](A18_multiModelRAG.py) for a runnable vision-LLM demo and the full ColPali pipeline reference implementation (embedding, search, and vision-based answering).*
