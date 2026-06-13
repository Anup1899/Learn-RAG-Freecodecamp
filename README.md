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