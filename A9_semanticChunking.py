from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb import EphemeralClient
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

document = """
# Authentication Guide

## 1. OAuth2 Authentication
OAuth2 is the industry-standard protocol for authorization. It enables third-party applications
to obtain limited access to a user's account without exposing credentials.

### Security Best Practices
- Use PKCE (Proof Key for Code Exchange) for public clients.
- Always validate the `state` parameter to prevent CSRF attacks.
- Store `client_secret` server-side only — never expose it in client-side code.
- Use HTTPS for all OAuth2 endpoints.

## 2. Rate Limiting
Rate limiting controls how many requests a client can make in a given time window to protect
the API from abuse and ensure fair usage.

### Handling Rate Limits
- Always check for `429 Too Many Requests` responses.
- Respect the `Retry-After` header before retrying.
- Implement **exponential backoff** with jitter for retries:
  - Wait = base_delay * (2 ^ attempt) + random_jitter
- Cache responses where possible to reduce API calls.
- Use bulk or batch endpoints when available to reduce request count.

## 3. Error Handling
Robust error handling ensures your integration is resilient and provides meaningful feedback
to users and developers.
### Best Practices
- **Log `request_id`** from error responses to aid debugging with API support.
- **Differentiate retryable vs. non-retryable errors**: 5xx and 429 are retryable; 4xx (except 429) are not.
- **Surface clear messages** to users for client errors (e.g., validation failures).
- **Never log sensitive data** such as tokens, passwords, or PII in error logs.
- Implement **circuit breakers** to stop sending requests to a failing service temporarily.

## 4. Webhooks
Webhooks allow an API to push real-time notifications to your application when events occur,
eliminating the need for polling.
### Webhook Best Practices
- **Respond quickly** (within 5-10 seconds). Offload processing to a background queue.
- **Handle retries**: Make your handlers **idempotent** — use `event_id` to detect duplicates.
- **Use HTTPS only** — never accept webhooks over plain HTTP.
- **Validate the payload schema** before processing to guard against malformed data.
- **Monitor and alert** on failed webhook deliveries from the provider dashboard.
- Store raw event payloads for auditing and replay.
"""

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n" ". ", " "],
    chunk_size=400,
    chunk_overlap=50
)

recursive_chunks = recursive_splitter.split_text(document)

chroma_client = EphemeralClient()

recrursive_vectorstore = Chroma.from_texts(
    recursive_chunks,
    embedding=embeddings,
    collection_name="recursive_vectorstore",
    client=chroma_client,
)

# print("Recursive Chunking....")
# for i, chunk in enumerate(recursive_chunks):
#     print(f"\n\nChunk {i + 1}: {chunk[:100]}")


semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_amount=90,
    breakpoint_threshold_type="percentile",
)

semantic_chunks = semantic_splitter.split_text(document)

semantic_vectorstore = Chroma.from_texts(
    semantic_chunks,
    embedding=embeddings,
    collection_name="semantic_chunks",
    client=chroma_client,
)

# print("Semantic Chunking....")
# for i, chunk in enumerate(semantic_chunks):
#     print(f"\n\nChunk {i + 1}: {chunk[:100]}")


def compare_chunking_strategies(recursive_chunks: list[str], semantic_chunks: list[str]) -> str:
    def avg_length(chunks):
        return sum(len(c) for c in chunks) / len(chunks) if chunks else 0

    def size_variance(chunks):
        if not chunks:
            return 0
        avg = avg_length(chunks)
        return sum((len(c) - avg) ** 2 for c in chunks) / len(chunks)

    r_count = len(recursive_chunks)
    s_count = len(semantic_chunks)
    r_avg = avg_length(recursive_chunks)
    s_avg = avg_length(semantic_chunks)
    r_min = min(len(c) for c in recursive_chunks)
    r_max = max(len(c) for c in recursive_chunks)
    s_min = min(len(c) for c in semantic_chunks)
    s_max = max(len(c) for c in semantic_chunks)
    r_variance = size_variance(recursive_chunks)
    s_variance = size_variance(semantic_chunks)

    summary = f"""
========================================
   CHUNKING STRATEGY COMPARISON SUMMARY
========================================

{'METRIC':<30} {'RECURSIVE':>15} {'SEMANTIC':>15}
{'-'*60}
{'Total Chunks':<30} {r_count:>15} {s_count:>15}
{'Avg Chunk Length (chars)':<30} {r_avg:>15.1f} {s_avg:>15.1f}
{'Min Chunk Length (chars)':<30} {r_min:>15} {s_min:>15}
{'Max Chunk Length (chars)':<30} {r_max:>15} {s_max:>15}
{'Size Variance':<30} {r_variance:>15.1f} {s_variance:>15.1f}

OBSERVATIONS
------------
• Chunk count   : {'Recursive produces more chunks' if r_count > s_count else 'Semantic produces more chunks' if s_count > r_count else 'Both produce equal chunks'}
• Avg size      : {'Recursive chunks are larger on average' if r_avg > s_avg else 'Semantic chunks are larger on average' if s_avg > r_avg else 'Both have equal average size'}
• Size variance : {'Recursive is more consistent in size' if r_variance < s_variance else 'Semantic is more consistent in size' if s_variance < r_variance else 'Both have equal size variance'}

========================================
"""
    return summary


print(compare_chunking_strategies(recursive_chunks, semantic_chunks))


def test_retrieval(query: str, k: int = 3) -> None:
    recursive_results = recrursive_vectorstore.similarity_search(query, k=k)
    semantic_results = semantic_vectorstore.similarity_search(query, k=k)

    print(f"\n{'='*60}")
    print(f"QUERY: {query!r}")
    print(f"{'='*60}")

    print(f"\n--- RECURSIVE (top {k}) ---")
    for i, doc in enumerate(recursive_results, 1):
        print(f"\n[{i}] {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}")

    print(f"\n--- SEMANTIC (top {k}) ---")
    for i, doc in enumerate(semantic_results, 1):
        print(f"\n[{i}] {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}")

    recursive_texts = {doc.page_content for doc in recursive_results}
    semantic_texts = {doc.page_content for doc in semantic_results}
    overlap = recursive_texts & semantic_texts

    print(f"\n--- COMPARISON ---")
    if not recursive_results and not semantic_results:
        print("No results returned from either vectorstore.")
        return

    r_avg = sum(len(d.page_content) for d in recursive_results) / len(recursive_results) if recursive_results else 0
    s_avg = sum(len(d.page_content) for d in semantic_results) / len(semantic_results) if semantic_results else 0
    print(f"Recursive avg chunk size : {r_avg:.0f} chars ({len(recursive_results)} results)")
    print(f"Semantic avg chunk size  : {s_avg:.0f} chars ({len(semantic_results)} results)")
    print(f"Overlapping results      : {len(overlap)} / {k}")
    if overlap:
        print("Shared chunks:")
        for text in overlap:
            print(f"  • {text[:80]}...")


test_queries = [
    "How do I handle rate limiting in API calls?",
    "What are the best practices for OAuth2 security?",
    "How should I handle webhook retries?",
    "What errors are retryable vs non-retryable?",
]

for query in test_queries:
    test_retrieval(query, k=2)