from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Config ---
USE_SEMANTIC_CHUNKING = True
MIN_SEMANTIC_CHUNK_COUNT = 2   # fallback if semantic produces too few chunks
MAX_SEMANTIC_CHUNK_SIZE = 3000  # fallback if any chunk is unreasonably large


def get_recursive_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=400,
        chunk_overlap=50,
    )
    return splitter.split_text(text)


def get_semantic_chunks(text: str) -> list[str]:
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    return splitter.split_text(text)


def chunk_document(text: str, use_semantic_chunking: bool = USE_SEMANTIC_CHUNKING) -> tuple[list[str], str]:
    """
    Returns (chunks, strategy_used).

    Primary  : SemanticChunker  (when use_semantic_chunking=True)
    Fallback : RecursiveCharacterTextSplitter

    Fallback triggers when:
      - use_semantic_chunking is False
      - SemanticChunker raises an exception
      - Result has fewer than MIN_SEMANTIC_CHUNK_COUNT chunks
      - Any single chunk exceeds MAX_SEMANTIC_CHUNK_SIZE chars
    """
    if not use_semantic_chunking:
        print("[Chunking] Semantic chunking disabled — using recursive.")
        return get_recursive_chunks(text), "recursive"

    try:
        print("[Chunking] Attempting semantic chunking...")
        chunks = get_semantic_chunks(text)

        if len(chunks) < MIN_SEMANTIC_CHUNK_COUNT:
            print(
                f"[Chunking] Semantic produced only {len(chunks)} chunk(s) "
                f"(min={MIN_SEMANTIC_CHUNK_COUNT}) — falling back to recursive."
            )
            return get_recursive_chunks(text), "recursive (fallback: too few chunks)"

        oversized = [c for c in chunks if len(c) > MAX_SEMANTIC_CHUNK_SIZE]
        if oversized:
            print(
                f"[Chunking] {len(oversized)} semantic chunk(s) exceed "
                f"{MAX_SEMANTIC_CHUNK_SIZE} chars — falling back to recursive."
            )
            return get_recursive_chunks(text), "recursive (fallback: oversized chunks)"

        print(f"[Chunking] Semantic chunking succeeded — {len(chunks)} chunks.")
        return chunks, "semantic"

    except Exception as e:
        print(f"[Chunking] Semantic chunking failed ({e}) — falling back to recursive.")
        return get_recursive_chunks(text), "recursive (fallback: exception)"


# ---------------------------------------------------------------------------

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

chunks, strategy = chunk_document(document)

print(f"\nStrategy used : {strategy}")
print(f"Total chunks  : {len(chunks)}")
print("\n--- Chunks ---")
for i, chunk in enumerate(chunks, 1):
    print(f"\n[{i}] ({len(chunk)} chars) {chunk[:120]}{'...' if len(chunk) > 120 else ''}")


