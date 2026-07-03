import hashlib

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from dotenv import load_dotenv
from qdrant_client.models import Optional

load_dotenv()

class TokenBudget:
    """Track and limit token usage."""

    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_per_request = max_tokens_per_request
        self.usage = {"total_input": 0, "total_output": 0, "requests": 0}

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (actual would use tiktoken)."""
        return int(len(text.split()) * 1.3)

    def check_budget(self, text: str) -> tuple[bool, int]:
        """Check if request is within budget."""
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage."""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:
        return {
            **self.usage,
            "total_tokens": self.usage["total_input"] + self.usage["total_output"],
            "avg_per_request": (
                (self.usage["total_input"] + self.usage["total_output"])
                / max(self.usage["requests"], 1)
            ),
        }


class BudgetedLLM:
    """LLM with token budgeting."""

    def __init__(self, max_tokens: int = 4000):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    @traceable(name="budgeted_invoke")
    def invoke(self, query: str) -> str:
        # Check budget
        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {tokens} > {self.budget.max_per_request}"
            )

        # Execute
        response = self.llm.invoke(query)
        result = response.content

        # Record usage
        output_tokens = self.budget.estimate_tokens(result)
        self.budget.record_usage(tokens, output_tokens)

        return result

    def get_stats(self) -> dict:
        return self.budget.get_stats()


def demo_token_budgeting():
    """Demonstrate token budgeting."""

    llm = BudgetedLLM(max_tokens=100)

    queries = [
        "What is AI?",  # Within budget
        "Explain " + "very " * 100 + "complex topic",  # Over budget
    ]

    print("\nToken Budgeting Demo:\n")

    for query in queries:
        try:
            result = llm.invoke(query)
            print(f"✅ {query[:40]}... -> {result[:30]}...")
        except ValueError as e:
            print(f"❌ {query[:40]}... -> {e}")

    print(f"\nUsage: {llm.get_stats()}")

# Semantic Caching
# Two Layer Cachec
    # Normalized Caching -- Converting to Lowercase, stripping whitespace, removing punctuation, etc.   
        # Example: "What is AI?" and "what is ai?" would be treated as the same query.
    # Hash -- Truns normalized query into a hash value (e.g., SHA-256) and uses this hash as the key in the cache. This ensures that even if the normalized query is long, the cache key remains fixed-length and efficient to look up.
        # Example: "What is AI?" and "what is ai?" would both produce the same hash value, ensuring they map to the same cache entry.


class SemanticCache:
    """A simple semantic cache for LLM responses."""

    def __init__(self, similarity_threshold: float = 0.9):
        self.cache = {}
        self.threshold = similarity_threshold
        self.embedder = ChatOpenAI(model="text-embedding-3-small", temperature=0)

    def hash_query(self, query: str) -> str:
        """Create hash of normalized query."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """"Get cached response if available."""
        query_hash = self.hash_query(query)

        # Exact Match
        if query_hash in self.cache:
            return self.cache[query_hash]["response"]

        return None

    def set(self, query: str, response: str):
        """Store response in cache."""
        query_hash = self.hash_query(query)
        self.cache[query_hash] = {
            "query": query,
            "response": response,
        }

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "cache_queries": len(self.cache)
        }

class CachedLLM:
    """LLM wrapper with Semantic Caching"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.cache = SemanticCache()
        self.cache_hits = 0
        self.cache_misses = 0
    
    @traceable(name="cached_invoke")
    def invoke(self, query: str) -> tuple[str, bool]:
        """
            Invoke LLM with caching.
            Returns:
                - response: str
                - from_cache: bool"""
        
        # Check cache first
        cached_response = self.cache.get(query)
        if cached_response:
            self.cache_hits += 1
            return cached_response, True
        
        # If not cached, call LLM
        response = self.llm.invoke(query)
        result = response.content

        # Store in cache
        self.cache.set(query, result)
        self.cache_misses += 1

        return result, False
    
    def get_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2%}",
        }
    
def demo_caching():
    """Demonstrate caching."""

    llm = CachedLLM()

    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",  # Cache hit
        "What is python?",  # Cache hit (normalized)
        "What is Rust?",
    ]

    print("\nCaching Demo:\n")

    for query in queries:
        result, from_cache = llm.invoke(query)
        source = "CACHE" if from_cache else "LLM"
        print(f"[{source}] {query} -> {result[:30]}...")

    print(f"\nStats: {llm.get_stats()}")


# Drawback of the above approach:
# Cache miss for semantically similar queries that are not identical.
    # Example: "What is AI?" and "Explain artificial intelligence" would be treated as different queries, even though they are semantically similar.
        # Differnt hash values would be generated for these queries, leading to a cache miss and an unnecessary LLM call.
# Solution: Implement semantic caching using embeddings to compare the similarity of queries rather than relying on exact string matches.
# Step 1 : Embed the query into a vector
# Step 2 : Search the cache by vector similarity (e.g., cosine similarity) to find semantically similar queries.
# Step 3 : If a similarity > threshold is found, return the cached response; otherwise, call the LLM and store the new query-response pair in the cache.


class EmbeddingSemanticCache:
    """Semantic cache that matches queries by embedding similarity instead of exact hashes."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.cache: list[dict] = []
        self.threshold = similarity_threshold
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    @staticmethod
    def cosine_similarity(vec1, vec2) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get(self, query: str) -> tuple[Optional[str], float]:
        """Find the closest cached query. Returns (response, similarity) or (None, best_similarity)."""
        if not self.cache:
            return None, 0.0

        query_vector = self.embedder.embed_query(query)

        best_entry, best_score = None, 0.0
        for entry in self.cache:
            score = self.cosine_similarity(query_vector, entry["embedding"])
            if score > best_score:
                best_entry, best_score = entry, score

        if best_entry and best_score >= self.threshold:
            return best_entry["response"], best_score

        return None, best_score

    def set(self, query: str, response: str):
        """Store the query's embedding alongside its response."""
        embedding = self.embedder.embed_query(query)
        self.cache.append({"query": query, "embedding": embedding, "response": response})

    def stats(self) -> dict:
        return {"cached_queries": len(self.cache)}


class SemanticCachedLLM:
    """LLM wrapper that reuses responses for semantically similar queries."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.cache = EmbeddingSemanticCache(similarity_threshold=similarity_threshold)
        self.cache_hits = 0
        self.cache_misses = 0

    @traceable(name="semantic_cached_invoke")
    def invoke(self, query: str) -> tuple[str, bool, float]:
        """
            Invoke LLM with embedding-based semantic caching.
            Returns:
                - response: str
                - from_cache: bool
                - similarity: float (similarity to the closest cached query)"""

        # Check cache by embedding similarity
        cached_response, similarity = self.cache.get(query)
        if cached_response is not None:
            self.cache_hits += 1
            return cached_response, True, similarity

        # If no sufficiently similar query is cached, call the LLM
        response = self.llm.invoke(query)
        result = response.content

        # Store the new query-response pair in the cache
        self.cache.set(query, result)
        self.cache_misses += 1

        return result, False, similarity

    def get_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2%}",
            **self.cache.stats(),
        }


def demo_semantic_caching():
    """Demonstrate embedding-based semantic caching."""

    llm = SemanticCachedLLM(similarity_threshold=0.7)

    queries = [
        "What is AI?",
        "Explain artificial intelligence",  # Semantically similar -> cache hit
        "What is the capital of France?",
        "What's the capital city of France?",  # Semantically similar -> cache hit
        "What is quantum computing?",
    ]

    print("\nSemantic Caching Demo:\n")

    for query in queries:
        result, from_cache, similarity = llm.invoke(query)
        source = f"CACHE ({similarity:.2f})" if from_cache else f"LLM ({similarity:.2f})"
        print(f"[{source}] {query} -> {result[:30]}...")

    print(f"\nStats: {llm.get_stats()}")




if __name__ == "__main__":
    # demo_model_routing()
    # demo_caching()
    # demo_token_budgeting()
    demo_semantic_caching()