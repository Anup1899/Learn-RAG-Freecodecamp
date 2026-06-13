import os
from dotenv import load_dotenv
from langsmith import traceable, Client
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "Freecodecamp-RAG")


# --- Documents ---
documents = [
    Document(page_content="Python is a high-level programming language known for its readability.", metadata={"id": "doc1", "category": "programming"}),
    Document(page_content="LangChain is a framework for building LLM-powered applications.", metadata={"id": "doc2", "category": "langchain"}),
    Document(page_content="LangSmith is the observability and evaluation platform for LangChain.", metadata={"id": "doc3", "category": "langsmith"}),
    Document(page_content="RAG stands for Retrieval-Augmented Generation — it grounds LLM answers in real documents.", metadata={"id": "doc4", "category": "rag"}),
    Document(page_content="Vector databases store embeddings for fast semantic similarity search.", metadata={"id": "doc5", "category": "rag"}),
    Document(page_content="Tracing captures every step of a chain — inputs, outputs, latency, and token usage.", metadata={"id": "doc6", "category": "observability"}),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="langsmith_demo_collection",
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = init_chat_model(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LangChain chain — automatically traced end-to-end when LANGCHAIN_TRACING_V2=true.
# LangSmith captures each node: retriever, prompt, llm, parser — with inputs/outputs/latency.
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --- @traceable: wraps plain Python functions as named spans in the trace ---
# Use this for any logic that is NOT a LangChain component but is still part of the pipeline.
@traceable(name="validate_query")
def validate_query(query: str) -> str:
    """Rejects empty queries before they reach the chain."""
    if not query.strip():
        raise ValueError("Query must not be empty.")
    return query.strip()


@traceable(name="rag_pipeline")
def run_rag(query: str) -> dict:
    """Full pipeline: validate → retrieve → generate. Appears as a single parent span."""
    clean_query = validate_query(query)
    answer = rag_chain.invoke(clean_query)
    return {"query": clean_query, "answer": answer}


# --- Manual feedback: log thumbs-up/down on a run from user signals ---
def log_feedback(run_id: str, score: int, comment: str = ""):
    """
    score = 1 (positive) or 0 (negative).
    run_id comes from the trace — visible in LangSmith UI or via the run metadata.
    """
    client = Client()
    client.create_feedback(
        run_id=run_id,
        key="user_feedback",
        score=score,
        comment=comment,
    )
    print(f"Feedback logged for run {run_id}: score={score}")


# --- Run ---
if __name__ == "__main__":
    test_queries = [
        "What is LangSmith used for?",
        "How does RAG work?",
        "What is a vector database?",
    ]

    for query in test_queries:
        print(f"\nQ: {query}")
        result = run_rag(query)
        print(f"A: {result['answer']}")

    # To log feedback on a specific run, grab the run_id from the LangSmith UI
    # and call: log_feedback(run_id="<run-id>", score=1, comment="Correct and concise")
