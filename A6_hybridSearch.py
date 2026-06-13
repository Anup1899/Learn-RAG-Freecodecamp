from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Documents with both semantic and specific indentifiers with metadata
documents = [
     # Add Documents which had both semantic and specific identifiers with metadata
    Document(page_content="This is a document about Python programming.", metadata={"id": "doc1", "category": "programming"}),
    Document(page_content="This document discusses the benefits of using AI in healthcare.", metadata={"id": "doc2", "category": "healthcare"}),
    Document(page_content="This is a document about Java programming.", metadata={"id": "doc3", "category": "programming"}),
    Document(page_content="This is a document about JavaScript programming.", metadata={"id": "doc4", "category": "programming"}),
    Document(page_content="Product SKU-7742X is our flagship product in the electronics category.", metadata={"id": "doc5", "category": "electronics"}),
    Document(page_content="For network connectivity, first check the ethernet cable and then the Wi-Fi settings.", metadata={"id": "doc6", "category": "troubleshooting"}),
    Document(page_content="Error code E_CONN_REFUSED indicates that the connection was refused by the server.", metadata={"id": "doc7", "category": "error"}),
    Document(page_content="Router configuration guide: Access the admin panel at 192.168.1.1 to modify settings.", metadata={"id": "doc8", "category": "config"})
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents, 
    embeddings,
    collection_name="hybrid_search_collection",
)

# Create vector retriever
vector_retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}
)

# BM25
bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3
)

# Hybrid: 50% vector (semantic match) + 50% BM25 (exact keyword match)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

def test_query(query, name, retriver):
    '''Test a query and show results'''
    results = retriver.invoke(query)
    print(f'{name} - Query: \"{query}\"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f'  {i+1}.{preview}')
    
    return results

# Test queries designed to challenge vecto
test_queries = [
    'SKU-7742X specifications',
    'E_CONN_REFUSED error',
    'How do I check network connectivity?',
    'router configuration',
]

for query in test_queries:
    print("="*60)

    # Vector only
    vector_results = test_query(query, "VECTOR", vector_retriever)

    # BM25 only
    bm25_results = test_query(query, "BM25", bm25_retriever)

    # Hybrid 
    hybrid_result = test_query(query, "HYBRID", ensemble_retriever)