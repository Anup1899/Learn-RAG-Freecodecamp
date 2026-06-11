from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def basic_embeddings():

    # Single text
    text = "What is Machine Learning?"
    single_embedding = embeddings_model.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 dimension: {single_embedding[:5]}")
    # Check if the vector is normalized.
    # What is np.linalg.norm(single_embedding):.4f? It should be close to 1 if the vector is normalized.
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}") 


def batch_embeddings():
    text = [
        "What is Machine Learning?",
        "What is Deep Learning?",
        "What is Natural Language Processing?"
    ]

    batch_embeddings = embeddings_model.embed_documents(text)

    for i, emb in enumerate(batch_embeddings):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm {np.linalg.norm(emb):.4f}\n")  # Check if the vector is normalized


def similarity_search():
    # Documents
    docs = [
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers.",
        "Natural language processing is a field of AI that enables computers to understand and process human language.",
        "Capital of India is New Delhi.",
    ]

    query = "What is Deep learning?"    

    # Embedd documents and query
    doc_vectors = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)

    # Compute cosine similarity
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vectors]

    # Rank documents based on similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    # Display results
    print(f"Query: {query}\n")
    print("Ranked Documents:")
    for doc, sim in ranked_docs:
        print(f"Similarity: {sim:.4f} - {doc}")

if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    similarity_search()   
