

from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def create_kb():
    """ Create a vector store from knowledge base """

    # Load PDF and split into chunks
    loader = PyPDFLoader("./docs/langchain.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(loader.load())

    # Create vector store from the chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory="./vector_store",
            )
    
    return vector_store

def demo_basic_rag():
    """ A basic RAG pipeline that retrives relevant chunks from the knowledge base and uses them to answer a question. """

    # create the vector store from the knowledge base
    vector_store = create_kb()

    # retrieve relevant chunks from the vector store
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    llm = init_chat_model(
        model="gpt-4o-mini",
        temperature=0.2
        )
    
    # RAG Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the following context:
        {context}
        Question : {question}
        Answer :
        
        Make sure to answer in concise manner, and if you don't know the answer, say you don't know.
        """
    )

    # Format retrived documents 
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # RAG Chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test the RAG chain with a question
    # Test
    questions = [
        "What is langchain?",
        "What created Langchain?",
        "What is LangGraph used for?",
    ]

    for question in questions:
        print(f"Question: {question}")
        answer = rag_chain.invoke(question)
        print(f"Answer: {answer}\n\n")


if __name__ == "__main__":
    demo_basic_rag()