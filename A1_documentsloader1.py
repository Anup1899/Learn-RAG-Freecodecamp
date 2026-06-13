import os
import json
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from dotenv import load_dotenv

load_dotenv()


def load_text_files():
    # Crete a temporary file with some sample text data for demonstration purposes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing the TextLoader.")
        temp_file_path = temp_file.name

    try:
        # Use TextLoader to load the content of the temporary file
        text_loader = TextLoader(temp_file_path)
        documents = text_loader.load()

        print(f"Loaded {len(documents)} document(s) from the temporary file.")

        for doc in documents:
            print(f"Loaded document: {doc}")
            print(f"Loaded document metadata: {doc.metadata}")
            print(f"Loaded document content: {doc.page_content}")
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


def pdf_loader(pdf_path):
    # Use PyPDFLoader to load the content of a PDF file
    pdf_loader = PyPDFLoader(pdf_path)
    documents = pdf_loader.load()
    print(f"Loaded {len(documents)} document(s) from the PDF file.")
    for doc in documents:
        # print(f"Loaded document: {doc}")
        print(f"Loaded document metadata:\n{json.dumps(doc.metadata, indent=2)}")
        # print(f"Loaded document content: {doc.page_content}")
        break


if __name__ == "__main__":
    # load_text_files()
    pdf_loader("./docs/langchain.pdf")
