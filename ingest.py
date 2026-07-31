import os
import sys
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import (
    OLLAMA_BASE_URL, EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS
)

def load_file(path: str):
    """Load a single file based on its extension."""
    try:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
        return loader.load()
    except Exception as e:
        print(f"Skipped {path}: {e}")
        return []

def scan_folder(folder_path: str):
    """Walk the folder and load all supported files, tagging metadata with full path."""
    all_docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                full_path = os.path.join(root, file)
                docs = load_file(full_path)
                for doc in docs:
                    doc.metadata["source"] = full_path
                    doc.metadata["filename"] = file
                all_docs.extend(docs)
    return all_docs

def index_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid folder.")
        return

    print(f"Scanning: {folder_path}")
    docs = scan_folder(folder_path)

    if not docs:
        print("No supported files found. Nothing indexed.")
        return

    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks. Embedding and storing...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"Indexed {len(chunks)} chunks from {len(docs)} files into ChromaDB.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        folder = input("Enter the folder path to index: ").strip()
    else:
        folder = sys.argv[1]
    index_folder(folder)