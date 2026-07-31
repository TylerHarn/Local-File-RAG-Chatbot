import os

# Ollama connection
OLLAMA_BASE_URL = "http://localhost:11434"

# Models (make sure you've run: ollama pull <model>)
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"

# Storage
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "local_files"

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval settings
TOP_K = 5

# Supported file extensions to index
SUPPORTED_EXTENSIONS = (".txt", ".md", ".py", ".json", ".csv", ".pdf")

# Ensure the persist directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
