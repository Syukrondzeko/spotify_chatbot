# Simple test script to retrieve the embedding for FAISS index 0
import os
import faiss
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# FAISS index path
FAISS_PATH = os.getenv("FAISS_PATH")

def load_faiss_index(faiss_path):
    """Load the FAISS index from the specified path."""
    assert os.path.exists(faiss_path), f"FAISS index file not found at {faiss_path}"
    logging.info("Loading FAISS index")
    index = faiss.read_index(faiss_path)
    logging.info("FAISS index loaded successfully")
    return index

def get_embedding_by_index(index, faiss_index):
    """Retrieve the embedding for the specified FAISS index."""
    embedding = index.reconstruct(faiss_index)
    logging.info(f"Retrieved embedding for FAISS index {faiss_index}")
    return embedding

if __name__ == "__main__":
    # Load FAISS index
    index = load_faiss_index(FAISS_PATH)
    
    # Retrieve embedding for FAISS index 0
    faiss_index = 0
    embedding = get_embedding_by_index(index, faiss_index)
    print(f"Embedding for FAISS index {faiss_index}: {embedding}")
