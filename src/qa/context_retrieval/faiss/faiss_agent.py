import faiss
import numpy as np
import json
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
METADATA_FAISS_PATH = os.getenv("METADATA_FAISS_PATH")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Global variables to avoid reloading
index = None
metadata = None
model = None

def load_faiss_index_and_metadata():
    """Load the FAISS index and metadata from environment-specified paths."""
    global index, metadata
    if index is None or metadata is None:
        logging.info("Loading FAISS index and metadata")

        # Load FAISS index
        index = faiss.read_index(FAISS_PATH)
        logging.info("FAISS index loaded successfully")

        # Load metadata for text retrieval
        with open(METADATA_FAISS_PATH, "r") as f:
            metadata = json.load(f)
        logging.info("Metadata loaded successfully")
    else:
        logging.info("FAISS index and metadata already loaded")

def load_model():
    """Load the SentenceTransformer model specified by the environment."""
    global model
    if model is None:
        logging.info("Loading SentenceTransformer model")
        model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        logging.info("Model loaded successfully")
    else:
        logging.info("Model already loaded")

def search_similar_sentences(user_question, top_k=5, nprobe=10):
    """Main function to handle FAISS-based retrieval given a question."""
    # Load model, FAISS index, and metadata if not already loaded
    load_faiss_index_and_metadata()
    load_model()

    # Encode the question to create a query embedding
    logging.info(f"Encoding the question: '{user_question}'")
    query_embedding = model.encode(user_question)

    # Set nprobe for partition search
    index.nprobe = nprobe

    # Normalize query embedding for cosine similarity
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    # Perform similarity search
    logging.info(f"Performing similarity search with nprobe={nprobe}")
    D, I = index.search(query_embedding, k=top_k)
    
    # Retrieve closest sentences from metadata
    closest_sentences = [metadata[idx]['text'] for idx in I[0]]
    return closest_sentences
