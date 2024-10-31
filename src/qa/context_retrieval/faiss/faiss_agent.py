import faiss
import numpy as np
import json
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import logging
logging.basicConfig(level=logging.INFO)
logging.info("faiss_agent.py has been loaded.")


# Load environment variables
load_dotenv()
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
METADATA_FAISS_PATH = os.getenv("METADATA_FAISS_PATH")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class FaissAgent:
    def __init__(self):
        self.index = None
        self.metadata = None
        self.model = None
        self._load_faiss_index_and_metadata()
        self._load_model()

    def _load_faiss_index_and_metadata(self):
        """Load the FAISS index and metadata from environment-specified paths."""
        if self.index is None or self.metadata is None:
            logging.info("Loading FAISS index and metadata")
            # Load FAISS index
            self.index = faiss.read_index(FAISS_PATH)
            logging.info("FAISS index loaded successfully")

            # Load metadata for text retrieval
            with open(METADATA_FAISS_PATH, "r") as f:
                self.metadata = json.load(f)
            logging.info("Metadata loaded successfully")
        else:
            logging.info("FAISS index and metadata already loaded")

    def _load_model(self):
        """Load the SentenceTransformer model specified by the environment."""
        if self.model is None:
            logging.info("Loading SentenceTransformer model")
            self.model = SentenceTransformer(EMBEDDING_MODEL_PATH)
            logging.info("Model loaded successfully")
        else:
            logging.info("Model already loaded")

    def search_similar_sentences(self, user_question, top_k=5, nprobe=10):
        """Perform similarity search on the loaded FAISS index using the query embedding."""
        # Encode the question to create a query embedding
        logging.info(f"Encoding the question: '{user_question}'")
        query_embedding = self.model.encode(user_question)

        # Set nprobe for partition search
        self.index.nprobe = nprobe

        # Normalize query embedding for cosine similarity
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Perform similarity search
        logging.info(f"Performing similarity search with nprobe={nprobe}")
        D, I = self.index.search(query_embedding, k=top_k)
        
        # Retrieve closest sentences from metadata
        closest_sentences = [self.metadata[idx]['text'] for idx in I[0]]
        return closest_sentences
