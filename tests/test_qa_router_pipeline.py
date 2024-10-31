# tests/test_qa_router_pipeline.py

import sys
import os
import logging
import json
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Add src directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qa.qa_router_pipeline import RouterPipeline

# Load environment variables
load_dotenv()
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
METADATA_FAISS_PATH = os.getenv("METADATA_FAISS_PATH")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize required components
logging.info("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL_PATH)

logging.info("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_PATH)

logging.info("Loading metadata...")
with open(METADATA_FAISS_PATH, "r") as f:
    metadata = json.load(f)

# Create a metadata map for quick access by faiss_index
metadata_map = {entry['faiss_index']: entry for entry in metadata}

# Initialize RouterPipeline with loaded components
router_pipeline = RouterPipeline(model, faiss_index, metadata_map)

# Define a single test question and agent type
question = "How many negative sentiment for our product?"
agent_type = "cohere"

# Route the question using RouterPipeline and output the result
logging.info(f"Processing question: '{question}' with agent: {agent_type}")
answer = router_pipeline.route_question(question, agent_type)

# Output the answer
if answer:
    logging.info(f"Final answer for '{question}': {answer}")
else:
    logging.warning(f"No answer generated for '{question}'")

print("Test completed.")
