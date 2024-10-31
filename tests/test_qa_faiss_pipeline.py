# tests/test_qa_faiss_pipeline.py

import sys
import os
import logging
import faiss
import json
from sentence_transformers import SentenceTransformer

# Add src directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qa.qa_faiss_pipeline import QAFaissPipeline

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
METADATA_FAISS_PATH = os.getenv("METADATA_FAISS_PATH")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load FAISS index, metadata, and model
logging.info("Loading FAISS index, metadata, and model for testing.")
index = faiss.read_index(FAISS_PATH)

with open(METADATA_FAISS_PATH, "r") as f:
    metadata = json.load(f)

model = SentenceTransformer(EMBEDDING_MODEL_PATH)
logging.info("FAISS resources loaded successfully for testing.")

# Initialize QAFaissPipeline with loaded resources
pipeline = QAFaissPipeline(model=model, index=index, metadata=metadata)

# Define a test question
question = "In comparison to our application, which music streaming platform are users most likely to compare ours with"
agent_type = "gemini"  # Choose from "cohere", "llama", or "gemini"

# Run the pipeline
logging.info(f"Processing question: '{question}' with agent type: '{agent_type}'")
answer = pipeline.answer_question(question, 5, 10, agent_type=agent_type)

# Output the answer
if answer:
    print("Answer:", answer)
else:
    logging.warning("No answer generated for the question.")
