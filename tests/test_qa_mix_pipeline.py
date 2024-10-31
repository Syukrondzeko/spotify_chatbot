import sys
import os
import logging
import faiss
import json
from sentence_transformers import SentenceTransformer
import pandas as pd

# Add src directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qa.qa_mix_pipeline import QAMixPipeline

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
metadata_by_id = {entry['id']: entry for entry in metadata}

model = SentenceTransformer(EMBEDDING_MODEL_PATH)
logging.info("FAISS resources loaded successfully for testing.")

# Initialize QAMixPipeline with loaded resources
pipeline = QAMixPipeline()
pipeline.model = model
pipeline.faiss_index = index
pipeline.metadata_by_id = metadata_by_id

# Define a test question and parameters
question = "What features do users mention the most?"
query_type = "filtering"
agent_type = "cohere"  # Options: "cohere", "llama", or "gemini"

# Run the pipeline
logging.info(f"Processing question: '{question}' with agent type: '{agent_type}'")
response = pipeline.answer_question(
    user_question="What are the most mentioned features?",
    query_type="filtering",
    agent_type="gemini",
    top_k=3  # Retrieve top 3 similar reviews
)
print("Generated Response:", response)