# src/streamlit_app.py

import streamlit as st
import logging
import json
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qa.qa_router_pipeline import RouterPipeline
import os

# Load environment variables
load_dotenv()
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
METADATA_FAISS_PATH = os.getenv("METADATA_FAISS_PATH")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the embedding model, FAISS index, and metadata once
@st.cache_resource
def load_resources():
    logging.info("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    
    logging.info("Loading FAISS index...")
    faiss_index = faiss.read_index(FAISS_PATH)
    
    logging.info("Loading metadata...")
    with open(METADATA_FAISS_PATH, "r") as f:
        metadata = json.load(f)
    
    # Initialize RouterPipeline with the loaded components
    router_pipeline = RouterPipeline(model, faiss_index, metadata)
    
    return router_pipeline

# Initialize the RouterPipeline
router_pipeline = load_resources()

# Streamlit App UI
st.title("Q&A with RouterPipeline")
st.write("Enter a question, and the RouterPipeline will find the best answer.")

# Input box for the user's question
question = st.text_input("Your Question", "")

# Dropdown for selecting the agent type
agent_type = st.selectbox("Select Agent Type", ["cohere", "llama", "gemini"])

# Process the question when the user clicks the button
if st.button("Get Answer"):
    if question.strip():
        logging.info(f"Processing question: '{question}' with agent: {agent_type}")
        
        # Route the question through RouterPipeline
        answer = router_pipeline.route_question(question, agent_type)
        
        # Display the answer
        if answer:
            st.write("**Answer:**", answer)
        else:
            st.write("No answer generated for your question.")
    else:
        st.warning("Please enter a question.")
