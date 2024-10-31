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
st.sidebar.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_CMYK_Green.png", width=200)
st.sidebar.title("Spotify Bot")
st.sidebar.write("Ask questions about Spotify's user feedback and insights.")

st.title("Spotify Bot Q&A")
st.write("Enter a question, and the Spotify Bot will find the best answer for you.")

# Input box for the user's question
question = st.text_input("Your Question", "")

# Dropdown for selecting the agent type
agent_type = st.selectbox("Select Agent Type", ["cohere", "llama", "gemini"])

# Initialize the session state for the answer
if "answer" not in st.session_state:
    st.session_state["answer"] = ""

# Process the question when the user clicks the button
if st.button("Get Answer"):
    if question.strip():
        logging.info(f"Processing question: '{question}' with agent: {agent_type}")
        
        # Route the question through RouterPipeline
        st.session_state["answer"] = router_pipeline.route_question(question, agent_type)
    else:
        st.warning("Please enter a question.")

# Display the answer if it's available
if st.session_state["answer"]:
    st.write("**Answer:**", st.session_state["answer"])

# Button to clear the answer without reloading the page
if st.button("Clear Answer"):
    st.session_state["answer"] = ""
