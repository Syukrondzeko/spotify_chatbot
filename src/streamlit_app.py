import streamlit as st
import logging
from qa.qa_router_pipeline import RouterPipeline

# Configure logging to display in the Streamlit app
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize RouterPipeline once when the app starts
st.info("Initializing RouterPipeline...")
router_pipeline = RouterPipeline()
st.info("RouterPipeline initialized successfully.")

# Streamlit UI
st.title("Question Processing with RouterPipeline")

# Input for user question
question = st.text_input("Enter your question:", "What are the specific features or aspects that users appreciate the most in our application?")
agent = st.selectbox("Choose an agent model:", ["cohere", "llama", "gemini"])

# Process the question when the user clicks the button
if st.button("Process Question"):
    st.info(f"Processing question: '{question}' with agent: {agent}")
    answer = router_pipeline.route_question(question, agent)

    # Display the answer or a warning if no answer was generated
    if answer:
        st.success(f"Answer: {answer}")
    else:
        st.warning("No answer generated for this question.")
