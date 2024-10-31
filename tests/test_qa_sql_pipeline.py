import sys
import os
import logging

# Add the src directory to sys.path to allow importing from the qa module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qa.qa_sql_pipeline import QASQLPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize QASQLPipeline
pipeline = QASQLPipeline()

# Define a test question and parameters
question = "How many negative sentiment for our product?"
query_type = "aggregating"  # Ensure this matches the requirements in QASQLPipeline
agent_type = "gemini"  # Options: "cohere", "llama", "gemini"

# Run the pipeline
logging.info(f"Processing question: '{question}' with agent type: '{agent_type}' and query type: '{query_type}'")
answer = pipeline.answer_question(question, query_type=query_type, agent_type=agent_type)

# Output the answer
if answer:
    print("Answer:", answer)
else:
    logging.warning("No answer generated for the question.")
