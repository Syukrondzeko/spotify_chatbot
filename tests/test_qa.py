import sys
import logging

# Add the src directory to sys.path so we can import modules from qa
sys.path.insert(0, './src')

from qa.qa_router_pipeline import RouterPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize RouterPipeline
router_pipeline = RouterPipeline()

# Define a single question and agent
question = "What are the specific features or aspects that users appreciate the most in our application?"
agent = "cohere"

# Process the question using RouterPipeline's route_question method
logging.info(f"Processing question: '{question}' with agent: {agent}")
answer = router_pipeline.route_question(question, agent)

# Output the answer
if answer:
    logging.info(f"Final answer for '{question}': {answer}")
else:
    logging.warning(f"No answer generated for '{question}'")
