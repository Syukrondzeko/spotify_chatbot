import pandas as pd
import logging
from qa.context_retrieval.sql.retrieval_agent.my_cohere import CohereQueryRetriever
from qa.context_retrieval.sql.retrieval_agent.llama_3 import LlamaQueryRetriever
from qa.context_retrieval.sql.retrieval_agent.gemini_flash import GeminiQueryRetriever
from qa.context_retrieval.sql.post_processing.query_extractor import extract_query
from qa.context_retrieval.sql.post_processing.query_executor import run_query

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve_and_execute_pipeline(user_question, query_type, agent_type="cohere"):
    # Initialize the appropriate agent based on agent_type
    api_key = os.getenv("COHERE_API") if agent_type == "cohere" else os.getenv("GEMINI_API_KEY") if agent_type == "gemini" else os.getenv("LLAMA_API")
    
    if agent_type == "cohere":
        retriever = CohereQueryRetriever(api_key=api_key)
    elif agent_type == "llama":
        retriever = LlamaQueryRetriever(api_key=api_key)
    elif agent_type == "gemini":
        retriever = GeminiQueryRetriever(api_key=api_key)
    else:
        raise ValueError(f"Unsupported agent_type: {agent_type}")

    # Step 1: Get raw output from agent
    raw_response = retriever.get_query(user_question, query_type)
    logging.info("Step 1 - Raw Output from Agent Retriever:\n%s", raw_response)

    # Step 2: Clean SQL extraction
    clean_query = extract_query(raw_response)
    logging.info("Step 2 - Clean SQL Query:\n%s", clean_query)

    # Step 3: Run the query and retrieve data
    if clean_query:
        query_result = run_query(clean_query)

        if isinstance(query_result, pd.DataFrame):
            logging.info("Step 3 - Data Retrieved:\n%s", query_result)
            # Check if data is empty; if so, retrieve a relaxed query
            if query_result.empty:
                logging.warning("No results found. Attempting a relaxed query.")

                # Get a more relaxed query based on the previous query
                relaxed_query_response = retriever.get_relax_query(user_question, clean_query)
                relaxed_query = extract_query(relaxed_query_response)
                logging.info("Relaxed SQL Query:\n%s", relaxed_query)

                # Execute the relaxed query if it exists
                if relaxed_query:
                    relaxed_results_df = run_query(relaxed_query)
                    logging.info("Step 4 - Data Retrieved from Relaxed Query:\n%s", relaxed_results_df)
                else:
                    logging.warning("No valid relaxed SQL query was generated.")
            else:
                return query_result
        else:
            # If `query_result` is an error message, pass it to `solved_error_query`
            logging.error("Step 3 - Error Encountered:\n%s", query_result)
            solved_query_response = retriever.solved_error_query(user_question, clean_query, query_result)
            solved_query = extract_query(solved_query_response)
            logging.info("Step 4 - Resolved Query:\n%s", solved_query)

            # Attempt to run the resolved query
            if solved_query:
                fixed_results_df = run_query(solved_query)
                logging.info("Step 5 - Data Retrieved from Resolved Query:\n%s", fixed_results_df)
            else:
                logging.warning("No valid resolved SQL query was generated.")

    else:
        logging.warning("No valid SQL query was extracted.")

# Example usage
if __name__ == "__main__":
    user_question = "What is the dissatisfaction in the august 2014?"
    query_type = "aggregating"
    agent_type = "cohere"  # Change this to "llama" or "gemini" as needed
    retrieve_and_execute_pipeline(user_question, query_type, agent_type)