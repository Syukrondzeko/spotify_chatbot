import pandas as pd
from retrieval_agent.my_cohere import CohereQueryRetriever
from post_processing.query_extractor import extract_query
from post_processing.query_executor import run_query
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def retrieve_and_execute_pipeline(user_question):
    # Initialize CohereQueryRetriever instance
    api_key = os.getenv("COHERE_API")
    retriever = CohereQueryRetriever(api_key=api_key)

    # Step 1: Get raw output from agent
    raw_response = retriever.get_query_from_cohere(user_question)
    print("Step 1 - Raw Output from Agent Retriever:\n", raw_response)

    # Step 2: Clean SQL extraction
    clean_query = extract_query(raw_response)
    print("Step 2 - Clean SQL Query:\n", clean_query)

    # Step 3: Run the query and retrieve data
    if clean_query:
        results_df = run_query(clean_query)
        print("Step 3 - Data Retrieved:\n", results_df)

        # Check if data is empty; if so, retrieve a relaxed query
        if results_df.empty:
            print("No results found. Attempting a relaxed query.")

            # Get a more relaxed query based on the previous query
            relaxed_query_response = retriever.get_relax_query_from_cohere(user_question, clean_query)
            relaxed_query = extract_query(relaxed_query_response)
            print("Relaxed SQL Query:\n", relaxed_query)

            # Execute the relaxed query if it exists
            if relaxed_query:
                relaxed_results_df = run_query(relaxed_query)
                print("Step 4 - Data Retrieved from Relaxed Query:\n", relaxed_results_df)
            else:
                print("No valid relaxed SQL query was generated.")
    else:
        print("No valid SQL query was extracted.")

# Example usage
if __name__ == "__main__":
    user_question = "In comparison to our application, which music streaming platform are users most likely to compare ours with?"
    retrieve_and_execute_pipeline(user_question)
