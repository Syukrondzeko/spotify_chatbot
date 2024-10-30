import os
import cohere
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Cohere API key from the .env file
api_key = os.getenv("COHERE_API")

class CohereQueryRetriever:
    def __init__(self, api_key):
        # Initialize Cohere client
        self.client = cohere.ClientV2(api_key=api_key)

    def get_query_from_cohere(self, user_question):
        # Define the prompt with the user question
        prompt = f"""
        Question: {user_question}

        Table: 'user_review'
        Columns: review_id, pseudo_author_id, author_name, review_text, review_rating, review_likes, year, month, day

        Instructions: Identify the relevant only essential columns to retrieve and apply only essential filters. Then, provide the SQL query in backticks. Use a simple query, avoiding any complex structures.
        """
        # Send the chat request
        res = self.client.chat(
            model="command-r-plus-08-2024",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        # Return the response content text
        return res.message.content[0].text

    def get_relax_query_from_cohere(self, user_question, previous_query):
        # Define a relaxed prompt with the previous query
        prompt = f"""
        Question: {user_question}
        Previous query: {previous_query}

        Instructions: It produced an empty result. Edit your query to make the filter more relaxed or remove unnecessary filters, but still fulfill my question.
        """
        # Send the chat request
        res = self.client.chat(
            model="command-r-plus-08-2024",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        # Return the response content text
        return res.message.content[0].text

