import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the .env file
api_key = os.getenv("GEMINI_API")

# Set the API URL
url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'

def get_query_from_gemini(user_question):
    # Define the prompt with the user question as an argument
    prompt = f"""
    Question: {user_question}

    Table: 'user_review'
    Columns: review_id, pseudo_author_id, author_name, review_text, review_rating, review_likes, year, month, date

    Instructions: Identify the relevant only essential columns to retrieve and apply only essential filters. Then, provide the SQL query in backticks. Use a simple query, avoiding any complex structures.
    """

    # Set up the payload
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    # Set up the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Make the POST request
    response = requests.post(url, json=payload, headers=headers)

    # Check the response status and print only the text result
    if response.status_code == 200:
        result_text = response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        print("Result:", result_text)
        return result_text
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Example usage
if __name__ == "__main__":
    user_question = "In comparison to our application, which music streaming platform are users most likely to compare ours with?"
    get_query_from_gemini(user_question)
