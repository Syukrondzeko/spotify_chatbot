import os
import requests
import json  # Import json directly
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the Ollama API base URL
ollama_api_url = os.getenv("OLLAMA_PATH")

def get_query_from_llama(user_question):
    # Prompt template
    prompt = f"""
    Question: {user_question}

    Table: 'user_review'
    Columns: review_id, pseudo_author_id, author_name, review_text, review_rating, review_likes, year, month, day

    Instructions: Identify the relevant only essential columns to retrieve and apply only essential filters. Then, provide the SQL query in backticks. Use a simple query, avoiding any complex structures. I will copy paste that query, so make sure it is executable without any need to manual replacement in the query. I use sqlite
    """

    # Payload for the Ollama API
    payload = {
        "model": "llama3.1",
        "prompt": prompt
    }

    # Set headers if any authentication is required
    headers = {
        "Content-Type": "application/json"
    }

    # Make the API request with stream enabled
    response = requests.post(ollama_api_url, json=payload, headers=headers, stream=True)

    # Check for successful response
    if response.status_code == 200:
        query_result = ""
        # Read each line in the streamed response
        for line in response.iter_lines():
            if line:
                # Decode each line to JSON, using the standard json library
                try:
                    line_data = json.loads(line.decode("utf-8"))
                    query_result += line_data.get("response", "")
                except json.JSONDecodeError:
                    print("Warning: Could not decode line as JSON")
        
        # Output the final accumulated SQL query
        print("Generated SQL Query:", query_result)
        return query_result
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

if __name__ == "__main__":
    user_question = "In comparison to our application, which music streaming platform are users most likely to compare ours with?"
    get_query_from_llama(user_question)
