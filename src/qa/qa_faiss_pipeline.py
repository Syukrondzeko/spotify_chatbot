# qa/qa_faiss_pipeline.py

import os
import logging
import pandas as pd
import json
from dotenv import load_dotenv
import requests
import cohere
import google.generativeai as genai
from qa.context_retrieval.faiss.faiss_agent import search_similar_sentences

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
LLAMA_API = os.getenv("LLAMA_API")

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize models
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

def _send_request(payload, headers):
    """
    Sends a request to the Llama API and processes the streaming response.

    Parameters:
    - payload (dict): The request payload.
    - headers (dict): The headers for the request.

    Returns:
    - str: The combined response text from the streaming output.
    """
    response = requests.post(LLAMA_API, json=payload, headers=headers, stream=True)
    if response.status_code == 200:
        query_result = ""
        for line in response.iter_lines():
            if line:
                try:
                    line_data = json.loads(line.decode("utf-8"))
                    query_result += line_data.get("response", "")
                except json.JSONDecodeError:
                    logging.warning("Could not decode line as JSON")
        return query_result
    else:
        logging.error(f"Error: {response.status_code}")
        return None

def generate_response(agent_type: str, prompt: str) -> str:
    """
    Generates a response based on the agent type and prompt.

    Parameters:
    - agent_type (str): Model to use ("cohere", "llama", "gemini").
    - prompt (str): Input prompt for the model.

    Returns:
    - str: Model's response or None if no response.
    """
    if agent_type == "cohere":
        response = cohere_client.chat(model="command-r-plus-08-2024", messages=[{"role": "user", "content": prompt}])
        return response.message.content[0].text if response.message else None
    elif agent_type == "llama":
        # Define the payload and headers for the Llama API request
        payload = {"model": "llama3.2", "prompt": prompt}
        headers = {"Content-Type": "application/json"}
        return _send_request(payload, headers)
    elif agent_type == "gemini":
        response = gemini_model.generate_content(prompt)
        return response.text if response else None
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

def qa_faiss_pipelines(user_question: str, agent_type: str = "cohere") -> str:
    """
    Retrieves FAISS context and generates a response with the specified model.

    Parameters:
    - user_question (str): User's question.
    - agent_type (str): Model type ("cohere", "llama", "gemini").

    Returns:
    - str: Generated answer or None if no answer.
    """
    logging.info("Retrieving context from FAISS for the question.")
    context_from_faiss = search_similar_sentences(user_question)

    if not context_from_faiss:
        logging.warning("No context found.")
        return None

    context_text = "\n".join(context_from_faiss)
    logging.info("Context retrieved and formatted.")

    prompt = f"Using the following context:\nContext: {context_text}\nAnswer the question:\nQuestion: {user_question}"
    logging.info("Prompt generated:\n%s", prompt)

    response = generate_response(agent_type, prompt)
    return response

# Example usage
if __name__ == "__main__":
    question = "What is the best feature in Spotify?"
    agent = "llama"  # Options: "cohere", "llama", "gemini"
    answer = qa_faiss_pipelines(question, agent_type=agent)
    print("Answer:", answer if answer else "No answer generated.")
