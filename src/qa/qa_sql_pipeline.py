import os
import logging
import pandas as pd
from dotenv import load_dotenv
import requests
import cohere
import google.generativeai as genai
from qa.context_retrieval.retrieval_pipeline import retrieve_and_execute_pipeline

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize models
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

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
        headers = {"Authorization": f"Bearer {LLAMA_API_KEY}"}
        data = {"prompt": prompt, "max_tokens": 100}
        response = requests.post("https://api.llama.ai/v1/generate", json=data, headers=headers)
        return response.json().get("text") if response.ok else None
    elif agent_type == "gemini":
        response = gemini_model.generate_content(prompt)
        return response.text if response else None
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

def qa_sql_pipelines(user_question: str, agent_type: str = "cohere") -> str:
    """
    Retrieves SQL context and generates a response with the specified model.

    Parameters:
    - user_question (str): User's question.
    - agent_type (str): Model type ("cohere", "llama", "gemini").

    Returns:
    - str: Generated answer or None if no answer.
    """
    logging.info("Retrieving SQL context for the question.")
    context_from_sql = retrieve_and_execute_pipeline(user_question, agent_type)

    if context_from_sql is None:
        logging.warning("No context found.")
        return None

    context_text = context_from_sql.to_string(index=False) if isinstance(context_from_sql, pd.DataFrame) else str(context_from_sql)
    logging.info("Context retrieved and formatted.")

    prompt = f"Using the following context:\nContext: {context_text}\nAnswer the question:\nQuestion: {user_question}"
    logging.info("Prompt generated:\n%s", prompt)

    response = generate_response(agent_type, prompt)
    return response

# Example usage
if __name__ == "__main__":
    question = "How many comments for each month in 2014?"
    agent = "cohere"  # Options: "cohere", "llama", "gemini"
    answer = qa_sql_pipelines(question, agent_type=agent)
    print("Answer:", answer if answer else "No answer generated.")
