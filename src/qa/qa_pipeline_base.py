import os
import logging
import requests
import cohere
import google.generativeai as genai
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import json

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
LLAMA_API = os.getenv("LLAMA_API")

# Initialize models
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QAPipelineBase(ABC):
    def _send_request(self, payload, headers):
        """Sends a request to the Llama API and processes the streaming response."""
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

    def generate_response(self, agent_type: str, prompt: str) -> str:
        """Generates a response based on the agent type and prompt."""
        if agent_type == "cohere":
            response = cohere_client.chat(model="command-r-plus-08-2024", messages=[{"role": "user", "content": prompt}])
            return response.message.content[0].text if response.message else None
        elif agent_type == "llama":
            # Define the payload and headers for the Llama API request
            payload = {"model": "llama3.2", "prompt": prompt}
            headers = {"Content-Type": "application/json"}
            return self._send_request(payload, headers)
        elif agent_type == "gemini":
            response = gemini_model.generate_content(prompt)
            return response.text if response else None
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    @abstractmethod
    def retrieve_context(self, user_question: str):
        """Abstract method to retrieve context, to be implemented by subclasses."""
        pass
