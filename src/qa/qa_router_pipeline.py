# src/qa/qa_router_pipeline.py

import logging
import json
import requests
import os
import cohere
from dotenv import load_dotenv
from qa.router.task_router import router_question, post_processing_router
from qa.qa_sql_pipeline import QASQLPipeline
from qa.qa_mix_pipeline import QAMixPipeline
from qa.qa_faiss_pipeline import QAFaissPipeline
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RouterPipeline:
    def __init__(self, model, faiss_index, metadata):
        logging.info("Initializing RouterPipeline")
        self.model = model  # Model passed in from outside
        self.faiss_index = faiss_index  # FAISS index passed in from outside
        self.metadata = metadata
        self.metadata_by_id = {entry['id']: entry for entry in metadata}
        logging.info("RouterPipeline initialized with model, FAISS index, and metadata.")

    def classify_user_question(self, user_question, agent_type="llama"):
        """Classifies the user's question and returns the classification."""
        logging.info("Generating prompt for classification.")
        prompt = router_question(user_question)
        logging.info(f"Sending prompt to {agent_type} model for classification.")
        
        # Generate response based on the agent type
        if agent_type == "llama":
            response = self.generate_response_llama(prompt)
        elif agent_type == "cohere":
            response = self.generate_response_cohere(prompt)
        elif agent_type == "gemini":
            response = self.generate_response_gemini(prompt)
        else:
            logging.error(f"Unsupported agent type: {agent_type}")
            return None

        # Process and return classification
        if response:
            classification = post_processing_router(response)
            logging.info(f"Classification result: {classification}")
            return classification
        else:
            logging.error("Failed to get a response from the model")
            return None

    def route_question(self, question, agent_type):
        """Routes the question to the appropriate pipeline based on the classification."""
        classification = self.classify_user_question(question, agent_type)
        
        if classification == "aggregate":
            logging.info("Routing to QASQLPipeline for aggregation.")
            pipeline = QASQLPipeline()
            return pipeline.answer_question(question, query_type="aggregating", agent_type=agent_type)
        
        elif classification == "filter":
            logging.info("Routing to QAMixPipeline for filtering.")
            pipeline = QAMixPipeline(self.model, self.faiss_index, self.metadata_by_id)
            return pipeline.answer_question(question, query_type="filtering", agent_type=agent_type)
        
        elif classification == "direct":
            logging.info("Routing to QAFaissPipeline for direct answer.")
            pipeline = QAFaissPipeline(self.model, self.faiss_index, self.metadata) 
            answer = pipeline.answer_question(question, top_k=5, nprobe=10, agent_type=agent_type)
            logging.info("Received answer from QAFaissPipeline.")
            return answer
        
        else:
            logging.error("Invalid classification for question.")
            return None

    def generate_response_llama(self, prompt):
        """Generate a response from the Llama API."""
        payload = {"model": "llama3.2", "prompt": prompt}
        headers = {"Content-Type": "application/json"}
        response = requests.post(os.getenv("LLAMA_API"), json=payload, headers=headers, stream=True)
        return self._process_response(response)

    def generate_response_cohere(self, prompt):
        """Generate a response from the Cohere API."""
        cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        response = cohere_client.chat(model="command-r-plus-08-2024", messages=[{"role": "user", "content": prompt}])
        return response.message.content[0].text if response.message else None

    def generate_response_gemini(self, prompt):
        """Generate a response from the Gemini API."""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        return response.text if response else None

    def _process_response(self, response):
        """Process streaming response from Llama API."""
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
