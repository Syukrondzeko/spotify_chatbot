# src/qa/qa_mix_pipeline.py

import logging
import pandas as pd
import numpy as np
import faiss
import requests
import cohere
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
from qa.context_retrieval.retrieval_pipeline import retrieve_and_execute_pipeline

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
LLAMA_API = os.getenv("LLAMA_API")

# Initialize models
genai.configure(api_key=GEMINI_API_KEY)
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QAMixPipeline:
    def __init__(self, model, faiss_index, metadata_by_id):
        """
        Initialize QAMixPipeline with model, FAISS index, and metadata.
        
        Args:
        - model: The embedding model (e.g., SentenceTransformer).
        - faiss_index: The FAISS index for similarity search.
        - metadata_by_id: A dictionary mapping FAISS index entries to metadata.
        """
        self.model = model
        self.faiss_index = faiss_index
        self.metadata_by_id = metadata_by_id
        logging.info("QAMixPipeline initialized with model, FAISS index, and metadata.")

    def retrieve_context(self, user_question: str, query_type: str, agent_type: str):
        """Retrieve context using SQL-based retrieval."""
        if query_type != "filtering":
            raise ValueError("Invalid query_type. Only 'filtering' is accepted in QAMixPipeline.")
        
        # This function should call retrieve_and_execute_pipeline, assuming it’s defined elsewhere.
        return retrieve_and_execute_pipeline(user_question, query_type, agent_type)

    def answer_question(self, user_question: str, query_type: str, agent_type: str = "cohere", top_k: int = 5):
        """Retrieves SQL context, filters FAISS embeddings, performs similarity search, and generates a response."""
        sql_query, context = self.retrieve_context(user_question, query_type, agent_type)
        logging.info(f"Retrieving context for the question using SQL:\n{sql_query}")

        if context is None or (isinstance(context, pd.DataFrame) and context.empty):
            logging.warning("No context found.")
            print("No context found.")
            return
        
        logging.info(f"Context:\n{context}")

        # Embed the user question
        question_embedding = self.model.encode(user_question).astype('float32').reshape(1, -1)
        faiss.normalize_L2(question_embedding)

        # Filter FAISS indices to only include those in context and retrieve relevant embeddings
        context_ids = set(context['id'])
        filtered_embeddings = []
        metadata_map = []

        for faiss_index in range(self.faiss_index.ntotal):
            metadata_entry = self.metadata_by_id.get(faiss_index)
            if metadata_entry and metadata_entry['id'] in context_ids:
                embedding_vector = np.array(metadata_entry['embedding']).astype('float32')
                filtered_embeddings.append(embedding_vector)
                metadata_map.append(metadata_entry)

        if not filtered_embeddings:
            print("No embeddings found for the retrieved IDs.")
            return

        # Stack embeddings and compute similarity
        filtered_embeddings = np.vstack(filtered_embeddings)
        faiss.normalize_L2(filtered_embeddings)
        similarities = np.dot(filtered_embeddings, question_embedding.T).flatten()

        # Sort by similarity and get top_k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Format the top_k results into a context string for the prompt
        context_text = ""
        for idx in top_indices:
            similarity_score = similarities[idx]
            metadata_entry = metadata_map[idx]
            text = metadata_entry.get("text", "Text not found")
            review_id = metadata_entry.get("id", "ID not found")
            context_text += f"Text: {text}\n\n"
        
        # Generate the final prompt
        prompt = f"Using the following context:\n{context_text}\nAnswer the question for our spotify management team:\nQuestion: {user_question}"
        logging.info("Prompt generated:\n%s", prompt)

        # Generate response using the chosen agent
        response = self.generate_response(agent_type, prompt)
        return response

    def generate_response(self, agent_type: str, prompt: str) -> str:
        """Generates a response based on the agent type and prompt."""
        if agent_type == "cohere":
            response = cohere_client.chat(
                model="command-r-plus-08-2024", messages=[{"role": "user", "content": prompt}]
            )
            return response.message.content[0].text if response.message else None
        elif agent_type == "llama":
            payload = {"model": "llama3.2", "prompt": prompt}
            headers = {"Content-Type": "application/json"}
            return self._send_request(payload, headers)
        elif agent_type == "gemini":
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
            return response.text if response else None
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

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