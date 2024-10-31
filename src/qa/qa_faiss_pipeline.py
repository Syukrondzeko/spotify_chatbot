import json
import logging
import os

import cohere
import google.generativeai as genai
import requests
import streamlit as st

from qa.context_retrieval.faiss.faiss_agent import FaissAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class QAFaissPipeline:
    def __init__(self, model=None, index=None, metadata=None):
        self.faiss_agent = FaissAgent()
        self.model = model
        self.index = index
        self.metadata = metadata
        logging.info("QAFaissPipeline initialized successfully.")

    def retrieve_context(self, user_question: str, top_k: int, nprobe: int):
        """Retrieve context using FAISS-based retrieval."""
        context = self.faiss_agent.search_similar_sentences(
            user_question=user_question,
            model=self.model,
            index=self.index,
            metadata=self.metadata,
            top_k=top_k,
            nprobe=nprobe,
        )
        return context

    def answer_question(
        self,
        user_question: str,
        top_k: int = 5,
        nprobe: int = 10,
        agent_type: str = "cohere",
    ) -> str:
        """Retrieves FAISS context and generates a response."""
        st.write("Step 1: Retrieving relevant context from FAISS index...")
        context = self.retrieve_context(user_question, top_k, nprobe)

        if not context:
            logging.warning("No context found.")
            st.warning("No relevant context was found.")
            return None

        # Join list context for FAISS with newline for prompt formatting
        context_text = "\n".join(context)
        logging.info("FAISS context retrieved and formatted.")

        prompt = f"Using the following context:\nContext: {context_text}\nAnswer this question for our Spotify management team:\nQuestion: {user_question}"
        logging.info("Prompt generated:\n%s", prompt)
        st.write("Step 2: Generating response...")

        response = self.generate_response(agent_type, prompt)
        return response

    def generate_response(self, agent_type: str, prompt: str) -> str:
        """Generates a response based on the agent type and prompt."""
        if agent_type == "cohere":
            response = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY")).chat(
                model="command-r-plus-08-2024",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.message.content[0].text if response.message else None
        elif agent_type == "llama":
            payload = {"model": "llama3.2", "prompt": prompt}
            headers = {"Content-Type": "application/json"}
            return self._send_request(payload, headers)
        elif agent_type == "gemini":
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
                prompt
            )
            return response.text if response else None
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    def _send_request(self, payload, headers):
        """Sends a request to the Llama API and processes the streaming response."""
        response = requests.post(
            os.getenv("LLAMA_API"), json=payload, headers=headers, stream=True
        )
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
