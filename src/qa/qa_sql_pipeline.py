import json
import logging
import os

import cohere
import google.generativeai as genai
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from qa.context_retrieval.retrieval_pipeline import retrieve_and_execute_pipeline

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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class QASQLPipeline:
    def retrieve_context(self, user_question: str, query_type: str, agent_type: str):
        """Retrieve context using SQL-based retrieval."""
        if query_type != "aggregating":
            raise ValueError(
                "Invalid query_type. Only 'aggregating' is accepted in QASQLPipeline."
            )

        return retrieve_and_execute_pipeline(user_question, query_type, agent_type)

    def answer_question(
        self, user_question: str, query_type: str, agent_type: str = "cohere"
    ) -> str:
        """Retrieves SQL context and generates a response."""
        st.write("Step 1: Retrieving context from SQL database...")
        logging.info("Retrieving context for the question using SQL.")
        sql_query, context = self.retrieve_context(
            user_question, query_type, agent_type
        )

        if context is None or (isinstance(context, pd.DataFrame) and context.empty):
            logging.warning("No context found.")
            return None

        # Format context based on type: DataFrame to string if SQL-based
        st.write("Step 2: Formatting retrieved context for response generation...")
        context_text = (
            context.to_string(index=False)
            if isinstance(context, pd.DataFrame)
            else str(context)
        )
        logging.info("SQL context retrieved and formatted.")

        prompt = f"""From this query:\n{sql_query}\n
        We got result:\n {context_text}
        \nAnswer this question based on the result above to make comprehensive but not verbose answer for our spotify management team:\nQuestion: {user_question}
        """
        logging.info("Prompt generated:\n%s", prompt)

        st.write("Step 3: Prompt generated. Generating response...")
        response = self.generate_response(agent_type, prompt)
        return response

    def generate_response(self, agent_type: str, prompt: str) -> str:
        """Generates a response based on the agent type and prompt."""
        if agent_type == "cohere":
            response = cohere_client.chat(
                model="command-r-plus-08-2024",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.message.content[0].text if response.message else None
        elif agent_type == "llama":
            # Send request to Llama API
            payload = {"model": "llama3.2", "prompt": prompt}
            headers = {"Content-Type": "application/json"}
            return self._send_request(payload, headers)
        elif agent_type == "gemini":
            response = gemini_model.generate_content(prompt)
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
