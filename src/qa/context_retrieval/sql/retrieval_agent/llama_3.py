import json
import os

import requests
from dotenv import load_dotenv

from .agent_base import AgentBase

load_dotenv()


class LlamaQueryRetriever(AgentBase):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.api_url = api_key
        self.ollama_model = "llama3.2"

    def get_query(self, user_question, query_type):
        if query_type == "filtering":
            prompt = self.build_filter_query(user_question)
        elif query_type == "aggregating":
            prompt = self.build_aggregate_query(user_question)
        payload = {"model": self.ollama_model, "prompt": prompt}
        headers = {"Content-Type": "application/json"}
        return self._send_request(payload, headers)

    def get_relax_query(self, user_question, previous_query):
        prompt = self.build_relax_query(user_question, previous_query)
        payload = {"model": self.ollama_model, "prompt": prompt}
        headers = {"Content-Type": "application/json"}
        return self._send_request(payload, headers)

    def solved_error_query(self, user_question, query, error_message):
        prompt = self.build_fixed_error_query_prompt(
            user_question, query, error_message
        )
        payload = {"model": self.ollama_model, "prompt": prompt}
        headers = {"Content-Type": "application/json"}
        return self._send_request(payload, headers)

    def _send_request(self, payload, headers):
        response = requests.post(
            self.api_url, json=payload, headers=headers, stream=True
        )
        if response.status_code == 200:
            query_result = ""
            for line in response.iter_lines():
                if line:
                    try:
                        line_data = json.loads(line.decode("utf-8"))
                        query_result += line_data.get("response", "")
                    except json.JSONDecodeError:
                        print("Warning: Could not decode line as JSON")
            return query_result
        else:
            print(f"Error: {response.status_code}")
            return None
