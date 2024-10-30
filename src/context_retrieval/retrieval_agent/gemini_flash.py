import requests
from dotenv import load_dotenv
from .agent_base import AgentBase

load_dotenv()

class GeminiQueryRetriever(AgentBase):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.api_url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'

    def get_query(self, user_question):
        prompt = self.build_query(user_question)
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        return self._send_request(payload, headers)

    def get_relax_query(self, user_question, previous_query):
        prompt = self.build_relax_query(user_question, previous_query)
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        return self._send_request(payload, headers)

    def solved_error_query(self, user_question, query, error_message):
        prompt = self.build_fixed_error_query_prompt(user_question, query, error_message)
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}
        return self._send_request(payload, headers)

    def _send_request(self, payload, headers):
        response = requests.post(self.api_url, json=payload, headers=headers)
        if response.status_code == 200:
            result_text = response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            return result_text
        else:
            print(f"Error: {response.status_code}")
            return None
