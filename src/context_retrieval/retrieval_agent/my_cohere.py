import cohere
from dotenv import load_dotenv
from .agent_base import AgentBase

load_dotenv()

class CohereQueryRetriever(AgentBase):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.client = cohere.ClientV2(api_key=api_key)

    def get_query(self, user_question):
        prompt = self.build_query(user_question)
        response = self.client.chat(model="command-r-plus-08-2024", messages=[{"role": "user", "content": prompt}])
        return response.message.content[0].text

    def get_relax_query(self, user_question, previous_query):
        prompt = self.build_relax_query(user_question, previous_query)
        response = self.client.chat(model="command-r-plus-08-2024", messages=[{"role": "user", "content": prompt}])
        return response.message.content[0].text
