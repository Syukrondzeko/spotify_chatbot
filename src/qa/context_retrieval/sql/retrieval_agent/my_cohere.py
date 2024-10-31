import cohere
from dotenv import load_dotenv
from .agent_base import AgentBase

load_dotenv()

class CohereQueryRetriever(AgentBase):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.client = cohere.ClientV2(api_key=api_key)

    def get_query(self, user_question, query_type):
        if query_type == 'filtering':
            prompt = self.build_filter_query(user_question)
        elif query_type == 'aggregating':
            prompt = self.build_aggregate_query(user_question)
        response = self.client.chat(model="command-r-plus-08-2024", messages=[{"role": "user", "content": prompt}])
        return response.message.content[0].text

    def get_relax_query(self, user_question, previous_query):
        prompt = self.build_relax_query(user_question, previous_query)
        response = self.client.chat(model="command-r-plus-08-2024", messages=[{"role": "user", "content": prompt}])
        return response.message.content[0].text

    def solved_error_query(self, user_question, query, error_message):
        prompt = self.build_fixed_error_query_prompt(user_question, query, error_message)
        response = self.client.chat(model="command-r-plus-08-2024", messages=[{"role": "user", "content": prompt}])
        return response.message.content[0].text
