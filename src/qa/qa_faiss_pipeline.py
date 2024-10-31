import logging
from qa.qa_pipeline_base import QAPipelineBase
from qa.context_retrieval.faiss.faiss_agent import FaissAgent

class QAFaissPipeline(QAPipelineBase):
    def __init__(self):
        # Instantiate the FaissAgent to load FAISS resources and model
        self.faiss_agent = FaissAgent()
        logging.info("QAFaissPipeline initialized successfully.")

    def retrieve_context(self, user_question: str):
        """Retrieve context using FAISS-based retrieval."""
        return self.faiss_agent.search_similar_sentences(user_question)

    def answer_question(self, user_question: str, agent_type: str = "cohere") -> str:
        """Retrieves FAISS context and generates a response."""
        logging.info("Retrieving context for the question using FAISS.")
        context = self.retrieve_context(user_question)

        if not context:
            logging.warning("No context found.")
            return None

        # Join list context for FAISS with newline for prompt formatting
        context_text = "\n".join(context)
        logging.info("FAISS context retrieved and formatted.")

        prompt = f"Using the following context:\nContext: {context_text}\nAnswer the question:\nQuestion: {user_question}"
        logging.info("Prompt generated:\n%s", prompt)

        response = self.generate_response(agent_type, prompt)
        return response
