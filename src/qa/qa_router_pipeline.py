# src/qa/qa_router_pipeline.py

import logging
from dotenv import load_dotenv
from qa.router.task_router import router_question, post_processing_router
from qa.qa_sql_pipeline import QASQLPipeline


# Load environment variables
load_dotenv()

class RouterPipeline(QAPipelineBase):
    def __init__(self):
        logging.info("Initializing RouterPipeline")

    def classify_user_question(self, user_question, agent_type="llama"):
        """Classifies the user's question and returns the classification."""
        logging.info("Generating prompt for classification.")
        prompt = router_question(user_question)
        logging.info(f"Sending prompt to {agent_type} model for classification.")
        response = self.generate_response(agent_type, prompt)
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
        
        # elif classification == "filter":
        #     logging.info("Routing to QAMixPipeline for filtering.")
        #     pipeline = QAMixPipeline()
        #     return pipeline.answer_question(question, query_type="filtering", agent_type=agent_type)
        
        # elif classification == "direct":
        #     logging.info("Routing to QAFaissPipeline for direct answer.")
        #     pipeline = QAFaissPipeline(faiss_agent=self.faiss_agent)  # Pass existing FaissAgent instance
        #     answer = pipeline.answer_question(question, agent_type=agent_type)
        #     logging.info("Received answer from QAFaissPipeline.")
        #     return answer
        
        # else:
        #     logging.error("Invalid classification for question.")
        #     return None
