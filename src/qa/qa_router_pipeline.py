import logging
from dotenv import load_dotenv
from qa.router.task_router import router_question, post_processing_router
from qa.qa_pipeline_base import QAPipelineBase
from qa.qa_sql_pipeline import QASQLPipeline
from qa.qa_mix_pipeline import QAMixPipeline
from qa.qa_faiss_pipeline import QAFaissPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class RouterPipeline(QAPipelineBase):
    def classify_user_question(self, user_question, agent_type="llama"):
        """
        Classifies the user's question using the specified agent type and returns the classification.
        
        Parameters:
        - user_question (str): The user's question to classify.
        - agent_type (str): The model to use for generating the response (options: "llama", "cohere", "gemini").
        
        Returns:
        - classification (str): "aggregate," "filter," or "direct" based on the question's classification.
        """
        logging.info("Generating prompt for classification.")
        prompt = router_question(user_question)

        logging.info(f"Sending prompt to {agent_type} model for classification.")
        response = self.generate_response(agent_type, prompt)

        if response:
            logging.info("Received response from model, processing classification.")
            classification = post_processing_router(response)
            logging.info(f"Classification result: {classification}")
            return classification
        else:
            logging.error("Failed to get a response from the model")
            return None

    def route_question(self, question, agent_type):
        """
        Routes the question to the appropriate pipeline based on the classification.
        
        Parameters:
        - question (str): The user's question to route.
        - agent_type (str): The model to use for generating the response (options: "llama", "cohere", "gemini").
        
        Returns:
        - answer (str): The answer generated by the appropriate pipeline.
        """
        logging.info(f"Classifying question: '{question}'")
        classification = self.classify_user_question(question, agent_type)

        if classification == "aggregate":
            logging.info("Routing to QASQLPipeline for aggregation.")
            pipeline = QASQLPipeline()
            answer = pipeline.answer_question(question, query_type="aggregating", agent_type=agent_type)
            logging.info("Received answer from QASQLPipeline.")
            return answer

        elif classification == "filter":
            logging.info("Routing to QAMixPipeline for filtering.")
            pipeline = QAMixPipeline()
            answer = pipeline.answer_question(question, query_type="filtering", agent_type=agent_type)
            logging.info("Received answer from QAMixPipeline.")
            return answer

        elif classification == "direct":
            logging.info("Routing to QAFaissPipeline for direct answer.")
            pipeline = QAFaissPipeline()
            answer = pipeline.answer_question(question, agent_type=agent_type)
            logging.info("Received answer from QAFaissPipeline.")
            return answer

        else:
            logging.error("Invalid classification for question.")
            return None

