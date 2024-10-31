import logging
from dotenv import load_dotenv
from qa.router.task_router import router_question, post_processing_router
from qa.qa_pipeline_base import QAPipelineBase  # Import QAPipelineBase

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
        # Generate the prompt using `router_question`
        prompt = router_question(user_question)

        # Generate the response using the specified agent type
        response = self.generate_response(agent_type, prompt)

        if response:
            # Post-process the response to extract the classification
            classification = post_processing_router(response)
            return classification
        else:
            logging.error("Failed to get a response from the model")
            return None

# Example usage
pipeline = RouterPipeline()
classification = pipeline.classify_user_question("Can you me how many users give negative comment", agent_type="cohere")
print("Classification:", classification)  # Expected output: "aggregate" (based on example context)
