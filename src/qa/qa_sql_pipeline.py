import logging
import pandas as pd
from qa.qa_pipeline_base import QAPipelineBase
from qa.context_retrieval.retrieval_pipeline import retrieve_and_execute_pipeline

class QASQLPipeline(QAPipelineBase):
    def retrieve_context(self, user_question: str, agent_type: str):
        """Retrieve context using SQL-based retrieval."""
        return retrieve_and_execute_pipeline(user_question, agent_type)

    def answer_question(self, user_question: str, agent_type: str = "cohere") -> str:
        """Retrieves SQL context and generates a response."""
        logging.info("Retrieving context for the question using SQL.")
        context = self.retrieve_context(user_question, agent_type)

        if context is None or (isinstance(context, pd.DataFrame) and context.empty):
            logging.warning("No context found.")
            return None

        # Format context based on type: DataFrame to string if SQL-based
        context_text = context.to_string(index=False) if isinstance(context, pd.DataFrame) else str(context)
        logging.info("SQL context retrieved and formatted.")

        prompt = f"Using the following context:\nContext: {context_text}\nAnswer the question:\nQuestion: {user_question}"
        logging.info("Prompt generated:\n%s", prompt)

        response = self.generate_response(agent_type, prompt)
        return response

# Example usage
if __name__ == "__main__":
    pipeline = QASQLPipeline()
    question = "How many comments for each month in 2014?"
    agent = "cohere"  # Options: "cohere", "llama", "gemini"
    answer = pipeline.answer_question(question, agent_type=agent)
    print("Answer:", answer if answer else "No answer generated.")
