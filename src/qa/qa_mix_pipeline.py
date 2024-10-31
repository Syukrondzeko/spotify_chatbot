import logging
import pandas as pd
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from qa.qa_pipeline_base import QAPipelineBase
from qa.context_retrieval.retrieval_pipeline import retrieve_and_execute_pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
METADATA_FAISS_PATH = os.getenv("METADATA_FAISS_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")

class QAMixPipeline(QAPipelineBase):
    def __init__(self):
        # Load FAISS index
        self.faiss_index = faiss.read_index(FAISS_PATH)
        logging.info("FAISS index loaded successfully.")
        self.model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        
        # Load metadata JSON file and create a mapping from FAISS index to metadata entry
        with open(METADATA_FAISS_PATH, 'r') as f:
            metadata = json.load(f)
        self.metadata_by_id = {entry['id']: entry for entry in metadata}
        logging.info("Metadata loaded successfully.")

    def retrieve_context(self, user_question: str, query_type: str, agent_type: str):
        """Retrieve context using SQL-based retrieval."""
        if query_type != "filtering":
            raise ValueError("Invalid query_type. Only 'filtering' is accepted in QAMixPipeline.")
        
        return retrieve_and_execute_pipeline(user_question, query_type, agent_type)

    def answer_question(self, user_question: str, query_type: str, agent_type: str = "cohere"):
        """Retrieves SQL context, filters FAISS embeddings, and performs similarity search."""
        logging.info("Retrieving context for the question using SQL.")
        context = self.retrieve_context(user_question, query_type, agent_type)

        if context is None or (isinstance(context, pd.DataFrame) and context.empty):
            logging.warning("No context found.")
            print("No context found.")
            return

        # Embed the user question
        question_embedding = self.model.encode(user_question).astype('float32').reshape(1, -1)
        faiss.normalize_L2(question_embedding)

        # Filter FAISS indices to only include those in `context` and retrieve relevant embeddings
        context_ids = set(context['id'])
        filtered_embeddings = []
        metadata_map = []

        for faiss_index in range(self.faiss_index.ntotal):
            metadata_entry = self.metadata_by_id.get(faiss_index)
            if metadata_entry and metadata_entry['id'] in context_ids:
                embedding_vector = np.array(metadata_entry['embedding']).astype('float32')
                filtered_embeddings.append(embedding_vector)
                metadata_map.append(metadata_entry)

        if not filtered_embeddings:
            print("No embeddings found for the retrieved IDs.")
            return

        # Stack embeddings and compute similarity
        filtered_embeddings = np.vstack(filtered_embeddings)
        faiss.normalize_L2(filtered_embeddings)
        similarities = np.dot(filtered_embeddings, question_embedding.T).flatten()

        # Sort by similarity and get top 5 results
        top_indices = np.argsort(similarities)[-5:][::-1]
        
        print(f"Total reviews retrieved: {len(context)}")
        print("Top 5 most similar reviews:")
        for rank, idx in enumerate(top_indices, start=1):
            similarity_score = similarities[idx]
            metadata_entry = metadata_map[idx]
            text = metadata_entry.get("text", "Text not found")
            review_id = metadata_entry.get("id", "ID not found")
            print(f"Rank {rank}: Text: {text}, ID: {review_id}, Similarity Score: {similarity_score:.4f}")


# Example usage
if __name__ == "__main__":
    pipeline = QAMixPipeline()
    question = "What features that has many bad review? limit max 5 rows"
    query_type = "filtering"
    agent = "cohere"  # Options: "cohere", "llama", "gemini"
    pipeline.answer_question(question, query_type, agent)
