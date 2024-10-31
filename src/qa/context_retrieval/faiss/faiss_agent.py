# src/qa/context_retrieval/faiss/faiss_agent.py

import faiss
import numpy as np
import logging


class FaissAgent:
    def search_similar_sentences(self, user_question, model, index, metadata, top_k=5, nprobe=10):
        """Perform similarity search on the provided FAISS index using the query embedding."""
        # Encode the question to create a query embedding
        logging.info(f"Encoding the question: '{user_question}'")
        query_embedding = model.encode(user_question)

        # Set nprobe for partition search
        index.nprobe = nprobe

        # Normalize query embedding for cosine similarity
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Perform similarity search
        logging.info(f"Performing similarity search with nprobe={nprobe}")
        D, I = index.search(query_embedding, k=top_k)
        
        # Retrieve closest sentences from metadata
        closest_sentences = [metadata[idx]['text'] for idx in I[0]]
        return closest_sentences
