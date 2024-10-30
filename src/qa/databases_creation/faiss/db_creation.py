import json
import pandas as pd
import numpy as np
import faiss
import logging
import os
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_VECTOR_PATH = os.getenv("EMBEDDING_VECTOR_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
METADATA_FAISS_PATH = os.getenv("METADATA_FAISS_PATH")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def load_all_embeddings(directory_path):
    logging.info("Loading all embeddings from directory")
    embeddings = []

    # Iterate over each file in the directory
    for file_name in os.listdir(directory_path):
        if file_name.startswith("embeddings_batch_") and file_name.endswith(".json"):
            file_path = os.path.join(directory_path, file_name)
            logging.info(f"Loading embeddings from file: {file_name}")
            
            # Load embeddings from the file
            with open(file_path, 'r') as f:
                for line in f:
                    batch = json.loads(line)
                    embeddings.extend(batch)
                    
    # Convert to DataFrame
    df = pd.DataFrame(embeddings)
    logging.info("All embeddings loaded into DataFrame")
    return df

def create_faiss_index_and_save_metadata(df, index_path, metadata_path):
    # Prepare embeddings and normalize them
    logging.info("Preparing embeddings and normalizing for FAISS")
    embeddings = np.array(df['embedding'].tolist()).astype('float32')
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

    # Create and populate FAISS index
    logging.info("Creating FAISS index")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    logging.info(f"FAISS index saved at {index_path}")
    
    # Save metadata (text and ID) for later retrieval
    metadata = df[['id', 'text']].to_dict(orient='records')
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    logging.info(f"Metadata saved at {metadata_path}")


# Load data, create index, and save metadata
df = load_all_embeddings(EMBEDDING_VECTOR_PATH)
create_faiss_index_and_save_metadata(df, FAISS_PATH, METADATA_FAISS_PATH)
logging.info("FAISS index and metadata saved successfully.")
