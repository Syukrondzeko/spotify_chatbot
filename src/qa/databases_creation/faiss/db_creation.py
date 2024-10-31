import json
import pandas as pd
import numpy as np
import faiss
import logging
import os
from dotenv import load_dotenv

# Load environment variables
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

# Modify this function in your FAISS creation script
def create_partitioned_faiss_index_and_save_metadata(df, index_path, metadata_path, nlist=100):
    # Prepare embeddings and normalize them
    logging.info("Preparing embeddings and normalizing for FAISS")
    embeddings = np.array(df['embedding'].tolist()).astype('float32')
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

    # Define the dimension of embeddings
    d = embeddings.shape[1]
    
    # Create an IVF index with partitioning
    logging.info(f"Creating partitioned FAISS IVF index with {nlist} clusters")
    quantizer = faiss.IndexFlatIP(d)  # Quantizer used to find clusters
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Enable a direct map
    index.make_direct_map()

    # Train the index with a subset of embeddings (required for IVF)
    logging.info("Training FAISS index on embeddings")
    index.train(embeddings)
    logging.info("Index training completed")

    # Add embeddings to the index
    index.add(embeddings)
    logging.info("Embeddings added to the FAISS index")
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    logging.info(f"Partitioned FAISS index with direct map saved at {index_path}")

    # Save metadata (including ID, text, etc.)
    metadata = [
        {
            "id": row["id"],
            "faiss_index": idx,
            "text": row["text"],
            "review_rating": row["review_rating"],
            "year": row["year"],
            "month": row["month"],
            "day": row["day"],
            "embedding": row["embedding"]  # Include the embedding for verification
        }
        for idx, row in df.iterrows()
    ]
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    logging.info(f"Metadata with ID-to-FAISS mapping saved at {metadata_path}")


# Load data, create partitioned index, and save metadata
df = load_all_embeddings(EMBEDDING_VECTOR_PATH)
create_partitioned_faiss_index_and_save_metadata(df, FAISS_PATH, METADATA_FAISS_PATH, nlist=100)
logging.info("Partitioned FAISS index and metadata saved successfully.")
