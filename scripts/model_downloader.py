from sentence_transformers import SentenceTransformer
import os

# Define the model path
model_name = "all-MiniLM-L6-v2"
model_path = os.path.join("models", model_name)

# Create directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Download and save the model to the specified directory
model = SentenceTransformer(model_name)
model.save(model_path)

print(f"Model '{model_name}' downloaded and saved to '{model_path}'.")
