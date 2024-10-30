import os
import sqlite3
import pandas as pd
import re
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def remove_emojis(text):
    """Remove emojis and special characters from text."""
    return re.sub(r'[^\w\s,]', '', text)

def preprocess_text(df):
    """Apply uniform preprocessing: remove emojis, filter text length, and extract date components."""
    logging.info("Starting text preprocessing")

    # Rename 'Unnamed: 0' to 'id' if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'id'})
    else:
        # Create unique IDs if 'Unnamed: 0' column does not exist
        df = df.reset_index().rename(columns={"index": "id"})

    # Drop rows with missing 'review_text' values
    df = df.dropna(subset=['review_text'])

    # Clean 'review_text' and overwrite it with cleaned content
    df['review_text'] = df['review_text'].apply(remove_emojis)

    # Filter for rows with at least 10 words in 'review_text' after cleaning
    df = df[df['review_text'].apply(lambda x: len(x.split()) >= 10)].reset_index(drop=True)

    # Convert 'review_timestamp' to datetime and extract year, month, and day
    df['review_timestamp'] = pd.to_datetime(df['review_timestamp'], errors='coerce')
    df['year'] = df['review_timestamp'].dt.year
    df['month'] = df['review_timestamp'].dt.month
    df['day'] = df['review_timestamp'].dt.day

    # Drop unnecessary columns
    df = df.drop(columns=['author_app_version', 'review_timestamp'])

    logging.info("Text preprocessing completed")
    return df

# Paths to database and CSV file
database_path = os.getenv("SQLITE_PATH")
csv_file_path = os.getenv("DATASET_PATH")

# Load the CSV data
df = pd.read_csv(csv_file_path)
logging.info("CSV data loaded successfully")

# Preprocess the data
df_filtered = preprocess_text(df)

# Connect to the SQLite database
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Create the user_review table with cleaned text under review_text and date components
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_review (
        id INTEGER PRIMARY KEY,
        pseudo_author_id TEXT,
        review_id TEXT,
        author_name TEXT,
        review_text TEXT,
        review_rating INTEGER,
        review_likes INTEGER,
        year INTEGER,
        month INTEGER,
        day INTEGER
    )
''')

# Insert preprocessed data from DataFrame to the SQLite table
df_filtered.to_sql('user_review', conn, if_exists='replace', index=False)

# Commit changes and close the connection
conn.commit()
conn.close()

logging.info("Filtered and preprocessed data successfully inserted into the SQLite database.")
