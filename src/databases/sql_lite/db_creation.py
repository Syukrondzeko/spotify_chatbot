import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database and CSV file paths from .env file
database_path = os.getenv("SQLITE_PATH")
csv_file_path = os.getenv("DATASET_PATH")

# Load the CSV data
df = pd.read_csv(csv_file_path)

# Filter DataFrame to only include rows with at least 10 words in review_text
df['word_count'] = df['review_text'].apply(lambda x: len(str(x).split()))
df_filtered = df[df['word_count'] >= 10]

# Drop the author_app_version column
df_filtered = df_filtered.drop(columns=['author_app_version'])

# Convert review_timestamp to datetime and extract year, month, and day
df_filtered['review_timestamp'] = pd.to_datetime(df_filtered['review_timestamp'])
df_filtered['year'] = df_filtered['review_timestamp'].dt.year
df_filtered['month'] = df_filtered['review_timestamp'].dt.month
df_filtered['day'] = df_filtered['review_timestamp'].dt.day

# Drop the original review_timestamp column
df_filtered = df_filtered.drop(columns=['review_timestamp'])

# Connect to the SQLite database
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Create the user_review table with year, month, and day columns
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_review (
        pseudo_author_id TEXT,
        author_name TEXT,
        review_text TEXT,
        review_rating INTEGER,
        review_likes INTEGER,
        year INTEGER,
        month INTEGER,
        day INTEGER
    )
''')

# Insert filtered data from DataFrame to the SQLite table
df_filtered.to_sql('user_review', conn, if_exists='replace', index=False)

# Commit changes and close the connection
conn.commit()
conn.close()

print("Filtered data successfully inserted into the SQLite database.")
