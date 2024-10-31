import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database path from .env file
database_path = os.getenv("DATABASE_PATH")

# Connect to the SQLite database
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Example query to test the connection
query = '''
SELECT review_text
FROM user_review
WHERE review_text LIKE '%music streaming%'
AND review_rating > 3;
'''
df_results = pd.read_sql_query(query, conn)
conn.close()

print(df_results)