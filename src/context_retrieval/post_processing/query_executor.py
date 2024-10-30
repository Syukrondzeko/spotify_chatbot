import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the database path from the .env file
database_path = os.getenv("DATABASE_PATH")

def run_query(query):
    """
    Connects to the SQLite database, executes the given query, and returns the results in a DataFrame.
    """
    if not query:
        raise ValueError("No SQL query provided.")

    conn = sqlite3.connect(database_path)
    
    try:
        df_results = pd.read_sql_query(query, conn)
    except Exception as e:
        print("Error executing query:", e)
        df_results = pd.DataFrame()  # Return an empty DataFrame on error
    finally:
        conn.close()

    return df_results
