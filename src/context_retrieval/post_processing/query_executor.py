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
    If an error occurs, returns the error message as a string.
    """
    if not query:
        return "No SQL query provided."

    conn = sqlite3.connect(database_path)
    
    try:
        df_results = pd.read_sql_query(query, conn)
        return df_results
    except Exception as e:
        error_message = f"Error executing query: {e}"
        return error_message  # Return the error message instead of an empty DataFrame
    finally:
        conn.close()
