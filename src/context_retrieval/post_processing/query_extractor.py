import re

def extract_query(response_text):
    # Use regex to find the SQL query between 'SELECT' and the first semicolon
    match = re.search(r"(SELECT.*?;)", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return None  # Return None if no SQL query is found
