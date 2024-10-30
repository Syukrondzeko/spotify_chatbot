import re

def extract_query(response_text):
    # Remove any enclosing triple backticks and whitespace
    response_text = response_text.strip('`').strip()

    # Use regex to find the SQL query between 'SELECT' and either a semicolon or the end of the text
    match = re.search(r"(SELECT.*?)(;|$)", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + ";"  # Ensure the query ends with a semicolon
    else:
        return None  # Return None if no SQL query is found
