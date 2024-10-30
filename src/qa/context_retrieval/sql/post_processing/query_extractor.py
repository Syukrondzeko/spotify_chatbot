import re

def extract_query(response_text):
    # Match SQL query inside ```...``` blocks and remove `sql` or `sqlite` tag if present
    match = re.search(r"```(?:sql|sqlite)?\s+(SELECT|WITH)[\s\S]*?```", response_text, re.IGNORECASE)
    if match:
        # Remove opening ``` and optional tag, and closing ```
        sql_query = match.group(0)
        sql_query = re.sub(r"```(?:sql|sqlite)?\s+", "", sql_query)  # Remove opening ``` and any tags
        sql_query = sql_query.replace("```", "").strip()  # Remove closing ```
        return sql_query

    return None
