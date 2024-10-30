import re

def extract_query(response_text):
    # Match SQL query within ```...``` block, starting with SELECT or WITH and ending with ;
    match = re.search(r"```(?:sql)?\s*`?(SELECT|WITH)[\s\S]*?;?\s*```", response_text, re.IGNORECASE)
    if match:
        # Return only the SQL query portion, excluding any `sql` or backticks
        sql_query = match.group(0)
        sql_query = re.sub(r"```(?:sql)?\s*`?", "", sql_query)  # Remove starting ```sql if present
        sql_query = sql_query.replace("```", "").replace("`", "").strip()  # Remove trailing ``` and backticks
        return sql_query

    # Fallback: Capture any standalone SQL starting with SELECT or WITH and ending with ;
    match_alt = re.search(r"(?<!`)(SELECT|WITH)[\s\S]*?;", response_text, re.IGNORECASE)
    if match_alt:
        return match_alt.group(0).strip()
    
    return None
