import re
import logging


def router_question(user_question):
    """
    Creates the prompt to classify the question as either "aggregate," "filter," or "direct."
    """
    prompt = f"""
    User question: "{user_question}"

    Classify the question as one of the following:
    - "`aggregate`" if it requires counting or averaging number and begin with "how many" or similar question
    - "`filter`" if it needs filtering by sentiment or date
    - "`direct`" if it doesn’t need filtering or aggregation

    Give brief explain max 50 words and give final answer inside backtick with only: `aggregate`, `filter`, or `direct`
    Format
    Brief Explanation: ...
    Final Answer: `...`
    """
    return prompt


def post_processing_router(response):
    """
    Processes the response from the language model to find the last occurrence of "aggregate," "filter," or "direct"
    within backticks and returns it. If not found, defaults to "direct."
    """
    # Search for the last occurrence of "aggregate", "filter", or "direct" within backticks at the end of the response
    match = re.search(r'`(aggregate|filter|direct)`\s*$', response.lower())

    if match:
        return match.group(1)  # Returns only "aggregate", "filter", or "direct" without any extra characters
    
    # If no valid classification term is found, log an error and default to 'direct'
    logging.error("Unexpected response format: defaulting to 'direct'")
    return "direct"  # Default to 'direct' if response is unexpected
