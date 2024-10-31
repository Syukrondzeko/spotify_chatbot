class AgentBase:
    def __init__(self, api_key):
        self.api_key = api_key

    def build_aggregate_query(self, user_question):
        return f"""
        Question: {user_question}

        Table: 'user_review'
        Columns: id, review_id, pseudo_author_id, author_name, review_text, review_rating, year, month, day, sentiment
        Example values: 1, 14a011a8-7544-47b4-8480-c502af0ac26f, 152618553977019693742, "Use it every day", 5, 2014, 5, 27, negative

        Instructions: Build a query based on user question. Identify the relevant and only essential columns to retrieve and apply only essential filters. Then, provide directly the one best SQL Lite query in backticks to answer the question. Use a simple query, avoiding any complex structures, and don't use any table or columns outside those mentioned above. Don't add limit if it is not mentioned in the question.
        """
    
    def build_filter_query(self, user_question):
        return f"""
        Question: {user_question}

        Table: 'user_review'
        Columns: id, review_id, pseudo_author_id, author_name, review_text, review_rating, year, month, day, sentiment
        Example values: 1, 14a011a8-7544-47b4-8480-c502af0ac26f, 152618553977019693742, "Use it every day", 5, 2014, 5, 27, negative

        Instructions: Fill below query using correct filter based on user question. Provide that one best SQL Lite query in backticks. Use a simple query, avoiding any complex structures, and don't use any table or columns outside those mentioned above. Don't add limit if it is not mentioned in the question.
        Fill this query:
        ```
        SELECT id FROM user_review
        WHERE [Apply Your Filter Here]

        Don't you ever change SELECT id FROM user_review, just edit the where clause to fit the user question
        """

    def build_relax_query(self, user_question, previous_query):
        return f"""
        Question: {user_question}
        Previous query: {previous_query}

        Instructions: It produced an empty result. Edit your query to make the filter more relaxed or remove unnecessary filters, but still fulfill my question. Don't add limit if it is not mentioned in the question.
        """

    def build_fixed_error_query_prompt(self, user_question, query, error):
        """Builds a prompt to correct errors in the generated SQL query."""
        return f"""
        I have a question: {user_question}
        And you provided this query based on the question: {query}
        However, I encountered this error: {error}

        Please fix the query to address the error and provide the correct version. Don't add limit if it is not mentioned in the question.
        """

    def get_query(self, user_question, query_type):
        """Abstract method to retrieve SQL query based on a user question."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_relax_query(self, user_question, previous_query):
        """Abstract method to retrieve relaxed SQL query based on a user question."""
        raise NotImplementedError("Subclasses should implement this method.")
