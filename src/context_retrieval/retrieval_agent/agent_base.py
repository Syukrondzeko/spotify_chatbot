class AgentBase:
    def __init__(self, api_key):
        self.api_key = api_key

    def build_query(self, user_question):
        return f"""
        Question: {user_question}

        Table: 'user_review'
        Columns: review_id, pseudo_author_id, author_name, review_text, review_rating, review_likes, year, month, date

        Instructions: Identify the relevant only essential columns to retrieve and apply only essential filters. Then, provide directly the one best SQL Lite query in backticks. Use a simple query, avoiding any complex structures, don't use any table or columns outside that I mention above.
        """

    def build_relax_query(self, user_question, previous_query):
        return f"""
        Question: {user_question}
        Previous query: {previous_query}

        Instructions: It produced an empty result. Edit your query to make the filter more relaxed or remove unnecessary filters, but still fulfill my question.
        """

    def get_query(self, user_question):
        """Abstract method to retrieve SQL query based on a user question."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_relax_query(self, user_question, previous_query):
        """Abstract method to retrieve relaxed SQL query based on a user question."""
        raise NotImplementedError("Subclasses should implement this method.")
