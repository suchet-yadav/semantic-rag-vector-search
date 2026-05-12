class MockGenerativeModel:
    """
    Mock implementation simulating a generative AI query expansion model.
    """

    def __init__(self):
        self.expansion_map = {
            "How does the system handle peak load?":
                "traffic spikes autoscaling load balancing "
                "infrastructure scaling concurrent requests",

            "How is user data protected?":
                "authentication encryption TLS JWT OAuth "
                "secure communication credentials",

            "How does semantic retrieval work?":
                "vector embeddings semantic similarity "
                "nearest neighbor vector database retrieval"
        }

    def expand_query(self, query: str) -> str:
        """
        Expand user query using semantic enrichment.
        """

        return self.expansion_map.get(query, query)
