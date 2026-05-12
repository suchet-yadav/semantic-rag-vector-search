from typing import List, Dict

from embedding import EmbeddingModel
from vector_store import VectorStore
from query_expander import MockGenerativeModel


class SemanticRetriever:
    """
    Core retrieval engine supporting:
    - Strategy A: Raw Vector Search
    - Strategy B: AI-Enhanced Query Expansion
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        query_expander: MockGenerativeModel
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.query_expander = query_expander

    def retrieve_raw(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:

        query_embedding = self.embedding_model.encode([query])

        results = self.vector_store.search(
            query_embedding,
            top_k
        )

        return [
            {
                "document": doc,
                "score": round(score, 4)
            }
            for doc, score in results
        ]

    def retrieve_enhanced(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict:

        expanded_query = self.query_expander.expand_query(query)

        query_embedding = self.embedding_model.encode(
            [expanded_query]
        )

        results = self.vector_store.search(
            query_embedding,
            top_k
        )

        return {
            "expanded_query": expanded_query,
            "results": [
                {
                    "document": doc,
                    "score": round(score, 4)
                }
                for doc, score in results
            ]
        }