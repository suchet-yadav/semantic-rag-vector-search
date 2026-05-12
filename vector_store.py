import faiss
import numpy as np
from typing import List, Tuple


class VectorStore:
    """
    FAISS-based vector storage and retrieval.
    """

    def __init__(self, embedding_dimension: int):
        self.dimension = embedding_dimension
        self.index = faiss.IndexFlatIP(embedding_dimension)
        self.documents = []

    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[str]
    ) -> None:
        """
        Add embeddings and corresponding documents to FAISS index.
        """

        self.index.add(embeddings.astype("float32"))
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Perform similarity search.
        """

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            top_k
        )

        results = []

        for idx, score in zip(indices[0], scores[0]):
            results.append(
                (self.documents[idx], float(score))
            )

        return results