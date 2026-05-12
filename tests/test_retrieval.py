import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../src"
        )
    )
)

from embedding import EmbeddingModel
from vector_store import VectorStore
from query_expander import MockGenerativeModel
from retriever import SemanticRetriever


def test_raw_retrieval():

    documents = [
        "Autoscaling handles traffic spikes.",
        "JWT secures authentication."
    ]

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(documents)

    vector_store = VectorStore(
        embedding_dimension=embeddings.shape[1]
    )

    vector_store.add_documents(
        embeddings,
        documents
    )

    retriever = SemanticRetriever(
        embedding_model,
        vector_store,
        MockGenerativeModel()
    )

    results = retriever.retrieve_raw(
        "How does the system scale?"
    )

    assert len(results) > 0
