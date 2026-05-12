import json
from tabulate import tabulate

from embedding import EmbeddingModel
from vector_store import VectorStore
from query_expander import MockGenerativeModel
from retriever import SemanticRetriever


DOCUMENT_PATH = "../data/documents.txt"


def load_documents(path: str):

    with open(path, "r", encoding="utf-8") as file:

        documents = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]

    return documents


def main():

    documents = load_documents(DOCUMENT_PATH)

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(documents)

    vector_store = VectorStore(
        embedding_dimension=embeddings.shape[1]
    )

    vector_store.add_documents(
        embeddings,
        documents
    )

    query_expander = MockGenerativeModel()

    retriever = SemanticRetriever(
        embedding_model,
        vector_store,
        query_expander
    )

    queries = [
        "How does the system handle peak load?",
        "How is user data protected?",
        "How does semantic retrieval work?"
    ]

    benchmark_results = []

    for query in queries:

        raw_results = retriever.retrieve_raw(query)

        enhanced_results = retriever.retrieve_enhanced(query)

        benchmark_results.append({
            "query": query,
            "strategy_a_raw_search": raw_results,
            "strategy_b_enhanced_search": enhanced_results
        })

    print("\n" + "=" * 80)
    print("SEMANTIC RAG BENCHMARK RESULTS")
    print("=" * 80)

    for result in benchmark_results:

        print(f"\nQuery: {result['query']}")

        print("\nStrategy A - Raw Vector Search")

        for idx, item in enumerate(
            result["strategy_a_raw_search"],
            start=1
        ):

            print(
                f"{idx}. Score: {item['score']}"
            )

            print(
                f"   {item['document']}\n"
            )

        print("\nStrategy B - AI Enhanced Retrieval")

        print(
            f"Expanded Query: "
            f"{result['strategy_b_enhanced_search']['expanded_query']}"
        )

        for idx, item in enumerate(
            result["strategy_b_enhanced_search"]["results"],
            start=1
        ):

            print(
                f"{idx}. Score: {item['score']}"
            )

            print(
                f"   {item['document']}\n"
            )

    summary_rows = []

    for result in benchmark_results:

        raw_score = (
            result["strategy_a_raw_search"][0]["score"]
        )

        enhanced_score = (
            result["strategy_b_enhanced_search"]["results"][0]["score"]
        )

        improvement = (
            (enhanced_score - raw_score) / raw_score
        ) * 100

        summary_rows.append([
            result["query"],
            raw_score,
            enhanced_score,
            f"{round(improvement, 2)}%"
        ])

    print("\n")
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(
        tabulate(
            summary_rows,
            headers=[
                "Query",
                "Raw Score",
                "Enhanced Score",
                "Improvement"
            ],
            tablefmt="grid"
        )
    )

    with open(
        "../benchmark_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            benchmark_results,
            file,
            indent=4
        )

    print("\nBenchmark results saved to benchmark_results.json")
    print("\nSemantic RAG benchmark completed successfully.")


if __name__ == "__main__":
    main()
