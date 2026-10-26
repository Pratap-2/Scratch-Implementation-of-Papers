def recall_at_k(retrieved, relevant, k):
    retrieved_top_k = retrieved[:k]

    if not relevant:
        return 0.0

    hits = len(set(retrieved_top_k) & set(relevant))

    return hits / len(relevant)


def precision_at_k(retrieved, relevant, k):
    retrieved_top_k = retrieved[:k]

    if k == 0:
        return 0.0

    hits = len(set(retrieved_top_k) & set(relevant))

    return hits / k


def reciprocal_rank(retrieved, relevant):
    relevant = set(relevant)

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def evaluate_retriever(
    query_results,
    ground_truth,
    k=5
):
    recalls = []
    precisions = []
    reciprocal_ranks = []

    for query_id, retrieved in query_results.items():

        relevant = ground_truth[query_id]

        recalls.append(
            recall_at_k(retrieved, relevant, k)
        )

        precisions.append(
            precision_at_k(retrieved, relevant, k)
        )

        reciprocal_ranks.append(
            reciprocal_rank(retrieved, relevant)
        )

    number_of_queries = len(query_results)

    if number_of_queries == 0:
        raise ValueError("No evaluation queries provided")

    return {
        f"Recall@{k}": sum(recalls) / number_of_queries,
        f"Precision@{k}": sum(precisions) / number_of_queries,
        "MRR": sum(reciprocal_ranks) / number_of_queries
    }


# Example evaluation data.

query_results = {
    "Q1": ["A", "B", "C"],
    "Q2": ["D", "E", "F"],
    "Q3": ["G", "H", "I"]
}

ground_truth = {
    "Q1": ["B"],
    "Q2": ["F", "D"],
    "Q3": ["X"]
}

metrics = evaluate_retriever(
    query_results,
    ground_truth,
    k=3
)

print(metrics)