from .hybrid_search import HybridSearch
from search_utils import load_movies, load_golden_dataset
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
model = "openrouter/free"

# class LLMSearchResult(SearchResult, total=False):

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Calculate precision at k.

    Args:
        retrieved: List of retrieved document titles.
        relevant: Set of relevant document titles.
        k: Number of top results to consider.

    Returns:
        Precision at k as a float.
    """
    if k <= 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    true_positives = sum(1 for doc in retrieved_at_k if doc in relevant)
    return true_positives / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Calculate recall at k.

    Args:
        retrieved: List of retrieved document titles.
        relevant: Set of relevant document titles.
        k: Number of top results to consider.

    Returns:
        Recall at k as a float.
    """
    if not relevant:
        return 0.0
    retrieved_at_k = retrieved[:k]
    true_positives = sum(1 for doc in retrieved_at_k if doc in relevant)
    return true_positives / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    """Calculate the F1 score given precision and recall.

    Args:
        precision: Precision value.
        recall: Recall value.
    Returns:
        F1 score as a float.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_command(limit: int = 5) -> dict:
    """Evaluate the search system using the golden dataset.

    Args:
        limit: Number of results to evaluate (k for precision@k, recall@k).

    Returns:
        A dictionary containing evaluation results.
    """
    movies = load_movies()
    golden_dataset = load_golden_dataset()
    test_cases = golden_dataset["test_cases"]

    # semantic_search = SemanticSearch()
    # semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    results_by_query = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])

        # Perform the search
        results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_docs = []
        for result in results:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)

            precision = precision_at_k(retrieved_docs, relevant_docs, limit)
            recall = recall_at_k(retrieved_docs, relevant_docs, limit)
            f1_score_value = f1_score(precision, recall)

            results_by_query[query] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score_value,
                "retrieved": retrieved_docs,
                "relevant": list(relevant_docs),
            }

    return {
        "test_cases_count": len(test_cases),
        "results": results_by_query,
        "limit": limit,
    }
