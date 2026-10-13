import os
from typing import TypedDict

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from search_utils import load_movies


class WeightedSearchResult(TypedDict):
    id: int
    title: str
    description: str
    score: float
    bm25_score: float
    semantic_score: float
    hybrid_score: float


class CombinedScoreData(TypedDict):
    title: str
    bm25_score: float
    description: str
    semantic_score: float


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(
        self, query: str, limit: int
    ) -> list[dict]:
        self.idx.load()
        results = self.idx.bm25_search(query, limit)

        return results

    def weighted_search(
        self, query: str, alpha: float = 0.5, limit: int = 5
    ) -> list[WeightedSearchResult]:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha doit être compris entre 0 et 1.")

        if limit <= 0:
            return []

        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * 500
        )

        combined = combine_search_results(
            bm25_results,
            semantic_results,
            alpha,
        )

        return combined[:limit]

    def rrf_search(
        self, query: str, k: int, limit: int = 10
    ) -> list[WeightedSearchResult]:
        raise NotImplementedError(
            "RRF hybrid search is not implemented yet."
        )


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalise les scores dans l'intervalle [0, 1]."""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def normalize_search_results(
    results: list[dict],
) -> list[tuple[dict, float]]:
    """Associe chaque résultat à son score normalisé."""
    normalized_scores = normalize_scores(
        [result["score"] for result in results]
    )

    return list(zip(results, normalized_scores))


def hybrid_score(
    bm25_score: float,
    semantic_score: float,
    alpha: float = 0.5,
) -> float:
    """Combine les scores BM25 et sémantique."""
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[dict],
    semantic_results: list[dict],
    alpha: float = 0.5,
) -> list[WeightedSearchResult]:
    """Fusionne les résultats BM25 et sémantiques par ID de document."""
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores: dict[int, CombinedScoreData] = {}

    # Intègre les résultats BM25.
    for result, normalized_score in bm25_normalized:
        doc_id = result["id"]

        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "description": result["description"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }

        combined_scores[doc_id]["bm25_score"] = max(
            combined_scores[doc_id]["bm25_score"],
            normalized_score,
        )

    # Intègre les résultats sémantiques.
    for result, normalized_score in semantic_normalized:
        doc_id = result["id"]

        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "description": result["description"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }

        combined_scores[doc_id]["semantic_score"] = max(
            combined_scores[doc_id]["semantic_score"],
            normalized_score,
        )

    hybrid_results: list[WeightedSearchResult] = []

    for doc_id, data in combined_scores.items():
        score = hybrid_score(
            data["bm25_score"],
            data["semantic_score"],
            alpha,
        )

        hybrid_results.append({
            "id": doc_id,
            "title": data["title"],
            "description": data["description"],
            "bm25_score": data["bm25_score"],
            "semantic_score": data["semantic_score"],
            "hybrid_score": score,
        })

    return sorted(
        hybrid_results,
        key=lambda result: result["hybrid_score"],
        reverse=True,
    )


def weighted_hybrid_search_command(
    query: str,
    alpha: float = 0.5,
    limit: int = 5,
) -> None:
    """Effectue une recherche hybride et affiche les résultats."""
    documents = load_movies()
    hybrid_search = HybridSearch(documents)

    results = hybrid_search.weighted_search(query, alpha, limit)

    for i, result in enumerate(results, start=1):
        print(f"{i}. ({result['title']})")
        print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
        print(
            f"   BM25: {result['bm25_score']:.3f}, "
            f"Semantic: {result['semantic_score']:.3f}"
        )
        print(f"   {result['description'][:100]}...\n")
