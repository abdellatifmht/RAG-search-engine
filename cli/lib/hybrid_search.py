import os
from typing import Literal, TypedDict

from lib.reranking import rerank, llm_evaluate_results
from lib.query_enhacement import enhance_query

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from search_utils import DEFAULT_SEARCH_LIMIT, SearchResult, format_search_result, load_movies, RRF_K



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

class RRFSearchCommandResult(TypedDict):
    original_query: str
    enhanced_query: str | None
    enhance_method: Literal["spell", "rewrite", "expand"] | None
    rerank_method: Literal["individual"] | None
    query: str
    k: int
    results: list[dict]



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
        self, query: str, k: int = 60, limit: int = 5
    ) -> list[dict]:
        if limit <= 0:
            return []

        if k <= 0:
            raise ValueError("k doit être supérieur à 0.")

        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * 500
        )

        combined = combined_rrf_results(
            bm25_results,
            semantic_results,
            k,
        )
        return combined[:limit]


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


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


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


def combined_rrf_results(
    bm25_results: list[dict],
    semantic_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Fusionne les résultats BM25 et sémantiques avec RRF."""

    ranked_bm25 = rank_scores(bm25_results)
    ranked_semantic = rank_scores(semantic_results)

    combined_scores: dict[int, dict] = {}

    def get_entry(result: dict) -> dict:
        doc_id = result["id"]

        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "description": result["description"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }

        return combined_scores[doc_id]

    # Résultats BM25
    for rank, result in ranked_bm25:
        entry = get_entry(result)

        entry["bm25_rank"] = rank
        entry["rrf_score"] += rrf_score(rank, k)

        entry["bm25_score"] = result["score"]

    # Résultats sémantiques
    for rank, result in ranked_semantic:
        entry = get_entry(result)

        if (
            entry["semantic_rank"] is None
            or rank < entry["semantic_rank"]
        ):
            old_rank = entry["semantic_rank"]

            if old_rank is not None:
                entry["rrf_score"] -= rrf_score(old_rank, k)

            entry["semantic_rank"] = rank
            entry["rrf_score"] += rrf_score(rank, k)

        entry["semantic_score"] = max(
            entry["semantic_score"],
            result["score"],
        )

    sorted_items =  sorted(
        combined_scores.items(),
        key=lambda item: item[1]["rrf_score"],
        reverse=True,
    )

    rrf_results: list[SearchResult] = []
    for doc_id, data in sorted_items:
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["description"],
            score=data["rrf_score"],
            rrf_score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        rrf_results.append(result)

    return rrf_results


def rank_scores(results: list[dict]) -> list[tuple[int, dict]]:
    """Classe les résultats par score et limite le nombre de résultats."""
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return [(rank + 1, result) for rank, result in enumerate(sorted_results, start=1)]


def rrf_search_command(
    query: str,
    k: int = RRF_K,
    enhance: Literal["spell", "rewrite"] | None = None,
    rerank_method: Literal["individual"] | None = None,
    evaluate: bool = False,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> RRFSearchCommandResult:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query

    search_limit = limit
    results = searcher.rrf_search(query, k, search_limit)

    if rerank_method:
        results = rerank(query, results, rerank_method, limit)

    if evaluate:
        results = llm_evaluate_results(query, results)

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "rerank_method": rerank_method,
        "query": query,
        "k": k,
        "results": results,
    }
