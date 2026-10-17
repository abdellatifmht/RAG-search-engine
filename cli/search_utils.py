import os
import json
from typing import Any, TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str

class SearchResult(TypedDict):
    id: int
    title: str
    description: str
    score: float
    metadata: dict[str, Any]

class GoldenTestCase(TypedDict):
    query: str
    relevant_docs: list[str]

class GoldenDataset(TypedDict):
    test_cases: list[GoldenTestCase]

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
DEFAULT_CHUNK_SIZE = 200
DOCUMENT_PREVIEW_LENGTH = 100
OVERLEAP_SIZE = 40

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "description": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


def load_golden_dataset() -> GoldenDataset:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        data = json.load(f)
    return data
