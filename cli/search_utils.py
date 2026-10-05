import os
import json
from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]
