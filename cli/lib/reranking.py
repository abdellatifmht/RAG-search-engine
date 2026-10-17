import os
import json
from time import sleep
from typing import Literal, NotRequired
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from openai import OpenAI
from search_utils import SearchResult


class RerankedSearchResult(SearchResult, total=False):
    individual_score: NotRequired[int]


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"


def llm_rerank_individual(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[RerankedSearchResult]:
    scored_docs: list[RerankedSearchResult] = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        score_text = (response.choices[0].message.content or "").strip()
        try:
            score = int(score_text)
        except ValueError:
            score = 0
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]

def llm_rerank_batch(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[RerankedSearchResult]:
    if not documents:
        return []

    # Prepare compact movie data for the LLM
    movies = [
        {
            "id": doc["id"],
            "title": doc.get("title", ""),
            "document": doc.get("document", ""),
        }
        for doc in documents
    ]

    prompt = f"""Rank the movies listed below by relevance to the search query.

                Query: "{query}"

                Consider:
                - Direct relevance to the query
                - User intent and preferences
                - Tone, style, and genre when relevant
                - Content appropriateness

                Movies:
                {json.dumps(movies, ensure_ascii=False)}

                Return ONLY a raw JSON array of movie IDs, ordered from most relevant
                to least relevant. Include each movie ID at most once.
                Use only IDs from the provided movies.

                Example: [75, 12, 34, 2, 1]

                Ranking:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = (response.choices[0].message.content or "").strip()

    try:
        ranked_ids = json.loads(response_text)
        if not isinstance(ranked_ids, list):
            raise ValueError("Expected a JSON array")
    except (json.JSONDecodeError, ValueError):
        # If the model returns invalid JSON, preserve the original ranking
        return [
            {**doc, "batch_rank": rank}
            for rank, doc in enumerate(documents[:limit], start=1)
        ]

    # Map IDs to their original documents
    docs_by_id = {doc["id"]: doc for doc in documents}
    ranked_docs = []
    seen_ids = set()

    for movie_id in ranked_ids:
        # Ignore unknown IDs and duplicates
        if movie_id not in docs_by_id or movie_id in seen_ids:
            continue

        seen_ids.add(movie_id)
        ranked_docs.append({
            **docs_by_id[movie_id],
            "batch_rank": len(ranked_docs) + 1,
        })

    # Append any movies omitted by the LLM, preserving their original order
    for doc in documents:
        if doc["id"] not in seen_ids:
            ranked_docs.append({
                **doc,
                "batch_rank": len(ranked_docs) + 1,
            })

    return ranked_docs[:limit]

def cross_encoder_rerank(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[RerankedSearchResult]:
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    scores = cross_encoder.predict(pairs)

    scored_docs = [
        {**doc, "cross_encoder_score": score}
        for doc, score in zip(documents, scores)
    ]

    sorted_docs = sorted(scored_docs, key=lambda x: x["cross_encoder_score"], reverse=True)
    
    return sorted_docs[:limit]


def rerank(
    query: str,
    documents: list[SearchResult],
    method: Literal["individual", "batch", "cross_encoder"],
    limit: int = 5,
) -> list[SearchResult] | list[RerankedSearchResult]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    elif method == "batch":
        return llm_rerank_batch(query, documents, limit)
    elif method == "cross_encoder":
        return cross_encoder_rerank(query, documents, limit)

    else:
        return documents[:limit]
