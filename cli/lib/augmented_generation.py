import os
from search_utils import load_movies
from .hybrid_search import HybridSearch
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"


def augmented_generation_command(query: str) -> None:
    """ Performs retrieval augmented generation (RAG) by searching for relevant documents and generating an answer. """
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(query)

    print("Search results for:")
    for result in search_results:
        print(f"- {result['title']}")

    # Generate an answer using the search results
    docs = []
    for i, result in enumerate(search_results, 1):
        docs.append(f"{i}. {result['title']}: {result['description']}")

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {docs}

            Answer:"""
    
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    answer = (response.choices[0].message.content or "").strip().strip('"')

    if answer:
        print("\nRAG Answer:")
        print(answer)
