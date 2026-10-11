import os
import re
import numpy as np
import json
from pydoc import doc
from sentence_transformers import SentenceTransformer
from search_utils import (
    CACHE_PATH,
    DOCUMENT_PREVIEW_LENGTH,
    SearchResult,
    format_search_result,
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    OVERLEAP_SIZE
)
from typing import TypedDict


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class ChunkScore(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float


class SemanticSearch:
    """Class for performing semantic search using sentence transformers."""
    def __init__(self):
        """Initialize the SemanticSearch class with a pre-trained model."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embedding_cache_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")

    def generate_embedding(self, text: str):
        """Generate an embedding for the given text using the model."""
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Build embeddings for a list of documents."""
        self.documents = documents

        contents = []
        for id, doc in enumerate(documents):
            self.document_map[id] = doc
            content = f"{doc['title']}: {doc['description']}"
            contents.append(content)

        self.embeddings = self.model.encode(contents, show_progress_bar=True)
        np.save(self.embedding_cache_path, self.embeddings)
        
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Load embeddings from cache or create them if not available."""
        self.documents = documents

        for id, doc in enumerate(documents):
            self.document_map[id] = doc

        if os.path.exists(self.embedding_cache_path):
            self.embeddings = np.load(self.embedding_cache_path)

            if len(self.embeddings) == len(documents):
                return self.embeddings 

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SemanticSearchResult]:
        """Search for documents similar to the query."""
        if not query.strip():
            raise ValueError("Query text cannot be empty.")

        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        embedding = self.generate_embedding(query)
        similarities_score = np.array([cosine_similarity(embedding, doc_embedding) for doc_embedding in self.embeddings])

        results = [(similarity, self.document_map[i]) for i, similarity in enumerate(similarities_score)]
        results.sort(key=lambda x: x[0], reverse=True)

        return [{
                "score": score,
                "title": doc['title'],
                "description": doc['description']
                }
                for score, doc in results[:limit]
            ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata: list[ChunkMetadata] = None
        self.chunk_embeddings_cache_path = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
        self.chunk_metadata_cache_path = os.path.join(CACHE_PATH, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Build embeddings for chunks of documents."""
        self.documents = documents

        chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []
        for id, doc in enumerate(documents):
            self.document_map[id] = doc
            if not doc['description']:
                continue
            doc_chunks = semantic_chunk_document(doc['description'], max_chunk_size=4, overlap=1)
            chunks.extend(doc_chunks)

            for chunk in doc_chunks:
                chunk_metadata.append({
                    "movie_idx": id,
                    "chunk_idx": chunks.index(chunk),
                    "total_chunks": len(doc_chunks)
                })

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(CACHE_PATH, exist_ok=True)
        np.save(self.chunk_embeddings_cache_path, self.chunk_embeddings)

        with open(self.chunk_metadata_cache_path, 'w') as f:
            json.dump(self.chunk_metadata, f)

        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        """Search for chunks similar to the query."""
        if not query.strip():
            raise ValueError("Query text cannot be empty.")

        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        chunk_scores: list[ChunkScore] = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)

            chun_idx = self.chunk_metadata[i]['chunk_idx']
            movie_idx = self.chunk_metadata[i]['movie_idx']
            chunk_scores.append({
                "chunk_idx": chun_idx,
                "movie_idx": movie_idx,
                "score": score
            })

        movie_scores: dict[int, ChunkScore] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score['movie_idx']
            score = chunk_score['score']

            if movie_idx not in movie_scores or score > movie_scores[movie_idx]['score']:
                movie_scores[movie_idx] = chunk_score

        sorted_movies = sorted(
            movie_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
            )

        if self.documents is None:
            raise ValueError("No documents loaded. Call `load_or_create_chunk_embeddings` first.")
        
        results: list[SearchResult] = []
        for movie_idx, chunk_score in sorted_movies[:limit]:
            movie = self.document_map[movie_idx]

            movie["description"] = movie['description'][:DOCUMENT_PREVIEW_LENGTH]
            movie["score"] = round(chunk_score['score'], 3)

            results.append(
                format_search_result(
                    doc_id=movie_idx,
                    title=movie['title'],
                    document=movie['description'][:DOCUMENT_PREVIEW_LENGTH],
                    score=movie['score'],
                    )
            )
        return results

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Load chunk embeddings from cache or create them if not available."""
        self.documents = documents

        for id, doc in enumerate(documents):
            self.document_map[id] = doc

        if os.path.exists(self.chunk_embeddings_cache_path) and os.path.exists(self.chunk_metadata_cache_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_cache_path)

            with open(self.chunk_metadata_cache_path, 'r') as f:
                self.chunk_metadata = json.load(f)

            if len(self.chunk_embeddings) == len(self.chunk_metadata):
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

                
def verify_model() -> None:
    """Verify that the semantic search model is loaded correctly."""
    semantic_search = SemanticSearch()
    model = semantic_search.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")

def embed_text(text: str) -> None:
    """Generate an embedding for the given text."""
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    """Verify that embeddings can be generated for a sample text."""
    semantic_search = SemanticSearch()

    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query: str) -> None:
    """Generate an embedding for the given query text."""
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_documents(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    """Search for documents similar to the query."""
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)

    results = semantic_search.search(query, limit)

    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc['title']} (Score: {doc['score']:.4f})")
        print(doc['document'] + "..." if len(doc['document']) > DOCUMENT_PREVIEW_LENGTH else doc['document'], end="\n\n")

def chunk_document(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = OVERLEAP_SIZE) -> list[str]:
    """Chunk a document into smaller pieces."""
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be a non-negative integer less than the chunk size.")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]

        if chunk and len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))

    return chunks


def chunk_command(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = OVERLEAP_SIZE) -> None:
    """Chunk a document into smaller pieces and print the results."""
    chunks = chunk_document(text, chunk_size, overlap)

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def semantic_chunk_document(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    """Chunk a document semantically into smaller pieces."""

    if max_chunk_size <= 0:
        raise ValueError("Max chunk size must be a positive integer.")
    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError("Overlap must be a non-negative integer less than the max chunk size.")

    # 1. Strip leading and trailing whitespace
    text = text.strip()

    # 2. Return an empty list if the text is empty
    if not text:
        return []

    # 3. Split the text into sentences using regex
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # 4. Preserve text without terminal punctuation
    if len(sentences) == 1 and not text.endswith(('.', '!', '?')):
        sentences = [text]

    chunks = []
    step = max_chunk_size - overlap

    for start in range(0, len(sentences), step):
        chunk = sentences[start:start + max_chunk_size]

        if chunk and len(chunk) <= overlap:
            break
        sentence_chunk = " ".join(chunk).strip()

        if sentence_chunk:
            chunks.append(sentence_chunk)

    return chunks


def semantic_chunk_command(text: str, max_chunk_size: int = 4, overlap: int = 0) -> None:
    """Chunk a document semantically into smaller pieces and print the results."""
    chunks = semantic_chunk_document(text, max_chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def embed_chunks_command() -> None:
    """Embed chunks of documents and print the results."""
    semantic_search = ChunkedSemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_chunk_embeddings(documents)

    embeddings = semantic_search.chunk_embeddings

    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    """Search for chunked documents similar to the query and print the results."""
    semantic_search = ChunkedSemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_chunk_embeddings(documents)

    results = semantic_search.search_chunks(query, limit)

    for i, movie in enumerate(results, start=1):
        print(f"\n{i}. {movie['title']} (score: {movie['score']:.4f})")
        print(f"   {movie['document'][:DOCUMENT_PREVIEW_LENGTH]}...")
