import os
import numpy as np
from sentence_transformers import SentenceTransformer
from search_utils import CACHE_PATH, load_movies


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
