from sentence_transformers import SentenceTransformer
from torch import embedding


class SemanticSearch:
    """Class for performing semantic search using sentence transformers."""
    def __init__(self):
        """Initialize the SemanticSearch class with a pre-trained model."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text: str):
        """Generate an embedding for the given text using the model."""
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        return self.model.encode([text])[0]

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
