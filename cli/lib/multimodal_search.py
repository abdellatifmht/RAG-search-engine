import numpy as np
import os
from PIL import Image
from sentence_transformers import SentenceTransformer

from search_utils import load_movies


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents: list[dict] = None) -> None:
        """
        Initialize the MultimodalSearch class with a pre-trained model.
        """
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [
            f"{doc['title']} {doc['description']}" for doc in documents
        ]
        self.texts_embeddings = self.model.encode(self.texts, show_progress_bar=True, convert_to_numpy=True)

    def embed_image(self, image_path):
        """
        Embed an image using the model.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: The embedding of the image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)
        image_embedding = self.model.encode([image], convert_to_numpy=True)
        return image_embedding[0]

    def search_with_image(self, image_path: str):
        image = Image.open(image_path)
        image_embedding = self.model.encode([image], convert_to_numpy=True)

        results = []
        for idx, text_embedding in enumerate(self.texts_embeddings):
            doc = self.documents[idx]
            similarity = cosine_similarity(image_embedding[0], text_embedding)

            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "similarity": round(similarity, 1),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]


def verify_image_embedding(image_path: str) -> None:
    """
    Verify that the image embedding is valid.

    Args:
        image_path (str): Path to the image file.
    """

    try:
        search = MultimodalSearch()
        embedding = search.embed_image(image_path)
        print(f"Embedding shape: {embedding.shape[0]} dimensions")
            
    except Exception as e:
        print(f"Error during embedding: {e}")


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)


def image_search_command(image_path: str) -> None:
    """
    Perform an image search and display the results.

    Args:
        image_path (str): Path to the image file.
    """
    documents = load_movies()
    multimodal_search = MultimodalSearch(documents=documents)

    results = multimodal_search.search_with_image(image_path)

    return results
