import os
import base64
import mimetypes
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"

def load_image(image_path: str) -> tuple[bytes, str]:
    """
    Load an image from the specified path.

    Args:
        image_path (str): Path to the image file.
    Returns:
        tuple[bytes, str]: The image data in bytes and its MIME type.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"File is not a valid image: {image_path}")

    with open(image_path, "rb") as f:
        image_data = f.read()

    return image_data, mime_type


def image_description(image_path: str, query: str) -> dict:
    """
    Describe an image based on a query.

    Args:
        image_path (str): Path to the image file.
        query (str): Query text to describe the image.

    Returns:
        dict: Description of the image based on the query.
    """
    img, mime = load_image(image_path)
    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
            - Synthesize visual and textual information
            - Focus on movie-specific details (actors, scenes, style, etc.)
            - Return only the rewritten query, without any additional commentary"""

    messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": system_prompt.strip()},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": query.strip()},
        ],
    }
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response
