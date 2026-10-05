from nltk.stem import PorterStemmer
from search_utils import DEFAULT_SEARCH_LIMIT,PROJECT_ROOT, load_movies
import string
import os



def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """
    Search for movies using keywords.

    Args:
        query (str): The search query.
        limit (int): The maximum number of results to return.

    Returns:
        list[dict]: A list of matching movies.
    """
    movies = load_movies()
    results = []
    stop_words = load_stop_words()

    query_tokens = tokenize_text(query)
    query_tokens = remove_stop_words(query_tokens, stop_words)

    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        title_tokens = remove_stop_words(title_tokens, stop_words)
        if has_matching_token(title_tokens, query_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by removing punctuation, converting to lowercase, and splitting into words.
    Args:
        text (str): The input text to preprocess.
    Returns:
        list[str]: A list of preprocessed words.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize the input text into words.
    Args:
        text (str): The input text to tokenize.
    Returns:
        list[str]: A list of tokenized words.
    """
    stemmer = PorterStemmer()
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(stemmer.stem(token))
    return valid_tokens


def has_matching_token(title_tokens: list[str], query_tokens: list[str]) -> bool:
    """
    Check if the title contains any of the keywords from the query.
    Args:
        title_tokens (list[str]): The tokenized movie title.
        query_tokens (list[str]): The tokenized search query.
    Returns:
        bool: True if there is a match, False otherwise.
    """
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def load_stop_words() -> list[str]:
    """Load stop words from a file and preprocess them.
    Returns:
        list[str]: A list of preprocessed stop words.
    """
    stop_words_path = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
    with open(stop_words_path, "r") as f:
        stop_words = f.read().splitlines()
    return [preprocess_text(word) for word in stop_words]


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> list[str]:
    """
    Remove stop words from the list of tokens.
    Args:
        tokens (list[str]): The list of tokens to filter.
        stop_words (list[str]): The list of stop words to remove.
    Returns:
        list[str]: A list of tokens with stop words removed.
    """
    return [token for token in tokens if token not in stop_words]
