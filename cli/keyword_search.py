import math

from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from search_utils import DEFAULT_SEARCH_LIMIT,PROJECT_ROOT, CACHE_PATH, load_movies
import string
import pickle
import os


class InvertedIndex:
    """A class representing an inverted index for efficient keyword search."""
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_freq_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")

    def _add_document(self, doc_id: int, text: str):
        """Add a document ID to the inverted index for a given token."""
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id].update(tokens)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        """Retrieve the list of document IDs associated with a given term."""
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        """Get the term frequency of a term in a specific document."""
        return self.term_frequencies[doc_id].get(term, 0)

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            content = f"{movie['title']} {movie['description']}"
            self._add_document(doc_id, content)

    def save(self):
        """Save the inverted index to a pickle file."""
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.term_freq_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)


    @classmethod
    def load(cls):
        """Load the inverted index from a pickle file."""
        index = cls()
        with open(index.index_path, "rb") as f:
            index.index = pickle.load(f)

        with open(index.docmap_path, "rb") as f:
            index.docmap = pickle.load(f)

        with open(index.term_freq_path, "rb") as f:
            index.term_frequencies = pickle.load(f)

        return index


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def get_term_frequency(doc_id: int, term: str) -> int:
    """
    Get the term frequency of a term in a specific document.
    Args:
        doc_id (int): The document ID.
        term (str): The term to get frequency for.
    Returns:
        int: The term frequency of the term in the document.
    """
    inverted_index = InvertedIndex.load()
    term_frequency = inverted_index.term_frequencies
    if not term_frequency:
        print("Term frequencies are empty. Please build the index first.")
        exit(1)
    term = tokenize_term(term)
    return inverted_index.get_tf(doc_id, term)


def tokenize_term(term: str) -> str:
    """
    Tokenize a single term.
    Args:
        term (str): The input term to tokenize.
    Returns:
        str: The tokenized term.
    Raises:
        ValueError: If the input term does not result in exactly one token.
    """
    tokens = tokenize_text(term)

    if len(tokens) != 1:
        raise ValueError(f"Expected a single token, but got {len(tokens)} tokens: {tokens}")

    return tokens[0]

def calculate_idf(token: str, inverted_index: InvertedIndex) -> float:
    """
    Calculate the inverse document frequency (IDF) for a given token.
    Args:
        token (str): The token to calculate IDF for.
        inverted_index (InvertedIndex): The inverted index instance.
    Returns:
        float: The IDF value for the token.
    """
    total_docs = len(inverted_index.docmap)
    doc_freq = len(inverted_index.get_documents(token))
    
    return math.log((total_docs + 1) / (doc_freq + 1))

def idf_command(term: str) -> float:
    """
    Calculate the inverse document frequency (IDF) for a given term.
    Args:
        term (str): The term to calculate IDF for.
    Returns:
        float: The IDF value for the term.
    """
    inverted_index = InvertedIndex.load()
    term = tokenize_term(term)
    return calculate_idf(term, inverted_index)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """
    Search for movies using keywords.

    Args:
        query (str): The search query.
        limit (int): The maximum number of results to return.

    Returns:
        list[dict]: A list of matching movies.
    """
    results = []
    stop_words = load_stop_words()

    inverted_index = InvertedIndex.load()
    index = inverted_index.index
    docmap = inverted_index.docmap

    if not index or not docmap:
        print("Inverted index is empty. Please build the index first.")
        exit(1)

    query_tokens = tokenize_text(query)
    query_tokens = remove_stop_words(query_tokens, stop_words)

    for query_token in query_tokens:
        if query_token in index:
            doc_ids = inverted_index.get_documents(query_token)
            for doc_id in doc_ids:
                if doc_id in docmap:
                    movie = docmap[doc_id]
                    if movie not in results:
                        results.append(movie)
                if len(results) >= limit:
                    break
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


# def has_matching_token(title_tokens: list[str], query_tokens: list[str]) -> bool:
#     """
#     Check if the title contains any of the keywords from the query.
#     Args:
#         title_tokens (list[str]): The tokenized movie title.
#         query_tokens (list[str]): The tokenized search query.
#     Returns:
#         bool: True if there is a match, False otherwise.
#     """
#     for query_token in query_tokens:
#         for title_token in title_tokens:
#             if query_token in title_token:
#                 return True
#     return False


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
