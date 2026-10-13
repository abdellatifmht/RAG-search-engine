import argparse
from search_utils import BM25_B, BM25_K1
from lib.keyword_search import (
    search_command,
    build_command,
    get_term_frequency,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25_search_command,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build the inverted index from the movie dataset")

    # Add subparsers for tf command
    tf_parser = subparsers.add_parser("tf", help="Display term frequencies for a given term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")


    # Add subparsers for idf command
    idf_parser = subparsers.add_parser("idf", help="Display inverse document frequencies for a given term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    # Add subparsers for search command
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    # Add subparsers for tfidf command
    tfidf_parser = subparsers.add_parser("tfidf", help="Display TF-IDF scores for a given term in a specific document") 
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF for")

    # Add subparsers for bm25_idf command
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    # Add subparsers for bm25_tf command
    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
    "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    # Add subparsers for bm25search command
    bm25search_parser = subparsers.add_parser(
    "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results")

    args = parser.parse_args()

    match args.command:
        case "search":
            
            print('Searching for:', args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['id']}: {res['title']}")
        case "build":
            print('Building the inverted index...')
            build_command()
            print('Inverted index built and saved successfully.')

        case "tf":
            try:
                term_frequency = get_term_frequency(args.doc_id, args.term)
                print(f"Term frequency of '{args.term}' in document {args.doc_id}: {term_frequency}")
            except ValueError as e:
                print(f"Error: {e}")
        
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            try:
                tf_idf = tfidf_command(args.doc_id, args.term)
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            except ValueError as e:
                print(f"Error: {e}")

        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except ValueError as e:
                print(f"Error: {e}")

        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except ValueError as e:
                print(f"Error: {e}")

        case "bm25search":
            print('Searching for:', args.query)
            results = bm25_search_command(args.query, limit=args.limit)

            for i, doc in enumerate(results, start=1):
                print(f"{i}. ({doc['id']}) {doc['title']} - (Score: {doc['score']:.2f})")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
