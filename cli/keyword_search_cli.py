import argparse
from keyword_search import search_command, build_command, get_term_frequency

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build the inverted index from the movie dataset")

    tf_parser = subparsers.add_parser("tf", help="Display term frequencies for a given term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
