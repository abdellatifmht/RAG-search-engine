import argparse
from lib.augmented_generation import augmented_generation_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    # summarize_parser = subparsers.add_parser(
    #     "summarize", help="Summarize a given text"
    # )
    # summarize_parser.add_argument("query", type=str, help="Query text to summarize")
    # summarize_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            if not query:
                print("Error: Query cannot be empty.")
                return
            augmented_generation_command(query)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()