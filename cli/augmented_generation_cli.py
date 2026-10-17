import argparse
from lib.augmented_generation import rag_command, summarize_command, question_command, citations_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize a given text"
    )
    summarize_parser.add_argument("query", type=str, help="Query text to summarize")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 5)")

    citation_parser = subparsers.add_parser(
        "citations", help="Generate citations for a given text"
    )
    citation_parser.add_argument("query", type=str, help="Text to generate citations for")
    citation_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 5)")

    question_parser = subparsers.add_parser(
        "question", help="Ask a question based on the retrieved documents"
    )
    question_parser.add_argument("question", type=str, help="Question to ask based on the retrieved documents")
    question_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("RAG Response:")
            print(result["answer"])

        case "summarize":
            result = summarize_command(args.query, limit=args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Summary:")
            print(result["summary"])

        case "citations":
            result = citations_command(args.query, limit=args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Answer:")
            print(result["answer"])

        case "question":
            result = question_command(args.question, limit=args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()

            print("Answer:")
            print(result["answer"])

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
