import argparse
from lib.semantic_search import embed_query_text, verify_model, embed_text, verify_embeddings

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Commands")
    subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for the given text")
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")

    subparsers.add_parser("verify_embeddings", help="Verify that embeddings can be generated for a sample text")

    query_parser = subparsers.add_parser("embed_query", help="Generate an embedding for the given query text")
    query_parser.add_argument("query", type=str, help="Query text to generate embedding for")

    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Verifying the semantic search model...")
            verify_model()

        case "embed_text":
            embed_text(args.text)
        
        case "verify_embeddings":
            print("Verifying that embeddings can be generated for a sample text...")
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
