import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Commands")
    subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for the given text")
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")

    subparsers.add_parser("verify_embeddings", help="Verify that embeddings can be generated for a sample text")

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
