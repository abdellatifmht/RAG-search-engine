import argparse
from lib.semantic_search import verify_model, embed_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    verify_parser = parser.add_subparsers(dest="command", help="Commands")
    verify_parser.add_parser("verify", help="Verify the semantic search model")

    embed_parser = verify_parser.add_parser("embed_text", help="Generate an embedding for the given text")
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")

    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Verifying the semantic search model...")
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
