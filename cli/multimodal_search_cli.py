import argparse
from lib.multimodal_search import image_search_command, verify_image_embedding

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparser.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_parser.add_argument("image_path", type=str, help="Path to the image file")

    image_search_parser = subparser.add_parser("image_search", help="Perform image search")
    image_search_parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            print(f"Verifying image embedding for: {args.image_path}")
            verify_image_embedding(args.image_path)

        case "image_search":
            results = image_search_command(args.image_path)

            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} (similarity: {result['similarity']})")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
