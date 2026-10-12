import argparse
from lib.hybrid_search import normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize text")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            print("Normalizing scores...")

            normalized_scores = normalize_scores(args.scores)
            for score in normalized_scores:
                print(score)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
