import argparse
from lib.hybrid_search import normalize_scores, weighted_hybrid_search_command, rrf_hybrid_search_command
from query_enhacement import enhance_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    # Add subparsers for each command
    normalize_parser = subparser.add_parser("normalize", help="Normalize text")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    # Add subparser for weighted hybrid search
    weighted_search_parser = subparser.add_parser("weighted-search", help="Perform a weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Query text to search for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 score in the hybrid search (default: 0.5)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 5)")

    # Add subparser for RRF search
    rrf_search_parser = subparser.add_parser("rrf-search", help="Perform a Reciprocal Rank Fusion (RRF) search")
    rrf_search_parser.add_argument("query", type=str, help="Query text to search for")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="K value for RRF (default: 60)")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 5)")
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            print("Normalizing scores...")

            normalized_scores = normalize_scores(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            weighted_hybrid_search_command(args.query, args.alpha, args.limit)

        case "rrf-search":
            if args.enhance:
                enhanced_query = enhance_query(args.query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")

            rrf_hybrid_search_command(enhanced_query, args.k, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
