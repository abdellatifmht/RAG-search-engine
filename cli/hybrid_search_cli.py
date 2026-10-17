import argparse
from lib.hybrid_search import normalize_scores, weighted_hybrid_search_command, rrf_search_command, rerank


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
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit the number of search results (default: 5)"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method to use",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the search results using llm-eval",
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
            result = rrf_search_command(args.query, args.k, args.enhance, args.rerank_method, args.evaluate, args.limit)

            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )

            if result["rerank_method"]:
                print(f"Re-ranking top {len(result['results'])} results using {result['rerank_method']} method...")

            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}: {res.get('llm_score', 'N/A')}/3")
                if res.get("individual_score") is not None:
                    print(f"   Rerank Score: {res['individual_score']:.3f}/10")

                if res.get("batch_rank"):
                    print(f"   Re-rank Rank: {res['batch_rank']}")

                if res.get("cross_encoder_score") is not None:
                    print(f"   Cross Encoder Score: {res['cross_encoder_score']:.3f}")

                print(f"   RRF Score: {res.get('score', 0):.3f}")
                ranks = []
                metadata = res.get("metadata", {})
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['description'][:100]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
