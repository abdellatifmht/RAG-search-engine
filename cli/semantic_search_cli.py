import argparse
from lib.semantic_search import (
    chunk_command,
    embed_chunks_command,
    embed_query_text,
    semantic_chunk_command,
    search_documents,
    verify_model,
    embed_text,
    verify_embeddings,
    search_chunked_command,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    subparsers.add_parser("verify", help="Verify the semantic search model")

    # Add a new subparser for embedding text
    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for the given text")
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")

    # Add a new subparser for verifying embeddings
    subparsers.add_parser("verify_embeddings", help="Verify that embeddings can be generated for a sample text")

    # Add a new subparser for embedding query text
    query_parser = subparsers.add_parser("embed_query", help="Generate an embedding for the given query text")
    query_parser.add_argument("query", type=str, help="Query text to generate embedding for")

    # Add a new subparser for searching documents
    search_parser = subparsers.add_parser("search", help="Search for documents similar to the query")
    search_parser.add_argument("query", type=str, help="Query text to search for similar documents")
    search_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 10)")

    # Add a new subparser for chunking documents
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a document into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Document text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk (default: 200 characters)")
    chunk_parser.add_argument("--overlap", type=int, default=40, help="Overlap between chunks (default: 40 characters)")

    # Add a new subparser for semantic chunking
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a document semantically into smaller pieces")
    semantic_chunk_parser.add_argument("text", type=str, help="Document text to chunk semantically")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of sentences in each chunk (default: 4)")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks in number of sentences (default: 0)")

    # Add a new subparser for embedding all chunks in the database
    subparsers.add_parser("embed_chunks", help="Generate embeddings for all chunks in the database")

    # Add a new subparser for searching chunked documents
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for chunked documents similar to the query")
    search_chunked_parser.add_argument("query", type=str, help="Query text to search for similar chunked documents")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 5)")

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

        case "search":
            try:
                search_documents(args.query, args.limit)
            except Exception as e:
                print(f"Error occurred while searching: {e}")

        case "chunk":
            try:
                chunk_command(args.text, args.chunk_size, args.overlap)
            except Exception as e:
                print(f"Error occurred while chunking: {e}")

        case "semantic_chunk":
            try:
                semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
            except Exception as e:
                print(f"Error occurred while semantic chunking: {e}")

        case "embed_chunks":
            print("Embedding chunks...")
            embed_chunks_command()

        case "search_chunked":
            search_chunked_command(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
