# RAG Search Engine

A command-line retrieval-augmented generation (RAG) search engine for a movie
catalogue. The project combines traditional keyword retrieval with semantic
embeddings, then optionally uses rank fusion, query enhancement, reranking, and
an LLM to produce grounded answers.

The repository is designed as an educational and experimental search project:
each retrieval stage can be run independently and inspected from the terminal.

## Features

- **Keyword search** using an inverted index, TF-IDF, and BM25.
- **Semantic search** using Sentence Transformers and cosine similarity.
- **Chunked semantic search** for more granular document retrieval.
- **Hybrid retrieval** with weighted score fusion or Reciprocal Rank Fusion
  (RRF).
- **Optional query enhancement** for spelling correction, rewriting, and query
  expansion.
- **Optional reranking** using individual, batch, or cross-encoder strategies.
- **RAG generation** through OpenRouter-compatible OpenAI chat completions.
- **Summaries, question answering, and cited answers** grounded in retrieved
  movie descriptions.
- **Multimodal image search** using a CLIP-compatible Sentence Transformer.
- **Search evaluation** with precision@k, recall@k, and F1 score against a
  golden dataset.

## How it works

The main retrieval flow is:

```text
Movie records
    ├── Inverted index ──> BM25 results
    └── Sentence embeddings ──> Semantic results
                                      │
                         Weighted fusion or RRF
                                      │
                         Optional enhancement/reranking
                                      │
                         Retrieved context for RAG
                                      │
                         OpenRouter-generated answer
```

Movie records are read from `data/movies.json`. Embeddings and search indexes
are stored locally in `cache/` when they are generated. The first semantic,
hybrid, or multimodal search may take longer because models and indexes can
need to be downloaded or built.

## Requirements

- Python **3.14 or newer**
- [`uv`](https://docs.astral.sh/uv/) recommended for dependency management
- An OpenRouter API key for RAG, query enhancement, reranking, evaluation, and
  image-description features

The project currently pins or constrains its Python dependencies in
[`pyproject.toml`](./pyproject.toml).

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/abdellatifmht/RAG-search-engine.git
cd RAG-search-engine
```

Create the environment and install dependencies with `uv`:

```bash
uv sync
```

Alternatively, create a virtual environment and install the project
dependencies with your preferred Python package manager.

## Configuration

Create a `.env` file in the project root:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
```

Never commit `.env` or expose the API key publicly. The repository ignores this
file by default.

The LLM features use the OpenAI Python client with the following OpenRouter
configuration:

```text
Base URL: https://openrouter.ai/api/v1
Model: openrouter/free
```

## Data and generated files

The application expects:

```text
data/movies.json
data/stopwords.txt
data/golden_dataset.json
```

The local development dataset and generated search artifacts are ignored by
Git because they can be large or project-specific. For a fresh clone, provide
your own compatible `data/movies.json`, then generate the indexes and
embeddings as needed. A movie record should contain at least:

```json
{
  "id": 1,
  "title": "Example Movie",
  "description": "A movie description used for retrieval."
}
```

Generated files such as `cache/index.pkl`, `cache/movie_embeddings.npy`, and
`cache/chunk_embeddings.npy` can be recreated locally and should not be
committed unless you intentionally want to version a specific dataset snapshot.

## Usage

All commands are run from the repository root. Because the CLI modules import
from `cli/`, invoke them with `PYTHONPATH=cli`:

```bash
export PYTHONPATH=cli
```

With `uv`, the equivalent form is:

```bash
uv run env PYTHONPATH=cli python cli/<command>.py ...
```

### Keyword search

Build the inverted index:

```bash
PYTHONPATH=cli uv run python cli/keyword_search_cli.py build
```

Search with the default keyword search:

```bash
PYTHONPATH=cli uv run python cli/keyword_search_cli.py search "space adventure"
```

Search with BM25 and limit the number of results:

```bash
PYTHONPATH=cli uv run python cli/keyword_search_cli.py bm25search "talking bear comedy" --limit 5
```

Inspect scoring components:

```bash
PYTHONPATH=cli uv run python cli/keyword_search_cli.py tf 1 bear
PYTHONPATH=cli uv run python cli/keyword_search_cli.py idf bear
PYTHONPATH=cli uv run python cli/keyword_search_cli.py tfidf 1 bear
PYTHONPATH=cli uv run python cli/keyword_search_cli.py bm25idf bear
PYTHONPATH=cli uv run python cli/keyword_search_cli.py bm25tf 1 bear
```

### Semantic search

Verify the Sentence Transformer model:

```bash
PYTHONPATH=cli uv run python cli/semantic_search_cli.py verify
```

Search movie descriptions by semantic similarity:

```bash
PYTHONPATH=cli uv run python cli/semantic_search_cli.py search "a magical school for young wizards" --limit 5
```

Chunk a document and generate chunk embeddings:

```bash
PYTHONPATH=cli uv run python cli/semantic_search_cli.py chunk "Your document text"
PYTHONPATH=cli uv run python cli/semantic_search_cli.py semantic_chunk "Your document text"
PYTHONPATH=cli uv run python cli/semantic_search_cli.py embed_chunks
PYTHONPATH=cli uv run python cli/semantic_search_cli.py search_chunked "a dangerous journey" --limit 5
```

### Hybrid search

Weighted hybrid search combines normalized BM25 and semantic scores:

```bash
PYTHONPATH=cli uv run python cli/hybrid_search_cli.py weighted-search \
  "family-friendly animated adventure" --alpha 0.5 --limit 5
```

RRF combines rankings from both retrieval systems:

```bash
PYTHONPATH=cli uv run python cli/hybrid_search_cli.py rrf-search \
  "zombie apocalypse comedy" --limit 5
```

RRF also supports query enhancement, reranking, and optional LLM evaluation:

```bash
PYTHONPATH=cli uv run python cli/hybrid_search_cli.py rrf-search \
  "romantic comdy wedding" \
  --enhance rewrite \
  --rerank-method cross_encoder \
  --evaluate \
  --limit 5
```

### RAG and LLM features

These commands require `OPENROUTER_API_KEY`:

```bash
PYTHONPATH=cli uv run python cli/augmented_generation_cli.py rag \
  "What movies feature a superhero saving the world?"

PYTHONPATH=cli uv run python cli/augmented_generation_cli.py summarize \
  "movies about friendship and bears" --limit 5

PYTHONPATH=cli uv run python cli/augmented_generation_cli.py question \
  "Which movie would you recommend for a family movie night?" --limit 5

PYTHONPATH=cli uv run python cli/augmented_generation_cli.py citations \
  "Which movies involve magic?" --limit 5
```

The RAG pipeline retrieves candidates with hybrid search before sending the
selected movie descriptions to the configured LLM.

### Image search

Image search embeds an image and compares it with movie title/description
embeddings:

```bash
PYTHONPATH=cli uv run python cli/multimodal_search_cli.py image_search \
  data/paddington.jpeg
```

To rewrite a text query using visual information from an image, use the
OpenRouter-backed image description command:

```bash
PYTHONPATH=cli uv run python cli/describe_image_cli.py \
  --image data/paddington.jpeg \
  --query "movies with a character like this"
```

### Evaluation

Run the bundled evaluation cases and calculate precision, recall, and F1:

```bash
PYTHONPATH=cli uv run python cli/evaluation_cli.py --limit 5
```

Evaluation cases are defined in `data/golden_dataset.json`. Update that file
when evaluating a different catalogue.

## Project structure

```text
.
├── cli/
│   ├── *_cli.py                 # Command-line entry points
│   └── lib/
│       ├── keyword_search.py    # Inverted index, TF-IDF, and BM25
│       ├── semantic_search.py   # Dense and chunked semantic retrieval
│       ├── hybrid_search.py     # Weighted fusion and RRF
│       ├── reranking.py         # Result reranking and LLM evaluation
│       ├── query_enhacement.py  # Query enhancement strategies
│       ├── augmented_generation.py
│       └── multimodal_search.py
├── data/                        # Local movie data and evaluation cases
├── cache/                       # Generated indexes and embeddings
├── pyproject.toml               # Project metadata and dependencies
└── uv.lock                     # Locked dependency versions
```

## Limitations

- This is a CLI application; it does not currently provide a web UI or API
  server.
- Search quality depends on the catalogue, chunking parameters, embedding
  models, and the quality of the cached indexes.
- LLM-backed features require network access and may incur OpenRouter costs or
  be subject to provider rate limits.
- Model downloads and embedding generation can require substantial disk space
  and startup time.
- The local data and cache directories are intentionally excluded from version
  control; provide them when publishing or cloning the project.

## Contributing

1. Create a feature branch.
2. Make focused changes and document new commands or configuration.
3. Run the relevant CLI command or evaluation before opening a pull request.
4. Do not commit API keys, `.env`, generated caches, or private datasets.


[![Boot.dev Learn Retrieval Augmented Generation certificate](https://qvault-webapp-dynamic-assets.storage.googleapis.com/certificates/0d23b904-07bc-4e7e-b0e3-ea7077ddbc39.jpeg?v=1792347782)](https://www.boot.dev/certificates/0d23b904-07bc-4e7e-b0e3-ea7077ddbc39)
