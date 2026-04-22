# SearchX: Hybrid Semantic Search Engine

SearchX is a full-stack search system that crawls a documentation corpus, builds lexical and semantic indexes, and serves ranked results through a FastAPI API with a polished React frontend.

It is designed to feel stronger than a toy app in three ways:
- the data pipeline is separated into crawl, index, and serve stages
- the backend exposes search, health, and runtime stats endpoints
- the UI is a React app that behaves like a small search product dashboard instead of a bare form

## What It Does

- Crawls a target website asynchronously and stores normalized page content in SQLite
- Builds a BM25-style inverted index for exact-term retrieval
- Generates sentence embeddings for semantic retrieval
- Fuses lexical and semantic scores into a hybrid ranking mode
- Caches ranked responses in Redis to reduce repeated-query latency
- Surfaces search health, cache hit rate, and indexed corpus stats in the frontend

## System Design

```text
                  ┌──────────────────────────┐
                  │    React Frontend UI     │
                  │ search, telemetry, UX    │
                  └────────────┬─────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │   FastAPI API    │
                     │ /search /stats   │
                     │ /health          │
                     └───────┬──────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
     ┌────────────┐  ┌──────────────┐  ┌──────────────┐
     │   Redis    │  │    SQLite    │  │ Embedding    │
     │ cache      │  │ pages/index  │  │ model        │
     └────────────┘  └──────────────┘  └──────────────┘
                             ▲
                             │
                     ┌──────────────┐
                     │ crawler +    │
                     │ indexer jobs │
                     └──────────────┘
```

## Tech Stack

| Layer | Choices |
|---|---|
| API | FastAPI, Uvicorn |
| Search | BM25-style lexical ranking, sentence-transformers, NumPy |
| Data | SQLite for pages and indexes, Redis for cache |
| Crawling | `httpx`, `asyncio`, BeautifulSoup |
| Frontend | React, Vite, modular CSS dashboard |
| Tooling | Docker, Docker Compose, Pytest, Locust |

## Repository Layout

```text
api/              FastAPI application
crawler/          Async crawler pipeline
indexer/          BM25 + embedding indexing
search/           Query-time retrieval logic
frontend/         React frontend application
db/               SQLite schema and generated database
tests/            Unit and API tests
```

## Local Development

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt
```

### 2. Crawl content

By default the crawler targets the Python documentation corpus, but you can point it at any website you want to index.

```bash
python crawler/crawler.py
```

To crawl a different site:

```bash
python crawler/crawler.py https://docs.python.org/3/ --max-pages 300
```

### 3. Build indexes

```bash
python indexer/indexer.py
```

This stage creates:
- `inverted_index` rows for lexical search
- `embeddings` rows for semantic search
- `stats` rows used by the UI and operational endpoints

### 4. Run the API

```bash
uvicorn api.main:app --reload
```

The API is available at:
- `http://localhost:8000/docs`
- `http://localhost:8000/search?q=decorators&mode=hybrid`
- `http://localhost:8000/stats`
- `http://localhost:8000/health`

### 5. Run the React frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite frontend runs at:
- `http://localhost:5173`

Routing behavior:
- on localhost, the app talks to `http://localhost:8000`
- in production, the app talks to `/api` and uses a Vercel rewrite

## Docker

```bash
docker-compose up --build
```

The container setup runs:
- FastAPI on port `8000`
- Redis on port `6379`

The SQLite database is mounted from `./db` so you can rebuild indexes locally and keep serving them from the container.

## Deployment

### Railway

Use Railway for the FastAPI service and attach a Redis instance.

Relevant files:
- [Dockerfile](Dockerfile)
- [railway.toml](railway.toml)

### Vercel

Deploy the React frontend and point `/api/*` to your Railway backend with [vercel.json](vercel.json) or [frontend/vercel.json](frontend/vercel.json).

Current rewrite shape:

```json
{
  "rewrites": [
    { "source": "/api/(.*)", "destination": "https://your-railway-app-url.up.railway.app/$1" }
  ]
}
```

Replace the placeholder URL before deploying.

If you deploy from the repo root, [vercel.json](vercel.json) builds `frontend/` and serves `frontend/dist`.
If you set `frontend/` as the Vercel project root, [frontend/vercel.json](frontend/vercel.json) contains the same rewrite.

## API Contract

### `GET /search`

Query params:
- `q`: search text
- `mode`: `keyword`, `semantic`, or `hybrid`

Example:

```bash
curl "http://localhost:8000/search?q=decorators&mode=hybrid"
```

Response shape:

```json
{
  "query": "decorators",
  "mode": "hybrid",
  "results": [
    {
      "url": "https://docs.python.org/3/glossary.html#term-decorator",
      "title": "decorator - Python documentation",
      "score": 0.98,
      "mode": "hybrid"
    }
  ],
  "response_time_ms": 27.14,
  "cached": false,
  "total_results": 10
}
```

### `GET /stats`

Returns indexed page count, term count, embedding count, and cache hit metrics.

### `GET /health`

Returns service, Redis, and database status.

## Engineering Notes

This project now intentionally aims for "production-style" rather than claiming true large-scale production readiness.

Current strengths:
- clear separation between ingestion, indexing, retrieval, and serving
- hybrid search with cacheable query paths
- semantic vectors loaded into process memory instead of pulled from SQLite on every query
- React frontend that exposes operational signals rather than hiding them
- tests plus a Locust file for performance exploration

Current limits:
- SQLite is still the index store, so corpus scale is intentionally bounded
- semantic search is process-local, not distributed
- ranking quality is heuristic, not ML-evaluated
- observability is lightweight and developer-focused rather than enterprise-grade

## How To Take It Further

The next serious upgrades would be:

1. Replace SQLite embeddings with `pgvector`, Qdrant, Weaviate, or FAISS-backed retrieval.
2. Add offline relevance evaluation using labeled queries and metrics such as MRR or nDCG.
3. Precompute richer document fields like summaries, anchors, and section-level embeddings.
4. Add structured logging, tracing, and request-level metrics.
5. Introduce background jobs for scheduled recrawls and reindexing.
6. Add authentication, rate limiting, and safer public deployment defaults.
