from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import sqlite3
import threading
import time
from typing import Literal

import redis
from pydantic import BaseModel, Field

from crawler.crawler import AsyncCrawler
from indexer.indexer import index_bm25, index_semantic, load_pages, update_stats
from search.search import search_keyword, search_semantic, search_hybrid

app = FastAPI(
    title="SearchX API",
    description="Hybrid semantic search API with keyword, semantic, and fused ranking modes.",
    version="1.0.0",
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Adds a process time header to the response and logs the request."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    return response

# --- Configuration ---
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
DB_PATH = os.getenv('DB_PATH', './db/search.db')

ingestion_lock = threading.Lock()
ingestion_status = {
    "status": "idle",
    "source_url": None,
    "max_pages": None,
    "pages_crawled": 0,
    "started_at": None,
    "completed_at": None,
    "error": None,
}

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    print("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to Redis: {e}")
    redis_client = None

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

# --- Helper Functions ---
def increment_cache_counter(hit: bool):
    """Increments cache hit or miss counters in Redis."""
    if not redis_client:
        return
    try:
        if hit:
            redis_client.incr('cache:hits')
        else:
            redis_client.incr('cache:misses')
    except redis.exceptions.RedisError:
        pass # Ignore if Redis is down


class IngestRequest(BaseModel):
    source_url: str = Field(..., description="Root URL to crawl and index")
    max_pages: int = Field(default=100, ge=1, le=2000)
    replace_existing: bool = Field(default=True)


def clear_search_data() -> None:
    """Clears indexed data so a new crawl can replace the current corpus."""
    db_conn = get_db_connection()
    if not db_conn:
        raise RuntimeError("Database is unavailable.")
    try:
        cursor = db_conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS pages (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT UNIQUE NOT NULL, title TEXT, content TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS inverted_index (term TEXT NOT NULL, page_id INTEGER NOT NULL, bm25_score REAL NOT NULL, FOREIGN KEY (page_id) REFERENCES pages(id))")
        cursor.execute("CREATE TABLE IF NOT EXISTS embeddings (page_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL, FOREIGN KEY (page_id) REFERENCES pages(id))")
        cursor.execute("CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        cursor.execute("DELETE FROM inverted_index")
        cursor.execute("DELETE FROM embeddings")
        cursor.execute("DELETE FROM pages")
        cursor.execute("DELETE FROM stats")
        db_conn.commit()
    finally:
        db_conn.close()


def write_stat(key: str, value: str) -> None:
    db_conn = get_db_connection()
    if not db_conn:
        raise RuntimeError("Database is unavailable.")
    try:
        cursor = db_conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        cursor.execute("INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)", (key, value))
        db_conn.commit()
    finally:
        db_conn.close()


def clear_runtime_caches() -> None:
    """Clears Redis counters/results after a reindex."""
    if not redis_client:
        return
    try:
        keys = redis_client.keys("kw:*") + redis_client.keys("sem:*") + redis_client.keys("hy:*")
        if keys:
            redis_client.delete(*keys)
        redis_client.delete("cache:hits", "cache:misses")
    except redis.exceptions.RedisError:
        pass


def run_ingestion_job(source_url: str, max_pages: int, replace_existing: bool) -> None:
    """Crawls a site and rebuilds the search corpus in a background thread."""
    with ingestion_lock:
        ingestion_status.update(
            {
                "status": "running",
                "source_url": source_url,
                "max_pages": max_pages,
                "pages_crawled": 0,
                "started_at": int(time.time()),
                "completed_at": None,
                "error": None,
            }
        )

    try:
        if replace_existing:
            clear_search_data()

        crawler = AsyncCrawler(start_url=source_url, max_pages=max_pages)
        import asyncio
        asyncio.run(crawler.crawl())

        pages = load_pages()
        if not pages:
            raise RuntimeError("Crawler completed but no pages were stored.")

        index_bm25(pages)
        index_semantic(pages)
        update_stats()
        write_stat("source_url", source_url)
        write_stat("crawl_mode", "replace" if replace_existing else "append")
        clear_runtime_caches()

        with ingestion_lock:
            ingestion_status.update(
                {
                    "status": "completed",
                    "pages_crawled": len(crawler.crawled_data),
                    "completed_at": int(time.time()),
                    "error": None,
                }
            )
    except Exception as exc:
        with ingestion_lock:
            ingestion_status.update(
                {
                    "status": "failed",
                    "completed_at": int(time.time()),
                    "error": str(exc),
                }
            )


def start_ingestion_job(source_url: str, max_pages: int, replace_existing: bool) -> None:
    with ingestion_lock:
        if ingestion_status["status"] == "running":
            raise RuntimeError("An ingestion job is already running.")
        worker = threading.Thread(
            target=run_ingestion_job,
            args=(source_url, max_pages, replace_existing),
            daemon=True,
        )
        worker.start()

# --- API Endpoints ---
@app.get("/search")
async def search(
    q: str,
    mode: Literal['keyword', 'semantic', 'hybrid'] = 'hybrid',
    page: int = 1,
    page_size: int = 10,
):
    """
    Performs a search based on the query and mode.
    """
    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")

    search_functions = {
        'keyword': search_keyword,
        'semantic': search_semantic,
        'hybrid': search_hybrid,
    }
    
    if mode not in search_functions:
        raise HTTPException(status_code=400, detail="Invalid search mode.")

    search_function = search_functions[mode]
    search_result = search_function(q, page=page, page_size=page_size)
    
    increment_cache_counter(search_result['cached'])

    return {
        "query": q,
        "mode": mode,
        "results": search_result['results'],
        "response_time_ms": search_result['response_time_ms'],
        "cached": search_result['cached'],
        "total_results": search_result['total_results'],
        "page": search_result['page'],
        "page_size": search_result['page_size'],
        "total_pages": search_result['total_pages'],
    }


@app.get("/")
async def root():
    """Returns a lightweight service manifest."""
    return {
        "name": "SearchX API",
        "status": "ok",
        "docs": "/docs",
        "endpoints": ["/search", "/stats", "/health", "/ingest", "/ingest/status"],
    }

@app.get("/stats")
async def get_stats():
    """
    Retrieves and returns search engine statistics.
    """
    db_conn = get_db_connection()
    if not db_conn:
        raise HTTPException(status_code=503, detail="Database is unavailable.")
    
    stats = {}
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT key, value FROM stats")
        rows = cursor.fetchall()
        stats = {row['key']: row['value'] for row in rows}
    except sqlite3.Error:
        pass # Table might not exist yet
    finally:
        db_conn.close()

    cache_hits = 0
    cache_misses = 0
    if redis_client:
        try:
            hits = redis_client.get('cache:hits')
            misses = redis_client.get('cache:misses')
            cache_hits = int(hits) if hits else 0
            cache_misses = int(misses) if misses else 0
        except redis.exceptions.RedisError:
            pass # Redis down

    total_requests = cache_hits + cache_misses
    cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0

    return {
        "total_pages_indexed": int(stats.get('total_pages_indexed', 0)),
        "total_terms": int(stats.get('total_terms', 0)),
        "total_embeddings": int(stats.get('total_embeddings', 0)),
        "cache_hit_rate": f"{cache_hit_rate:.2f}%",
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "source_url": stats.get("source_url"),
        "last_indexed_at": int(stats["last_indexed_at"]) if stats.get("last_indexed_at") else None,
    }

@app.get("/health")
async def health_check():
    """
    Performs a health check on the API and its dependencies.
    """
    redis_status = "unavailable"
    if redis_client:
        try:
            if redis_client.ping():
                redis_status = "ok"
        except redis.exceptions.ConnectionError:
            pass

    db_status = "error"
    db_conn = get_db_connection()
    if db_conn:
        try:
            db_conn.cursor().execute("SELECT 1")
            db_status = "ok"
        except sqlite3.Error:
            pass
        finally:
            db_conn.close()

    return {"status": "ok", "redis": redis_status, "db": db_status}


@app.post("/ingest", status_code=202)
async def ingest_site(payload: IngestRequest):
    """Starts crawling and indexing for a target website."""
    source_url = payload.source_url.strip()
    if not source_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="source_url must start with http:// or https://")

    try:
        start_ingestion_job(source_url, payload.max_pages, payload.replace_existing)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "status": "accepted",
        "source_url": source_url,
        "max_pages": payload.max_pages,
        "replace_existing": payload.replace_existing,
    }


@app.get("/ingest/status")
async def get_ingest_status():
    """Returns crawl/index job status for the UI."""
    with ingestion_lock:
        return dict(ingestion_status)
