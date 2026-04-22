import json
import os
import sqlite3
import string
import time
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import redis
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

DB_PATH = os.getenv("DB_PATH", "db/search.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "3600"))
TOP_K = int(os.getenv("SEARCH_TOP_K", "10"))
MAX_PAGE_SIZE = int(os.getenv("SEARCH_MAX_PAGE_SIZE", "20"))

try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
except redis.exceptions.RedisError:
    redis_client = None

_embedding_cache: Dict[str, object] = {
    "signature": None,
    "page_ids": np.array([], dtype=np.int64),
    "urls": [],
    "titles": [],
    "contents": [],
    "matrix": np.empty((0, 0), dtype=np.float32),
}


def get_db_connection():
    """Establishes a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@lru_cache(maxsize=1)
def get_stop_words() -> set:
    """Caches the stop-word set to avoid repeated NLTK lookups."""
    return set(stopwords.words("english"))


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Loads the embedding model once per process."""
    return SentenceTransformer(MODEL_NAME)


def normalize_query(query: str) -> str:
    """Trims repeated whitespace so cache keys stay stable."""
    return " ".join(query.strip().split())


def tokenize(text: str) -> List[str]:
    """Lowercases and tokenizes text while removing stop words and punctuation."""
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [
        token
        for token in word_tokenize(cleaned)
        if token.isalpha() and token not in get_stop_words()
    ]


def build_snippet(content: str, query: str, max_length: int = 220) -> str:
    """Builds a short excerpt around the first likely match in the document."""
    normalized_content = " ".join((content or "").split())
    if not normalized_content:
        return ""

    query_terms = [term for term in tokenize(query) if term]
    lowered_content = normalized_content.lower()

    match_index = -1
    for term in query_terms:
        match_index = lowered_content.find(term.lower())
        if match_index != -1:
            break

    if match_index == -1:
        snippet = normalized_content[:max_length].strip()
        return f"{snippet}..." if len(normalized_content) > max_length else snippet

    start = max(0, match_index - max_length // 3)
    end = min(len(normalized_content), start + max_length)
    if end - start < max_length and start > 0:
        start = max(0, end - max_length)

    snippet = normalized_content[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(normalized_content):
        snippet = f"{snippet}..."
    return snippet


def get_cache(cache_key: str) -> Optional[List[Dict]]:
    """Returns cached search results when Redis is available."""
    if not redis_client:
        return None
    try:
        cached_results = redis_client.get(cache_key)
        return json.loads(cached_results) if cached_results else None
    except (redis.exceptions.RedisError, json.JSONDecodeError):
        return None


def set_cache(cache_key: str, results: List[Dict]) -> None:
    """Stores search results in Redis without failing the request path."""
    if not redis_client:
        return
    try:
        redis_client.set(cache_key, json.dumps(results), ex=CACHE_TTL_SECONDS)
    except redis.exceptions.RedisError:
        return


def paginate_results(results: List[Dict], page: int, page_size: int) -> Dict:
    """Returns a stable page slice and pagination metadata."""
    safe_page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    safe_page = max(1, page)
    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size
    return {
        "results": results[start:end],
        "total_results": len(results),
        "page": safe_page,
        "page_size": safe_page_size,
        "total_pages": max(1, (len(results) + safe_page_size - 1) // safe_page_size),
    }


def get_embedding_signature(conn: sqlite3.Connection) -> Optional[str]:
    """Builds a lightweight signature so the in-memory embedding cache refreshes when needed."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*), COALESCE(MAX(page_id), 0)
        FROM embeddings
        """
    )
    count, max_page_id = cursor.fetchone()
    if not count:
        return None
    return f"{count}:{max_page_id}"


def load_embedding_index() -> Dict[str, object]:
    """Loads semantic search vectors into memory and reuses them across queries."""
    conn = get_db_connection()
    try:
        signature = get_embedding_signature(conn)
        if signature and signature == _embedding_cache["signature"]:
            return _embedding_cache

        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT e.page_id, e.embedding, p.url, p.title, p.content
            FROM embeddings e
            JOIN pages p ON e.page_id = p.id
            ORDER BY e.page_id
            """
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        _embedding_cache.update(
            {
                "signature": None,
                "page_ids": np.array([], dtype=np.int64),
                "urls": [],
                "titles": [],
                "contents": [],
                "matrix": np.empty((0, 0), dtype=np.float32),
            }
        )
        return _embedding_cache

    page_ids, embedding_blobs, urls, titles, contents = zip(*rows)
    matrix = np.vstack(
        [np.frombuffer(blob, dtype=np.float32) for blob in embedding_blobs]
    ).astype(np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    _embedding_cache.update(
        {
            "signature": signature,
            "page_ids": np.array(page_ids, dtype=np.int64),
            "urls": list(urls),
            "titles": list(titles),
            "contents": list(contents),
            "matrix": matrix,
        }
    )
    return _embedding_cache


def search_keyword_all(query: str) -> List[Dict]:
    """Builds the full lexical ranking list for a query."""
    tokens = tokenize(query)
    if not tokens:
        return []

    placeholders = ",".join(["?"] * len(tokens))
    query_sql = f"""
        SELECT p.url, p.title, p.content, SUM(ii.bm25_score) AS score
        FROM inverted_index ii
        JOIN pages p ON ii.page_id = p.id
        WHERE ii.term IN ({placeholders})
        GROUP BY p.id, p.url, p.title, p.content
        ORDER BY score DESC
    """

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query_sql, tokens)
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {
            "url": row[0],
            "title": row[1],
            "score": float(row[3]),
            "mode": "keyword",
            "snippet": build_snippet(row[2], query),
        }
        for row in rows
    ]


def search_semantic_all(query: str) -> List[Dict]:
    """Builds the full semantic ranking list for a query."""
    embedding_index = load_embedding_index()
    matrix = embedding_index["matrix"]
    if matrix.size == 0:
        return []

    query_embedding = get_model().encode(query)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []

    normalized_query_embedding = query_embedding / query_norm
    similarities = matrix @ normalized_query_embedding
    top_indices = np.argsort(similarities)[::-1]
    return [
        {
            "url": embedding_index["urls"][index],
            "title": embedding_index["titles"][index],
            "score": float(similarities[index]),
            "mode": "semantic",
            "snippet": build_snippet(embedding_index["contents"][index], query),
        }
        for index in top_indices
    ]


def search_keyword(query: str, page: int = 1, page_size: int = TOP_K) -> Dict:
    """Performs a keyword search using the BM25 inverted index."""
    start_time = time.time()
    normalized_query = normalize_query(query)
    cache_key = f"kw:{normalized_query}"

    cached_results = get_cache(cache_key)
    if cached_results is not None:
        paginated = paginate_results(cached_results, page, page_size)
        return {
            "results": paginated["results"],
            "response_time_ms": (time.time() - start_time) * 1000,
            "cached": True,
            "total_results": paginated["total_results"],
            "page": paginated["page"],
            "page_size": paginated["page_size"],
            "total_pages": paginated["total_pages"],
        }

    results = search_keyword_all(normalized_query)
    if not results:
        return {
            "results": [],
            "response_time_ms": (time.time() - start_time) * 1000,
            "cached": False,
            "total_results": 0,
            "page": max(1, page),
            "page_size": max(1, min(page_size, MAX_PAGE_SIZE)),
            "total_pages": 1,
        }

    set_cache(cache_key, results)
    paginated = paginate_results(results, page, page_size)

    return {
        "results": paginated["results"],
        "response_time_ms": (time.time() - start_time) * 1000,
        "cached": False,
        "total_results": paginated["total_results"],
        "page": paginated["page"],
        "page_size": paginated["page_size"],
        "total_pages": paginated["total_pages"],
    }


def search_semantic(query: str, page: int = 1, page_size: int = TOP_K) -> Dict:
    """Performs semantic search against the in-memory embedding index."""
    start_time = time.time()
    normalized_query = normalize_query(query)
    cache_key = f"sem:{normalized_query}"

    cached_results = get_cache(cache_key)
    if cached_results is not None:
        paginated = paginate_results(cached_results, page, page_size)
        return {
            "results": paginated["results"],
            "response_time_ms": (time.time() - start_time) * 1000,
            "cached": True,
            "total_results": paginated["total_results"],
            "page": paginated["page"],
            "page_size": paginated["page_size"],
            "total_pages": paginated["total_pages"],
        }

    results = search_semantic_all(normalized_query)
    if not results:
        return {
            "results": [],
            "response_time_ms": (time.time() - start_time) * 1000,
            "cached": False,
            "total_results": 0,
            "page": max(1, page),
            "page_size": max(1, min(page_size, MAX_PAGE_SIZE)),
            "total_pages": 1,
        }

    set_cache(cache_key, results)
    paginated = paginate_results(results, page, page_size)

    return {
        "results": paginated["results"],
        "response_time_ms": (time.time() - start_time) * 1000,
        "cached": False,
        "total_results": paginated["total_results"],
        "page": paginated["page"],
        "page_size": paginated["page_size"],
        "total_pages": paginated["total_pages"],
    }


def normalize_scores(results: List[Dict]) -> List[Dict]:
    """Normalizes scores in a list of results to a [0, 1] range."""
    if not results:
        return []

    normalized_results = [dict(result) for result in results]
    scores = [result["score"] for result in normalized_results]
    min_score, max_score = min(scores), max(scores)

    if max_score == min_score:
        for result in normalized_results:
            result["score"] = 1.0
        return normalized_results

    for result in normalized_results:
        result["score"] = (result["score"] - min_score) / (max_score - min_score)
    return normalized_results


def search_hybrid(query: str, page: int = 1, page_size: int = TOP_K) -> Dict:
    """Performs a weighted hybrid search across lexical and semantic results."""
    start_time = time.time()
    normalized_query = normalize_query(query)
    cache_key = f"hy:{normalized_query}"

    cached_results = get_cache(cache_key)
    if cached_results is not None:
        paginated = paginate_results(cached_results, page, page_size)
        return {
            "results": paginated["results"],
            "response_time_ms": (time.time() - start_time) * 1000,
            "cached": True,
            "total_results": paginated["total_results"],
            "page": paginated["page"],
            "page_size": paginated["page_size"],
            "total_pages": paginated["total_pages"],
        }

    keyword_results = get_cache(f"kw:{normalized_query}")
    if keyword_results is None:
        keyword_results = search_keyword_all(normalized_query)
        set_cache(f"kw:{normalized_query}", keyword_results)

    semantic_results = get_cache(f"sem:{normalized_query}")
    if semantic_results is None:
        semantic_results = search_semantic_all(normalized_query)
        set_cache(f"sem:{normalized_query}", semantic_results)

    keyword_results = normalize_scores(keyword_results)
    semantic_results = normalize_scores(semantic_results)

    combined_results: Dict[str, Dict[str, object]] = {}
    for result in keyword_results:
        combined_results[result["url"]] = {
            "title": result["title"],
            "bm25_score": result["score"],
            "semantic_score": 0.0,
            "snippet": result.get("snippet", ""),
        }

    for result in semantic_results:
        if result["url"] not in combined_results:
            combined_results[result["url"]] = {
                "title": result["title"],
                "bm25_score": 0.0,
                "semantic_score": result["score"],
                "snippet": result.get("snippet", ""),
            }
            continue
        combined_results[result["url"]]["semantic_score"] = result["score"]
        if not combined_results[result["url"]]["snippet"]:
            combined_results[result["url"]]["snippet"] = result.get("snippet", "")

    final_results = []
    for url, scores in combined_results.items():
        hybrid_score = (0.4 * float(scores["bm25_score"])) + (
            0.6 * float(scores["semantic_score"])
        )
        final_results.append(
            {
                "url": url,
                "title": str(scores["title"]),
                "score": hybrid_score,
                "mode": "hybrid",
                "snippet": str(scores.get("snippet", "")),
            }
        )

    final_results.sort(key=lambda result: result["score"], reverse=True)
    set_cache(cache_key, final_results)
    paginated = paginate_results(final_results, page, page_size)

    return {
        "results": paginated["results"],
        "response_time_ms": (time.time() - start_time) * 1000,
        "cached": False,
        "total_results": paginated["total_results"],
        "page": paginated["page"],
        "page_size": paginated["page_size"],
        "total_pages": paginated["total_pages"],
    }
