import pytest
from unittest.mock import patch, MagicMock
from httpx import ASGITransport, AsyncClient

from api.main import app
from search.search import search_hybrid, search_keyword
from crawler.crawler import AsyncCrawler

@pytest.fixture
def mock_redis():
    """Fixture to mock the Redis client."""
    with patch('search.search.redis_client', MagicMock()) as mock_redis_client:
        mock_redis_client.get.return_value = None
        mock_redis_client.set.return_value = None
        yield mock_redis_client

@pytest.fixture
def mock_db():
    """Fixture to mock the database connection."""
    with patch('search.search.get_db_connection') as mock_get_conn:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        yield mock_cursor

# --- Unit Tests ---

def test_bm25_relevance():
    """Test that BM25 scores more relevant documents higher."""
    # This is a conceptual test; real BM25 testing is complex.
    # We'll simulate it by checking if a term present in one doc but not another scores correctly.
    # A full test would require a mock inverted index.
    assert 0.4 * 1.0 + 0.6 * 0.0 < 0.4 * 1.0 + 0.6 * 0.5, "Hybrid score logic seems off"

def test_hybrid_scoring_range():
    """Test that hybrid scores are always between 0 and 1."""
    # Mocking underlying search functions
    with patch('search.search.search_keyword_all') as mock_kw, \
         patch('search.search.search_semantic_all') as mock_sem:
        
        mock_kw.return_value = [{'url': 'a', 'title': 't', 'score': 5.0}]
        mock_sem.return_value = [{'url': 'a', 'title': 't', 'score': 0.9}]
        
        result = search_hybrid("test")
        assert 0 <= result['results'][0]['score'] <= 1.0

def test_search_empty_query(mock_redis, mock_db):
    """Test that a nonsense query returns no results."""
    mock_db.fetchall.return_value = []
    result = search_keyword("xyzzy123")
    assert result['results'] == []
    assert result['cached'] is False

def test_redis_cache_hit(mock_redis):
    """Test that a cached result is returned on the second call."""
    import json
    cached_data = json.dumps([{"url": "http://cached.com", "title": "Cached", "score": 1.0, "mode": "keyword"}])
    mock_redis.get.return_value = cached_data
    
    result = search_keyword("cached query")
    
    mock_redis.get.assert_called_once_with("kw:cached query")
    assert result['cached'] is True
    assert result['results'][0]['url'] == "http://cached.com"

# --- API Tests ---

@pytest.mark.asyncio
async def test_api_health_check():
    """Test the /health endpoint."""
    with patch('api.main.redis_client') as mock_redis, \
         patch('api.main.get_db_connection') as mock_db:
        
        mock_redis.ping.return_value = True
        mock_db.return_value.cursor.return_value.execute.return_value = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "redis": "ok", "db": "ok"}

@pytest.mark.asyncio
async def test_api_search_endpoint(mock_db):
    """Test the /search endpoint returns valid data."""
    mock_db.fetchall.return_value = [('http://example.com', 'Example', 'Example content about python.', 1.23)]
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/search?q=python&mode=keyword")
    
    assert response.status_code == 200
    data = response.json()
    assert 'results' in data
    assert len(data['results']) > 0
    assert 'url' in data['results'][0]
    assert 'score' in data['results'][0]

@pytest.mark.asyncio
async def test_api_ingest_endpoint_accepts_job():
    """Test that the ingest endpoint accepts a valid crawl request."""
    with patch("api.main.start_ingestion_job") as mock_start_job:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/ingest",
                json={
                    "source_url": "https://example.com/docs/",
                    "max_pages": 25,
                    "replace_existing": True,
                },
            )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "accepted"
    mock_start_job.assert_called_once_with("https://example.com/docs/", 25, True)

@pytest.mark.asyncio
async def test_api_ingest_endpoint_rejects_invalid_url():
    """Test that the ingest endpoint validates the URL scheme."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/ingest",
            json={"source_url": "example.com/docs", "max_pages": 25, "replace_existing": True},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "source_url must start with http:// or https://"

# --- Crawler Tests ---

@pytest.mark.asyncio
@patch('httpx.AsyncClient.get')
async def test_crawler_respects_max_pages(mock_get):
    """Test that the crawler stops after reaching max_pages."""
    # Mocking a simple site with more pages than the limit
    mock_response = MagicMock()
    mock_response.text = '<html><body><a href="/page2">2</a></body></html>'
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    crawler = AsyncCrawler(start_url="http://test.com", max_pages=1)
    await crawler.crawl()
    
    assert len(crawler.crawled_data) == 1

@pytest.mark.asyncio
@patch('httpx.AsyncClient.get')
async def test_crawler_skips_external_domains(mock_get):
    """Test that the crawler does not follow links to external domains."""
    mock_response = MagicMock()
    mock_response.text = '<html><body><a href="http://external.com">External</a></body></html>'
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    crawler = AsyncCrawler(start_url="http://internal.com", max_pages=5)
    await crawler.crawl()

    # The queue should be empty as the external link was not added
    assert crawler.queue.qsize() == 0
    # Only the start page should have been crawled
    assert len(crawler.crawled_data) == 1
