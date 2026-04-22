-- Pages table
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    content TEXT
);

-- BM25 inverted index
CREATE TABLE IF NOT EXISTS inverted_index (
    term TEXT NOT NULL,
    page_id INTEGER NOT NULL,
    bm25_score REAL NOT NULL,
    FOREIGN KEY (page_id) REFERENCES pages(id)
);
CREATE INDEX IF NOT EXISTS idx_term ON inverted_index(term);

-- Embeddings table (for semantic search)
CREATE TABLE IF NOT EXISTS embeddings (
    page_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (page_id) REFERENCES pages(id)
);

-- Stats table
CREATE TABLE IF NOT EXISTS stats (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
