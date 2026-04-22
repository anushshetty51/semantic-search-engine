import sqlite3
import time
import string
import os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

DB_PATH = os.getenv('DB_PATH', 'db/search.db')
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def load_pages():
    """
    Loads all pages from the SQLite database.

    Returns:
        list: A list of tuples, where each tuple contains (id, content).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM pages")
    pages = cursor.fetchall()
    conn.close()
    return pages

def tokenize(text):
    """
    Tokenizes text by lowercasing, removing punctuation, and removing stopwords.

    Args:
        text (str): The text to tokenize.

    Returns:
        list: A list of tokens.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.isalpha() and word not in stop_words]

def index_bm25(pages):
    """
    Creates a BM25 index and stores it in the database.

    Args:
        pages (list): A list of page data from the database.
    """
    print("Starting BM25 indexing...")
    start_time = time.time()

    tokenized_corpus = [tokenize(doc) for _, doc in pages]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS inverted_index (term TEXT NOT NULL, page_id INTEGER NOT NULL, bm25_score REAL NOT NULL, FOREIGN KEY (page_id) REFERENCES pages(id))")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_term ON inverted_index(term)")
    cursor.execute("DELETE FROM inverted_index")

    terms = set(term for doc in tokenized_corpus for term in doc)
    
    for term in terms:
        for i, doc in enumerate(tokenized_corpus):
            if term in doc:
                page_id = pages[i][0]
                score = bm25.get_scores([term])[i]
                if score > 0:
                     cursor.execute("INSERT INTO inverted_index (term, page_id, bm25_score) VALUES (?, ?, ?)", (term, page_id, score))

    conn.commit()
    conn.close()
    
    print(f"BM25 indexing completed in {time.time() - start_time:.2f} seconds.")

def index_semantic(pages):
    """
    Generates and stores semantic embeddings for pages.

    Args:
        pages (list): A list of page data from the database.
    """
    print("Starting semantic indexing...")
    start_time = time.time()

    model = SentenceTransformer(MODEL_NAME)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS embeddings (page_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL, FOREIGN KEY (page_id) REFERENCES pages(id))")
    cursor.execute("DELETE FROM embeddings")

    batch_size = 32
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i+batch_size]
        page_ids = [p[0] for p in batch]
        # Truncate content to first 512 tokens
        contents = [" ".join(p[1].split()[:512]) for p in batch]
        
        embeddings = model.encode(contents, show_progress_bar=False)
        
        for page_id, embedding in zip(page_ids, embeddings):
            embedding_blob = embedding.astype(np.float32).tobytes()
            cursor.execute("INSERT INTO embeddings (page_id, embedding) VALUES (?, ?)", (page_id, embedding_blob))
        
        if (i // batch_size) % (50 // batch_size) == 0:
            print(f"Processed {i + len(batch)} pages...")

    conn.commit()
    conn.close()

    print(f"Semantic indexing completed in {time.time() - start_time:.2f} seconds.")

def update_stats():
    """Updates the statistics table with the latest data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    total_pages = cursor.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    total_terms = cursor.execute("SELECT COUNT(DISTINCT term) FROM inverted_index").fetchone()[0]
    total_embeddings = cursor.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    cursor.execute("INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)", ('total_pages_indexed', str(total_pages)))
    cursor.execute("INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)", ('total_terms', str(total_terms)))
    cursor.execute("INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)", ('total_embeddings', str(total_embeddings)))
    cursor.execute("INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)", ('embedding_model', MODEL_NAME))
    cursor.execute("INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)", ('last_indexed_at', str(int(time.time()))))
    
    conn.commit()
    conn.close()
    print("Stats table updated.")

def main():
    """
    Main function to run the indexing process.
    """
    pages = load_pages()
    if not pages:
        print("No pages found in the database. Run the crawler first.")
        return

    index_bm25(pages)
    index_semantic(pages)
    update_stats()
    print("\nIndexing complete.")

if __name__ == '__main__':
    main()
