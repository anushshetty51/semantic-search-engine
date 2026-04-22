import asyncio
import time
import sqlite3
import argparse
import os
from urllib.parse import urljoin, urlparse
from typing import Set, List, Dict, Optional

import httpx
from bs4 import BeautifulSoup

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    return sqlite3.connect(os.getenv("DB_PATH", "db/search.db"))

class AsyncCrawler:
    """
    A production-grade asynchronous web crawler that performs a breadth-first search (BFS)
    traversal of a website, designed for efficiency and robustness.
    """

    def __init__(self, start_url: str, max_pages: int = 500, concurrency: int = 10, request_timeout: int = 5, rate_limit_delay: float = 0.5):
        """
        Initializes the crawler with specified parameters.

        Args:
            start_url (str): The entry point URL for the crawl.
            max_pages (int): The maximum number of pages to crawl.
            concurrency (int): The number of concurrent requests to make.
            request_timeout (int): The timeout in seconds for each HTTP request.
            rate_limit_delay (float): The delay in seconds between requests to the same domain.
        """
        self.start_url = start_url
        self.max_pages = max_pages
        self.base_domain = urlparse(start_url).netloc
        self.queue = asyncio.Queue()
        self.visited_urls: Set[str] = set()
        self.crawled_data: List[Dict[str, str]] = []
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rate_limit_delay = rate_limit_delay
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout),
            follow_redirects=True,
            headers={'User-Agent': 'SemanticSearchBot/1.0'}
        )
        self.start_time = 0.0

    async def crawl(self):
        """
        Main entry point to start the crawling process.
        """
        self.start_time = time.time()
        await self.queue.put(self.start_url)
        self.visited_urls.add(self.start_url)

        tasks = [asyncio.create_task(self.worker()) for _ in range(self.semaphore._value)]
        
        await self.queue.join()

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        await self.client.aclose()
        self.save_to_database()
        self.print_summary()

    async def worker(self):
        """
        The worker task that continuously fetches URLs from the queue and processes them.
        """
        while len(self.crawled_data) < self.max_pages:
            try:
                url = await self.queue.get()
                async with self.semaphore:
                    page_content = await self.fetch_page(url)
                    if page_content:
                        parsed_data = self.parse_page(url, page_content)
                        if parsed_data and len(self.crawled_data) < self.max_pages:
                            self.crawled_data.append(parsed_data)
                            self.print_progress()
                            await self.enqueue_links(parsed_data['links'])
                
                await asyncio.sleep(self.rate_limit_delay)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                # Silently ignore all other exceptions to ensure robustness
                if not self.queue.empty():
                    self.queue.task_done()
                continue

    async def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetches the content of a single page.

        Args:
            url (str): The URL of the page to fetch.

        Returns:
            Optional[str]: The HTML content of the page, or None if an error occurs.
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.text
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException):
            return None

    def parse_page(self, url: str, html_content: str) -> Optional[Dict]:
        """
        Parses the HTML content of a page to extract title, text, and links.

        Args:
            url (str): The URL of the page.
            html_content (str): The HTML content to parse.

        Returns:
            Optional[Dict]: A dictionary with url, title, text, and internal links.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            title = soup.title.string.strip() if soup.title else ''
            text_content = soup.get_text(separator=' ', strip=True)
            
            internal_links = set()
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                parsed_url = urlparse(full_url)
                # Keep only the path and query, remove fragment
                clean_url = parsed_url._replace(fragment="").geturl()
                
                if urlparse(clean_url).netloc == self.base_domain:
                    internal_links.add(clean_url)
            
            return {'url': url, 'title': title, 'text': text_content, 'links': internal_links}
        except Exception:
            return None

    async def enqueue_links(self, links: Set[str]):
        """
        Adds new, unvisited links to the crawling queue.

        Args:
            links (Set[str]): A set of URLs to potentially add to the queue.
        """
        for link in links:
            if link not in self.visited_urls and len(self.visited_urls) < self.max_pages:
                self.visited_urls.add(link)
                await self.queue.put(link)

    def save_to_database(self):
        """
        Saves all crawled page data into the SQLite database.
        """
        if not self.crawled_data:
            return
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            content TEXT
        )
        ''')
        
        try:
            cursor.executemany(
                "INSERT OR IGNORE INTO pages (url, title, content) VALUES (?, ?, ?)",
                [(page['url'], page['title'], page['text']) for page in self.crawled_data]
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            conn.close()

    def print_progress(self):
        """
        Prints the current progress of the crawl to the console.
        """
        elapsed_time = time.time() - self.start_time
        pages_crawled = len(self.crawled_data)
        queue_size = self.queue.qsize()
        print(
            f"\rProgress: {pages_crawled}/{self.max_pages} pages | "
            f"Queue: {queue_size} | "
            f"Elapsed: {elapsed_time:.2f}s",
            end=""
        )

    def print_summary(self):
        """
        Prints a final summary of the crawling session.
        """
        total_time = time.time() - self.start_time
        print(f"\n\n--- Crawl Complete ---")
        print(f"Crawled {len(self.crawled_data)} pages in {total_time:.2f} seconds.")
        print(f"Saved data to 'db/search.db'.")
        print("----------------------")


def main():
    """
    Main function to run the crawler from the command line.
    """
    parser = argparse.ArgumentParser(description="A production-grade asynchronous web crawler.")
    parser.add_argument(
        "start_url", 
        nargs='?', 
        default="https://docs.python.org/3/", 
        help="The starting URL for the crawl. Defaults to Python 3 docs."
    )
    parser.add_argument(
        "--max-pages", 
        type=int, 
        default=500, 
        help="The maximum number of pages to crawl."
    )
    args = parser.parse_args()

    crawler = AsyncCrawler(start_url=args.start_url, max_pages=args.max_pages)
    try:
        asyncio.run(crawler.crawl())
    except KeyboardInterrupt:
        print("\nCrawler interrupted by user. Shutting down gracefully...")

if __name__ == '__main__':
    main()
