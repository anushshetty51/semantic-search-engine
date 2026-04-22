from locust import HttpUser, task, between
import random

class SearchUser(HttpUser):
    wait_time = between(1, 3)
    
    queries = [
        "python decorators", "asyncio", "fastapi tutorial", "data classes",
        "context managers", "python list comprehension", "generators vs iterators",
        "what is the gil", "python type hints", "collections module",
        "functools", "itertools", "python logging", "virtual environments",
        "pip install", "magic methods", "python exceptions", "lambda functions",
        "multiprocessing", "threading"
    ]

    @task(6)
    def hybrid_search(self):
        """Simulates a user performing a hybrid search."""
        query = random.choice(self.queries)
        self.client.get(f"/search?q={query}&mode=hybrid", name="/search?mode=hybrid")

    @task(2)
    def keyword_search(self):
        """Simulates a user performing a keyword search."""
        query = random.choice(self.queries)
        self.client.get(f"/search?q={query}&mode=keyword", name="/search?mode=keyword")

    @task(1)
    def semantic_search(self):
        """Simulates a user performing a semantic search."""
        query = random.choice(self.queries)
        self.client.get(f"/search?q={query}&mode=semantic", name="/search?mode=semantic")

    @task(1)
    def health_check(self):
        """Simulates a health check call."""
        self.client.get("/health", name="/health")

    def on_start(self):
        """on_start is called when a Locust start before any task is scheduled"""
        self.client.verify = False # Disables SSL verification for local testing if needed
