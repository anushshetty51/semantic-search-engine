import { useEffect, useState } from "react";

const MODES = [
  {
    id: "keyword",
    title: "Keyword",
    note: "Exact-term retrieval with BM25-style scoring",
  },
  {
    id: "semantic",
    title: "Semantic",
    note: "Embedding similarity for concept matching",
  },
  {
    id: "hybrid",
    title: "Hybrid",
    note: "Weighted fusion across lexical and semantic signals",
  },
];

const SAMPLE_QUERIES = [
  "decorators",
  "async context manager",
  "type hints for protocols",
  "python logging configuration",
];

function getApiBase() {
  if (window.location.protocol === "file:") {
    return "http://localhost:8000";
  }
  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    return "http://localhost:8000";
  }
  return "/api";
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value ?? 0);
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function ResultCard({ result, index, cached }) {
  const scoreWidth = Math.max(6, Math.round(result.score * 100));

  return (
    <article className="result-card">
      <div className="result-rank">#{index + 1}</div>
      <h3 className="result-title">{result.title || "Untitled document"}</h3>
      <a
        className="result-url"
        href={result.url}
        target="_blank"
        rel="noopener noreferrer"
      >
        {result.url}
      </a>
      {result.snippet && <p className="result-snippet">{result.snippet}</p>}
      <div className="score-bar">
        <div className="score-fill" style={{ width: `${scoreWidth}%` }} />
      </div>
      <div className="result-meta">
        <span className="tag">Score {result.score.toFixed(4)}</span>
        <span className="tag">{result.mode}</span>
        <span className={`tag ${cached ? "cached" : ""}`}>
          {cached ? "Redis cache" : "Live ranking"}
        </span>
      </div>
    </article>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [currentPage, setCurrentPage] = useState(1);
  const [sourceUrl, setSourceUrl] = useState("https://docs.python.org/3/");
  const [maxPages, setMaxPages] = useState(100);
  const [replaceExisting, setReplaceExisting] = useState(true);
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [ingestStatus, setIngestStatus] = useState(null);
  const [results, setResults] = useState([]);
  const [responseMeta, setResponseMeta] = useState(null);
  const [status, setStatus] = useState("idle");
  const [errorMessage, setErrorMessage] = useState("");

  const apiBase = getApiBase();

  useEffect(() => {
    async function loadRuntimeData() {
      try {
        const [healthResponse, statsResponse] = await Promise.all([
          fetch(`${apiBase}/health`),
          fetch(`${apiBase}/stats`),
        ]);

        if (!healthResponse.ok || !statsResponse.ok) {
          throw new Error("Failed to load runtime telemetry");
        }

        const [healthData, statsData] = await Promise.all([
          healthResponse.json(),
          statsResponse.json(),
        ]);

        setHealth(healthData);
        setStats(statsData);
      } catch (error) {
        setErrorMessage(error.message);
      }
    }

    loadRuntimeData();
  }, [apiBase]);

  useEffect(() => {
    let pollTimer;

    async function loadIngestStatus() {
      try {
        const response = await fetch(`${apiBase}/ingest/status`);
        if (!response.ok) {
          throw new Error("Failed to load ingestion status");
        }
        const data = await response.json();
        setIngestStatus(data);
        if (data.status === "running" || data.status === "completed" || data.status === "failed") {
          const [healthResponse, statsResponse] = await Promise.all([
            fetch(`${apiBase}/health`),
            fetch(`${apiBase}/stats`),
          ]);
          if (healthResponse.ok && statsResponse.ok) {
            setHealth(await healthResponse.json());
            setStats(await statsResponse.json());
          }
        }
      } catch (error) {
        setErrorMessage(error.message);
      }
    }

    loadIngestStatus();
    pollTimer = window.setInterval(loadIngestStatus, 4000);

    return () => window.clearInterval(pollTimer);
  }, [apiBase]);

  async function runSearch(nextQuery, page = 1) {
    const normalizedQuery = nextQuery.trim();
    if (!normalizedQuery) {
      return;
    }

    setStatus("loading");
    setErrorMessage("");
    setCurrentPage(page);

    try {
      const response = await fetch(
        `${apiBase}/search?q=${encodeURIComponent(normalizedQuery)}&mode=${mode}&page=${page}&page_size=10`
      );
      if (!response.ok) {
        throw new Error(`Search failed with status ${response.status}`);
      }

      const data = await response.json();
      setResults(data.results);
      setResponseMeta(data);
      setStatus(data.results.length ? "ready" : "empty");
    } catch (error) {
      setResults([]);
      setResponseMeta(null);
      setStatus("error");
      setErrorMessage(error.message);
    }
  }

  async function startIngestion(event) {
    event.preventDefault();
    setErrorMessage("");
    setResults([]);
    setResponseMeta(null);
    setStatus("idle");

    try {
      const response = await fetch(`${apiBase}/ingest`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          source_url: sourceUrl.trim(),
          max_pages: Number(maxPages),
          replace_existing: replaceExisting,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to start ingestion");
      }

      if (replaceExisting) {
        setQuery("");
        setStats({
          total_pages_indexed: 0,
          total_terms: 0,
          total_embeddings: 0,
          cache_hit_rate: "0.00%",
          cache_hits: 0,
          cache_misses: 0,
          source_url: payload.source_url,
          last_indexed_at: null,
        });
      }

      setIngestStatus({
        status: "running",
        source_url: payload.source_url,
        max_pages: payload.max_pages,
        pages_crawled: 0,
        error: null,
      });
    } catch (error) {
      setErrorMessage(error.message);
    }
  }

  const healthy = health?.redis === "ok" && health?.db === "ok";
  const ingestRunning = ingestStatus?.status === "running";

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">SX</div>
          <div className="brand-copy">
            <strong>SearchX</strong>
            <span>Web search workspace</span>
          </div>
        </div>
        <div className="system-chip">
          {health ? (healthy ? "API online" : "API online with degraded dependency") : "Connecting to API..."}
        </div>
      </header>

      <section className="hero">
        <div className="hero-card">
          <div className="eyebrow">React frontend, search-first workflow</div>
          <h1>Search across the site you actually want to index.</h1>
          <p className="hero-copy">
            SearchX crawls a website, builds lexical and semantic indexes, and lets you
            explore results with keyword, semantic, and hybrid ranking modes from one interface.
          </p>

          <form
            className="search-form"
            onSubmit={(event) => {
              event.preventDefault();
              setCurrentPage(1);
              runSearch(query, 1);
            }}
          >
            <div className="search-row">
              <input
                type="text"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Try: decorators, async context manager, tuple unpacking"
              />
              <button type="submit">Run Search</button>
            </div>

            <div className="mode-grid">
              {MODES.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`mode-button ${mode === item.id ? "active" : ""}`}
                  onClick={() => {
                    setMode(item.id);
                    setCurrentPage(1);
                  }}
                >
                  <span className="mode-title">{item.title}</span>
                  <span className="mode-note">{item.note}</span>
                </button>
              ))}
            </div>
          </form>

          <div className="hero-footer">
            <div className="pill">React + Vite</div>
            <div className="pill">FastAPI backend</div>
            <div className="pill">Redis-backed caching</div>
            <div className="pill">Site ingestion + hybrid retrieval</div>
          </div>

          <div className="samples">
            <div className="sample-label">Starter queries</div>
            <div className="sample-row">
              {SAMPLE_QUERIES.map((sample) => (
                <button
                  key={sample}
                  className="sample-chip"
                  type="button"
                  onClick={() => {
                    setQuery(sample);
                    setCurrentPage(1);
                    runSearch(sample, 1);
                  }}
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        </div>

        <aside className="panel">
          <div>
            <div className="panel-kicker">Corpus management</div>
            <h2 className="insights-title">Index a website</h2>
          </div>

          <form className="ingest-form" onSubmit={startIngestion}>
            <label className="field">
              <span className="field-label">Website URL</span>
              <input
                type="url"
                value={sourceUrl}
                onChange={(event) => setSourceUrl(event.target.value)}
                placeholder="https://example.com/docs/"
                required
              />
            </label>

            <label className="field">
              <span className="field-label">Max pages</span>
              <input
                type="number"
                min="1"
                max="2000"
                value={maxPages}
                onChange={(event) => setMaxPages(event.target.value)}
                required
              />
            </label>

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={replaceExisting}
                onChange={(event) => setReplaceExisting(event.target.checked)}
              />
              <span>Replace current corpus before indexing</span>
            </label>

            <button className="primary-action" type="submit" disabled={ingestRunning}>
              {ingestRunning ? "Indexing in progress" : "Start indexing"}
            </button>
          </form>

          <div className="signal">
            <span className="signal-dot" />
            <div className="signal-copy">
              <strong>Current source</strong>
              <span>{stats?.source_url || "No source indexed yet"}</span>
            </div>
          </div>

          <div className="ingest-status-card">
            <strong>Indexer status</strong>
            <div className={`health-badge ${ingestStatus?.status === "failed" ? "warn" : ""}`}>
              {ingestStatus?.status || "idle"}
            </div>
            <div className="metric-list">
              <div className="metric-row">
                <span>Target</span>
                <span>{ingestStatus?.source_url || stats?.source_url || "--"}</span>
              </div>
              <div className="metric-row">
                <span>Max pages</span>
                <span>{ingestStatus?.max_pages ?? "--"}</span>
              </div>
              <div className="metric-row">
                <span>Pages crawled</span>
                <span>{ingestStatus?.pages_crawled ?? "--"}</span>
              </div>
            </div>
            {ingestStatus?.error && <p className="inline-error">{ingestStatus.error}</p>}
          </div>

          <div className="stat-grid">
            <StatCard label="Indexed pages" value={stats ? formatNumber(stats.total_pages_indexed) : "--"} />
            <StatCard label="Unique terms" value={stats ? formatNumber(stats.total_terms) : "--"} />
            <StatCard label="Embeddings" value={stats ? formatNumber(stats.total_embeddings) : "--"} />
            <StatCard label="Cache hit rate" value={stats?.cache_hit_rate ?? "--"} />
          </div>

          <div className="signal">
            <span className="signal-dot" />
            <div className="signal-copy">
              <strong>Search endpoint ready</strong>
              <span>
                {health
                  ? `Redis: ${health.redis} · DB: ${health.db}`
                  : "Health and stats load automatically when the page opens."}
              </span>
            </div>
          </div>

          <p className="footer-note">
            Use the form above to crawl a new site, rebuild the index, and then run searches
            against that corpus from the same UI.
          </p>
        </aside>
      </section>

      <section className="workspace">
        <section className="results-panel">
          <div className="results-header">
            <div>
              <h2 className="results-title">Search Results</h2>
              <div className="query-summary">
                {responseMeta
                  ? `"${responseMeta.query}" returned ${responseMeta.total_results} ranked results in ${responseMeta.response_time_ms.toFixed(2)} ms.`
                  : "Run a query to inspect ranking output."}
              </div>
            </div>
            <div className={`health-badge ${responseMeta && !responseMeta.cached ? "warn" : ""}`}>
              {responseMeta ? (responseMeta.cached ? "Cache hit" : "Fresh compute") : "Awaiting request"}
            </div>
          </div>

          {status === "loading" && (
            <div className="loader visible">
              <span className="loader-ring" />
              <span>Querying lexical and semantic ranking paths...</span>
            </div>
          )}

          {status === "idle" && (
            <div className="state visible">
              <div className="empty-illustration">⌕</div>
              <strong>Search-ready workspace</strong>
              <p className="empty-copy">
                Start with one of the sample prompts or enter your own query to see ranked
                results, response time, and cache behavior.
              </p>
            </div>
          )}

          {status === "error" && (
            <div className="state visible">
              <div className="empty-illustration">!</div>
              <strong>API request failed</strong>
              <p className="empty-copy">{errorMessage || "The frontend could not reach the backend."}</p>
            </div>
          )}

          {status === "empty" && (
            <div className="state visible">
              <div className="empty-illustration">∅</div>
              <strong>No ranked matches</strong>
              <p className="empty-copy">
                Try broader wording, fewer modifiers, or switch ranking modes to compare behavior.
              </p>
            </div>
          )}

          {status === "ready" && (
            <>
              <div className="results-stack">
                {results.map((result, index) => (
                  <ResultCard
                    key={`${result.url}-${index}`}
                    result={result}
                    index={(currentPage - 1) * (responseMeta?.page_size || 10) + index}
                    cached={Boolean(responseMeta?.cached)}
                  />
                ))}
              </div>

              <div className="pagination-bar">
                <button
                  type="button"
                  className="pagination-button"
                  disabled={currentPage <= 1}
                  onClick={() => runSearch(query, currentPage - 1)}
                >
                  Previous
                </button>
                <span className="pagination-copy">
                  Page {responseMeta?.page || 1} of {responseMeta?.total_pages || 1}
                </span>
                <button
                  type="button"
                  className="pagination-button"
                  disabled={!responseMeta || currentPage >= responseMeta.total_pages}
                  onClick={() => runSearch(query, currentPage + 1)}
                >
                  Next
                </button>
              </div>
            </>
          )}
        </section>

        <aside className="insights-panel">
          <div className="insights-header">
            <div>
              <div className="panel-kicker">Request intelligence</div>
              <h2 className="insights-title">Search diagnostics</h2>
            </div>
            <div className={`health-badge ${healthy ? "" : "warn"}`}>
              {health ? (healthy ? "System healthy" : "Degraded dependency") : "Checking health"}
            </div>
          </div>

          <div className="insight-block">
            <strong>Latest response</strong>
            <div className="insight-value">
              {responseMeta ? `${responseMeta.response_time_ms.toFixed(2)} ms` : "--"}
            </div>
            <div className="stat-label">Observed end-to-end latency from the API payload</div>
          </div>

          <div className="insight-block">
            <strong>Mode in use</strong>
            <div className="insight-value">{mode[0].toUpperCase() + mode.slice(1)}</div>
            <div className="stat-label">Switch ranking strategies to compare result shapes</div>
          </div>

          <div className="insight-block">
            <strong>System overview</strong>
            <div className="metric-list">
              <div className="metric-row">
                <span>API base</span>
                <span>{apiBase}</span>
              </div>
              <div className="metric-row">
                <span>Redis health</span>
                <span>{health?.redis ?? "--"}</span>
              </div>
              <div className="metric-row">
                <span>Database health</span>
                <span>{health?.db ?? "--"}</span>
              </div>
              <div className="metric-row">
                <span>Result window</span>
                <span>Top 10</span>
              </div>
            </div>
          </div>
        </aside>
      </section>
    </main>
  );
}
