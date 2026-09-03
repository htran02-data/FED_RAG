"""
Embed chunks and write them to the store.

The store is two files that must stay in step:

    fed.db       SQLite: chunk text, metadata, and each chunk's vector row
    vectors.npy  float32 matrix, L2-normalized, one row per chunk

`chunks.vector_index` is the row number in the matrix. Keeping the vectors in a
plain numpy array means retrieval is a single matrix multiply -- at ~7,000
chunks that is well under a millisecond, faster than a network round trip to a
hosted index, and every intermediate is inspectable.

Ingestion is idempotent. A chunk's identity is its `content_hash`, so rerunning
this script re-embeds only what is genuinely new. FOMC minutes repeat
near-identical sentences year after year, so the hash covers the meeting date
and section as well as the text -- otherwise 2025's boilerplate would collide
with 2016's and silently vanish.

Usage:
    python embed.py --dry-run        # report what would be embedded, call nothing
    python embed.py --limit 200      # embed a pilot batch
    python embed.py                  # embed everything outstanding
"""

import argparse
import json
import os
import pathlib
import collections
import sqlite3
import time

import numpy as np
from dotenv import load_dotenv

# Secrets come from .env or the environment, never from source.
load_dotenv()

CHUNKS_PATH = pathlib.Path("data/chunks.jsonl")
DB_PATH = pathlib.Path("fed.db")
VECTORS_PATH = pathlib.Path("vectors.npy")

MODEL = "voyage-3-large"
# Voyage caps free accounts (no payment method on file) at 3 requests/min and
# 10K tokens/min. A 128-chunk batch is ~17K tokens, which exceeds the per-minute
# token ceiling on its own, so the batch has to stay small enough to fit under it.
BATCH_SIZE = 128
# Chunks range from ~14 to ~515 tokens, so batching by count gives requests of
# wildly different sizes -- 40 chunks was 7,092 tokens in one slice of the
# corpus and ~5,300 in another. Batches are packed to a token ceiling instead.
TOKENS_PER_WORD = 1.35        # measured 1.29 on this corpus; rounded up
FREE_TIER_BATCH_TOKENS = 2800
# Stay under the documented 3 RPM / 10K TPM with margin. A fixed sleep is not
# enough: the limiter uses a rolling window, and a retried request still spends
# budget, so bursts slip over the line even when the average looks safe.
FREE_TIER_TPM = 8400
FREE_TIER_RPM = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    content_hash TEXT PRIMARY KEY,
    chunk_id     TEXT NOT NULL,
    meeting_date TEXT NOT NULL,
    year         INTEGER NOT NULL,
    doc_type     TEXT NOT NULL,
    section      TEXT NOT NULL,
    text         TEXT NOT NULL,
    embed_text   TEXT NOT NULL,
    source_url   TEXT,
    word_count   INTEGER,
    vector_index INTEGER NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_chunks_date    ON chunks(meeting_date);
CREATE INDEX IF NOT EXISTS idx_chunks_year    ON chunks(year);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);
CREATE INDEX IF NOT EXISTS idx_chunks_doctype ON chunks(doc_type);

-- Lexical half of retrieval. FOMC questions often reuse the minutes' own
-- wording ("many participants", "25 basis points"), and dense vectors smooth
-- exactly that signal away, so BM25 over the raw text is kept alongside them.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content_hash UNINDEXED,
    text
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def load_chunks(path=CHUNKS_PATH):
    if not path.exists():
        raise SystemExit(f"{path} not found. Run chunk.py first.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def existing_hashes(conn):
    return {row[0] for row in conn.execute("SELECT content_hash FROM chunks")}


def pending(chunks, conn):
    """Chunks not yet in the store, de-duplicated within this batch too."""
    known = existing_hashes(conn)
    out, seen = [], set()
    for chunk in chunks:
        digest = chunk["content_hash"]
        if digest in known or digest in seen:
            continue
        seen.add(digest)
        out.append(chunk)
    return out


def normalize_rows(matrix):
    """Unit-length rows, so a dot product is a cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class RateBudget:
    """
    Rolling one-minute budget for requests and tokens.

    Voyage enforces both a request/minute and a token/minute ceiling over a
    sliding window. Sleeping a fixed interval between calls approximates that
    badly -- it ignores how many tokens each call actually spent, so a run of
    dense batches overruns the token ceiling while looking well behaved.
    """

    def __init__(self, tokens_per_minute, requests_per_minute):
        self.tpm = tokens_per_minute
        self.rpm = requests_per_minute
        self.events = collections.deque()

    def _prune(self, now):
        while self.events and now - self.events[0][0] >= 60.0:
            self.events.popleft()

    def spend(self, tokens):
        """Block until this many tokens fit inside the window, then record them."""
        while True:
            now = time.monotonic()
            self._prune(now)
            used = sum(spent for _, spent in self.events)
            if (used + tokens <= self.tpm) and (len(self.events) < self.rpm):
                self.events.append((now, tokens))
                return
            oldest = self.events[0][0]
            time.sleep(max(60.0 - (now - oldest) + 0.5, 1.0))


def count_tokens(client, texts):
    """Exact count where possible; a deliberately high estimate otherwise."""
    try:
        return client.count_tokens(texts, model=MODEL)
    except Exception:
        return int(sum(len(t.split()) for t in texts) * 1.6)


def voyage_embedder(budget=None, max_retries=8):
    """
    The real embedder. Imported lazily so --dry-run needs no API key.

    Embeds exactly the texts it is given -- batching is the caller's job, so a
    failed batch costs one batch rather than the whole run.

    Retries cover more than rate limits. A run that paces itself to a free-tier
    ceiling takes hours, and over that span a dropped connection or a brief 5xx
    is routine; treating those as fatal throws away the whole remaining run for
    a blip. Auth and malformed-request errors are not retried, because repeating
    them cannot help.
    """
    import voyageai
    import voyageai.error as verror

    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit("VOYAGE_API_KEY is not set")
    client = voyageai.Client()

    transient = (
        verror.RateLimitError,
        verror.APIConnectionError,
        verror.ServiceUnavailableError,
        verror.ServerError,
        verror.Timeout,
    )

    def embed(texts, input_type):
        tokens = count_tokens(client, texts)
        for attempt in range(max_retries):
            if budget:
                budget.spend(tokens)
            try:
                return normalize_rows(
                    client.embed(texts, model=MODEL, input_type=input_type).embeddings
                )
            except transient as error:
                if attempt == max_retries - 1:
                    raise
                backoff = min(90.0, 20.0 * (attempt + 1))
                print(f"    {type(error).__name__}; waiting {backoff:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})", flush=True)
                time.sleep(backoff)
        raise RuntimeError("unreachable")

    return embed


def load_vectors(path=VECTORS_PATH, dimension=None):
    if path.exists():
        return np.load(path)
    return np.zeros((0, dimension or 0), dtype=np.float32)


def estimate_tokens(chunk):
    """Deliberately high estimate, so a batch never overshoots its ceiling."""
    return max(1, int(chunk.get("word_count", 0) * TOKENS_PER_WORD) + 8)


def token_batches(chunks, max_tokens, max_count):
    """
    Pack chunks into batches that stay under a token ceiling.

    A single oversized chunk still gets its own batch rather than being
    dropped -- the ceiling is a target, not a guarantee, and the embedder's
    retry handles the rare case where one passage alone is too large.
    """
    batch, total = [], 0
    for chunk in chunks:
        tokens = estimate_tokens(chunk)
        if batch and (total + tokens > max_tokens or len(batch) >= max_count):
            yield batch
            batch, total = [], 0
        batch.append(chunk)
        total += tokens
    if batch:
        yield batch


def _write_batch(batch, vectors, conn, vectors_path):
    """Append one batch's vectors and rows together, so the two cannot drift."""
    existing = load_vectors(vectors_path, dimension=vectors.shape[1])
    if existing.size and existing.shape[1] != vectors.shape[1]:
        raise SystemExit(
            f"vector width changed ({existing.shape[1]} -> {vectors.shape[1]}). "
            "Delete fed.db and vectors.npy and re-embed the corpus."
        )
    offset = existing.shape[0]
    rows = [
        (c["content_hash"], c["chunk_id"], c["meeting_date"], c["year"],
         c["doc_type"], c["section"], c["text"], c["embed_text"],
         c.get("source_url", ""), c.get("word_count", 0), offset + i)
        for i, c in enumerate(batch)
    ]
    combined = np.vstack([existing, vectors]) if existing.size else vectors
    np.save(vectors_path, combined)
    with conn:
        conn.executemany(
            "INSERT INTO chunks_fts (content_hash, text) VALUES (?,?)",
            [(c["content_hash"], c["text"]) for c in batch],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO chunks (content_hash, chunk_id, meeting_date, "
            "year, doc_type, section, text, embed_text, source_url, word_count, "
            "vector_index) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('embedding_model', ?)", (MODEL,))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('dimension', ?)",
                     (str(vectors.shape[1]),))


def ingest(chunks, conn, embedder, vectors_path=VECTORS_PATH,
           batch_size=BATCH_SIZE, batch_tokens=None):
    """
    Embed and store the chunks that are new. Returns how many were added.

    Each batch is committed as it completes rather than at the end of the run,
    so a rate-limit failure or a Ctrl+C costs one batch instead of everything.
    Re-running picks up where it stopped, because `pending` keys on
    `content_hash`.
    """
    todo = pending(chunks, conn)
    if not todo:
        return 0

    ceiling = batch_tokens or 10 ** 9
    added = 0
    for batch in token_batches(todo, ceiling, batch_size):
        # Embed the date- and section-prefixed text, never the bare passage.
        vectors = embedder([c["embed_text"] for c in batch], "document")
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for {len(batch)} chunks"
            )
        _write_batch(batch, vectors, conn, vectors_path)
        added += len(batch)
        print(f"    stored {added}/{len(todo)}", flush=True)
    return added


def backfill_fts(conn):
    """Populate the lexical index for rows embedded before it existed."""
    missing = conn.execute(
        "SELECT content_hash, text FROM chunks WHERE content_hash NOT IN "
        "(SELECT content_hash FROM chunks_fts)"
    ).fetchall()
    if missing:
        with conn:
            conn.executemany(
                "INSERT INTO chunks_fts (content_hash, text) VALUES (?,?)",
                [(row[0], row[1]) for row in missing],
            )
    return len(missing)


def verify(conn, vectors_path=VECTORS_PATH):
    """The table and the matrix must agree, or retrieval returns the wrong text."""
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if not vectors_path.exists():
        return count, 0, ["vectors.npy is missing"]
    matrix = np.load(vectors_path)
    problems = []
    if matrix.shape[0] != count:
        problems.append(f"{count} rows in fed.db but {matrix.shape[0]} vectors")
    indices = [r[0] for r in conn.execute("SELECT vector_index FROM chunks")]
    if indices and sorted(indices) != list(range(count)):
        problems.append("vector_index values are not a contiguous 0..n-1 range")
    norms = np.linalg.norm(matrix, axis=1) if matrix.size else np.array([1.0])
    if matrix.size and not np.allclose(norms, 1.0, atol=1e-3):
        problems.append("vectors are not unit length")
    return count, matrix.shape[0], problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is outstanding without calling the API")
    parser.add_argument("--limit", type=int, help="embed at most N new chunks")
    parser.add_argument("--free-tier", action="store_true",
                        help="pace requests for an account with no payment method "
                             "(3 RPM / 10K TPM): smaller batches, 30s apart")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-tokens", type=int,
                        help="token ceiling for a single request")
    parser.add_argument("--tpm", type=int, help="token/minute ceiling to respect")
    parser.add_argument("--rpm", type=int, help="request/minute ceiling to respect")
    args = parser.parse_args()

    chunks = load_chunks()
    conn = connect()
    todo = pending(chunks, conn)

    stored = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"{len(chunks)} chunks on disk | {stored} already in {DB_PATH} "
          f"| {len(todo)} outstanding")

    if args.dry_run:
        ceiling = args.batch_tokens or (
            FREE_TIER_BATCH_TOKENS if args.free_tier else None)
        size = args.batch_size or BATCH_SIZE
        batches = list(token_batches(todo, ceiling or 10 ** 9, size))
        tokens = sum(estimate_tokens(c) for c in todo)
        print(f"\nWould embed {len(todo)} chunks in {len(batches)} batch(es) "
              f"with {MODEL}")
        if ceiling:
            print(f"  <={ceiling} tokens per request, <={size} chunks")
        print(f"  ~{tokens:,} estimated tokens")
        if args.free_tier:
            print(f"  at {FREE_TIER_TPM} tokens/min: ~{tokens / FREE_TIER_TPM:.0f} min")
        print("\nDry run. Nothing embedded, nothing written.")
        return

    if args.limit:
        todo = todo[:args.limit]
        print(f"  limiting this run to {len(todo)} chunks")

    # A count cap as well as the token ceiling: whichever binds first.
    batch_size = args.batch_size or BATCH_SIZE
    tpm = args.tpm or (FREE_TIER_TPM if args.free_tier else None)
    rpm = args.rpm or (FREE_TIER_RPM if args.free_tier else None)
    budget = RateBudget(tpm, rpm) if tpm and rpm else None

    batch_tokens = args.batch_tokens or (
        FREE_TIER_BATCH_TOKENS if args.free_tier else None)

    if todo:
        batches = len(list(token_batches(todo, batch_tokens or 10 ** 9, batch_size)))
        if budget:
            tokens = sum(estimate_tokens(c) for c in todo)
            print(f"  pacing to {tpm} tokens/min and {rpm} requests/min")
            print(f"  {batches} batches, <={batch_tokens} tokens each, "
                  f"{tokens:,} tokens total (~{tokens / tpm:.0f} min)")
            print("  safe to interrupt -- each batch is committed as it lands",
                  flush=True)
        added = ingest(todo, conn, voyage_embedder(budget=budget),
                       batch_size=batch_size, batch_tokens=batch_tokens)
        print(f"  added {added} chunks")
    else:
        print("  nothing to do")

    count, vectors, problems = verify(conn)
    print(f"\nStore: {count} rows in {DB_PATH}, {vectors} vectors in {VECTORS_PATH}")
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    if not problems:
        print("  table and matrix agree")


if __name__ == "__main__":
    main()
