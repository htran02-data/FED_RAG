"""
Retrieve passages and generate a cited answer. This is the core loop.

FOMC minutes are formulaic: the same sections say nearly the same things in
nearly the same words, meeting after meeting. Semantic search alone happily
returns the right sentence from the wrong year, and that is the dominant
failure mode. The defence is to constrain by date in SQL *before* the vector
search runs, so the wrong years are never candidates in the first place.

Usage:
    python ask.py "What did participants say about the labor market in 2025?"
    python ask.py --retrieve-only "inflation expectations since September 2024"
    python ask.py --section "Committee Policy Action" "why did the vote dissent?"
"""

import argparse
import datetime as dt
import os
import pathlib
import re
import sqlite3
import textwrap

import numpy as np
from dotenv import load_dotenv

import rates

# Secrets come from .env or the environment, never from source.
load_dotenv()

DB_PATH = pathlib.Path("fed.db")
VECTORS_PATH = pathlib.Path("vectors.npy")

MODEL = "claude-opus-5"
EMBED_MODEL = "voyage-3-large"
DEFAULT_K = 8

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
MONTH_RE = "|".join(MONTHS)

SYSTEM_PROMPT = """You answer questions about Federal Open Market Committee \
communications using only the passages supplied to you.

Attribution rules, which matter more than fluency:
- The staff, the participants, and the Committee are three different voices. \
The staff's forecast is not the Committee's view. Participants' views are not \
Committee decisions. Each passage is labelled with the section it came from, \
and that label tells you whose view it carries. Attributing one to another is \
a factual error.
- Preserve the FOMC's own quantifiers exactly as written: "a couple", "a few", \
"several", "some", "many", "most", "almost all". They are quasi-ordinal and \
load-bearing. Never paraphrase one into another, and never soften one into \
"some".
- Every claim carries a citation like [S1], or [S2, S4] for several. A sentence \
without a citation is not allowed.
- Quote distinctive language rather than paraphrasing it when the wording is \
the point.
- If the passages do not answer the question, say so plainly and say what they \
do cover. Do not reach, and do not fill gaps from anything you know outside \
these passages.
- Dates matter. Each passage carries its meeting date; if the question is about \
a period, check that the passage you cite is actually from it."""


class TimeFilter:
    """A date window parsed out of the question, applied in SQL before search."""

    def __init__(self, start, end, label, source):
        self.start = start
        self.end = end
        self.label = label
        self.source = source

    def __repr__(self):
        return f"TimeFilter({self.start}..{self.end}, {self.label!r})"

    def __eq__(self, other):
        return (isinstance(other, TimeFilter)
                and (self.start, self.end) == (other.start, other.end))


def _year_bounds(year):
    return f"{year:04d}-01-01", f"{year:04d}-12-31"


def _month_bounds(year, month):
    last = 31
    if month in (4, 6, 9, 11):
        last = 30
    elif month == 2:
        last = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last:02d}"


def parse_temporal(question, today=None, era_names=None):
    """
    Pull a date window out of the question, or return None.

    Recognises explicit years and months, open-ended "since"/"before" bounds,
    relative windows, and the policy-era phrases derived from the corpus in
    rates.py. Returning None means no constraint was stated, in which case the
    whole corpus is searched.
    """
    today = today or dt.date.today()
    text = question.lower()
    horizon = today.isoformat()
    corpus_start = "1900-01-01"

    # Named policy eras, derived from the statements rather than assumed.
    for label, (start, end) in sorted((era_names or {}).items(),
                                      key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(label)}\b", text):
            return TimeFilter(start, end, label, "policy-era")

    # "between 2019 and 2021", "from 2018 to 2020"
    span = re.search(r"\b(?:between|from)\s+(\d{4})\s+(?:and|to|through|-)\s+(\d{4})", text)
    if span:
        first, second = sorted((int(span.group(1)), int(span.group(2))))
        return TimeFilter(_year_bounds(first)[0], _year_bounds(second)[1],
                          f"{first}-{second}", "year-range")

    # "since September 2024", "after March 2020"
    anchored = re.search(rf"\b(since|after|from)\s+({MONTH_RE})\s+(\d{{4}})", text)
    if anchored:
        start, _ = _month_bounds(int(anchored.group(3)), MONTHS[anchored.group(2)])
        return TimeFilter(start, horizon,
                          f"since {anchored.group(2).title()} {anchored.group(3)}",
                          "open-ended")

    # "in September 2024", "at the March 2020 meeting"
    month_year = re.search(rf"\b({MONTH_RE})\s+(\d{{4}})", text)
    if month_year:
        start, end = _month_bounds(int(month_year.group(2)), MONTHS[month_year.group(1)])
        return TimeFilter(start, end,
                          f"{month_year.group(1).title()} {month_year.group(2)}",
                          "month")

    # "since 2022", "before 2020", "prior to 2019"
    bound = re.search(r"\b(since|after|before|prior to|up to|until)\s+(\d{4})", text)
    if bound:
        year = int(bound.group(2))
        if bound.group(1) in ("since", "after"):
            return TimeFilter(_year_bounds(year)[0], horizon, f"since {year}", "open-ended")
        return TimeFilter(corpus_start, _year_bounds(year - 1)[1],
                          f"before {year}", "open-ended")

    # "in the last 18 months", "over the past 2 years"
    relative = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+(month|year)s?\b", text)
    if relative:
        amount = int(relative.group(1))
        days = amount * (365 if relative.group(2) == "year" else 30)
        start = (today - dt.timedelta(days=days)).isoformat()
        return TimeFilter(start, horizon,
                          f"last {amount} {relative.group(2)}s", "relative")

    if re.search(r"\bthis year\b", text):
        return TimeFilter(*_year_bounds(today.year), label=str(today.year),
                          source="relative")
    if re.search(r"\blast year\b", text):
        return TimeFilter(*_year_bounds(today.year - 1), label=str(today.year - 1),
                          source="relative")

    # "since September" with no year -- the most recent one that has passed.
    bare = re.search(rf"\b(since|after)\s+({MONTH_RE})\b", text)
    if bare:
        month = MONTHS[bare.group(2)]
        year = today.year if month <= today.month else today.year - 1
        start, _ = _month_bounds(year, month)
        return TimeFilter(start, horizon,
                          f"since {bare.group(2).title()} {year}", "open-ended")

    # A bare year, anywhere: "what did they say about tariffs in 2025"
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if years:
        chosen = sorted({int(y) for y in years})
        if len(chosen) == 1:
            return TimeFilter(*_year_bounds(chosen[0]), label=str(chosen[0]),
                              source="year")
        return TimeFilter(_year_bounds(chosen[0])[0], _year_bounds(chosen[-1])[1],
                          f"{chosen[0]}-{chosen[-1]}", "year-range")

    return None


# Words that carry no signal for lexical matching but appear in nearly every
# question and every passage.
STOPWORDS = {
    "the", "and", "for", "that", "with", "what", "did", "say", "about", "was",
    "were", "how", "who", "why", "their", "there", "this", "these", "those",
    "from", "have", "has", "had", "not", "but", "its", "his", "her", "they",
    "them", "which", "when", "where", "would", "could", "should", "been",
    "being", "are", "any", "all", "some", "more", "most", "over", "into",
    "than", "then", "also", "such", "does", "committee", "fomc", "fed",
    "federal", "reserve", "meeting", "minutes",
}

RRF_K = 60           # standard reciprocal-rank-fusion constant
# Dense similarity is the stronger signal on this corpus; BM25 is a corrective
# for the cases where a question reuses the minutes' exact wording. Weighting
# them equally lets lexical noise displace good dense hits.
LEXICAL_WEIGHT = 0.15


def fts_terms(question):
    """Content words from the question, safe to hand to FTS5."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def lexical_ranks(conn, question, hashes):
    """
    BM25 ranking *within the candidate set*, via SQLite's FTS5.

    Ranking over the whole corpus and then fusing against date-filtered
    candidates does not work: most survivors fall outside the global top of the
    lexical list and receive no lexical signal at all, so fusion adds noise
    instead of evidence. The candidate set is restricted first, exactly as it is
    for the vector search.
    """
    terms = fts_terms(question)
    if not terms or not hashes:
        return {}
    match = " OR ".join(f'"{term}"' for term in terms)
    placeholders = ",".join("?" * len(hashes))
    try:
        rows = conn.execute(
            f"SELECT content_hash FROM chunks_fts WHERE chunks_fts MATCH ? "
            f"AND content_hash IN ({placeholders}) ORDER BY bm25(chunks_fts)",
            (match, *hashes),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}          # no lexical index built yet
    return {row[0]: rank for rank, row in enumerate(rows, start=1)}


def fuse(dense_order, lexical, hashes, lexical_weight=LEXICAL_WEIGHT):
    """
    Reciprocal rank fusion of the dense and lexical rankings.

    Fusing ranks rather than scores avoids having to calibrate a cosine
    similarity against a BM25 score, which are not on comparable scales.
    """
    fused = {}
    for rank, index in enumerate(dense_order, start=1):
        fused[index] = 1.0 / (RRF_K + rank)
    for index, digest in enumerate(hashes):
        rank = lexical.get(digest)
        if rank:
            fused[index] = fused.get(index, 0.0) + lexical_weight / (RRF_K + rank)
    return fused


def parse_doc_type(question):
    """
    Pull an explicit document type out of the question, or return None.

    The minutes' Committee Policy Action section quotes the policy statement
    almost verbatim, so a question about "the December 2020 statement" retrieves
    the minutes' copy of it ahead of the statement itself. When the question
    names the document, that is a metadata constraint like any other and belongs
    in the SQL filter rather than in the ranking.
    """
    text = question.lower()
    # "Statement on Longer-Run Goals" is a different document that this corpus
    # deliberately excludes; don't let it trigger the filter.
    if "longer-run goals" in text or "longer run goals" in text:
        return None
    if re.search(r"\bstatements?\b", text):
        return "statement"
    if re.search(r"\bminutes\b", text):
        return "minutes"
    return None


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def candidates(conn, time_filter=None, section=None, doc_type=None):
    """Apply the metadata constraints in SQL, before any vector maths."""
    clauses, params = [], []
    if time_filter:
        clauses.append("meeting_date BETWEEN ? AND ?")
        params += [time_filter.start, time_filter.end]
    if section:
        clauses.append("section = ?")
        params.append(section)
    if doc_type:
        clauses.append("doc_type = ?")
        params.append(doc_type)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    query = ("SELECT content_hash, chunk_id, meeting_date, year, doc_type, section, "
             f"text, source_url, vector_index FROM chunks{where}")
    return [dict(row) for row in conn.execute(query, params)]


def embed_query(text):
    """Embed the question. Separate input_type from the documents' on purpose."""
    import voyageai

    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit("VOYAGE_API_KEY is not set")
    client = voyageai.Client()
    vector = client.embed([text], model=EMBED_MODEL, input_type="query").embeddings[0]
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def search(question, conn, matrix, k=DEFAULT_K, time_filter=None, section=None,
           doc_type=None, embedder=None, hybrid=True,
           lexical_weight=LEXICAL_WEIGHT):
    """
    Filter by metadata, then rank what survives.

    Ranking is dense cosine similarity fused with BM25 over the same candidate
    set. Brute-force over the filtered sub-matrix: at this corpus size it is a
    single matrix multiply.

    Returns (hits, diagnostics).
    """
    rows = candidates(conn, time_filter, section, doc_type)
    diagnostics = {
        "time_filter": time_filter,
        "section": section,
        "candidates": len(rows),
        "hybrid": hybrid,
        "corpus": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
    }
    if not rows:
        return [], diagnostics

    embed = embedder or embed_query
    query_vector = embed(question)

    indices = np.array([r["vector_index"] for r in rows])
    scores = matrix[indices] @ query_vector
    dense_order = list(np.argsort(-scores))

    if hybrid:
        hashes = [r["content_hash"] for r in rows]
        lexical = lexical_ranks(conn, question, hashes)
        diagnostics["lexical_matches"] = sum(1 for h in hashes if h in lexical)
        fused = fuse(dense_order, lexical, hashes, lexical_weight)
        order = sorted(fused, key=lambda i: -fused[i])[:k]
    else:
        order = dense_order[:k]

    hits = []
    for rank, position in enumerate(order, start=1):
        hit = dict(rows[int(position)])
        hit["score"] = float(scores[int(position)])
        hit["label"] = f"S{rank}"
        hits.append(hit)
    diagnostics["meetings_hit"] = sorted({h["meeting_date"] for h in hits})
    return hits, diagnostics


def format_passages(hits):
    return "\n\n".join(
        f"[{h['label']}] ({h['meeting_date']} | {h['section']})\n{h['text']}"
        for h in hits
    )


def answer(question, hits, model=MODEL):
    """Generate the answer. Passages carry their own date and section labels."""
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    prompt = (
        f"Passages:\n\n{format_passages(hits)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only these passages, with a citation on every claim."
    )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this request."
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


def sources(hits):
    """One line per distinct document behind the answer."""
    seen, lines = set(), []
    for hit in hits:
        key = (hit["meeting_date"], hit["doc_type"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {hit['meeting_date']}  {hit['doc_type']:<9} {hit['source_url']}")
    return lines


def load_store():
    if not DB_PATH.exists() or not VECTORS_PATH.exists():
        raise SystemExit(
            f"Store not found ({DB_PATH}, {VECTORS_PATH}). Run embed.py first."
        )
    return connect(), np.load(VECTORS_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--section", help="restrict retrieval to one section")
    parser.add_argument("--doc-type", choices=["minutes", "statement"])
    parser.add_argument("--retrieve-only", action="store_true",
                        help="show the retrieved passages without generating")
    parser.add_argument("--no-filter", action="store_true",
                        help="skip temporal parsing (to compare retrieval)")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="rank by embedding similarity alone, without BM25")
    args = parser.parse_args()

    conn, matrix = load_store()
    era_names = rates.named_eras(rates.load_path()) if pathlib.Path(
        "data/chunks.jsonl").exists() else {}
    time_filter = None if args.no_filter else parse_temporal(
        args.question, era_names=era_names)
    doc_type = args.doc_type or (None if args.no_filter
                                 else parse_doc_type(args.question))

    hits, diagnostics = search(args.question, conn, matrix, k=args.k,
                               time_filter=time_filter, section=args.section,
                               doc_type=doc_type,
                               hybrid=not args.no_hybrid)

    if time_filter:
        print(f"Time filter: {time_filter.label}  "
              f"({time_filter.start} .. {time_filter.end}, via {time_filter.source})")
    else:
        print("Time filter: none -- searching the whole corpus")
    if doc_type:
        print(f"Document filter: {doc_type} (named in the question)")
    print(f"Candidates after filtering: {diagnostics['candidates']} "
          f"of {diagnostics['corpus']}")

    if not hits:
        print("\nNo passages matched those constraints.")
        return

    print(f"\nTop {len(hits)} passages")
    for hit in hits:
        print(f"  [{hit['label']}] {hit['score']:.3f}  {hit['meeting_date']}  "
              f"{hit['section'][:40]}")
        print(f"        {textwrap.shorten(hit['text'], 110)}")

    print("\nSources")
    for line in sources(hits):
        print(line)

    if args.retrieve_only:
        return

    # Retrieval stands on its own. Without a generation key the passages above
    # are the answer -- cited, dated, and linked -- so say that plainly instead
    # of failing.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nANTHROPIC_API_KEY is not set, so no prose answer was generated.")
        print("The passages above are the retrieval result, each with its "
              "meeting date and section.")
        return

    print(f"\n{'-' * 70}\n")
    print(answer(args.question, hits))


if __name__ == "__main__":
    main()
