# Ask the Fed — RAG over FOMC communications

RAG system over ~17 years of FOMC minutes and statements (2009–2026), answering
natural-language questions with citations back to federalreserve.gov.

## Stack

Python 3.12. SQLite for chunks and metadata, numpy `.npy` for vectors.
Voyage `voyage-3-large` for embeddings, Claude for generation. Streamlit for the UI.

No LangChain, no LlamaIndex, no hosted vector DB. The corpus is ~3,000 chunks;
brute-force cosine over a numpy array is faster than a network round trip and
keeps retrieval debuggable.

## Layout

```
scrape.py    download and cache raw HTML          -> data/raw/{meeting_date}/
chunk.py     parse sections, split into passages  -> data/chunks.jsonl
embed.py     embed chunks, write to store         -> fed.db + vectors.npy
ask.py       retrieve + generate (the core loop)
app.py       Streamlit UI
eval.py      gold questions, recall@k, wrong-meeting rate
spike.py     original single-document proof. Reference only, do not modify.
```

## Corpus facts

Eight meetings per year, ~142 total. Minutes are released three weeks after the
policy decision. Minutes run ~7,000 words; statements ~400.

**The corpus starts in 2009, and that floor is load-bearing.** Minutes from 2008
and earlier are a different document: none of the six standard sections exist in
them (verified against 2000, 2006 and 2008). Since the section label is what
identifies whose view a passage carries, ingesting pre-2009 minutes would add
volume while destroying attribution. Do not extend the range without solving
that first.

URLs have moved twice and all three eras are still served:

| era | minutes | statements |
| --- | --- | --- |
| 2011– | `/monetarypolicy/fomcminutes{date}.htm` | `/newsevents/pressreleases/monetary{date}a.htm` |
| 2006–2010 | same | `/newsevents/press/monetary/{date}a.htm` |
| pre-2007 | `/fomc/minutes/{date}.htm` | `/boarddocs/` directory, not HTML |

The date is the **last** day of the meeting. Do not construct these — parse them
from `fomccalendars.htm` and `fomchistorical{YYYY}.htm`. The pattern has
exceptions.

Standard section headings, in order:

1. Developments in Financial Markets and Open Market Operations
2. Staff Review of the Economic Situation
3. Staff Review of the Financial Situation
4. Staff Economic Outlook
5. Participants' Views on Current Conditions and the Economic Outlook
6. Committee Policy Action

These have drifted, and an exact-match parser silently folds a renamed section
into the one above it. Observed variants: "Committee Policy Action**s**"; "Staff
Review of Financial Situation" (no "the"); "Participants' **View**"; "Discussion
of Financial Markets…" and "Financial Developments…"; and, throughout 2009–2015,
"Developments in Financial Markets and **the Federal Reserve's Balance Sheet**".
Some 2009 meetings merge sections outright ("Staff Review of the Economic **and
Financial** Situation", "Meeting Participants' Views **and Committee Policy
Action**) — those keep their own label, because the attribution really is
combined and flattening them would claim a precision the source lacks.

Match on normalized text, not HTML tags. Handle curly apostrophes, and note that
the section heading sits *inside* the first paragraph of its section, separated
by a `<br/>` — treating the whole `<p>` as body text files every section's
opening paragraph under the previous heading.

## The central problem

FOMC minutes are formulaic. The same sections say nearly the same things in
nearly the same words, meeting after meeting. Semantic search alone returns the
right sentence from the wrong year, and that is the dominant failure mode.

Three defenses, all mandatory:

- Parse temporal constraints out of the query ("in 2025", "since September",
  "during the hiking cycle") and filter by date in SQL **before** vector search.
- Chunk on section boundaries and store `section` as a filterable column.
- Prepend `[{meeting_date} | {section}]` to chunk text before embedding, so no
  passage is ever seen without its date attached.

## Domain rules for generation

- **Staff ≠ participants ≠ the Committee.** The staff forecast is not the
  Committee's view. Attributing one to the other is a factual error. The section
  label identifies whose view a passage is.
- **Preserve Fed-speak quantifiers verbatim.** "A couple", "a few", "several",
  "many", "most", "almost all" are quasi-ordinal and load-bearing. Never
  paraphrase them into "some".
- Every claim carries a citation. No citation, no claim.
- When the retrieved passages don't answer the question, say so. Do not reach.

## Conventions

- Cache raw HTML on first fetch. Never re-scrape during development.
- Ingestion is idempotent: reruns must not duplicate rows. Key on `content_hash`.
- Secrets from environment variables only. Never commit `.env`.
- pytest for anything with logic. Parsing and chunking especially.

## Working agreement

- Before scaling any stage from one document to eighty, print intermediate
  output — section histograms, coverage tables, sample chunks — so I can inspect
  it. Do not silently process the whole corpus.
- Never state a fact about what the FOMC said from your own knowledge. Every
  factual claim about the corpus comes from a document in `data/raw/`.
- Never generate gold eval questions or answers from memory. Gold pairs are
  derived from documents that have actually been read, and labeled with the
  meeting date and section they came from.
- When a change is meant to improve retrieval, run `eval.py` before and after and
  report both numbers. "This should help" is not evidence.
