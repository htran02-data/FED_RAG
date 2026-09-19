# Ask the Fed — notes for coding agents

RAG over FOMC minutes and statements, 2009–2026. Ask a question in plain
English, get passages back with citations to federalreserve.gov.

**The full specification is in [CLAUDE.md](CLAUDE.md). Read it before changing
anything.** This file exists because Codex reads `AGENTS.md` and Claude Code
reads `CLAUDE.md`; the rules below are the subset that, if broken, makes the
program produce answers that are wrong rather than merely worse.

## Run it

```bash
./fed                             # interactive prompt
.venv/bin/python -m pytest tests/   # all tests should pass
.venv/bin/python eval.py --compare  # retrieval quality, before and after
```

Never run `scrape.py --download` or `embed.py` casually. The corpus is cached in
`data/raw/` and the store is built; re-embedding costs about four hours on the
free Voyage tier.

## Rules that are not style preferences

**Staff ≠ participants ≠ the Committee.** The `section` label on a passage
identifies whose view it carries. The staff's forecast is not the Committee's
decision, and participants' opinions are not policy. Any change that lets a
chunk, or an expanded chunk, span two sections breaks this. Attributing one to
the other is a factual error, not a formatting one.

**Preserve Fed quantifiers verbatim.** "A couple", "a few", "several", "many",
"most", "almost all" are quasi-ordinal and load-bearing. Never paraphrase them
into "some". This is why the terminal answer selects sentences instead of
rewriting them.

**The corpus starts in 2009 and that floor is load-bearing.** Minutes from 2008
and earlier have none of the six standard sections, so ingesting them would add
volume while destroying attribution. Verified against 2000, 2006 and 2008.

**Filter before searching, not after.** The dominant failure of this corpus is
returning the right sentence from the wrong year, because the minutes say nearly
the same things every meeting. Temporal and document constraints are applied in
SQL *before* the vector search. Unfiltered, 71% of retrieved passages come from
a meeting the user did not ask about; filtered, 14%.

**Never state a fact about what the FOMC said from your own knowledge.** Every
factual claim about the corpus must come from a document in `data/raw/`. Gold
eval pairs are derived from passages actually read, never written from memory.

## How retrieval is scored

`eval.py` anchors on the answer quote, **not** on `content_hash`. Re-slicing the
corpus changes every hash, so a hash-keyed gold set silently drops to zero the
moment chunking changes and can never compare two schemes.

The gold set (`data/gold.jsonl`, 30 pairs) deliberately contains hard cases:
three answers that span a paragraph break, and five questions that state no
period. Each pair records `spans_paragraphs` and `has_temporal_cue`, and tests
assert both against the corpus and the parser. A gold set of only
single-paragraph, date-cued questions scores 1.000 on everything and measures
nothing.

**When a change is meant to improve retrieval, run `eval.py` before and after
and report both numbers.** "This should help" is not evidence. Three plausible
ideas have already been measured and rejected on this corpus:

| idea | result |
| --- | --- |
| overlapping chunks (share a boundary paragraph) | +30% tokens, no recall gain |
| contextual prefix on each chunk | recall 0.900 → 0.867, i.e. worse |
| reranking alone | +0.024 MRR, no recall change |

What did work: embedding one paragraph per chunk and widening the winners at
read time (`ask.expand_hits`), which took quote recall from 0.900 to 1.000 at no
embedding cost.

## Conventions

- Secrets from the environment only. `.env` is gitignored; never commit it.
- Every derived store (`*.db`, `*.npy`, `data/chunks*.jsonl`) is gitignored.
  Committing one puts tens of megabytes of vectors into history.
- Ingestion is idempotent, keyed on `content_hash`.
- pytest for anything with logic — parsing and chunking especially.
- `spike.py` is the original proof of concept. Reference only; do not modify.

## Modernized application

`fedrag/service.py` owns the shared terminal/browser pipeline. `fedrag/cli.py`
owns terminal behavior; `ask.py` keeps retrieval functions and delegates its CLI.
Offline keyword mode must never call providers. The corpus connection is read-only.
Settings supplied on launch must carry into the interactive session.
Generated answers require an explicit `ANTHROPIC_MODEL`; do not invent a model ID.
See `README.md` and `VALIDATION.md` for current launch paths and measured results.
