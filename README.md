# Ask the Fed · FED-RAG

Ask questions about FOMC minutes and statements directly in your terminal.
Answers include dates, speaker/section labels, and links to the original Fed
communications. The included corpus covers **142 meetings, 10,952 passages,
January 28, 2009 through July 29, 2026**. It is a snapshot, not a live news feed.

## Start on this computer

Open Terminal, drag the `fed` file into it, and press Return. Or double-click
`Start Fed.command` in Finder (macOS may ask you to approve opening it).

From the project directory:

```sh
./fed
```

You will see `fed>`. Type, for example:

```text
What did participants say about tariffs in 2018?
What did the December 2020 statement say about inflation?
:full 1
:save "my answer.md"
:quit
```

**This working copy is separate from the original iCloud project.** Its local
`.runtime-python` file points to the original project's existing Python runtime
so it runs immediately on this computer. No API secrets were copied. To make
this copy independent of that runtime, run `./setup.sh` (requires internet).
The launcher prefers its own `.venv` once installed.

## How answers work

- **Quoted answers** are the default. Relevant sentences are reproduced from
  retrieved passages, preserving wording and attribution. These are evidence
  excerpts, not a synthesized explanation or a guarantee of completeness.
- **Automatic retrieval** uses semantic search when `VOYAGE_API_KEY` is set;
  otherwise it clearly reports offline keyword search. API failures are shown,
  never silently disguised as offline results.
- **Offline mode** uses the local SQLite full-text index. It makes no API calls
  with quoted answers, but can miss synonyms or conceptually similar wording.
- **Generated answers** are optional. Configure both API providers and choose
  `--answer generated`. Every source is supplied with date and section labels.
  Generated claims still need review against those sources.

The date and document constraints apply **before** ranking. Winning paragraphs
are expanded only within their original document and section. Staff forecasts,
participants' opinions, and Committee decisions keep their distinct labels.

Each question is independent. For follow-ups, repeat the subject and date;
this version does not infer missing context from earlier questions.

## Terminal commands

```sh
./fed --doctor
./fed --offline
./fed "What did participants say about tariffs in 2018?"
./fed --offline --json "inflation in 2020"
./fed --offline --output answer.md "inflation in 2020"
./fed --mode semantic --answer generated "inflation in 2020"
./fed -i -k 5 --doc-type minutes --expand 1
```

`python ask.py` and `python -m fedrag` open the same interface. Launching with
an absolute path works from any directory. Relative export filenames resolve
against your current terminal directory. Existing files are never overwritten.
See [USAGE.md](USAGE.md) for all interactive commands.

## Install elsewhere

Use Python 3.12 or newer:

```sh
./setup.sh
cp .env.example .env
```

Edit `.env` only if you want API features:

```dotenv
VOYAGE_API_KEY=your_voyage_key
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_MODEL=a_model_id_available_in_your_account
```

The model is explicitly configurable; no unverified model ID is hardcoded.
Environment variables take precedence over `.env`. `FEDRAG_DATA_DIR` or
`--data-dir` can point to another directory containing `fed.db`, `vectors.npy`,
and `data/`. Ingestion scripts build the store in their project directory.
`--doctor` checks local integrity and configuration without displaying secrets
or contacting providers.

## Browser interface

```sh
.venv/bin/python -m streamlit run app.py
```

The browser and terminal share the same pipeline, filtering, quoted/generated
answer options, and source expansion. The browser retains the latest answer
through reruns and offers a Markdown download.

## Data pipeline

The supplied corpus and store are ready. Normal questions never scrape,
re-embed, or change them. To deliberately rebuild or update the corpus:

```sh
.venv/bin/python scrape.py
.venv/bin/python scrape.py --download --limit 1
.venv/bin/python chunk.py --inspect --show 3
# Inspect coverage and section boundaries before scaling up.
.venv/bin/python scrape.py --download
.venv/bin/python chunk.py
.venv/bin/python embed.py --dry-run
.venv/bin/python embed.py
```

Embedding uses an external API and can take hours. Preserve existing stores
before rebuilding. Never combine a database with vectors from another run.
The 2009 floor preserves the corpus's documented section-attribution rules.

## Project map

| Component | Responsibility |
|---|---|
| `fed`, `fedrag/cli.py` | Launcher, questions, commands, export, setup checks |
| `fedrag/service.py` | Shared application pipeline and read-only corpus access |
| `fedrag/config.py` | Project-relative configuration |
| `ask.py` | Date parsing, hybrid retrieval, expansion, answering; compatible entry point |
| `app.py` | Browser interface |
| `scrape.py`, `chunk.py`, `embed.py` | Cached ingestion and index construction |
| `rates.py` | Policy eras derived from cached statements |
| `eval.py`, `tests/` | Retrieval evaluation and regression tests |

Experimental stores (`ctx.*`, `w3.*`) and existing reports are retained from the
original project. They are not used by the new default pipeline. `spike.py`
remains the original proof of concept.

## Verification

```sh
.venv/bin/python -m pytest -q
.venv/bin/python eval.py --compare
```

See [VALIDATION.md](VALIDATION.md) for measured results and limitations of this
rebuild. The cached 30-question gold evaluation measures retrieval, not complete
answer correctness or general financial advice.
