# Terminal guide

Run `./fed` from the project folder, or drag the executable into Terminal and
press Return. Ask a complete question at `fed>`. Answers are printed right there,
followed by source labels, meeting dates, sections, and Federal Reserve links.

```text
fed> What did participants say about tariffs in 2018?
fed> :full 1
fed> :save "tariffs answer.md"
fed> :quit
```

## Commands

| Command | What it does |
|---|---|
| `:help` | List commands |
| `:status` | Show corpus dates and current settings |
| `:k 5` | Retrieve 1–50 passages |
| `:full 2` | Display source 2 completely |
| `:last` | Redisplay the last successful answer and its links |
| `:save "answer.md"` | Export answer plus full evidence |
| `:save answer.json` | Export structured answer, filters, and evidence |
| `:sections` | List all section labels, including historical combined sections |
| `:section participants` | Select an unambiguous partial section name |
| `:section any` | Remove the section restriction |
| `:doc minutes` | Limit to minutes; also `statement` or `any` |
| `:expand 1` | Add 0–4 neighboring paragraphs within each section |
| `:mode offline` | Local keyword search; also `auto` or `semantic` |
| `:answer quotes` | Verbatim sentence selection; also `generated` or `off` |
| `:filter on` | Use dates/document types stated in questions; also `off` |
| `:rerank on` | Additional API reranking; also `off` |
| `:reset` | Restore launch settings and clear the previous answer |
| `:quit` | Exit; Ctrl-D also exits |

Ctrl-C cancels input or the active question and keeps the session open. Invalid
settings or failed API requests display an error without ending the session.
History is available with arrow keys during the session; questions are not
persisted unless you explicitly save an answer. Exports never overwrite files.

## Better questions

Include a period: `in 2018`, `September 2025`, `between 2019 and 2021`,
`since September 2024`, `last year`, or `during the hiking cycle`. Cycle names
are calculated from the cached statements. Relative dates use your computer's
current date, so they may fall beyond the corpus's coverage.

Name `statement` or `minutes` to constrain the source document. Select a section
to isolate the staff, participants, or Committee. Sections with combined historical
attribution remain combined. Every source label shows the actual attribution.

Questions are independent, even in one session. Ask “What did participants say
about inflation in 2020?” instead of “What about the following year?”

If an answer is unavailable, try more specific words, a wider period, or inspect
`:full 1`. Offline search matches words and is less capable with paraphrases.
A retrieved sentence is evidence for its exact wording; it may not fully resolve
an analytical “why” question.

## API features

Semantic search needs `VOYAGE_API_KEY`. Generated answers additionally need
`ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`. Set them in the project's `.env` or
export them in your shell. Use `./fed --doctor` to check which are configured.
Actual account access and model availability are checked only when queried.

```sh
./fed --mode semantic --answer generated "What did the December 2020 statement say about inflation?"
```

Offline mode rejects generated answers and reranking to keep its no-network
promise. To switch from such settings, use `:answer quotes` and `:rerank off`.

## Troubleshooting

- **Runtime not found:** run `./setup.sh` to create this copy's Python environment.
- **Store missing:** check `--data-dir` or `FEDRAG_DATA_DIR`; the ready store is
  included in this working copy.
- **Vector mismatch:** restore a matching database/vector pair. Offline mode
  remains available when the full-text index is intact.
- **API request fails:** verify the named provider's key/account/model, or use
  `:mode offline`, `:answer quotes`, and `:rerank off`.
- **No answer:** inspect corpus coverage and section filters before broadening
  the question. No matching passage is not proof that an event never happened.
