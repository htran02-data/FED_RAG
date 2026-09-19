# Rebuild validation · September 19, 2026

The original iCloud project was read but not modified. The updated project is a
separate working copy. Existing documents, gold questions, production stores,
and experimental stores were preserved. No scraping or embedding calls were
made, and no API secrets were copied.

## Automated and application checks

- Original baseline: 259 tests passed.
- Rebuilt application: 281 tests passed, including terminal workflows, read-only
  access, temporal/document constraints, offline API isolation, exports,
  launch-time settings, error recovery, and browser answer persistence.
- Local setup check passed: SQLite integrity, offline index, vector row alignment.
- Real offline question returned verbatim excerpts, date/section attribution,
  and Federal Reserve links.
- Terminal entry point successfully ran from outside the project directory.
- Browser checked using Streamlit's application test runner: startup, question
  submission, displayed evidence, and retained answer after settings changed.

## Retrieval measurements

All evaluations use the existing 30-question gold set, k=8. Semantic queries
use cached question embeddings, so no provider requests are required.

The standard `eval.py --compare` run uses expansion=0. Results were unchanged:

| Metric, date-filtered | Before | After |
|---|---:|---:|
| Quote recall@8 | 0.900 | 0.900 |
| Quote MRR | 0.854 | 0.854 |
| Wrong-meeting share | 0.144 | 0.144 |

The new shared semantic pipeline was also compared directly with the existing
retrieval function on every gold question at expansion=1: identical winning
passages and expanded text for all 30 questions.

| Mode, date-filtered, expansion=1 | Quote recall@8 | Quote MRR | Wrong-meeting share |
|---|---:|---:|---:|
| Existing semantic/hybrid pipeline | 1.000 | 0.942 | 0.144 |
| New offline keyword mode | 0.967 | 0.668 | 0.159 |

Offline mode is a convenience fallback, not a measured improvement in ranking.
This small, corpus-specific evaluation does not establish general answer
correctness. In particular, generated answer quality was not evaluated.

## Limits and configuration

Live Voyage embedding/reranking and Anthropic generation were not exercised.
Generation configuration and routing were tested with a test double. Add your
own credentials and an available model ID to enable those features.

The current local launcher reuses the original project's Python environment.
Run `./setup.sh` to create an independent environment in this working copy.
The setup script requires network access and was not run during validation.

The corpus ends July 29, 2026. Questions beyond that date cannot be answered
from this snapshot. Sessions keep settings and the previous result; follow-up
questions do not inherit dates or meaning from earlier questions.
