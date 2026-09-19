"""
Generate a PDF evaluation report from the live store.

Every number in the report is read from fed.db or produced by running the eval
here and now -- nothing is transcribed by hand, so the document cannot drift
away from what the system actually does.

Rendering goes through headless Chrome, which is already on most machines and
avoids adding a PDF toolchain to the project's dependencies.

Usage:
    python report.py                    # -> report.html and report.pdf
    python report.py --html-only
"""

import argparse
import collections
import datetime as dt
import html
import json
import pathlib
import shutil
import sqlite3
import subprocess

import ask
import eval as V
import rates

OUT_HTML = pathlib.Path("report.html")
OUT_PDF = pathlib.Path("report.pdf")

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]


def find_chrome():
    for path in CHROME_PATHS:
        if pathlib.Path(path).exists():
            return path
    return shutil.which("chromium") or shutil.which("google-chrome")


def corpus_stats(conn):
    row = conn.execute(
        "SELECT COUNT(*) chunks, COUNT(DISTINCT meeting_date) meetings, "
        "MIN(meeting_date) first, MAX(meeting_date) last FROM chunks"
    ).fetchone()
    stats = dict(row)
    stats["by_doc_type"] = dict(conn.execute(
        "SELECT doc_type, COUNT(*) FROM chunks GROUP BY doc_type").fetchall())
    stats["by_section"] = conn.execute(
        "SELECT section, COUNT(*) c FROM chunks GROUP BY section "
        "ORDER BY c DESC LIMIT 8").fetchall()
    stats["by_year"] = conn.execute(
        "SELECT substr(meeting_date,1,4) y, COUNT(DISTINCT meeting_date) "
        "FROM chunks GROUP BY y ORDER BY y").fetchall()
    return stats


def run_eval(gold, conn, matrix, era_names, embedder, k):
    unfiltered = V.evaluate(gold, conn, matrix, k, use_filter=False,
                            era_names=era_names, embedder=embedder)
    filtered = V.evaluate(gold, conn, matrix, k, use_filter=True,
                          era_names=era_names, embedder=embedder)
    return V.summarize(unfiltered, k), V.summarize(filtered, k), filtered


def esc(text):
    return html.escape(str(text))


def build_html(stats, before, after, per_question, k, gold):
    generated = dt.date.today().isoformat()
    years = "".join(
        f"<td>{esc(y)}</td>" for y, _ in stats["by_year"])
    counts = "".join(
        f"<td>{esc(c)}</td>" for _, c in stats["by_year"])

    section_rows = "".join(
        f"<tr><td class='l'>{esc(name)}</td><td>{count:,}</td></tr>"
        for name, count in stats["by_section"])

    def pct(value):
        return f"{value * 100:.1f}%"

    metric_rows = f"""
      <tr class="headline">
        <td class="l">Passages from the wrong meeting</td>
        <td class="bad">{pct(before['wrong_meeting_rate'])}</td>
        <td class="good">{pct(after['wrong_meeting_rate'])}</td>
        <td class="delta">-{(before['wrong_meeting_rate']-after['wrong_meeting_rate'])*100:.1f} pts</td>
      </tr>
      <tr>
        <td class="l">Answer text reached the reader (quote recall@{k})</td>
        <td class="bad">{pct(before[f'quote_recall@{k}'])}</td>
        <td class="good">{pct(after[f'quote_recall@{k}'])}</td>
        <td class="delta">+{(after[f'quote_recall@{k}']-before[f'quote_recall@{k}'])*100:.1f} pts</td>
      </tr>
      <tr>
        <td class="l">Gold chunk retrieved (recall@{k})</td>
        <td class="bad">{pct(before[f'recall@{k}'])}</td>
        <td class="good">{pct(after[f'recall@{k}'])}</td>
        <td class="delta">+{(after[f'recall@{k}']-before[f'recall@{k}'])*100:.1f} pts</td>
      </tr>
      <tr>
        <td class="l">Some passage from the right meeting</td>
        <td class="bad">{pct(before[f'meeting_recall@{k}'])}</td>
        <td class="good">{pct(after[f'meeting_recall@{k}'])}</td>
        <td class="delta">+{(after[f'meeting_recall@{k}']-before[f'meeting_recall@{k}'])*100:.1f} pts</td>
      </tr>
      <tr>
        <td class="l">Mean reciprocal rank</td>
        <td class="bad">{before['mrr']:.3f}</td>
        <td class="good">{after['mrr']:.3f}</td>
        <td class="delta">+{after['mrr']-before['mrr']:.3f}</td>
      </tr>
      <tr>
        <td class="l">Chunks searched per question</td>
        <td class="bad">{before['mean_candidates']:,.0f}</td>
        <td class="good">{after['mean_candidates']:,.0f}</td>
        <td class="delta">-{before['mean_candidates']-after['mean_candidates']:,.0f}</td>
      </tr>"""

    ranked = collections.Counter(
        r["rank"] if r["rank"] and r["rank"] <= 3 else ("4+" if r["rank"] else "miss")
        for r in per_question)
    rank_rows = "".join(
        f"<tr><td class='l'>{esc(label)}</td><td>{ranked.get(key, 0)}</td></tr>"
        for key, label in ((1, "Ranked first"), (2, "Ranked second"),
                           (3, "Ranked third"), ("4+", "Rank 4 or worse"),
                           ("miss", "Not retrieved at all")))

    gold_years = collections.Counter(g["meeting_date"][:4] for g in gold)
    covered = sorted(gold_years)
    corpus_years = [y for y, _ in stats["by_year"]]
    uncovered = [y for y in corpus_years if y not in gold_years]

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Ask the Fed - Retrieval Evaluation</title>
<style>
  @page {{ size: A4; margin: 18mm 16mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
         color: #16202E; font-size: 10.5pt; line-height: 1.5; margin: 0; }}
  h1 {{ font-size: 22pt; margin: 0 0 4pt; letter-spacing: -0.4pt; }}
  .sub {{ color: #5A6675; margin: 0 0 4pt; font-size: 11pt; }}
  .meta {{ color: #8A94A3; font-size: 8.5pt; margin: 0 0 18pt;
          font-family: "SF Mono", Menlo, monospace; }}
  h2 {{ font-size: 12.5pt; margin: 20pt 0 7pt; padding-bottom: 4pt;
       border-bottom: 1.5px solid #16202E; }}
  h3 {{ font-size: 10.5pt; margin: 14pt 0 5pt; }}
  p {{ margin: 0 0 8pt; max-width: 62em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8pt 0 10pt;
          font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: right; padding: 5pt 8pt; border-bottom: 0.5px solid #D6DEE8; }}
  th {{ font-size: 8pt; text-transform: uppercase; letter-spacing: 0.4pt;
       color: #78879B; border-bottom: 1px solid #16202E; }}
  td.l, th.l {{ text-align: left; }}
  table.years {{ font-size: 7.5pt; }}
  table.years th, table.years td {{ padding: 4pt 2pt; }}
  table.years th.l, table.years td.l {{ padding-right: 8pt; white-space: nowrap; }}
  tr.headline td {{ font-weight: 700; background: #F2F5F9; }}
  td.bad {{ color: #A63D28; }}
  td.good {{ color: #2A6350; font-weight: 600; }}
  td.delta {{ color: #8A6A3B; font-size: 9.5pt; }}
  .callout {{ border-left: 3px solid #8A6A3B; background: #F7F3EB;
             padding: 8pt 12pt; margin: 10pt 0; }}
  .caveat {{ border: 0.5px solid #D6DEE8; padding: 10pt 14pt; margin-top: 8pt; }}
  .caveat li {{ margin-bottom: 6pt; }}
  ul {{ margin: 0; padding-left: 16pt; }}
  .small {{ font-size: 9pt; color: #5A6675; }}
  footer {{ margin-top: 16pt; padding-top: 6pt; border-top: 0.5px solid #D6DEE8;
           font-size: 8pt; color: #8A94A3; font-family: "SF Mono", Menlo, monospace; }}
</style></head><body>

<h1>Ask the Fed</h1>
<p class="sub">Retrieval evaluation over FOMC minutes and statements</p>
<p class="meta">Generated {generated} &middot; {stats['chunks']:,} passages &middot;
{stats['meetings']} meetings &middot; {esc(stats['first'])} to {esc(stats['last'])}</p>

<h2>What was measured</h2>
<p>FOMC minutes are formulaic: the same sections say nearly the same things,
meeting after meeting, for years. Semantic search alone therefore returns the
right sentence from the wrong year, and that is the dominant failure mode of
this corpus.</p>
<p>The system's defence is to parse the constraints a question already states -
its period, and the document it names - and apply them in SQL <em>before</em>
the vector search runs. The table below is the same {len(gold)} questions run
twice against the same store: once searching everything, once narrowing first.</p>

<h2>Result</h2>
<table>
  <thead><tr><th class="l">Metric</th><th>Search everything</th>
  <th>Narrow first</th><th>Change</th></tr></thead>
  <tbody>{metric_rows}</tbody>
</table>
<div class="callout"><strong>The first row is the finding.</strong> Without
metadata filtering, {pct(before['wrong_meeting_rate'])} of retrieved passages
come from a meeting other than the one asked about. Filtering first brings that
to {pct(after['wrong_meeting_rate'])}.</div>

<h3>Where the correct passage ranked (narrow first, k={k})</h3>
<table><thead><tr><th class="l">Position</th><th>Questions</th></tr></thead>
<tbody>{rank_rows}</tbody></table>

<h2>Corpus</h2>
<table class="years"><thead><tr><th class="l">Year</th>{years}</tr></thead>
<tbody><tr><td class="l">Meetings</td>{counts}</tr></tbody></table>
<h3>Passages by section</h3>
<table><thead><tr><th class="l">Section</th><th>Passages</th></tr></thead>
<tbody>{section_rows}</tbody></table>
<p class="small">Documents: {esc(json.dumps(stats['by_doc_type']))}. Every passage
carries its meeting date and section, and the section label is what identifies
whose view it is - the staff's, the participants', or the Committee's.</p>

<h2>Limits of this evaluation</h2>
<div class="caveat"><ul>
<li><strong>The gold set covers {len(gold_years)} of {len(corpus_years)} years.</strong>
No pairs exist for {esc(', '.join(uncovered)) if uncovered else 'any missing year'}.
This report therefore shows that expanding the corpus did not damage retrieval on
the years it does cover - not that retrieval works on the years it does not.</li>
<li><strong>Every gold question names its period.</strong> So this measures
"when a question states a period, does filtering help?", not "does filtering help
on arbitrary questions". Questions with no temporal cue fall back to searching
the whole corpus and are unrepresented here.</li>
<li><strong>{len(gold)} pairs is a small sample.</strong> The direction is solid;
the third decimal place is noise. Any change worth less than roughly two
questions cannot be distinguished from chance at this size.</li>
<li><strong>Answer generation is switched off</strong> and therefore untested.
The domain rules it must follow - preserve Fed quantifiers verbatim, never
attribute a staff forecast to the Committee - are enforced only by a prompt that
has not been run.</li>
</ul></div>

<footer>Ask the Fed &middot; retrieval evaluation &middot; numbers produced by
eval.py against fed.db at generation time</footer>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    gold = V.load_gold()
    conn, matrix = ask.load_store()
    era_names = rates.named_eras(rates.load_path())
    embedder = V.cached_query_embedder([g["question"] for g in gold])

    stats = corpus_stats(conn)
    before, after, per_question = run_eval(gold, conn, matrix, era_names,
                                           embedder, args.k)

    OUT_HTML.write_text(
        build_html(stats, before, after, per_question, args.k, gold),
        encoding="utf-8")
    print(f"wrote {OUT_HTML}")

    if args.html_only:
        return

    chrome = find_chrome()
    if not chrome:
        print("No Chrome/Chromium found -- open report.html and print to PDF.")
        return

    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={OUT_PDF.resolve()}", OUT_HTML.resolve().as_uri()],
        check=True, capture_output=True, timeout=120,
    )
    size = OUT_PDF.stat().st_size / 1024
    print(f"wrote {OUT_PDF} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
