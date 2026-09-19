"""
Discover FOMC meeting documents and cache their raw HTML.

Meeting URLs are never constructed. They are parsed out of the Fed's own index
pages -- fomccalendars.htm for recent years and fomchistorical{YYYY}.htm for
older ones -- because the /monetarypolicy/fomcminutes{YYYYMMDD}.htm pattern has
exceptions and the two index pages do not share a DOM structure.

Cache layout:
    data/raw/index/fomccalendars.htm
    data/raw/index/fomchistorical2018.htm
    data/raw/{meeting_date}/minutes.html
    data/raw/{meeting_date}/statement.html
    data/raw/{meeting_date}/meta.json

Nothing is ever re-fetched once cached.

Usage:
    python scrape.py                     # discover only, print a coverage table
    python scrape.py --download          # discover, then fetch every document
    python scrape.py --download --limit 3
"""

import argparse
import dataclasses
import json
import pathlib
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
CALENDAR_URL = f"{BASE}/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = f"{BASE}/monetarypolicy/fomchistorical{{year}}.htm"

RAW = pathlib.Path(__file__).resolve().parent / "data/raw"
INDEX_CACHE = RAW / "index"

USER_AGENT = "fomc-rag/0.1 (research; contact via repo)"
POLITE_DELAY = 0.5

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

# The minutes href carries the meeting's last day. This is the Fed's own
# identifier for the meeting, so it is what we key on -- but we read it out of
# a parsed href, never build it.
# Minutes moved path in 2007: /fomc/minutes/{date}.htm became
# /monetarypolicy/fomcminutes{date}.htm. Both are still served.
MINUTES_HREF = re.compile(r"/(?:fomcminutes|minutes/)(\d{8})\.htm$")
# Statements moved twice: /newsevents/press/monetary/{date}a.htm (2006-2010)
# became /newsevents/pressreleases/monetary{date}a.htm (2011 on). The optional
# slash covers both; the pre-2006 /boarddocs/ directory form is not HTML and is
# out of scope.
STATEMENT_HREF = re.compile(r"/monetary/?(\d{8})[a-z]*\.htm$")


@dataclasses.dataclass
class Meeting:
    meeting_date: str          # YYYY-MM-DD, the last day of the meeting
    year: int
    minutes_url: str | None
    statement_url: str | None
    source_index: str

    def as_dict(self):
        return dataclasses.asdict(self)


def normalize(text):
    """Curly quotes and stray whitespace break exact matching."""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def iso_from_href(href, pattern):
    """Pull YYYY-MM-DD out of a parsed href. Returns None if it does not match."""
    match = pattern.search(href)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def last_day_from_range(month_text, date_text, year):
    """
    Turn a calendar's month/day range into the meeting's final day.

    Handles "27-28", "30-May 1", "16-17*", and bare "15". Returns None when the
    text is not a date range we recognise, so the caller can fall back to the
    href.
    """
    month_text = normalize(month_text).lower().strip("* ")
    date_text = normalize(date_text).strip("* ")

    # A month cell may itself hold a range: "April/May" or "January".
    start_month = MONTHS.get(month_text.split("/")[0].strip())
    if start_month is None:
        return None

    # "30-May 1" -> the end half names its own month.
    cross = re.search(r"-\s*([A-Za-z]+)\s*(\d{1,2})", date_text)
    if cross:
        end_month = MONTHS.get(cross.group(1).lower())
        if end_month is None:
            return None
        return f"{year:04d}-{end_month:02d}-{int(cross.group(2)):02d}"

    days = re.findall(r"\d{1,2}", date_text)
    if not days:
        return None
    return f"{year:04d}-{start_month:02d}-{int(days[-1]):02d}"


def decode(response):
    """
    The Fed serves "content-type: text/html" with no charset, so requests falls
    back to ISO-8859-1 and turns every en-dash and curly apostrophe into
    mojibake. The pages themselves declare <meta charset="utf-8">, and some
    carry a BOM, so decode the bytes explicitly instead of trusting the header.
    """
    try:
        return response.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text


def fetch(url, cache_path, session=None):
    """Fetch once, then never hit the Fed's servers again."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8"), False
    getter = session or requests
    response = getter.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    cache_path.write_text(decode(response), encoding="utf-8")
    time.sleep(POLITE_DELAY)
    return response.text, True


def _statement_url(anchors):
    """
    Pick the policy statement out of a meeting's anchors.

    Three different documents share the /monetary{date}*.htm shape:
        monetary{date}a.htm   the policy statement          <- what we want
        monetary{date}a1.htm  the implementation note
        monetary{date}b.htm   Statement on Longer-Run Goals
    They are told apart by their link text, not their URL, so match on the
    label the Fed prints.
    """
    for anchor in anchors:
        label = normalize(anchor.get_text(" ")).lower()
        href = anchor.get("href", "")
        if not STATEMENT_HREF.search(href):
            continue
        if "longer-run goals" in label or "implementation note" in label:
            continue
        if label in ("statement", "html"):
            return BASE + href
    return None


def parse_calendar(html):
    """
    Recent years: one div.panel per year, one div.row.fomc-meeting per meeting.

    Inside a meeting, the statement links sit in an unlabelled column whose text
    begins "Statement:", so the anchors are scoped to that column before the
    label match runs.
    """
    soup = BeautifulSoup(html, "html.parser")
    meetings = []

    for panel in soup.select("div.panel"):
        heading = panel.find(["h3", "h4", "h5"])
        if not heading:
            continue
        year_match = re.search(r"(\d{4})", normalize(heading.get_text(" ")))
        if not year_match:
            continue
        year = int(year_match.group(1))

        for block in panel.select("div.fomc-meeting"):
            month_div = block.select_one("div.fomc-meeting__month")
            date_div = block.select_one("div.fomc-meeting__date")

            minutes_url = None
            for anchor in block.select("a"):
                href = anchor.get("href", "")
                if MINUTES_HREF.search(href):
                    minutes_url = BASE + href
                    break

            statement_anchors = []
            for column in block.find_all("div", recursive=False):
                if normalize(column.get_text(" ")).lower().startswith("statement:"):
                    statement_anchors.extend(column.find_all("a"))
            if not statement_anchors:
                statement_anchors = block.find_all("a")
            statement_url = _statement_url(statement_anchors)

            meeting_date = None
            if minutes_url:
                meeting_date = iso_from_href(minutes_url, MINUTES_HREF)
            if meeting_date is None and statement_url:
                meeting_date = iso_from_href(statement_url, STATEMENT_HREF)
            if meeting_date is None and month_div and date_div:
                meeting_date = last_day_from_range(
                    month_div.get_text(" "), date_div.get_text(" "), year
                )
            if meeting_date is None:
                continue

            meetings.append(
                Meeting(meeting_date, year, minutes_url, statement_url, "fomccalendars.htm")
            )
    return meetings


def parse_historical(html, year):
    """
    Older years: one div.panel per *meeting*, headed "January 30-31 Meeting - 2018".

    There is no fomc-meeting wrapper and no "Statement:" column here, so the
    anchors are matched by label across the whole panel.
    """
    soup = BeautifulSoup(html, "html.parser")
    meetings = []

    for panel in soup.select("div.panel"):
        heading = panel.find(["h3", "h4", "h5"])
        if not heading:
            continue
        title = normalize(heading.get_text(" "))
        if "meeting" not in title.lower():
            continue

        anchors = panel.find_all("a")
        minutes_url = None
        for anchor in anchors:
            href = anchor.get("href", "")
            if MINUTES_HREF.search(href):
                minutes_url = BASE + href
                break
        statement_url = _statement_url(anchors)

        meeting_date = None
        if minutes_url:
            meeting_date = iso_from_href(minutes_url, MINUTES_HREF)
        if meeting_date is None and statement_url:
            meeting_date = iso_from_href(statement_url, STATEMENT_HREF)
        if meeting_date is None:
            head = re.match(r"([A-Za-z]+)\s+([\d\-–A-Za-z ]+?)\s+Meeting", title)
            if head:
                meeting_date = last_day_from_range(head.group(1), head.group(2), year)
        if meeting_date is None:
            continue

        meetings.append(
            Meeting(meeting_date, year, minutes_url, statement_url,
                    f"fomchistorical{year}.htm")
        )
    return meetings


def discover(start_year, end_year, session=None):
    """Parse every index page in range and return meetings sorted by date."""
    calendar_html, _ = fetch(CALENDAR_URL, INDEX_CACHE / "fomccalendars.htm", session)
    found = parse_calendar(calendar_html)

    covered = {m.year for m in found}
    for year in range(start_year, end_year + 1):
        if year in covered:
            continue
        url = HISTORICAL_URL.format(year=year)
        try:
            html, _ = fetch(url, INDEX_CACHE / f"fomchistorical{year}.htm", session)
        except requests.HTTPError as exc:
            print(f"  no historical index for {year} ({exc.response.status_code})")
            continue
        found.extend(parse_historical(html, year))

    in_range = [m for m in found if start_year <= m.year <= end_year]
    by_date = {}
    for meeting in in_range:
        # A meeting can appear on both indexes; prefer the record with more links.
        existing = by_date.get(meeting.meeting_date)
        if existing is None or _score(meeting) > _score(existing):
            by_date[meeting.meeting_date] = meeting
    return [by_date[k] for k in sorted(by_date)]


def _score(meeting):
    return bool(meeting.minutes_url) * 2 + bool(meeting.statement_url)


def download(meetings, session=None):
    """Cache the minutes and statement HTML for each meeting."""
    fetched = cached = 0
    for meeting in meetings:
        folder = RAW / meeting.meeting_date
        for kind, url in (("minutes", meeting.minutes_url),
                          ("statement", meeting.statement_url)):
            if not url:
                continue
            _, was_fetched = fetch(url, folder / f"{kind}.html", session)
            fetched += was_fetched
            cached += not was_fetched
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "meta.json").write_text(
            json.dumps(meeting.as_dict(), indent=2), encoding="utf-8"
        )
    return fetched, cached


def print_coverage(meetings):
    """The corpus should be eight meetings a year. Show where it is not."""
    print(f"\nDiscovered {len(meetings)} meetings "
          f"({meetings[0].meeting_date} .. {meetings[-1].meeting_date})\n")
    print(f"  {'year':<6}{'meetings':>9}{'minutes':>9}{'statements':>12}   source")
    print("  " + "-" * 54)
    for year in sorted({m.year for m in meetings}):
        rows = [m for m in meetings if m.year == year]
        source = rows[0].source_index.replace(".htm", "")
        print(f"  {year:<6}{len(rows):>9}{sum(bool(m.minutes_url) for m in rows):>9}"
              f"{sum(bool(m.statement_url) for m in rows):>12}   {source}")

    gaps = [m for m in meetings if not m.minutes_url]
    if gaps:
        print(f"\n  {len(gaps)} meeting(s) without minutes "
              f"(unreleased or notation vote):")
        for m in gaps:
            print(f"    {m.meeting_date}  statement={'yes' if m.statement_url else 'no'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--download", action="store_true",
                        help="fetch documents; without it, only discover and report")
    parser.add_argument("--limit", type=int, default=None,
                        help="download at most N meetings (newest first)")
    args = parser.parse_args()

    with requests.Session() as session:
        meetings = discover(args.start_year, args.end_year, session)
        if not meetings:
            sys.exit("Discovered no meetings -- the index page structure has changed.")
        print_coverage(meetings)

        if not args.download:
            print("\nDiscovery only. Re-run with --download to fetch documents.")
            return

        targets = [m for m in meetings if m.minutes_url or m.statement_url]
        if args.limit:
            # Newest first, but only meetings that actually have documents --
            # scheduled future meetings would otherwise fill the whole limit.
            targets = sorted(targets, key=lambda m: m.meeting_date, reverse=True)[:args.limit]
        print(f"\nDownloading {len(targets)} meeting(s) into {RAW}/ ...")
        fetched, cached = download(targets, session)
        print(f"  {fetched} fetched, {cached} already cached")


if __name__ == "__main__":
    main()
