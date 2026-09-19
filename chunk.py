"""
Parse cached FOMC HTML into section-labelled passages.

Chunks are cut on section boundaries, never on a fixed token window, because
the section label is what tells generation whose view a passage carries -- the
staff's, the participants', or the Committee's.

Every chunk stores `embed_text`, which is the passage prefixed with
"[{meeting_date} | {section}]". No passage is ever embedded without its date
attached; that prefix is the main defence against retrieving the right sentence
from the wrong year.

Usage:
    python chunk.py --inspect            # parse, print histograms, write nothing
    python chunk.py                      # write data/chunks.jsonl
    python chunk.py --meeting 2026-07-29 --show 3
"""

import argparse
import collections
import hashlib
import itertools
import json
import pathlib
import re

from bs4 import BeautifulSoup

RAW = pathlib.Path(__file__).resolve().parent / "data/raw"
CHUNKS_PATH = pathlib.Path(__file__).resolve().parent / "data/chunks.jsonl"

# Canonical sections, in the order the Fed prints them, each with the pattern
# that identifies it. Headings have drifted over ten years and an exact-match
# parser silently folds a renamed section into the one above it -- which is how
# a Committee decision ends up labelled as a staff view. Observed drift in
# 2016-2026: "Committee Policy Action" -> "...Actions", "Staff Review of the
# Financial Situation" -> "Staff Review of Financial Situation", "Participants'
# Views" -> "Participants' View", and "Developments in Financial Markets and
# Open Market Operations" -> "Discussion of Financial Markets and Open Market
# Operations" / "Financial Developments and Open Market Operations".
SECTION_PATTERNS = [
    # 2009-2015 titled this section "...and the Federal Reserve's Balance
    # Sheet"; 2016 on says "...and Open Market Operations". Same section: the
    # Desk manager's report on markets and operations.
    ("Developments in Financial Markets and Open Market Operations",
     re.compile(r"open market operations\b"
                r"|^(?:developments in|discussion of) financial markets"
                r"|^financial developments")),
    ("Staff Review of the Economic Situation",
     re.compile(r"^staff review of (?:the )?economic situation")),
    ("Staff Review of the Financial Situation",
     re.compile(r"^staff review of (?:the )?financial situation")),
    ("Staff Economic Outlook",
     re.compile(r"^staff (?:economic outlook|review of (?:the )?economic outlook)")),
    ("Participants' Views on Current Conditions and the Economic Outlook",
     re.compile(r"^participants'? views?\b")),
    ("Committee Policy Action",
     re.compile(r"^committee policy actions?\b")),
]

PREAMBLE = "Preamble"
STATEMENT_SECTION = "Policy Statement"

MIN_WORDS = {"minutes": 25, "statement": 8}
MAX_WORDS = 400          # long participant paragraphs get split at sentences
SPLIT_TARGET = 250

# Paragraphs are grouped into overlapping windows: consecutive chunks share
# their boundary paragraph, so the last paragraph of one chunk is the first
# paragraph of the next. A claim that straddles a paragraph break therefore
# survives whole inside at least one chunk instead of being cut in half by an
# arbitrary boundary.
#
# WINDOW=1 disables grouping and restores one-chunk-per-paragraph, which is
# what the baseline measurement uses.
WINDOW = 3

# Carrying the tail of the previous chunk costs far less than repeating a whole
# paragraph: one or two sentences instead of ~90 words. Used when overlap is
# wanted in the embedded text itself rather than added at read time.
SENTENCE_OVERLAP = 0

EMPHASIS = ("strong", "b", "em")

# A signature block or rule line is shaped like a heading but is not one.
SEPARATOR = re.compile(r"^[_\s.-]*$")
PERSON = re.compile(r"^[A-Z][a-z]+(?: [A-Z]\.?)* [A-Z][a-z']+(?: Secretary)?$")
# The meeting date sits in the same bold-plus-break shape as a real heading.
DATELINE = re.compile(r"^[A-Z][a-z]+ \d{1,2}(-[A-Za-z]* ?\d{1,2})?, \d{4}$")
# Headings carry footnote markers: "Annual Organizational Matters 5".
FOOTNOTE_MARKER = re.compile(r"\s+\d{1,2}$")
# Vote tallies sit inside Committee Policy Action; they are not sections.
VOTE_LINE = re.compile(r"^voting (for|against) this action")


def normalize(text):
    """Curly quotes and stray whitespace break exact matching."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    # U+2011 non-breaking hyphen shows up inside rate ranges ("1\u20111/2 percent")
    # and breaks any pattern written with an ASCII hyphen.
    text = text.replace("\u2011", "-").replace("\u2012", "-").replace("\u2212", "-")
    text = text.replace("\xa0", " ").replace("\u2009", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", text).strip()


def match_section(text):
    """
    Return the canonical section for a heading, or None.

    Matching is on normalized lowercase text with trailing punctuation removed,
    so "Committee Policy Actions" and "Committee Policy Action:" land on the
    same canonical label.
    """
    candidate = normalize(text).lower().rstrip(":.").strip()
    if not candidate or len(candidate.split()) > 15:
        return None
    for canonical, pattern in SECTION_PATTERNS:
        if pattern.search(candidate):
            return canonical
    return None


def clean_heading(text):
    """Strip the footnote marker the Fed appends to some section titles."""
    return FOOTNOTE_MARKER.sub("", normalize(text)).strip()


def is_noise_heading(text):
    """Signature lines, rule separators, vote tallies and attendee names."""
    stripped = normalize(text).strip("_ ").strip()
    if not stripped or SEPARATOR.match(stripped):
        return True
    if VOTE_LINE.match(stripped.lower()):
        return True
    return bool(PERSON.match(stripped) or DATELINE.match(stripped))


def leading_heading(paragraph):
    """
    Return the heading a paragraph opens with, or None.

    FOMC minutes put the section title *inside* the first paragraph of the
    section:

        <p><strong>Staff Economic Outlook</strong><br/>The projection ...</p>

    so the title and the first body paragraph share one <p>. The <br/> is what
    separates a real heading from an attendee entry, which instead reads
    <p><strong>Jane Doe</strong>, Director, ...</p> with no break. Treating the
    whole <p> as body text -- or only matching standalone headings -- attributes
    every section's opening paragraph to the section above it.
    """
    first = next(
        (c for c in paragraph.children
         if getattr(c, "name", None) or (isinstance(c, str) and c.strip())),
        None,
    )
    if getattr(first, "name", None) not in EMPHASIS:
        return None

    text = normalize(first.get_text(" "))
    if not text or len(text.split()) > 15:
        return None

    following = first.next_sibling
    while isinstance(following, str) and not following.strip():
        following = following.next_sibling
    has_break = getattr(following, "name", None) == "br"

    # A canonical title counts even without the break; anything else needs it.
    if match_section(text) or has_break:
        return text
    return None


def parse_minutes(html):
    """
    Walk the paragraphs in order, tracking which section each falls under.

    A paragraph that opens with a heading switches the section and contributes
    its remaining text as the first body paragraph of that new section.
    Non-canonical headings ("Financial Stability Report", "Selection of
    Committee Officer") become sections under their own name rather than
    bleeding into the canonical section above them.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("div", id="article") or soup

    current = PREAMBLE
    paragraphs = []
    unmatched = []

    for tag in body.find_all(["p", "h2", "h3", "h4", "h5"]):
        text = normalize(tag.get_text(" "))
        if not text:
            continue

        if tag.name != "p":
            section = match_section(text)
            if section:
                current = section
            continue

        heading = leading_heading(tag)
        if heading:
            section = match_section(heading)
            if section:
                current = section
            elif is_noise_heading(heading):
                heading = None          # a signature line; keep the section as is
            else:
                current = clean_heading(heading)   # a special topic keeps its own label
                unmatched.append(current)
            if heading:
                text = text[len(heading):].lstrip(" :.-").strip()

        if len(text.split()) < MIN_WORDS["minutes"]:
            continue
        paragraphs.append({"section": current, "text": text})

    return paragraphs, unmatched


def parse_statement(html):
    """Statements have no sections; the whole document is the Committee's."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("div", id="article") or soup

    paragraphs = []
    seen = set()
    for tag in body.find_all("p"):
        text = normalize(tag.get_text(" "))
        if not text or text in seen:
            continue
        seen.add(text)
        if len(text.split()) < MIN_WORDS["statement"]:
            continue
        # Trailing boilerplate links, not policy content.
        if text.lower().startswith(("for media inquiries", "last update", "share")):
            continue
        paragraphs.append({"section": STATEMENT_SECTION, "text": text})
    return paragraphs


def split_long(text, max_words=MAX_WORDS, target=SPLIT_TARGET):
    """
    Split an over-long paragraph on sentence boundaries.

    Participants' Views paragraphs sometimes run past 400 words and cover
    several distinct arguments; one vector for all of them retrieves poorly.
    Splitting on sentences keeps quantifiers ("a couple", "several") attached
    to the clause they govern.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences = re.split(r"(?<=[.!?]) +", text)
    parts, current = [], []
    for sentence in sentences:
        current.append(sentence)
        if len(" ".join(current).split()) >= target:
            parts.append(" ".join(current))
            current = []
    if current:
        # Don't strand a fragment; fold it into the previous part.
        tail = " ".join(current)
        if parts and len(tail.split()) < 40:
            parts[-1] = parts[-1] + " " + tail
        else:
            parts.append(tail)
    return parts


def content_hash(meeting_date, doc_type, section, text):
    """Stable identity for a passage. Ingestion keys on this to stay idempotent."""
    payload = f"{meeting_date}|{doc_type}|{section}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def context_line(meeting_date, doc_type, section):
    """
    A sentence of context prepended before embedding.

    The bare "[2025-09-17 | Section]" tag carries the date, but as digits that
    a question phrased "in September 2025" does not match lexically and matches
    only weakly in vector space. Spelling the month out, and naming the document,
    puts the question's own vocabulary into the passage.
    """
    year, month, _ = meeting_date.split("-")
    month_name = MONTH_NAMES[int(month) - 1]
    kind = ("minutes of the Federal Open Market Committee meeting"
            if doc_type == "minutes" else "FOMC policy statement")
    return (f"This passage is from the {kind} of {month_name} {year} "
            f"({meeting_date}). Section: {section}.")


def tail_sentences(text, count):
    """The last `count` sentences of a passage, for cheap overlap."""
    if count <= 0 or not text:
        return ""
    sentences = re.split(r"(?<=[.!?]) +", text.strip())
    return " ".join(sentences[-count:])


def windows(units, window=WINDOW):
    """
    Group consecutive units into overlapping windows.

    Stride is window-1, so neighbouring windows share exactly one unit: the
    last of a chunk is the first of the next. Yields lists of units.

    A run shorter than the window becomes a single chunk rather than being
    padded, and the final window is not emitted when its units are already
    wholly contained in the previous one -- otherwise the tail of every section
    would be duplicated with nothing new added.
    """
    if window <= 1 or len(units) <= 1:
        return [[unit] for unit in units]

    stride = window - 1
    groups = []
    start = 0
    while start < len(units):
        group = units[start:start + window]
        if groups and set(id(u) for u in group) <= set(id(u) for u in groups[-1]):
            break
        groups.append(group)
        if start + window >= len(units):
            break
        start += stride
    return groups


def build_chunks(paragraphs, meeting_date, doc_type, source_url, window=WINDOW,
                 sentence_overlap=SENTENCE_OVERLAP, context=False):
    """
    The date and section are prepended to the embedded text on purpose: the
    embedding model must never see a passage without knowing when it was said.

    Windows are cut inside a single section, never across one. The section
    label is what identifies whose view a passage carries, and a chunk that
    spanned "Staff Economic Outlook" into "Participants' Views" would carry two
    voices under one label -- the misattribution the domain rules forbid.
    """
    # Long paragraphs are split first, so a window groups comparable units.
    units = []
    for order, para in enumerate(paragraphs):
        for part in split_long(para["text"]):
            units.append({"section": para["section"], "text": part,
                          "paragraph": order})

    chunks = []
    for section, group in itertools.groupby(units, key=lambda u: u["section"]):
        run = list(group)
        previous_tail = ""
        for members in windows(run, window):
            body = " ".join(u["text"] for u in members)
            # The carried tail is prepended to the embedded text but kept out of
            # `text`, so a citation still quotes only what the passage says.
            text = body
            header = f"[{meeting_date} | {section}]"
            if context:
                header = context_line(meeting_date, doc_type, section)
            embedded = f"{header} {previous_tail} {body}".replace("  ", " ").strip()
            previous_tail = tail_sentences(body, sentence_overlap)
            digest = content_hash(meeting_date, doc_type, section, text)
            chunks.append({
                "chunk_id": f"{meeting_date}:{doc_type}:{digest}",
                "content_hash": digest,
                "meeting_date": meeting_date,
                "year": int(meeting_date[:4]),
                "doc_type": doc_type,
                "section": section,
                "text": text,
                "embed_text": embedded,
                "source_url": source_url,
                "word_count": len(text.split()),
                "paragraphs": sorted({u["paragraph"] for u in members}),
                "units": len(members),
            })
    return chunks


def load_meeting(folder, window=WINDOW, sentence_overlap=SENTENCE_OVERLAP,
                 context=False):
    """Parse one cached meeting folder into chunks."""
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meeting_date = folder.name

    chunks, unmatched = [], []
    minutes_path = folder / "minutes.html"
    if minutes_path.exists():
        paragraphs, unmatched = parse_minutes(minutes_path.read_text(encoding="utf-8"))
        chunks += build_chunks(paragraphs, meeting_date, "minutes",
                               meta.get("minutes_url", ""), window=window,
                               sentence_overlap=sentence_overlap, context=context)

    statement_path = folder / "statement.html"
    if statement_path.exists():
        paragraphs = parse_statement(statement_path.read_text(encoding="utf-8"))
        chunks += build_chunks(paragraphs, meeting_date, "statement",
                               meta.get("statement_url", ""), window=window,
                               sentence_overlap=sentence_overlap, context=context)
    return chunks, unmatched


def iter_meetings(only=None):
    for folder in sorted(RAW.iterdir()):
        if not folder.is_dir() or folder.name == "index":
            continue
        if only and folder.name != only:
            continue
        yield folder


def report(all_chunks, unmatched_by_meeting, per_meeting):
    """Print enough for a human to see whether the parse actually worked."""
    print(f"\n{len(all_chunks)} chunks from {len(per_meeting)} meetings\n")

    print("  chunks by section")
    print("  " + "-" * 66)
    counts = collections.Counter(c["section"] for c in all_chunks)
    order = [PREAMBLE] + [name for name, _ in SECTION_PATTERNS] + [STATEMENT_SECTION]
    for name in order:
        if counts.get(name):
            words = [c["word_count"] for c in all_chunks if c["section"] == name]
            print(f"  {counts[name]:5d}  {name[:46]:<46} avg {sum(words)//len(words):4d}w")
    for name, count in counts.items():
        if name not in order:
            print(f"  {count:5d}  {name[:46]:<46} (UNRECOGNISED)")

    print("\n  section coverage per meeting (minutes only)")
    print("  " + "-" * 66)
    missing = []
    for date in sorted(per_meeting):
        sections = per_meeting[date]
        absent = [n for n, _ in SECTION_PATTERNS if n not in sections]
        if absent:
            missing.append((date, absent))
    if missing:
        print(f"  {len(missing)} of {len(per_meeting)} meetings are missing a standard section:")
        for date, absent in missing[:15]:
            print(f"    {date}  missing: {', '.join(s[:34] for s in absent)}")
        if len(missing) > 15:
            print(f"    ... and {len(missing) - 15} more")
    else:
        print("  all meetings have all six standard sections")

    leftovers = collections.Counter()
    for items in unmatched_by_meeting.values():
        leftovers.update(items)
    if leftovers:
        print("\n  heading-shaped lines that matched no known section (top 12)")
        print("  " + "-" * 66)
        for text, count in leftovers.most_common(12):
            print(f"  {count:5d}  {text[:60]}")

    lengths = sorted(c["word_count"] for c in all_chunks)
    if lengths:
        mid = lengths[len(lengths) // 2]
        print(f"\n  words per chunk: min {lengths[0]}  median {mid}  max {lengths[-1]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true",
                        help="parse and report without writing chunks.jsonl")
    parser.add_argument("--meeting", help="restrict to one meeting date")
    parser.add_argument("--show", type=int, default=0, help="print N sample chunks")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help="paragraphs per chunk; consecutive chunks share one. "
                             "1 disables grouping (one chunk per paragraph)")
    parser.add_argument("--out", default=str(CHUNKS_PATH),
                        help="where to write the chunks")
    parser.add_argument("--sentence-overlap", type=int, default=SENTENCE_OVERLAP,
                        help="carry this many sentences from the previous chunk "
                             "into the embedded text (cheaper than repeating a "
                             "whole paragraph)")
    parser.add_argument("--context", action="store_true",
                        help="prepend a spelled-out context sentence before embedding")
    args = parser.parse_args()

    all_chunks = []
    unmatched_by_meeting = {}
    per_meeting = {}

    for folder in iter_meetings(args.meeting):
        chunks, unmatched = load_meeting(
            folder, window=args.window,
            sentence_overlap=args.sentence_overlap, context=args.context)
        if not chunks:
            continue
        all_chunks += chunks
        if unmatched:
            unmatched_by_meeting[folder.name] = unmatched
        per_meeting[folder.name] = {
            c["section"] for c in chunks if c["doc_type"] == "minutes"
        }

    if not all_chunks:
        raise SystemExit("No chunks parsed. Run scrape.py --download first.")

    report(all_chunks, unmatched_by_meeting, per_meeting)

    if args.show:
        print("\n  sample chunks")
        print("  " + "-" * 66)
        step = max(1, len(all_chunks) // args.show)
        for chunk in all_chunks[::step][:args.show]:
            print(f"\n  {chunk['chunk_id']}  ({chunk['word_count']}w)")
            print(f"  embed_text: {chunk['embed_text'][:220]}...")

    if args.inspect:
        print("\nInspect only. Nothing written.")
        return

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    unique = len({c["content_hash"] for c in all_chunks})
    print(f"\nWrote {len(all_chunks)} chunks ({unique} unique hashes) -> {out_path}")
    if unique != len(all_chunks):
        print(f"  note: {len(all_chunks) - unique} duplicate passages collapse on ingest")


if __name__ == "__main__":
    main()
