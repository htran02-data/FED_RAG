"""
Single-document RAG spike over one set of FOMC minutes.

Purpose: prove the whole loop works end to end before building any pipeline.
No database, no vector store, no web app. Everything lives in memory.

Setup:
    pip install requests beautifulsoup4 numpy voyageai anthropic python-dotenv

Then create a .env file next to this script:
    ANTHROPIC_API_KEY=sk-ant-...
    VOYAGE_API_KEY=pa-...

Then:
    python spike.py
"""

import os
import re
import pathlib

import requests
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import voyageai
import anthropic

load_dotenv()

URL = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20250917.htm"
MEETING_DATE = "2025-09-17"
CACHE = pathlib.Path("cache_20250917.html")

SECTION_HEADINGS = [
    "Developments in Financial Markets and Open Market Operations",
    "Staff Review of the Economic Situation",
    "Staff Review of the Financial Situation",
    "Staff Economic Outlook",
    "Participants' Views on Current Conditions and the Economic Outlook",
    "Committee Policy Action",
]


def normalize(text):
    """Curly quotes and stray whitespace break exact matching."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def fetch(url, cache_path):
    """Fetch once, then never hit the Fed's servers again."""
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    response = requests.get(url, headers={"User-Agent": "fomc-rag-spike/0.1"}, timeout=30)
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def parse_sections(raw_html):
    """
    Walk the paragraphs in order. A paragraph that exactly matches a known
    heading switches the current section; everything else is body text.

    Matching on heading text rather than HTML tags survives the markup
    changes the Fed has made over the years.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    body = soup.find("div", id="article") or soup
    headings = {normalize(h) for h in SECTION_HEADINGS}

    current = "Preamble"
    paragraphs = []
    for tag in body.find_all(["p", "h2", "h3", "h4", "h5", "strong"]):
        text = normalize(tag.get_text())
        if not text:
            continue
        if text in headings:
            current = text
            continue
        if len(text.split()) < 25:  # skip footnote markers and stray fragments
            continue
        paragraphs.append({"section": current, "text": text})
    return paragraphs


def build_chunks(paragraphs, meeting_date):
    """
    One paragraph per chunk. FOMC paragraphs are already coherent units,
    so this is a better starting point than fixed token windows.

    The date and section are prepended to the embedded text on purpose:
    the model must never see a passage without knowing when it was said.
    """
    chunks = []
    for i, para in enumerate(paragraphs):
        header = f"[{meeting_date} | {para['section']}]"
        chunks.append(
            {
                "id": f"{meeting_date}#{i}",
                "meeting_date": meeting_date,
                "section": para["section"],
                "text": para["text"],
                "embed_text": f"{header} {para['text']}",
            }
        )
    return chunks


def embed(texts, input_type):
    client = voyageai.Client()
    out = []
    for i in range(0, len(texts), 128):
        batch = client.embed(texts[i : i + 128], model="voyage-3-large", input_type=input_type)
        out.extend(batch.embeddings)
    matrix = np.array(out, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def search(question, chunks, matrix, k=6):
    """Brute-force cosine similarity. At this scale it is instant."""
    query_vector = embed([question], input_type="query")[0]
    scores = matrix @ query_vector
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


def answer(question, hits):
    passages = "\n\n".join(
        f"[S{n}] ({hit['meeting_date']}, {hit['section']})\n{hit['text']}"
        for n, (hit, _) in enumerate(hits, start=1)
    )
    prompt = f"""Answer the question using only the passages below.

Rules:
- Cite the passage for every claim, like [S1] or [S2, S4].
- Preserve the FOMC's own quantifiers exactly. "Several participants" and
  "most participants" are not interchangeable.
- Never attribute a staff forecast to the Committee, or vice versa. The
  section label on each passage tells you whose view it is.
- If the passages do not answer the question, say so plainly.

Passages:
{passages}

Question: {question}"""

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


if __name__ == "__main__":
    for key in ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"{key} is not set")

    paragraphs = parse_sections(fetch(URL, CACHE))
    chunks = build_chunks(paragraphs, MEETING_DATE)
    print(f"Parsed {len(chunks)} chunks")
    for section in dict.fromkeys(c["section"] for c in chunks):
        count = sum(1 for c in chunks if c["section"] == section)
        print(f"  {count:3d}  {section}")

    if len(chunks) < 20:
        raise SystemExit("\nToo few chunks. The parser missed the body -- inspect the cached HTML.")

    matrix = embed([c["embed_text"] for c in chunks], input_type="document")

    question = "What did participants say about risks to the labor market?"
    hits = search(question, chunks, matrix)

    print(f"\nQ: {question}\n")
    for n, (hit, score) in enumerate(hits, start=1):
        print(f"  [S{n}] {score:.3f}  {hit['section']}: {hit['text'][:90]}...")
    print(f"\n{answer(question, hits)}")
