"""
Ask the Fed -- a Streamlit front end over the retrieval loop in ask.py.

The UI deliberately shows its work: the date window parsed out of the question,
how many chunks survived that filter, and every passage behind the answer. If
retrieval goes wrong, the reason should be visible without opening a terminal.

Usage:
    streamlit run app.py
"""

import os
import pathlib

import numpy as np
import streamlit as st

import ask
import rates

SECTIONS = [
    "Developments in Financial Markets and Open Market Operations",
    "Staff Review of the Economic Situation",
    "Staff Review of the Financial Situation",
    "Staff Economic Outlook",
    "Participants' Views on Current Conditions and the Economic Outlook",
    "Committee Policy Action",
    "Policy Statement",
]

EXAMPLES = [
    "What did the Fed say about labor markets in 2025?",
    "How did participants describe tariffs and prices in 2018?",
    "What worried participants during the hiking cycle?",
    "How has the staff's view of inflation risk changed since September 2024?",
    "Who dissented in 2024 and why?",
]


st.set_page_config(page_title="Ask the Fed", page_icon="🏛️", layout="wide")


@st.cache_resource
def load_store():
    """Open the store once per session, not once per question."""
    if not ask.DB_PATH.exists() or not ask.VECTORS_PATH.exists():
        return None, None
    conn = ask.connect()
    conn.execute("PRAGMA query_only = ON")
    return conn, np.load(ask.VECTORS_PATH)


@st.cache_data
def load_eras():
    if not pathlib.Path("data/chunks.jsonl").exists():
        return {}
    return rates.named_eras(rates.load_path())


@st.cache_data
def corpus_summary(_conn):
    row = _conn.execute(
        "SELECT COUNT(*) AS chunks, COUNT(DISTINCT meeting_date) AS meetings, "
        "MIN(meeting_date) AS first, MAX(meeting_date) AS last FROM chunks"
    ).fetchone()
    return dict(row)


def render_sources(hits):
    st.markdown("**Sources**")
    seen = set()
    for hit in hits:
        key = (hit["meeting_date"], hit["doc_type"])
        if key in seen:
            continue
        seen.add(key)
        st.markdown(
            f"- [{hit['meeting_date']} {hit['doc_type']}]({hit['source_url']})"
        )


def render_passages(hits):
    for hit in hits:
        header = (f"[{hit['label']}]  {hit['meeting_date']}  ·  {hit['section']}  "
                  f"·  {hit['score']:.3f}")
        with st.expander(header):
            st.write(hit["text"])
            st.caption(f"[{hit['doc_type']} on federalreserve.gov]({hit['source_url']})")


conn, matrix = load_store()

st.title("Ask the Fed")
st.caption("Retrieval-augmented question answering over FOMC minutes and "
           "statements, with citations back to federalreserve.gov.")

if conn is None:
    st.error(
        "No store found. Build it first:\n\n"
        "```\npython scrape.py --download\npython chunk.py\npython embed.py\n```"
    )
    st.stop()

summary = corpus_summary(conn)
era_names = load_eras()

with st.sidebar:
    st.header("Retrieval")
    st.metric("Chunks", f"{summary['chunks']:,}")
    st.metric("Meetings", summary["meetings"])
    st.caption(f"{summary['first']} to {summary['last']}")

    k = st.slider("Passages retrieved", 3, 20, ask.DEFAULT_K)
    use_filter = st.toggle(
        "Filter by date before searching", value=True,
        help="Parses a period out of the question and applies it in SQL before "
             "the vector search. Turning this off is the quickest way to see "
             "why it exists.",
    )
    section = st.selectbox("Section", ["Any"] + SECTIONS)
    doc_type = st.selectbox("Document", ["Any", "minutes", "statement"])

    st.divider()
    st.caption("Policy eras, derived from the target ranges in the statements:")
    for label in ("hiking cycle", "cutting cycle"):
        if label in era_names:
            start, end = era_names[label]
            st.caption(f"· {label}: {start} to {end}")

question = st.text_input("Question", placeholder=EXAMPLES[0])

columns = st.columns(len(EXAMPLES))
for column, example in zip(columns, EXAMPLES):
    if column.button(example[:26] + "…", help=example, use_container_width=True):
        question = example

if not question:
    st.info("Ask a question, or pick one of the examples above.")
    st.stop()

time_filter = ask.parse_temporal(question, era_names=era_names) if use_filter else None

with st.spinner("Retrieving…"):
    hits, diagnostics = ask.search(
        question, conn, matrix, k=k, time_filter=time_filter,
        section=None if section == "Any" else section,
        doc_type=None if doc_type == "Any" else doc_type,
    )

left, right = st.columns(2)
left.metric(
    "Date filter",
    time_filter.label if time_filter else "none",
    help=(f"{time_filter.start} to {time_filter.end} (via {time_filter.source})"
          if time_filter else "No period was stated, so the whole corpus was searched."),
)
right.metric("Candidates searched", f"{diagnostics['candidates']:,}",
             delta=f"of {diagnostics['corpus']:,} chunks", delta_color="off")

if not hits:
    st.warning("No passages matched those constraints. Try widening the period "
               "or clearing the section filter.")
    st.stop()

st.subheader("Retrieved passages")
render_passages(hits)

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.warning("ANTHROPIC_API_KEY is not set, so no answer was generated. "
               "The retrieved passages above are the retrieval half of the loop.")
    render_sources(hits)
    st.stop()

st.subheader("Answer")
with st.spinner("Generating…"):
    try:
        st.markdown(ask.answer(question, hits))
    except Exception as error:                      # surfaced, not swallowed
        st.error(f"Generation failed: {error}")
        st.stop()

render_sources(hits)
st.caption("Every claim above should carry a citation like [S1] pointing at one "
           "of the passages listed. If it does not, treat it as unsupported.")
