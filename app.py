"""Browser interface sharing the same pipeline as `./fed`."""
import streamlit as st
from fedrag.service import Engine, Options

st.set_page_config(page_title="Ask the Fed", page_icon="🏛️", layout="wide")
st.title("Ask the Fed")
st.caption("Explore FOMC communications with dated, attributed source passages.")

try:
    engine = Engine()
except Exception as error:
    st.error(f"Cannot open the corpus: {error}")
    st.stop()

try:
    summary = engine.summary
    with st.sidebar:
        st.header("Your corpus")
        st.write(f"{summary['passages']:,} passages · {summary['meetings']} meetings")
        st.caption(f"{summary['first']} to {summary['last']}")
        mode = st.selectbox("Search", ["auto", "offline", "semantic"])
        answer_mode = st.selectbox("Answer", ["quotes", "generated", "off"])
        st.caption("Quotes preserve original wording. Offline search with quoted answers needs no API keys.")
        k = st.slider("Passages", 1, 30, 8)
        expand = st.slider("Neighboring paragraphs", 0, 4, 1)
        sections = [r[0] for r in engine.conn.execute('SELECT DISTINCT section FROM chunks ORDER BY section')]
        section = st.selectbox("Speaker / section", ["Any"] + sections)
        doc = st.selectbox("Document", ["Any", "minutes", "statement"])
        use_filter = st.checkbox("Use dates mentioned in the question", True)
        rerank = st.checkbox("API reranking", False)

    with st.form("question_form"):
        question = st.text_input("Your question", placeholder="What did participants say about tariffs in 2018?")
        st.caption("Include a date or period. Each question is independent of previous questions.")
        submitted = st.form_submit_button("Ask the Fed", type="primary")

    if submitted:
        st.session_state.pop('result', None)
        with st.spinner("Finding evidence…"):
            try:
                options = Options(k=k, expand=expand, section=None if section == 'Any' else section,
                                  doc_type=None if doc == 'Any' else doc, use_filter=use_filter,
                                  rerank=rerank, retrieval=mode, answer_mode=answer_mode)
                st.session_state['result'] = engine.query(question, options)
            except (Exception, SystemExit) as error:
                st.error(f"Cannot answer: {error}")

    result = st.session_state.get('result')
    if result is not None:
        st.subheader(result.question)
        d = result.diagnostics
        window = d.get('time_filter')
        st.caption(f"{result.retrieval} search · {result.answer_mode} answer · "
                   f"{d['candidates']:,} of {d['corpus']:,} passages · "
                   + (f"{window.start} to {window.end}" if window else "all meetings"))
        st.markdown(result.answer)
        st.download_button("Download answer and sources", result.markdown(),
                           file_name="fed-answer.md", mime="text/markdown")
        st.subheader("Evidence")
        for hit in result.hits:
            with st.expander(f"[{hit['label']}] {hit['meeting_date']} | {hit['section']}"):
                st.write(hit['text'])
                st.markdown(f"[Read the {hit['doc_type']} at the Federal Reserve]({hit['source_url']})")
    else:
        st.info("Ask about inflation, employment, policy decisions, or the balance sheet within the corpus dates.")
finally:
    engine.close()
