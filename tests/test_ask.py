"""
Temporal parsing and retrieval tests.

The dominant failure mode of this corpus is retrieving the right sentence from
the wrong year, so the date filter is the part most worth pinning down.
"""

import datetime as dt
import sqlite3

import numpy as np
import pytest

import ask
import embed as E

TODAY = dt.date(2026, 9, 2)

# Derived from the corpus by rates.py, not assumed here.
ERAS = {
    "hiking cycle": ("2022-03-16", "2023-07-26"),
    "tightening cycle": ("2022-03-16", "2023-07-26"),
    "cutting cycle": ("2024-09-18", "2025-12-10"),
    "easing cycle": ("2024-09-18", "2025-12-10"),
}


def parse(question):
    return ask.parse_temporal(question, today=TODAY, era_names=ERAS)


class TestExplicitYears:
    def test_bare_year(self):
        window = parse("What did the Fed say about labor markets in 2025?")
        assert (window.start, window.end) == ("2025-01-01", "2025-12-31")

    def test_year_anywhere_in_the_sentence(self):
        assert parse("2016 discussion of the balance sheet").start == "2016-01-01"

    def test_two_years_span_both(self):
        window = parse("How did the view change from 2019 to 2021?")
        assert (window.start, window.end) == ("2019-01-01", "2021-12-31")

    def test_between_form(self):
        window = parse("inflation risks between 2019 and 2021")
        assert (window.start, window.end) == ("2019-01-01", "2021-12-31")

    def test_reversed_range_is_ordered(self):
        window = parse("between 2021 and 2019")
        assert (window.start, window.end) == ("2019-01-01", "2021-12-31")


class TestMonths:
    def test_month_and_year(self):
        window = parse("What happened at the September 2025 meeting?")
        assert (window.start, window.end) == ("2025-09-01", "2025-09-30")

    def test_february_in_a_leap_year(self):
        assert parse("February 2020 minutes").end == "2020-02-29"

    def test_february_in_a_common_year(self):
        assert parse("February 2019 minutes").end == "2019-02-28"

    def test_since_month_and_year_is_open_ended(self):
        window = parse("What has changed since September 2024?")
        assert window.start == "2024-09-01"
        assert window.end == TODAY.isoformat()

    def test_bare_since_month_picks_the_most_recent_one(self):
        # Asked on 2026-09-02, "since March" means March 2026.
        window = parse("How have views shifted since March?")
        assert window.start == "2026-03-01"

    def test_bare_since_month_rolls_back_a_year_when_not_yet_reached(self):
        # December has not happened yet in 2026, so it means December 2025.
        window = parse("How have views shifted since December?")
        assert window.start == "2025-12-01"


class TestOpenEndedBounds:
    def test_since_year(self):
        window = parse("What has the Committee said about tariffs since 2022?")
        assert window.start == "2022-01-01"
        assert window.end == TODAY.isoformat()

    def test_before_year_excludes_that_year(self):
        window = parse("What did they say about the balance sheet before 2020?")
        assert window.end == "2019-12-31"

    def test_prior_to_year(self):
        assert parse("views prior to 2018").end == "2017-12-31"


class TestRelativeWindows:
    def test_this_year(self):
        window = parse("What has the Committee said this year?")
        assert (window.start, window.end) == ("2026-01-01", "2026-12-31")

    def test_last_year(self):
        window = parse("What did the Committee say last year?")
        assert (window.start, window.end) == ("2025-01-01", "2025-12-31")

    def test_last_n_months(self):
        window = parse("labor market commentary in the last 6 months")
        assert window.start == (TODAY - dt.timedelta(days=180)).isoformat()

    def test_past_n_years(self):
        window = parse("how has the language changed over the past 2 years")
        assert window.start == (TODAY - dt.timedelta(days=730)).isoformat()


class TestPolicyEras:
    """Era bounds come from rates.py, which parses them out of the statements."""

    def test_hiking_cycle(self):
        window = parse("What worried participants during the hiking cycle?")
        assert (window.start, window.end) == ("2022-03-16", "2023-07-26")
        assert window.source == "policy-era"

    def test_tightening_is_a_synonym(self):
        assert parse("risks during the tightening cycle").start == "2022-03-16"

    def test_cutting_cycle(self):
        assert parse("dissents during the cutting cycle").start == "2024-09-18"

    def test_era_wins_over_a_stray_year_in_the_same_question(self):
        # The phrase is more specific than an incidental year mention.
        window = parse("Compared with 2019, what changed in the hiking cycle?")
        assert window.source == "policy-era"


class TestNoConstraint:
    @pytest.mark.parametrize("question", [
        "What does the Committee think about inflation?",
        "How does the staff describe financial conditions?",
        "Who dissented and why?",
    ])
    def test_returns_none(self, question):
        assert parse(question) is None

    def test_none_means_search_everything(self, store):
        conn, _ = store
        assert len(ask.candidates(conn, None)) == 6


@pytest.fixture
def store(tmp_path):
    """A small store whose vectors are deliberately meaningless."""
    conn = E.connect(":memory:")
    rows = [
        ("h1", "c1", "2019-06-19", 2019, "minutes", "Staff Economic Outlook",
         "The staff projected moderate growth.", "[..] staff", "u1", 5, 0),
        ("h2", "c2", "2022-06-15", 2022, "minutes",
         "Participants' Views on Current Conditions and the Economic Outlook",
         "Several participants judged inflation risks to the upside.",
         "[..] participants", "u2", 8, 1),
        ("h3", "c3", "2023-07-26", 2023, "minutes", "Committee Policy Action",
         "Members agreed to raise the target range.", "[..] members", "u3", 7, 2),
        ("h4", "c4", "2025-09-17", 2025, "minutes",
         "Participants' Views on Current Conditions and the Economic Outlook",
         "A few participants noted labor market softening.", "[..] a few", "u4", 7, 3),
        ("h5", "c5", "2025-09-17", 2025, "statement", "Policy Statement",
         "The Committee decided to lower the target range.", "[..] stmt", "u5", 8, 4),
        ("h6", "c6", "2026-07-29", 2026, "minutes", "Staff Economic Outlook",
         "The staff revised the projection down.", "[..] staff 2026", "u6", 6, 5),
    ]
    with conn:
        conn.executemany(
            "INSERT INTO chunks (content_hash, chunk_id, meeting_date, year, "
            "doc_type, section, text, embed_text, source_url, word_count, "
            "vector_index) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    matrix = np.eye(6, dtype=np.float32)
    return conn, matrix


def stub_embedder(index):
    """Return a query vector that scores row `index` highest."""
    def embed(_text):
        vector = np.zeros(6, dtype=np.float32)
        vector[index] = 1.0
        return vector
    return embed


class TestCandidateFiltering:
    def test_year_filter_narrows_the_pool(self, store):
        conn, _ = store
        window = ask.TimeFilter("2025-01-01", "2025-12-31", "2025", "year")
        rows = ask.candidates(conn, window)
        assert {r["meeting_date"] for r in rows} == {"2025-09-17"}

    def test_section_filter(self, store):
        conn, _ = store
        rows = ask.candidates(conn, None, section="Committee Policy Action")
        assert [r["content_hash"] for r in rows] == ["h3"]

    def test_doc_type_filter(self, store):
        conn, _ = store
        rows = ask.candidates(conn, None, doc_type="statement")
        assert [r["content_hash"] for r in rows] == ["h5"]

    def test_filters_combine(self, store):
        conn, _ = store
        window = ask.TimeFilter("2025-01-01", "2025-12-31", "2025", "year")
        rows = ask.candidates(conn, window, doc_type="minutes")
        assert [r["content_hash"] for r in rows] == ["h4"]


class TestSearch:
    def test_date_filter_excludes_the_best_global_match(self, store):
        """
        The whole point of filtering first: row 5 (2026) is the strongest match
        for this query vector, but a 2019 question must never surface it.
        """
        conn, matrix = store
        window = ask.TimeFilter("2019-01-01", "2019-12-31", "2019", "year")
        hits, diag = ask.search("staff projection", conn, matrix, k=3,
                                time_filter=window, embedder=stub_embedder(5))
        assert diag["candidates"] == 1
        assert [h["meeting_date"] for h in hits] == ["2019-06-19"]

    def test_without_the_filter_the_wrong_year_wins(self, store):
        # Same query, no date constraint -- this is the failure being defended
        # against, so it is worth asserting that it really does happen.
        conn, matrix = store
        hits, _ = ask.search("staff projection", conn, matrix, k=1,
                             embedder=stub_embedder(5))
        assert hits[0]["meeting_date"] == "2026-07-29"

    def test_hits_are_ranked_and_labelled(self, store):
        conn, matrix = store
        hits, _ = ask.search("q", conn, matrix, k=3, embedder=stub_embedder(1))
        assert [h["label"] for h in hits] == ["S1", "S2", "S3"]
        assert hits[0]["content_hash"] == "h2"
        assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]

    def test_empty_candidate_set_returns_no_hits(self, store):
        conn, matrix = store
        window = ask.TimeFilter("1999-01-01", "1999-12-31", "1999", "year")
        hits, diag = ask.search("q", conn, matrix, time_filter=window,
                                embedder=stub_embedder(0))
        assert hits == []
        assert diag["candidates"] == 0

    def test_diagnostics_report_which_meetings_were_hit(self, store):
        conn, matrix = store
        _, diag = ask.search("q", conn, matrix, k=2, embedder=stub_embedder(3))
        assert diag["meetings_hit"] == sorted(set(diag["meetings_hit"]))


class TestPassageFormatting:
    def test_every_passage_shows_its_date_and_section(self, store):
        conn, matrix = store
        hits, _ = ask.search("q", conn, matrix, k=3, embedder=stub_embedder(1))
        rendered = ask.format_passages(hits)
        for hit in hits:
            assert hit["meeting_date"] in rendered
            assert hit["section"] in rendered

    def test_labels_line_up_with_citations(self, store):
        conn, matrix = store
        hits, _ = ask.search("q", conn, matrix, k=2, embedder=stub_embedder(0))
        rendered = ask.format_passages(hits)
        assert "[S1]" in rendered and "[S2]" in rendered

    def test_sources_are_deduplicated_per_document(self, store):
        hits = [
            {"meeting_date": "2025-09-17", "doc_type": "minutes", "source_url": "u"},
            {"meeting_date": "2025-09-17", "doc_type": "minutes", "source_url": "u"},
            {"meeting_date": "2025-09-17", "doc_type": "statement", "source_url": "v"},
        ]
        assert len(ask.sources(hits)) == 2


class TestParseDocType:
    """
    The minutes' Committee Policy Action section reproduces the policy statement
    almost verbatim, so a question naming the statement otherwise retrieves the
    minutes' copy of it first.
    """

    @pytest.mark.parametrize("question,expected", [
        ("According to the December 2020 statement, what held down inflation?", "statement"),
        ("What did the May 2022 statement say about the first quarter?", "statement"),
        ("What do the minutes say about the labor market?", "minutes"),
        ("In the September 2019 minutes, who dissented?", "minutes"),
    ])
    def test_named_document_is_detected(self, question, expected):
        assert ask.parse_doc_type(question) == expected

    @pytest.mark.parametrize("question", [
        "What did participants say about inflation in 2025?",
        "How did the staff describe financial conditions?",
    ])
    def test_no_document_named(self, question):
        assert ask.parse_doc_type(question) is None

    def test_longer_run_goals_does_not_trigger_the_filter(self):
        # A different document, deliberately excluded from this corpus.
        question = "What is the Statement on Longer-Run Goals and Monetary Policy Strategy?"
        assert ask.parse_doc_type(question) is None


class TestFusion:
    def test_dense_only_when_there_is_no_lexical_match(self):
        fused = ask.fuse([2, 0, 1], {}, ["a", "b", "c"])
        assert max(fused, key=lambda i: fused[i]) == 2

    def test_lexical_evidence_can_lift_a_passage(self):
        # Index 1 is last by embedding but first lexically.
        fused = ask.fuse([0, 2, 1], {"b": 1}, ["a", "b", "c"], lexical_weight=5.0)
        assert max(fused, key=lambda i: fused[i]) == 1

    def test_lexical_weight_of_zero_changes_nothing(self):
        order = [2, 0, 1]
        plain = ask.fuse(order, {}, ["a", "b", "c"])
        weighted = ask.fuse(order, {"a": 1}, ["a", "b", "c"], lexical_weight=0.0)
        assert plain == weighted


class TestFtsTerms:
    def test_stopwords_and_short_words_are_dropped(self):
        assert "the" not in ask.fts_terms("What did the Fed say about it")
        assert "did" not in ask.fts_terms("What did the Fed say")

    def test_content_words_survive(self):
        terms = ask.fts_terms("What did many participants cite about tariffs in 2018?")
        assert "participants" in terms and "tariffs" in terms

    def test_corpus_boilerplate_is_dropped(self):
        # These appear in nearly every passage, so they carry no signal.
        for word in ("committee", "federal", "reserve", "minutes"):
            assert word not in ask.fts_terms(f"What did the {word} discuss")


class TestExpandHits:
    """
    Read-time expansion is the alternative to embedding overlapping copies:
    retrieve one precise paragraph, then widen it once it has won.
    """

    @pytest.fixture
    def store(self):
        conn = E.connect(":memory:")
        rows = [
            # one section, four paragraphs in order
            ("h0", "c0", "2025-09-17", 2025, "minutes", "Participants' Views",
             "Para zero.", "e", "u", 2, 0, 0),
            ("h1", "c1", "2025-09-17", 2025, "minutes", "Participants' Views",
             "Para one.", "e", "u", 2, 1, 1),
            ("h2", "c2", "2025-09-17", 2025, "minutes", "Participants' Views",
             "Para two.", "e", "u", 2, 2, 2),
            # a different section in the same meeting -- must never be pulled in
            ("h3", "c3", "2025-09-17", 2025, "minutes", "Staff Economic Outlook",
             "Staff para.", "e", "u", 2, 0, 3),
        ]
        with conn:
            conn.executemany(
                "INSERT INTO chunks (content_hash, chunk_id, meeting_date, year, "
                "doc_type, section, text, embed_text, source_url, word_count, "
                "para_index, vector_index) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        return conn

    def _hit(self, conn, content_hash):
        row = conn.execute(
            "SELECT content_hash, meeting_date, doc_type, section, text, para_index "
            "FROM chunks WHERE content_hash = ?", (content_hash,)).fetchone()
        return dict(row)

    def test_widens_to_both_neighbours(self, store):
        hit = self._hit(store, "h1")
        widened = ask.expand_hits([hit], store, radius=1)[0]
        assert widened["text"] == "Para zero. Para one. Para two."

    def test_keeps_the_original_passage_for_citation(self, store):
        hit = self._hit(store, "h1")
        widened = ask.expand_hits([hit], store, radius=1)[0]
        assert widened["core_text"] == "Para one."

    def test_stops_at_the_start_of_a_section(self, store):
        hit = self._hit(store, "h0")
        widened = ask.expand_hits([hit], store, radius=1)[0]
        assert widened["text"] == "Para zero. Para one."

    def test_never_crosses_into_another_section(self, store):
        # Widening into the staff's section would put two voices under one label.
        hit = self._hit(store, "h2")
        widened = ask.expand_hits([hit], store, radius=5)[0]
        assert "Staff para." not in widened["text"]

    def test_radius_zero_is_a_no_op(self, store):
        hit = self._hit(store, "h1")
        assert ask.expand_hits([hit], store, radius=0)[0]["text"] == "Para one."

    def test_missing_para_index_is_left_alone(self, store):
        hit = self._hit(store, "h1")
        hit["para_index"] = None
        widened = ask.expand_hits([hit], store, radius=1)[0]
        assert widened["text"] == "Para one."

    def test_word_count_is_recomputed(self, store):
        hit = self._hit(store, "h1")
        widened = ask.expand_hits([hit], store, radius=1)[0]
        assert widened["word_count"] == len(widened["text"].split())

    def test_duplicate_paragraphs_are_not_repeated(self, store):
        # 2009 minutes repeat a few paragraphs under two headings.
        with store:
            store.execute(
                "INSERT INTO chunks (content_hash, chunk_id, meeting_date, year, "
                "doc_type, section, text, embed_text, source_url, word_count, "
                "para_index, vector_index) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("h4", "c4", "2025-09-17", 2025, "minutes", "Participants' Views",
                 "Para one.", "e", "u", 2, 3, 4))
        hit = self._hit(store, "h2")
        widened = ask.expand_hits([hit], store, radius=1)[0]
        assert widened["text"].count("Para one.") == 1


class TestExtractiveAnswer:
    """
    Assembling an answer by selection rather than generation. The point is that
    omission is the only possible failure: nothing is rewritten, so a Fed
    quantifier cannot be softened and a staff view cannot become a Committee
    decision.
    """

    @staticmethod
    def hit(text, label="S1", date="2018-03-21", section="Participants' Views"):
        return {"text": text, "core_text": text, "label": label,
                "meeting_date": date, "section": section}

    def test_picks_the_sentence_that_answers_the_question(self):
        hits = [self.hit(
            "Participants discussed the labour market at length. "
            "Several participants noted that tariffs on imported steel raised costs.")]
        chosen = ask.extractive_answer("What did participants say about tariffs?", hits)
        assert "tariffs on imported steel" in chosen[0][2]

    def test_quantifiers_survive_verbatim(self):
        # "Several" must not become "some" -- they are not interchangeable.
        hits = [self.hit("Several participants judged that tariffs raised input costs.")]
        chosen = ask.extractive_answer("what did participants say about tariffs", hits)
        assert chosen[0][2].startswith("Several participants")

    def test_returns_nothing_when_no_sentence_matches(self):
        hits = [self.hit("The staff reviewed developments in foreign exchange markets.")]
        assert ask.extractive_answer("what did participants say about tariffs", hits) == []

    def test_keeps_the_attribution_of_each_sentence(self):
        hits = [self.hit("Several participants noted tariffs raised costs.",
                         section="Participants' Views"),
                self.hit("The staff projected tariffs would lower growth.",
                         label="S2", section="Staff Economic Outlook")]
        chosen = ask.extractive_answer("tariffs effect on costs and growth", hits,
                                       max_sentences=2)
        sections = {row[3]["section"] for row in chosen}
        assert "Participants' Views" in sections or "Staff Economic Outlook" in sections
        for _pos, _order, _sentence, hit in chosen:
            assert hit["section"]           # every sentence carries its label

    def test_does_not_repeat_the_same_sentence(self):
        sentence = "Several participants noted that tariffs raised input costs."
        hits = [self.hit(sentence), self.hit(sentence, label="S2", date="2018-06-13")]
        chosen = ask.extractive_answer("tariffs input costs participants", hits)
        assert len(chosen) == 1

    def test_respects_the_sentence_budget(self):
        text = " ".join(
            f"Participants judged that tariffs raised costs in district {i}." 
            for i in range(10))
        chosen = ask.extractive_answer("tariffs raised costs", [self.hit(text)],
                                       max_sentences=3)
        assert len(chosen) <= 3

    def test_output_order_follows_the_documents(self):
        text = ("Participants judged that tariffs raised costs. "
                "Participants also judged that tariffs lowered investment.")
        chosen = ask.extractive_answer("tariffs costs investment", [self.hit(text)],
                                       max_sentences=2)
        assert chosen[0][1] < chosen[1][1]

    def test_a_question_of_pure_stopwords_returns_nothing(self):
        hits = [self.hit("Several participants noted that tariffs raised costs.")]
        assert ask.extractive_answer("what did they do about it", hits) == []


class TestSplitSentences:
    def test_splits_on_terminators(self):
        assert ask.split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_does_not_split_mid_number(self):
        # "3-3/4 to 4 percent." must stay whole.
        text = "The range is 3-3/4 to 4 percent. The vote was unanimous."
        assert len(ask.split_sentences(text)) == 2

    def test_empty_text(self):
        assert ask.split_sentences("") == []


class TestQuestionStopwords:
    def test_question_words_are_dropped(self):
        assert ask.content_words("What did they say") == set()

    def test_subject_words_are_kept(self):
        assert "tariffs" in ask.content_words("What did they say about tariffs")

    def test_corpus_words_are_kept_unlike_fts_terms(self):
        # fts_terms drops "committee" as boilerplate; for choosing a sentence
        # it is meaningful, so the two lists are deliberately different.
        assert "committee" in ask.content_words("what did the committee decide")
