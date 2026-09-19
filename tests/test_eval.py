"""Scoring tests for the eval harness, plus gold-set integrity checks."""

import json
import pathlib

import numpy as np
import pytest

import ask
import embed as E
import eval as V
import rates

GOLD_PATH = pathlib.Path("data/gold.jsonl")
CHUNKS_PATH = pathlib.Path("data/chunks.jsonl")


@pytest.fixture(scope="module")
def gold():
    return V.load_gold(GOLD_PATH)


@pytest.fixture(scope="module")
def corpus():
    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        return {json.loads(line)["content_hash"]: json.loads(line) for line in handle}


class TestGoldSetIntegrity:
    """
    Every gold pair must point at a passage that actually exists, with the
    meeting date and section it claims. A pair written from memory would fail
    these.
    """

    def test_there_are_at_least_twenty_pairs(self, gold):
        assert len(gold) >= 20

    def test_ids_are_unique(self, gold):
        assert len({g["id"] for g in gold}) == len(gold)

    def test_every_gold_hash_exists_in_the_corpus(self, gold, corpus):
        missing = [g["id"] for g in gold if g["content_hash"] not in corpus]
        assert missing == []

    def test_meeting_date_matches_the_chunk(self, gold, corpus):
        for item in gold:
            assert corpus[item["content_hash"]]["meeting_date"] == item["meeting_date"]

    def test_section_matches_the_chunk(self, gold, corpus):
        for item in gold:
            assert corpus[item["content_hash"]]["section"] == item["section"]

    def test_answer_key_is_grounded_in_the_passage(self, gold, corpus):
        """The quoted span must really appear in the cited passage."""
        for item in gold:
            if item.get("spans_paragraphs"):
                continue      # checked by test_spanning_answers_really_span
            passage = corpus[item["content_hash"]]["text"]
            assert V.quote_matches(passage, item["answer_key"]), (
                f"{item['id']}: answer_key is not grounded in its passage")

    def test_spanning_answers_really_span(self, gold, corpus):
        """
        A pair marked spans_paragraphs must not fit inside any single chunk of
        its meeting -- otherwise it silently stops testing what it exists for.
        """
        by_meeting = {}
        for row in corpus.values():
            by_meeting.setdefault(row["meeting_date"], []).append(row)
        for item in gold:
            if not item.get("spans_paragraphs"):
                continue
            same = by_meeting.get(item["meeting_date"], [])
            assert not any(V.quote_matches(r["text"], item["answer_key"]) for r in same), (
                f"{item['id']} is marked spanning but fits inside one chunk")

    def test_temporal_cue_flag_matches_the_parser(self, gold):
        """The flag is documentation, so it has to agree with what ask.py sees."""
        eras = rates.named_eras(rates.load_path())
        for item in gold:
            parsed = ask.parse_temporal(item["question"], era_names=eras) is not None
            assert parsed == item["has_temporal_cue"], (
                f"{item['id']}: has_temporal_cue={item['has_temporal_cue']} "
                f"but the parser says {parsed}")

    def test_both_kinds_of_question_are_represented(self, gold):
        """
        The set must keep testing the hard case. Questions that state no period
        fall back to searching the whole corpus, which is where retrieval is
        weakest -- a gold set of only date-cued questions cannot see that.
        """
        cued = sum(1 for g in gold if g["has_temporal_cue"])
        assert cued >= 15, "too few date-cued questions to measure the filter"
        assert len(gold) - cued >= 3, "no date-free questions: the hard case is untested"
        assert sum(1 for g in gold if g.get("spans_paragraphs")) >= 3, (
            "no paragraph-spanning questions: chunk boundaries are untested")

    def test_pairs_span_many_years(self, gold):
        assert len({g["meeting_date"][:4] for g in gold}) >= 8

    def test_pairs_span_several_sections(self, gold):
        assert len({g["section"] for g in gold}) >= 5

    def test_a_cued_question_window_contains_its_own_gold_meeting(self, gold):
        """A cued question whose window excludes its answer would filter it away."""
        eras = rates.named_eras(rates.load_path())
        for item in gold:
            if not item["has_temporal_cue"]:
                continue
            window = ask.parse_temporal(item["question"], era_names=eras)
            assert window.start <= item["meeting_date"] <= window.end, (
                f"{item['id']}: gold meeting falls outside its own question window")


@pytest.fixture
def scored_store():
    """Two meetings, near-identical text -- the wrong-year failure in miniature."""
    conn = E.connect(":memory:")
    rows = [
        ("gold", "c1", "2025-09-17", 2025, "minutes", "Staff Economic Outlook",
         "The staff revised the projection down.", "e", "u", 6, 0),
        ("other", "c2", "2019-06-19", 2019, "minutes", "Staff Economic Outlook",
         "The staff revised the projection down.", "e", "u", 6, 1),
    ]
    with conn:
        conn.executemany(
            "INSERT INTO chunks (content_hash, chunk_id, meeting_date, year, "
            "doc_type, section, text, embed_text, source_url, word_count, "
            "vector_index) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    return conn, np.eye(2, dtype=np.float32)


def prefer(index):
    def embed(_text):
        vector = np.zeros(2, dtype=np.float32)
        vector[index] = 1.0
        return vector
    return embed


ITEM = {"id": "t1", "question": "What did the staff project in September 2025?",
        "meeting_date": "2025-09-17", "section": "Staff Economic Outlook",
        "content_hash": "gold",
        "answer_key": "revised the projection down"}


class TestEvaluate:
    def test_filter_rescues_a_question_the_wrong_year_would_win(self, scored_store):
        conn, matrix = scored_store
        # The embedder prefers the 2019 passage; the date filter removes it.
        unfiltered = V.evaluate([ITEM], conn, matrix, k=2, use_filter=False,
                                embedder=prefer(1))
        filtered = V.evaluate([ITEM], conn, matrix, k=2, use_filter=True,
                              embedder=prefer(1))
        assert unfiltered[0]["rank"] == 2
        assert filtered[0]["rank"] == 1

    def test_wrong_meeting_count_drops_with_the_filter(self, scored_store):
        conn, matrix = scored_store
        unfiltered = V.evaluate([ITEM], conn, matrix, k=2, use_filter=False,
                                embedder=prefer(1))
        filtered = V.evaluate([ITEM], conn, matrix, k=2, use_filter=True,
                              embedder=prefer(1))
        assert unfiltered[0]["wrong_meeting"] == 1
        assert filtered[0]["wrong_meeting"] == 0

    def test_miss_is_recorded_as_none(self, scored_store):
        conn, matrix = scored_store
        item = dict(ITEM, content_hash="absent")
        assert V.evaluate([item], conn, matrix, k=1, use_filter=True,
                          embedder=prefer(0))[0]["rank"] is None


class TestSummarize:
    def test_recall_and_mrr(self):
        results = [
            {"rank": 1, "quote_rank": 1, "meeting_hit": True, "wrong_meeting": 0,
             "retrieved": 4, "candidates": 10, "words_shown": 400},
            {"rank": 4, "quote_rank": 2, "meeting_hit": True, "wrong_meeting": 2,
             "retrieved": 4, "candidates": 10, "words_shown": 400},
            {"rank": None, "quote_rank": None, "meeting_hit": False,
             "wrong_meeting": 4, "retrieved": 4, "candidates": 10,
             "words_shown": 400},
        ]
        stats = V.summarize(results, 4)
        assert stats["recall@4"] == pytest.approx(2 / 3)
        assert stats["meeting_recall@4"] == pytest.approx(2 / 3)
        assert stats["mrr"] == pytest.approx((1 + 0.25) / 3)
        assert stats["wrong_meeting_rate"] == pytest.approx(6 / 12)

    def test_perfect_run(self):
        results = [{"rank": 1, "quote_rank": 1, "meeting_hit": True,
                    "wrong_meeting": 0, "retrieved": 3, "candidates": 5,
                    "words_shown": 300}]
        stats = V.summarize(results, 3)
        assert stats["recall@3"] == 1.0
        assert stats["wrong_meeting_rate"] == 0.0

    def test_empty_retrieval_does_not_divide_by_zero(self):
        results = [{"rank": None, "quote_rank": None, "meeting_hit": False,
                    "wrong_meeting": 0, "retrieved": 0, "candidates": 0,
                    "words_shown": 0}]
        assert V.summarize(results, 5)["wrong_meeting_rate"] == 0.0


class TestQuoteMatching:
    """
    Quote anchoring is what lets one gold set score two chunking schemes. A
    content_hash dies the moment slicing changes; the words do not.
    """

    def test_contiguous_quote(self):
        assert V.quote_matches("The staff revised the projection down.",
                               "revised the projection")

    def test_elided_quote_matches_across_the_gap(self):
        text = "The VIX rose sharply in February but remained low by historical standards."
        assert V.quote_matches(text, "The VIX ... remained low by historical standards")

    def test_elided_fragments_must_appear_in_order(self):
        text = "remained low by historical standards, and then the VIX rose."
        assert not V.quote_matches(text, "The VIX ... remained low by historical standards")

    def test_whitespace_and_case_are_normalized(self):
        assert V.quote_matches("The  staff\nrevised   the projection.",
                               "The staff revised the projection")

    def test_absent_quote_does_not_match(self):
        assert not V.quote_matches("Participants discussed the labour market.",
                                   "revised the projection down")

    def test_quote_rank_requires_the_right_meeting(self):
        # The same sentence genuinely recurs across meetings, so matching words
        # alone would credit a hit from the wrong year.
        item = {"meeting_date": "2025-09-17", "answer_key": "revised the projection"}
        hits = [
            {"meeting_date": "2019-06-19", "text": "The staff revised the projection down."},
            {"meeting_date": "2025-09-17", "text": "The staff revised the projection down."},
        ]
        assert V.quote_rank(hits, item) == 2

    def test_quote_rank_is_none_when_absent(self):
        item = {"meeting_date": "2025-09-17", "answer_key": "revised the projection"}
        hits = [{"meeting_date": "2025-09-17", "text": "Participants discussed tariffs."}]
        assert V.quote_rank(hits, item) is None
