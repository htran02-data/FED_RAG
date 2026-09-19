"""
Store tests: idempotency, and keeping fed.db and vectors.npy in step.
"""

import numpy as np
import pytest

import embed as E


def chunk(digest, date="2025-09-17", section="Staff Economic Outlook", text="t"):
    return {
        "content_hash": digest, "chunk_id": f"{date}:minutes:{digest}",
        "meeting_date": date, "year": int(date[:4]), "doc_type": "minutes",
        "section": section, "text": text,
        "embed_text": f"[{date} | {section}] {text}",
        "source_url": "http://example", "word_count": len(text.split()),
    }


def fake_embedder(dimension=4):
    """Deterministic unit vectors; the values are irrelevant to these tests."""
    def embed(texts, _input_type):
        matrix = np.array(
            [[(hash(t) >> s) % 97 + 1 for s in range(dimension)] for t in texts],
            dtype=np.float32,
        )
        return E.normalize_rows(matrix)
    return embed


@pytest.fixture
def conn():
    return E.connect(":memory:")


class TestNormalizeRows:
    def test_rows_become_unit_length(self):
        matrix = E.normalize_rows([[3.0, 4.0], [1.0, 0.0]])
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

    def test_zero_row_does_not_divide_by_zero(self):
        matrix = E.normalize_rows([[0.0, 0.0]])
        assert not np.isnan(matrix).any()

    def test_dtype_is_float32(self):
        assert E.normalize_rows([[1.0, 2.0]]).dtype == np.float32


class TestPending:
    def test_everything_is_pending_against_an_empty_store(self, conn):
        chunks = [chunk("a"), chunk("b")]
        assert len(E.pending(chunks, conn)) == 2

    def test_duplicates_within_one_batch_collapse(self, conn):
        assert len(E.pending([chunk("a"), chunk("a")], conn)) == 1


class TestIngestIsIdempotent:
    def test_second_run_adds_nothing(self, conn, tmp_path):
        vectors = tmp_path / "vectors.npy"
        chunks = [chunk("a"), chunk("b")]

        assert E.ingest(chunks, conn, fake_embedder(), vectors) == 2
        assert E.ingest(chunks, conn, fake_embedder(), vectors) == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
        assert np.load(vectors).shape[0] == 2

    def test_only_the_new_chunk_is_embedded_on_a_rerun(self, conn, tmp_path):
        vectors = tmp_path / "vectors.npy"
        seen = []

        def counting(texts, input_type):
            seen.append(len(texts))
            return fake_embedder()(texts, input_type)

        E.ingest([chunk("a")], conn, counting, vectors)
        E.ingest([chunk("a"), chunk("b")], conn, counting, vectors)
        assert seen == [1, 1]

    def test_vector_indices_stay_contiguous_across_runs(self, conn, tmp_path):
        vectors = tmp_path / "vectors.npy"
        E.ingest([chunk("a"), chunk("b")], conn, fake_embedder(), vectors)
        E.ingest([chunk("c")], conn, fake_embedder(), vectors)
        indices = sorted(r[0] for r in conn.execute("SELECT vector_index FROM chunks"))
        assert indices == [0, 1, 2]

    def test_a_row_points_at_its_own_vector(self, conn, tmp_path):
        vectors = tmp_path / "vectors.npy"
        E.ingest([chunk("a"), chunk("b")], conn, fake_embedder(), vectors)
        matrix = np.load(vectors)
        expected = fake_embedder()(
            [chunk("b")["embed_text"], chunk("a")["embed_text"]], "document")
        index = conn.execute(
            "SELECT vector_index FROM chunks WHERE content_hash='b'").fetchone()[0]
        assert np.allclose(matrix[index], expected[0])

    def test_repeated_boilerplate_from_different_meetings_both_survive(self, conn,
                                                                      tmp_path):
        # The same sentence recurs across years; the hash keeps them distinct.
        vectors = tmp_path / "vectors.npy"
        text = "The Committee reaffirmed its longer-run goals."
        a = chunk("h2024", date="2024-01-31", text=text)
        b = chunk("h2025", date="2025-01-29", text=text)
        assert E.ingest([a, b], conn, fake_embedder(), vectors) == 2


class TestIngestSafety:
    def test_mismatched_vector_count_raises(self, conn, tmp_path):
        def short(texts, _input_type):
            return E.normalize_rows(np.ones((len(texts) - 1, 4), dtype=np.float32))

        with pytest.raises(RuntimeError, match="vectors for"):
            E.ingest([chunk("a"), chunk("b")], conn, short, tmp_path / "v.npy")

    def test_changing_the_vector_width_is_refused(self, conn, tmp_path):
        vectors = tmp_path / "vectors.npy"
        E.ingest([chunk("a")], conn, fake_embedder(4), vectors)
        with pytest.raises(SystemExit, match="vector width changed"):
            E.ingest([chunk("b")], conn, fake_embedder(8), vectors)


class TestVerify:
    def test_a_healthy_store_reports_no_problems(self, conn, tmp_path):
        vectors = tmp_path / "vectors.npy"
        E.ingest([chunk("a"), chunk("b")], conn, fake_embedder(), vectors)
        count, stored, problems = E.verify(conn, vectors)
        assert (count, stored, problems) == (2, 2, [])

    def test_missing_matrix_is_reported(self, conn, tmp_path):
        _, _, problems = E.verify(conn, tmp_path / "absent.npy")
        assert problems == ["vectors.npy is missing"]

    def test_row_count_drift_is_caught(self, conn, tmp_path):
        vectors = tmp_path / "vectors.npy"
        E.ingest([chunk("a"), chunk("b")], conn, fake_embedder(), vectors)
        np.save(vectors, np.load(vectors)[:1])
        _, _, problems = E.verify(conn, vectors)
        assert any("but 1 vectors" in p for p in problems)


class TestRateBudget:
    """
    A fixed sleep between requests is not a rate limiter: it ignores how many
    tokens each request spent. These use a fake clock so nothing actually waits.
    """

    @pytest.fixture
    def clock(self, monkeypatch):
        state = {"now": 1000.0, "slept": []}

        def monotonic():
            return state["now"]

        def sleep(seconds):
            state["slept"].append(seconds)
            state["now"] += seconds

        monkeypatch.setattr(E.time, "monotonic", monotonic)
        monkeypatch.setattr(E.time, "sleep", sleep)
        return state

    def test_first_request_within_budget_does_not_wait(self, clock):
        budget = E.RateBudget(tokens_per_minute=9000, requests_per_minute=3)
        budget.spend(5000)
        assert clock["slept"] == []

    def test_second_request_over_the_token_ceiling_waits(self, clock):
        budget = E.RateBudget(tokens_per_minute=9000, requests_per_minute=3)
        budget.spend(5000)
        budget.spend(5000)          # 10000 > 9000, so this must wait out the window
        assert clock["slept"], "expected a wait before exceeding the token ceiling"

    def test_requests_under_the_ceiling_do_not_wait(self, clock):
        budget = E.RateBudget(tokens_per_minute=9000, requests_per_minute=3)
        budget.spend(3000)
        budget.spend(3000)
        assert clock["slept"] == []

    def test_request_ceiling_is_enforced_independently_of_tokens(self, clock):
        budget = E.RateBudget(tokens_per_minute=100000, requests_per_minute=3)
        for _ in range(3):
            budget.spend(1)
        budget.spend(1)             # 4th request in the window
        assert clock["slept"], "expected a wait on the request ceiling"

    def test_window_expires_after_sixty_seconds(self, clock):
        budget = E.RateBudget(tokens_per_minute=9000, requests_per_minute=3)
        budget.spend(9000)
        clock["now"] += 61.0        # the earlier spend falls out of the window
        budget.spend(9000)
        assert clock["slept"] == []

    def test_waiting_clears_the_window_and_then_proceeds(self, clock):
        budget = E.RateBudget(tokens_per_minute=9000, requests_per_minute=3)
        budget.spend(8000)
        budget.spend(8000)
        # After waiting, the first spend has aged out and the second is recorded.
        assert len(budget.events) == 1
        assert budget.events[0][1] == 8000


class TestTokenBatches:
    """
    Batching by chunk count produces requests of wildly different sizes -- this
    corpus runs from ~14 to ~515 tokens per chunk -- which is how a batch that
    looked safe came to 7,092 tokens against a 10K/min ceiling.
    """

    def sized(self, *word_counts):
        return [chunk(str(i), text="w " * n) | {"word_count": n}
                for i, n in enumerate(word_counts)]

    def test_batches_stay_under_the_token_ceiling(self):
        chunks = self.sized(*([100] * 20))          # ~143 tokens each
        for batch in E.token_batches(chunks, max_tokens=500, max_count=128):
            assert sum(E.estimate_tokens(c) for c in batch) <= 500

    def test_every_chunk_appears_exactly_once(self):
        chunks = self.sized(10, 200, 50, 400, 7, 90)
        batched = [c for batch in E.token_batches(chunks, 300, 128) for c in batch]
        assert [c["content_hash"] for c in batched] == \
               [c["content_hash"] for c in chunks]

    def test_order_is_preserved(self):
        chunks = self.sized(*range(1, 30))
        batched = [c for batch in E.token_batches(chunks, 200, 128) for c in batch]
        assert batched == chunks

    def test_an_oversized_chunk_gets_its_own_batch_rather_than_vanishing(self):
        chunks = self.sized(10, 5000, 10)
        batches = list(E.token_batches(chunks, 300, 128))
        assert sum(len(b) for b in batches) == 3
        big = [b for b in batches if b[0]["word_count"] == 5000]
        assert len(big) == 1 and len(big[0]) == 1

    def test_count_cap_binds_when_chunks_are_tiny(self):
        chunks = self.sized(*([1] * 50))
        batches = list(E.token_batches(chunks, max_tokens=10 ** 6, max_count=10))
        assert all(len(b) <= 10 for b in batches)
        assert len(batches) == 5

    def test_empty_input(self):
        assert list(E.token_batches([], 100, 10)) == []

    def test_estimate_is_not_an_underestimate(self):
        # Measured 1.29 tokens/word on this corpus; the estimate rounds up.
        assert E.estimate_tokens({"word_count": 100}) >= 129
