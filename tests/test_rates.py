"""
Rate-path tests.

The date bounds for "the hiking cycle" are a fact about what the FOMC did, so
they are derived from the statements rather than hardcoded. These tests cover
the derivation, using synthetic paths -- not remembered history.
"""

import pytest

import rates as R


class TestParseFraction:
    @pytest.mark.parametrize("token,expected", [
        ("5", 5.0), ("1/4", 0.25), ("1/2", 0.5),
        ("4-1/4", 4.25), ("4-3/4", 4.75), ("1-1/2", 1.5), ("2-1/2", 2.5),
    ])
    def test_fractions(self, token, expected):
        assert R.parse_fraction(token) == expected


class TestParseTargetRange:
    def test_maintain_phrasing(self):
        text = ("The Committee decided to maintain the target range for the "
                "federal funds rate at 4-1/4 to 4-1/2 percent.")
        assert R.parse_target_range(text) == (4.25, 4.5)

    def test_raise_phrasing(self):
        text = ("the Committee decided to raise the target range for the federal "
                "funds rate to 1-1/2 to 1-3/4 percent and anticipates that ongoing "
                "increases will be appropriate.")
        assert R.parse_target_range(text) == (1.5, 1.75)

    def test_phrasing_with_an_intervening_clause(self):
        # "by 1/2 percentage point to 4-3/4 to 5 percent"
        text = ("the Committee decided to lower the target range for the federal "
                "funds rate by 1/2 percentage point to 4-3/4 to 5 percent.")
        assert R.parse_target_range(text) == (4.75, 5.0)

    def test_zero_lower_bound_phrasing(self):
        text = ("decided to lower the target range for the federal funds rate to "
                "0 to 1/4 percent.")
        assert R.parse_target_range(text) == (0.0, 0.25)

    def test_non_breaking_hyphen_is_normalised_upstream(self):
        # chunk.normalize turns U+2011 into "-" before this ever runs; if it did
        # not, this sentence would silently fail to parse.
        text = ("maintain the target range for the federal funds rate at 1-1/2 to "
                "1-3/4 percent.")
        assert R.parse_target_range(text) is not None

    def test_unrelated_text_returns_none(self):
        assert R.parse_target_range("Participants discussed the labor market.") is None


def statement(date, low, high):
    return {
        "meeting_date": date, "doc_type": "statement",
        "text": (f"The Committee decided to set the target range for the federal "
                 f"funds rate at {low} to {high} percent."),
    }


class TestRatePath:
    def test_builds_an_ordered_path_of_midpoints(self):
        path = R.rate_path([
            statement("2022-03-16", "1/4", "1/2"),
            statement("2022-05-04", "3/4", "1"),
        ])
        assert list(path) == ["2022-03-16", "2022-05-04"]
        assert path["2022-03-16"] == 0.375
        assert path["2022-05-04"] == 0.875

    def test_minutes_are_ignored(self):
        chunks = [dict(statement("2022-03-16", "1/4", "1/2"), doc_type="minutes")]
        assert R.rate_path(chunks) == {}

    def test_first_statement_chunk_per_meeting_wins(self):
        chunks = [statement("2022-03-16", "1/4", "1/2"),
                  statement("2022-03-16", "3", "3-1/4")]
        assert R.rate_path(chunks)["2022-03-16"] == 0.375


class TestEras:
    PATH = {
        "2021-12-15": 0.125,   # flat
        "2022-01-26": 0.125,
        "2022-03-16": 0.375,   # first hike
        "2022-05-04": 0.875,
        "2022-06-15": 1.625,
        "2023-05-03": 5.125,
        "2023-06-14": 5.125,   # a pause inside the cycle
        "2023-07-26": 5.375,   # last hike
        "2023-09-20": 5.375,   # holding
        "2024-07-31": 5.375,
        "2024-09-18": 4.875,   # first cut
        "2024-11-07": 4.625,
    }

    def test_hiking_cycle_runs_from_first_to_last_hike(self):
        found = R.eras(self.PATH)
        hiking = [e for e in found if e["kind"] == R.HIKE]
        assert len(hiking) == 1
        assert (hiking[0]["start"], hiking[0]["end"]) == ("2022-03-16", "2023-07-26")

    def test_a_pause_does_not_split_the_cycle(self):
        # 2023-06-14 is flat but sits between two hikes.
        hiking = [e for e in R.eras(self.PATH) if e["kind"] == R.HIKE][0]
        assert hiking["start"] < "2023-06-14" < hiking["end"]

    def test_cutting_cycle_is_detected(self):
        cutting = [e for e in R.eras(self.PATH) if e["kind"] == R.CUT]
        assert (cutting[0]["start"], cutting[0]["end"]) == ("2024-09-18", "2024-11-07")

    def test_leading_flat_run_is_a_holding_era(self):
        assert R.eras(self.PATH)[0]["kind"] == R.HOLD

    def test_empty_path(self):
        assert R.eras({}) == []

    def test_single_meeting_has_no_moves(self):
        assert R.eras({"2022-03-16": 0.375}) == []


class TestNamedEras:
    def test_phrases_map_to_the_latest_cycle_of_each_kind(self):
        names = R.named_eras(TestEras.PATH)
        assert names["hiking cycle"] == ("2022-03-16", "2023-07-26")
        assert names["tightening cycle"] == names["hiking cycle"]
        assert names["cutting cycle"] == ("2024-09-18", "2024-11-07")
        assert names["easing cycle"] == names["cutting cycle"]

    def test_most_recent_cycle_wins_when_there_are_several(self):
        path = {
            "2016-12-14": 0.625, "2017-03-15": 0.875,     # an earlier hiking run
            "2019-07-31": 0.625, "2020-03-15": 0.125,     # cut back down
            "2022-03-16": 0.375, "2022-05-04": 0.875,     # the later hiking run
        }
        assert R.named_eras(path)["hiking cycle"] == ("2022-03-16", "2022-05-04")

    def test_no_cycle_of_a_kind_means_no_phrase(self):
        assert "cutting cycle" not in R.named_eras({"2022-03-16": 0.375,
                                                    "2022-05-04": 0.875})
