"""
Parsing and chunking tests.

Most of these pin bugs that were found by inspecting the real corpus, not
hypotheticals. Each one names the failure it prevents.
"""

import re

import pytest
from bs4 import BeautifulSoup

import chunk as C


def paragraph(html):
    """Return the first <p> of a fragment, for heading-detection tests."""
    return BeautifulSoup(html, "html.parser").find("p")


class TestNormalize:
    def test_curly_apostrophes_become_straight(self):
        # "Participants' Views" is spelled with a curly apostrophe in the source.
        assert C.normalize("Participants’ Views") == "Participants' Views"

    def test_curly_double_quotes(self):
        assert C.normalize("“ample”") == '"ample"'

    def test_en_dash_and_nbsp(self):
        assert C.normalize("July 28–29,\xa02026") == "July 28-29, 2026"

    def test_collapses_whitespace(self):
        assert C.normalize("a  \n b\t c") == "a b c"


class TestMatchSection:
    @pytest.mark.parametrize("heading,expected", [
        # The canonical spellings.
        ("Developments in Financial Markets and Open Market Operations",
         "Developments in Financial Markets and Open Market Operations"),
        ("Staff Review of the Economic Situation", "Staff Review of the Economic Situation"),
        ("Staff Review of the Financial Situation", "Staff Review of the Financial Situation"),
        ("Staff Economic Outlook", "Staff Economic Outlook"),
        ("Participants' Views on Current Conditions and the Economic Outlook",
         "Participants' Views on Current Conditions and the Economic Outlook"),
        ("Committee Policy Action", "Committee Policy Action"),
        # Drift actually observed in the 2016-2026 corpus.
        ("Committee Policy Actions", "Committee Policy Action"),
        ("Staff Review of Financial Situation", "Staff Review of the Financial Situation"),
        ("Participants' View on Current Conditions and the Economic Outlook",
         "Participants' Views on Current Conditions and the Economic Outlook"),
        ("Participants' Views on Current Economic Conditions and the Economic Outlook",
         "Participants' Views on Current Conditions and the Economic Outlook"),
        ("Discussion of Financial Markets and Open Market Operations",
         "Developments in Financial Markets and Open Market Operations"),
        ("Financial Developments and Open Market Operations",
         "Developments in Financial Markets and Open Market Operations"),
        # 2009-2015 named the same section after the balance sheet.
        ("Developments in Financial Markets and the Federal Reserve's Balance Sheet",
         "Developments in Financial Markets and Open Market Operations"),
        # Trailing punctuation and curly apostrophes.
        ("Committee Policy Action:", "Committee Policy Action"),
        ("Participants’ Views on Current Conditions and the Economic Outlook",
         "Participants' Views on Current Conditions and the Economic Outlook"),
    ])
    def test_known_headings_and_drift(self, heading, expected):
        assert C.match_section(heading) == expected

    @pytest.mark.parametrize("text", [
        # Genuinely merged sections keep their own label: the attribution is
        # combined in the source, so flattening them to one canonical name
        # would claim a precision the document does not have.
        "Staff Review of the Economic and Financial Situation",
        "Meeting Participants' Views and Committee Policy Action",
        "Financial Stability Report",
        "Selection of Committee Officer",
        "Voting for this action:",
        "Revisions to Documents Governing Foreign Currency Operations",
        "",
    ])
    def test_non_sections_return_none(self, text):
        assert C.match_section(text) is None

    def test_body_prose_is_not_a_heading(self):
        # A whole paragraph mentioning open market operations must not match.
        body = ("The Manager turned first to a discussion of open market operations "
                "over the intermeeting period, noting that conditions in money "
                "markets had remained stable throughout the period under review.")
        assert C.match_section(body) is None


class TestLeadingHeading:
    def test_strong_followed_by_break_is_a_heading(self):
        p = paragraph("<p><strong>Staff Economic Outlook</strong><br/>"
                      "The projection prepared by the staff ...</p>")
        assert C.leading_heading(p) == "Staff Economic Outlook"

    def test_attendee_entry_is_not_a_heading(self):
        # <p><strong>Name</strong>, Title</p> -- no <br/>, so not a section.
        p = paragraph("<p><strong>Jane Q. Doe</strong>, Director, Division of "
                      "Research and Statistics, Board of Governors</p>")
        assert C.leading_heading(p) is None

    def test_canonical_heading_without_break_still_counts(self):
        p = paragraph("<p><strong>Committee Policy Action</strong> "
                      "In their discussion ...</p>")
        assert C.leading_heading(p) == "Committee Policy Action"

    def test_paragraph_not_starting_with_emphasis(self):
        p = paragraph("<p>Participants noted that <strong>inflation</strong> "
                      "remained elevated over the period.</p>")
        assert C.leading_heading(p) is None


class TestNoiseHeadings:
    @pytest.mark.parametrize("text", [
        "_______________________",
        "Joshua Gallin",
        "James A. Clouse",
        "Brian F. Madigan Secretary",
        "July 28-29, 2026",
        "January 31-February 1, 2017",
        "Voting for this action:",
        "Voting against this action: None.",
    ])
    def test_noise(self, text):
        assert C.is_noise_heading(text) is True

    @pytest.mark.parametrize("text", [
        # Genuinely merged sections keep their own label: the attribution is
        # combined in the source, so flattening them to one canonical name
        # would claim a precision the document does not have.
        "Staff Review of the Economic and Financial Situation",
        "Meeting Participants' Views and Committee Policy Action",
        "Financial Stability Report",
        "Balance Sheet Normalization",
        "Long-Run Monetary Policy Implementation Frameworks",
    ])
    def test_real_special_topics_are_not_noise(self, text):
        assert C.is_noise_heading(text) is False

    def test_footnote_marker_is_stripped(self):
        assert C.clean_heading("Annual Organizational Matters 5") == \
            "Annual Organizational Matters"

    def test_clean_heading_leaves_ordinary_titles_alone(self):
        assert C.clean_heading("Balance Sheet Normalization") == \
            "Balance Sheet Normalization"


class TestParseMinutes:
    # The Fed puts the section title inside the section's first paragraph.
    # Treating the whole <p> as body text attributes that paragraph to the
    # *previous* section -- which labelled participants' views as the staff's.
    MINUTES = """
    <div id="article">
      <p><strong>Staff Economic Outlook</strong><br/>
      The projection prepared by the staff for this meeting was revised down a
      little, reflecting incoming data on spending and on the labor market that
      had come in softer than the staff had previously anticipated overall.</p>
      <p>In the staff's judgment, the risks around the forecast for real
      activity remained tilted somewhat to the downside relative to the modal
      projection that had been prepared for this particular meeting.</p>
      <p><strong>Participants' Views on Current Conditions and the Economic
      Outlook</strong><br/>
      Several participants observed that inflation remained elevated and that
      the recent readings had not yet provided sufficient confidence that
      inflation was moving down toward the Committee's stated objective.</p>
      <p><strong>Committee Policy Actions</strong><br/>
      In their discussion of monetary policy for this meeting, members agreed
      that it would be appropriate to maintain the target range for the federal
      funds rate at its current level for the time being.</p>
    </div>
    """

    def test_opening_paragraph_belongs_to_its_own_section(self):
        paragraphs, _ = C.parse_minutes(self.MINUTES)
        opening = next(p for p in paragraphs if "Several participants" in p["text"])
        assert opening["section"] == \
            "Participants' Views on Current Conditions and the Economic Outlook"

    def test_heading_text_is_stripped_from_the_body(self):
        paragraphs, _ = C.parse_minutes(self.MINUTES)
        opening = next(p for p in paragraphs if "Several participants" in p["text"])
        assert not opening["text"].startswith("Participants'")
        assert opening["text"].startswith("Several participants")

    def test_continuation_paragraph_keeps_the_section(self):
        paragraphs, _ = C.parse_minutes(self.MINUTES)
        cont = next(p for p in paragraphs if "risks around the forecast" in p["text"])
        assert cont["section"] == "Staff Economic Outlook"

    def test_drifted_plural_heading_is_canonicalised(self):
        paragraphs, _ = C.parse_minutes(self.MINUTES)
        action = next(p for p in paragraphs if "members agreed" in p["text"])
        assert action["section"] == "Committee Policy Action"

    def test_special_topic_does_not_bleed_into_the_section_above(self):
        html = """
        <div id="article">
          <p><strong>Staff Economic Outlook</strong><br/>
          The staff projection prepared for this meeting was little revised from
          the previous forecast round and continued to show moderate growth.</p>
          <p><strong>Financial Stability Report</strong><br/>
          The staff presented its assessment of financial stability, reviewing
          asset valuations, leverage in the financial sector, and funding risks
          across the banking system in some considerable detail.</p>
        </div>
        """
        paragraphs, unmatched = C.parse_minutes(html)
        stability = next(p for p in paragraphs if "assessment of financial" in p["text"])
        assert stability["section"] == "Financial Stability Report"
        assert "Financial Stability Report" in unmatched

    def test_attendee_paragraphs_do_not_switch_sections(self):
        html = """
        <div id="article">
          <p><strong>Committee Policy Action</strong><br/>
          In their discussion of monetary policy, members agreed that it would be
          appropriate to maintain the target range at its current level today.</p>
          <p><strong>Jane Q. Doe</strong>, Director, Division of Research and
          Statistics, Board of Governors of the Federal Reserve System, and a
          number of other officers attended the meeting in person this time.</p>
        </div>
        """
        paragraphs, _ = C.parse_minutes(html)
        assert {p["section"] for p in paragraphs} == {"Committee Policy Action"}

    def test_short_fragments_are_dropped(self):
        html = '<div id="article"><p>1. See the appendix.</p></div>'
        paragraphs, _ = C.parse_minutes(html)
        assert paragraphs == []


class TestSplitLong:
    def test_short_paragraph_is_untouched(self):
        text = "Participants noted that inflation remained elevated."
        assert C.split_long(text) == [text]

    def test_long_paragraph_splits_on_sentence_boundaries(self):
        sentence = "Several participants observed that inflation had eased somewhat. "
        text = (sentence * 80).strip()   # ~720 words, well past MAX_WORDS
        parts = C.split_long(text)
        assert len(parts) > 1
        for part in parts:
            assert part.strip().endswith(".")

    def test_split_preserves_every_word(self):
        sentence = "A few participants judged that the risks were roughly balanced. "
        text = (sentence * 80).strip()
        parts = C.split_long(text)
        assert " ".join(parts).split() == text.split()

    def test_no_stranded_fragment(self):
        text = ("Many participants noted that the labor market had cooled. " * 60
                + "Others disagreed.")
        parts = C.split_long(text)
        assert all(len(p.split()) >= 40 for p in parts)


class TestChunkRecords:
    PARAGRAPHS = [{
        "section": "Participants' Views on Current Conditions and the Economic Outlook",
        "text": "Several participants judged that the risks to inflation were "
                "tilted to the upside over the medium term horizon.",
    }]

    def test_embed_text_carries_date_and_section(self):
        chunks = C.build_chunks(self.PARAGRAPHS, "2025-09-17", "minutes", "http://x")
        prefix = ("[2025-09-17 | Participants' Views on Current Conditions "
                  "and the Economic Outlook]")
        assert chunks[0]["embed_text"].startswith(prefix)

    def test_no_passage_is_embedded_without_its_date(self):
        chunks = C.build_chunks(self.PARAGRAPHS, "2025-09-17", "minutes", "http://x")
        for chunk in chunks:
            assert re.match(r"^\[\d{4}-\d{2}-\d{2} \| ", chunk["embed_text"])

    def test_quantifiers_survive_chunking_verbatim(self):
        # "Several" is quasi-ordinal and load-bearing; it must not be reworded.
        chunks = C.build_chunks(self.PARAGRAPHS, "2025-09-17", "minutes", "http://x")
        assert chunks[0]["text"].startswith("Several participants")

    def test_year_is_derived_from_meeting_date(self):
        chunks = C.build_chunks(self.PARAGRAPHS, "2016-03-16", "minutes", "http://x")
        assert chunks[0]["year"] == 2016


class TestContentHash:
    def test_is_stable_across_calls(self):
        first = C.content_hash("2025-09-17", "minutes", "Staff Economic Outlook", "text")
        second = C.content_hash("2025-09-17", "minutes", "Staff Economic Outlook", "text")
        assert first == second

    def test_same_text_in_a_different_meeting_hashes_differently(self):
        # FOMC minutes repeat near-identical sentences year after year; the hash
        # must keep them distinct or ingestion would silently drop them.
        boilerplate = ("The Committee reaffirmed its Statement on Longer-Run "
                       "Goals and Monetary Policy Strategy.")
        a = C.content_hash("2024-01-31", "minutes", "Committee Policy Action", boilerplate)
        b = C.content_hash("2025-01-29", "minutes", "Committee Policy Action", boilerplate)
        assert a != b

    def test_same_text_in_a_different_section_hashes_differently(self):
        a = C.content_hash("2025-01-29", "minutes", "Staff Economic Outlook", "t")
        b = C.content_hash("2025-01-29", "minutes", "Committee Policy Action", "t")
        assert a != b


class TestWindows:
    """
    Overlapping windows: consecutive chunks share their boundary paragraph, so
    a claim straddling a break survives whole inside at least one chunk.
    """

    @staticmethod
    def units(n):
        return [{"section": "S", "text": f"p{i}", "paragraph": i} for i in range(n)]

    def test_window_one_is_one_chunk_per_paragraph(self):
        groups = C.windows(self.units(4), window=1)
        assert [[u["text"] for u in g] for g in groups] == [["p0"], ["p1"], ["p2"], ["p3"]]

    def test_consecutive_chunks_share_their_boundary_paragraph(self):
        for n in range(2, 30):
            groups = C.windows(self.units(n), window=3)
            for a, b in zip(groups, groups[1:]):
                assert a[-1]["text"] == b[0]["text"], f"broken at n={n}"

    def test_no_paragraph_is_ever_dropped(self):
        for n in range(1, 30):
            groups = C.windows(self.units(n), window=3)
            seen = {u["text"] for g in groups for u in g}
            assert seen == {f"p{i}" for i in range(n)}, f"lost one at n={n}"

    def test_a_run_shorter_than_the_window_is_one_chunk(self):
        assert len(C.windows(self.units(2), window=3)) == 1

    def test_no_trailing_chunk_that_adds_nothing(self):
        # Without the containment check the tail of every section is emitted
        # twice, the second time carrying no paragraph the first did not.
        groups = C.windows(self.units(5), window=3)
        assert [[u["text"] for u in g] for g in groups] == [
            ["p0", "p1", "p2"], ["p2", "p3", "p4"]]

    def test_single_paragraph(self):
        assert [[u["text"] for u in g] for g in C.windows(self.units(1), window=3)] == [["p0"]]


class TestContextLine:
    def test_spells_the_month_out(self):
        line = C.context_line("2025-09-17", "minutes", "Staff Economic Outlook")
        assert "September 2025" in line

    def test_keeps_the_iso_date_too(self):
        # The digits are what the date filter and the eval match on.
        assert "2025-09-17" in C.context_line("2025-09-17", "minutes", "X")

    def test_names_the_document_kind(self):
        assert "statement" in C.context_line("2025-09-17", "statement", "Policy Statement").lower()
        assert "minutes" in C.context_line("2025-09-17", "minutes", "X").lower()

    def test_carries_the_section(self):
        assert "Staff Economic Outlook" in C.context_line(
            "2025-09-17", "minutes", "Staff Economic Outlook")


class TestTailSentences:
    def test_returns_the_last_sentence(self):
        assert C.tail_sentences("One. Two. Three.", 1) == "Three."

    def test_returns_several(self):
        assert C.tail_sentences("One. Two. Three.", 2) == "Two. Three."

    def test_zero_returns_nothing(self):
        assert C.tail_sentences("One. Two.", 0) == ""

    def test_more_than_available_returns_all(self):
        assert C.tail_sentences("Only one.", 5) == "Only one."

    def test_empty_text(self):
        assert C.tail_sentences("", 2) == ""


class TestOverlapDoesNotCrossSections:
    def test_a_window_never_spans_two_sections(self):
        """
        A chunk covering Staff Economic Outlook and Participants' Views would
        carry two voices under one label -- the misattribution the domain rules
        forbid.
        """
        paragraphs = (
            [{"section": "Staff Economic Outlook", "text": f"staff {i} " + "w " * 30}
             for i in range(3)]
            + [{"section": "Participants' Views", "text": f"parts {i} " + "w " * 30}
               for i in range(3)]
        )
        chunks = C.build_chunks(paragraphs, "2025-09-17", "minutes", "u", window=3)
        for chunk in chunks:
            if chunk["section"] == "Staff Economic Outlook":
                assert "parts" not in chunk["text"]
            else:
                assert "staff" not in chunk["text"]
