"""
Index-parsing tests.

Meeting URLs are parsed from the Fed's index pages rather than constructed, so
these tests pin the shapes of both index layouts and the label rules that tell
the policy statement apart from the two other documents sharing its URL form.
"""

import pytest
from bs4 import BeautifulSoup

import scrape as S


class TestIsoFromHref:
    def test_minutes_href(self):
        assert S.iso_from_href("/monetarypolicy/fomcminutes20250917.htm",
                               S.MINUTES_HREF) == "2025-09-17"

    def test_statement_href_with_suffix(self):
        assert S.iso_from_href("/newsevents/pressreleases/monetary20180131a.htm",
                               S.STATEMENT_HREF) == "2018-01-31"

    def test_legacy_minutes_path(self):
        # Minutes lived at /fomc/minutes/{date}.htm before 2007.
        assert S.iso_from_href("/fomc/minutes/20060131.htm",
                               S.MINUTES_HREF) == "2006-01-31"

    def test_mid_era_statement_path(self):
        # Statements sat under /newsevents/press/monetary/ from 2006 to 2010.
        assert S.iso_from_href("/newsevents/press/monetary/20090128a.htm",
                               S.STATEMENT_HREF) == "2009-01-28"

    def test_modern_statement_path_still_matches(self):
        assert S.iso_from_href("/newsevents/pressreleases/monetary20260128a.htm",
                               S.STATEMENT_HREF) == "2026-01-28"

    def test_pdf_is_not_matched(self):
        # Only the HTML documents are cached; the PDF twin must not match.
        assert S.iso_from_href("/monetarypolicy/files/fomcminutes20250917.pdf",
                               S.MINUTES_HREF) is None


class TestLastDayFromRange:
    @pytest.mark.parametrize("month,days,year,expected", [
        ("January", "27-28", 2026, "2026-01-28"),
        ("September", "16-17*", 2025, "2025-09-17"),
        ("June", "15", 2021, "2021-06-15"),
        # A meeting that crosses a month boundary names the end month itself.
        ("April", "30-May 1", 2019, "2019-05-01"),
        ("January", "31-February 1", 2017, "2017-02-01"),
        ("October", "31-November 1", 2017, "2017-11-01"),
    ])
    def test_ranges(self, month, days, year, expected):
        assert S.last_day_from_range(month, days, year) == expected

    def test_unparseable_returns_none(self):
        assert S.last_day_from_range("Smarch", "1-2", 2020) is None
        assert S.last_day_from_range("January", "unscheduled", 2020) is None


class TestStatementSelection:
    """
    Three documents share the /monetary{date}*.htm shape and only the link
    label separates them:
        monetary{date}a.htm   the policy statement          <- wanted
        monetary{date}a1.htm  the implementation note
        monetary{date}b.htm   Statement on Longer-Run Goals
    """

    HISTORICAL = """
    <div>
      <a href="/newsevents/pressreleases/monetary20180131a.htm">Statement</a>
      <a href="/newsevents/pressreleases/monetary20180131b.htm">Statement on
         Longer-Run Goals and Monetary Policy Strategy</a>
    </div>
    """

    CALENDAR = """
    <div>
      <a href="/monetarypolicy/files/monetary20260128a1.pdf">PDF</a>
      <a href="/newsevents/pressreleases/monetary20260128a.htm">HTML</a>
      <a href="/newsevents/pressreleases/monetary20260128a1.htm">Implementation Note</a>
      <a href="/newsevents/pressreleases/monetary20260128b.htm">Statement on
         Longer-Run Goals and Monetary Policy Strategy</a>
    </div>
    """

    def _anchors(self, html):
        return BeautifulSoup(html, "html.parser").find_all("a")

    def test_historical_picks_the_policy_statement(self):
        assert S._statement_url(self._anchors(self.HISTORICAL)) == \
            S.BASE + "/newsevents/pressreleases/monetary20180131a.htm"

    def test_calendar_picks_the_policy_statement(self):
        assert S._statement_url(self._anchors(self.CALENDAR)) == \
            S.BASE + "/newsevents/pressreleases/monetary20260128a.htm"

    def test_longer_run_goals_is_never_chosen(self):
        html = """
        <div><a href="/newsevents/pressreleases/monetary20250822a.htm">Statement on
        Longer-Run Goals and Monetary Policy Strategy</a></div>
        """
        # The August 2025 notation vote published only this document.
        assert S._statement_url(self._anchors(html)) is None

    def test_implementation_note_alone_is_not_a_statement(self):
        html = ('<div><a href="/newsevents/pressreleases/monetary20260128a1.htm">'
                'Implementation Note</a></div>')
        assert S._statement_url(self._anchors(html)) is None


class TestParseCalendar:
    HTML = """
    <div class="panel panel-default">
      <h4>2026 FOMC Meetings</h4>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month"><strong>January</strong></div>
        <div class="fomc-meeting__date">27-28</div>
        <div class="col-xs-12">Statement:
          <a href="/monetarypolicy/files/monetary20260128a1.pdf">PDF</a> |
          <a href="/newsevents/pressreleases/monetary20260128a.htm">HTML</a>
          <a href="/newsevents/pressreleases/monetary20260128a1.htm">Implementation Note</a>
        </div>
        <div class="col-xs-12 fomc-meeting__minutes">Minutes:
          <a href="/monetarypolicy/files/fomcminutes20260128.pdf">PDF</a> |
          <a href="/monetarypolicy/fomcminutes20260128.htm">HTML</a>
        </div>
      </div>
    </div>
    """

    def test_extracts_one_meeting(self):
        meetings = S.parse_calendar(self.HTML)
        assert len(meetings) == 1

    def test_meeting_date_is_the_last_day(self):
        assert S.parse_calendar(self.HTML)[0].meeting_date == "2026-01-28"

    def test_urls_are_absolute_and_html(self):
        meeting = S.parse_calendar(self.HTML)[0]
        assert meeting.minutes_url == S.BASE + "/monetarypolicy/fomcminutes20260128.htm"
        assert meeting.statement_url == \
            S.BASE + "/newsevents/pressreleases/monetary20260128a.htm"

    def test_future_meeting_without_documents_still_parses(self):
        html = """
        <div class="panel panel-default">
          <h4>2026 FOMC Meetings</h4>
          <div class="row fomc-meeting">
            <div class="fomc-meeting__month"><strong>December</strong></div>
            <div class="fomc-meeting__date">8-9</div>
          </div>
        </div>
        """
        meeting = S.parse_calendar(html)[0]
        assert meeting.meeting_date == "2026-12-09"
        assert meeting.minutes_url is None


class TestParseHistorical:
    HTML = """
    <div class="panel panel-default">
      <h5>January 30-31 Meeting - 2018</h5>
      <a href="/monetarypolicy/beigebook201801.htm">HTML</a>
      <a href="/newsevents/pressreleases/monetary20180131a.htm">Statement</a>
      <a href="/newsevents/pressreleases/monetary20180131b.htm">Statement on
         Longer-Run Goals and Monetary Policy Strategy</a>
      <a href="/monetarypolicy/fomcminutes20180131.htm">HTML</a>
      <a href="/monetarypolicy/files/FOMC20180131meeting.pdf">Transcript</a>
    </div>
    """

    def test_parses_meeting_date_from_the_minutes_href(self):
        assert S.parse_historical(self.HTML, 2018)[0].meeting_date == "2018-01-31"

    def test_picks_minutes_html_not_the_beige_book(self):
        meeting = S.parse_historical(self.HTML, 2018)[0]
        assert meeting.minutes_url == S.BASE + "/monetarypolicy/fomcminutes20180131.htm"

    def test_picks_the_policy_statement(self):
        meeting = S.parse_historical(self.HTML, 2018)[0]
        assert meeting.statement_url == \
            S.BASE + "/newsevents/pressreleases/monetary20180131a.htm"

    def test_panels_that_are_not_meetings_are_skipped(self):
        html = '<div class="panel panel-default"><h5>Beige Book</h5></div>'
        assert S.parse_historical(html, 2018) == []


class TestDecode:
    class FakeResponse:
        """The Fed sends text/html with no charset, so requests guesses latin-1."""
        def __init__(self, content, apparent="UTF-8"):
            self.content = content
            self.encoding = "ISO-8859-1"
            self.apparent_encoding = apparent

        @property
        def text(self):
            return self.content.decode(self.encoding)

    def test_utf8_bytes_decode_without_mojibake(self):
        response = self.FakeResponse("July 28–29, 2026".encode("utf-8"))
        assert S.decode(response) == "July 28–29, 2026"
        assert "â" not in S.decode(response)

    def test_byte_order_mark_is_stripped(self):
        response = self.FakeResponse("﻿<html>".encode("utf-8"))
        assert S.decode(response) == "<html>"

    def test_falls_back_when_bytes_are_not_utf8(self):
        response = self.FakeResponse(b"caf\xe9", apparent="ISO-8859-1")
        assert S.decode(response) == "café"
