"""Terminal workflow regressions over the real, cached corpus; no network calls."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import ask
import eval as evaluation
from fedrag.cli import interactive, main, save
from fedrag.config import ROOT
from fedrag.service import Engine, Options


@pytest.fixture
def engine():
    value = Engine(ROOT)
    yield value
    value.close()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.delenv('VOYAGE_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('This test must not make API calls')
    monkeypatch.setattr(ask, 'embed_query', forbidden)
    monkeypatch.setattr(ask, 'rerank', forbidden)
    monkeypatch.setattr(ask, 'answer', forbidden)


def test_offline_real_answer_and_prefilter(engine):
    result = engine.query('What did participants say about tariffs in 2018?',
                          Options(retrieval='offline', doc_type='minutes', k=3))
    assert result.retrieval == 'offline'
    assert '[S1]' in result.answer
    assert result.diagnostics['candidates'] < result.diagnostics['corpus']
    assert result.hits
    for hit in result.hits:
        assert hit['meeting_date'].startswith('2018')
        assert hit['doc_type'] == 'minutes'
        assert hit['source_url'].startswith('https://www.federalreserve.gov/')
    for _, _, sentence, hit in ask.extractive_answer(result.question, result.hits):
        assert sentence in hit['text']


def test_auto_no_key_is_explicit_offline(engine):
    assert engine.query('inflation in 2020').retrieval == 'offline'


def test_statement_auto_filter(engine):
    result = engine.query('What did the December 2020 statement say about inflation?')
    assert result.hits
    assert all(h['doc_type'] == 'statement' and h['meeting_date'].startswith('2020-12') for h in result.hits)


def test_outside_corpus_no_invention(engine):
    result = engine.query('inflation in 1900')
    assert not result.hits
    assert result.answer.startswith('No passages matched')


@pytest.mark.parametrize('options', [Options(k=0), Options(k=51), Options(expand=-1),
                                     Options(doc_type='typo'), Options(answer_mode='typo'),
                                     Options(retrieval='typo')])
def test_invalid_options(engine, options):
    with pytest.raises(ValueError):
        engine.query('inflation', options)


@pytest.mark.parametrize('options', [Options(retrieval='offline', rerank=True),
                                     Options(retrieval='offline', answer_mode='generated')])
def test_offline_never_uses_apis(engine, options):
    with pytest.raises(ValueError, match='API'):
        engine.query('inflation', options)


def test_read_only_store(engine):
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        engine.conn.execute('DELETE FROM chunks')


def test_save_citations_and_prevent_overwrite(engine, tmp_path):
    result = engine.query('participants tariffs in 2018')
    filename = tmp_path / 'answer.json'
    save(result, filename)
    content = json.loads(filename.read_text())
    assert content['sources'][0]['source_url']
    assert content['diagnostics']['time_filter']['start'] == '2018-01-01'
    with pytest.raises(FileExistsError):
        save(result, filename)
    md = tmp_path / 'answer.md'
    save(result, md)
    assert result.hits[0]['text'] in md.read_text()


def test_interactive_recovers_from_bad_command_and_supports_save(engine, monkeypatch, capsys, tmp_path):
    destination = tmp_path / 'my answer.md'
    commands = iter([':doc typo', ':filter maybe', ':k 0', ':k 2',
                     'participants tariffs in 2018', ':last', ':full 1',
                     f':save "{destination}"', ':reset', ':last', ':quit'])
    monkeypatch.setattr('builtins.input', lambda _: next(commands))
    assert interactive(engine, Options(retrieval='offline')) == 0
    text = capsys.readouterr().out
    assert 'Document must be' in text
    assert 'Use on or off' in text
    assert 'between 1 and 50' in text
    assert destination.exists()
    assert 'Ask a question first.' in text
    assert 'tariffs' in text


def test_interactive_keeps_launch_settings(engine, monkeypatch):
    commands = iter(['inflation', ':quit'])
    monkeypatch.setattr('builtins.input', lambda _: next(commands))
    captured = []
    original = engine.query
    def query(question, options):
        captured.append(options)
        return original(question, options)
    monkeypatch.setattr(engine, 'query', query)
    interactive(engine, Options(k=2, expand=0, doc_type='statement', answer_mode='off', retrieval='offline'))
    assert captured[0].k == 2
    assert captured[0].expand == 0
    assert captured[0].doc_type == 'statement'
    assert captured[0].answer_mode == 'off'


def test_ctrl_c_recovers_and_eof_exits(engine, monkeypatch, capsys):
    events = iter([KeyboardInterrupt(), EOFError()])
    def read(_):
        raise next(events)
    monkeypatch.setattr('builtins.input', read)
    assert interactive(engine, Options()) == 0
    assert 'Cancelled' in capsys.readouterr().out


def test_cli_json_outside_project(tmp_path):
    process = subprocess.run([sys.executable, str(ROOT / 'ask.py'), '--offline',
                              '--json', 'participants tariffs in 2018'],
                             cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result['sources']
    assert result['retrieval'] == 'offline'


def test_cli_error_code_for_missing_store(tmp_path, capsys):
    assert main(['--data-dir', str(tmp_path), 'inflation']) == 1
    assert 'Store missing' in capsys.readouterr().err


def test_semantic_results_unchanged(engine, monkeypatch):
    gold = evaluation.load_gold()
    embedder = evaluation.cached_query_embedder([item['question'] for item in gold])
    monkeypatch.setattr(ask, 'embed_query', embedder)
    for item in gold:
        question = item['question']
        result = engine.query(question, Options(retrieval='semantic'))
        expected, _ = ask.search(question, engine.conn, engine.vectors(),
                                  time_filter=ask.parse_temporal(question, era_names=engine.eras),
                                  doc_type=ask.parse_doc_type(question), embedder=embedder)
        assert [h['content_hash'] for h in result.hits] == [h['content_hash'] for h in expected]
        assert [h['text'] for h in result.hits] == [h['text'] for h in expected]


def test_generated_mode_uses_explicit_model(engine, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_MODEL', 'configured-test-model')
    gold = evaluation.load_gold()
    question = gold[0]['question']
    embedder = evaluation.cached_query_embedder([g['question'] for g in gold])
    monkeypatch.setattr(ask, 'embed_query', embedder)
    calls = []
    def generate(question, hits, model):
        calls.append((question, hits, model))
        return 'A test answer [S1].'
    monkeypatch.setattr(ask, 'answer', generate)
    result = engine.query(question, Options(retrieval='semantic', answer_mode='generated'))
    assert result.answer == 'A test answer [S1].'
    assert calls[0][2] == 'configured-test-model'
    assert calls[0][1][0]['source_url']


def test_browser_answer_survives_settings_change():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=20)
    assert not app.exception
    app.selectbox[0].select('offline').run()
    app.text_input[0].set_value('What did participants say about tariffs in 2018?')
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert not app.error
    assert len(app.expander) > 0
    assert any('tariffs' in item.value for item in app.markdown)
    app.slider[0].set_value(3).run(timeout=20)
    assert not app.exception
    assert len(app.expander) > 0
