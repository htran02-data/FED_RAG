"""One question/answer pipeline for the terminal and browser."""
from dataclasses import dataclass
from pathlib import Path
import os
import sqlite3

import numpy as np
import ask
import rates
from .config import data_root


@dataclass
class Options:
    k: int = 8
    section: str | None = None
    doc_type: str | None = None
    use_filter: bool = True
    expand: int = 1
    rerank: bool = False
    retrieval: str = "auto"
    answer_mode: str = "quotes"
    hybrid: bool = True

    def validate(self):
        if not 1 <= self.k <= 50:
            raise ValueError("Passage count must be between 1 and 50.")
        if not 0 <= self.expand <= 4:
            raise ValueError("Expansion must be between 0 and 4.")
        if self.doc_type not in (None, "minutes", "statement"):
            raise ValueError("Document must be minutes, statement, or any.")
        if self.retrieval not in ("auto", "semantic", "offline"):
            raise ValueError("Retrieval must be auto, semantic, or offline.")
        if self.answer_mode not in ("quotes", "generated", "off"):
            raise ValueError("Answer mode must be quotes, generated, or off.")


@dataclass
class Result:
    question: str
    answer: str
    hits: list
    diagnostics: dict
    retrieval: str
    answer_mode: str

    def as_dict(self):
        diagnostics = dict(self.diagnostics)
        window = diagnostics.get("time_filter")
        if window:
            diagnostics["time_filter"] = vars(window)
        return {"question": self.question, "answer": self.answer,
                "retrieval": self.retrieval, "answer_mode": self.answer_mode,
                "diagnostics": diagnostics, "sources": self.hits}

    def markdown(self):
        lines = [f"# {self.question}", "", self.answer, "", "## Sources", ""]
        for hit in self.hits:
            lines += [f"[{hit['label']}] {hit['meeting_date']} | {hit['section']}",
                      hit["source_url"], "", hit["text"], ""]
        lines += [f"Retrieval: {self.retrieval}; answer: {self.answer_mode}"]
        return "\n".join(lines) + "\n"


class Engine:
    def __init__(self, root=None):
        self.root = Path(root) if root else data_root()
        db = self.root / "fed.db"
        if not db.is_file():
            raise ValueError(f"Store missing: {db}. See README.md for the build steps.")
        self.conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self.matrix = None
        try:
            self.summary = dict(self.conn.execute(
                "SELECT COUNT(*) passages, COUNT(DISTINCT meeting_date) meetings, "
                "MIN(meeting_date) first, MAX(meeting_date) last FROM chunks"
            ).fetchone())
            chunks = self.root / "data/chunks.jsonl"
            self.eras = rates.named_eras(rates.load_path(chunks)) if chunks.exists() else {}
        except Exception:
            self.close()
            raise

    def close(self):
        self.conn.close()

    def vectors(self):
        if self.matrix is None:
            path = self.root / "vectors.npy"
            if not path.is_file():
                raise ValueError("vectors.npy is missing. Use --offline or rebuild the vector store.")
            matrix = np.load(path, mmap_mode="r", allow_pickle=False)
            bounds = self.conn.execute(
                "SELECT MIN(vector_index), MAX(vector_index), COUNT(DISTINCT vector_index), "
                "COUNT(*) FROM chunks").fetchone()
            if (matrix.ndim != 2 or not bounds[3] or bounds[0] != 0
                    or bounds[1] != len(matrix) - 1 or bounds[2] != bounds[3]
                    or bounds[3] != len(matrix)):
                raise ValueError("Database and vector matrix do not match. Use a matching store pair.")
            self.matrix = matrix
        return self.matrix

    def query(self, question, options=None):
        options = options or Options()
        options.validate()
        question = question.strip()
        if not question:
            raise ValueError("Enter a question first.")
        mode = options.retrieval
        if mode == "auto":
            mode = "semantic" if os.environ.get("VOYAGE_API_KEY") else "offline"
        if mode == "offline" and options.rerank:
            raise ValueError("Reranking uses an API. Turn it off for offline retrieval.")
        if mode == "offline" and options.answer_mode == "generated":
            raise ValueError("Generated answers use an API. Use quotes for offline mode.")
        window = ask.parse_temporal(question, era_names=self.eras) if options.use_filter else None
        doc = options.doc_type or (ask.parse_doc_type(question) if options.use_filter else None)
        if mode == "semantic":
            hits, diagnostics = ask.search(
                question, self.conn, self.vectors(), k=options.k, time_filter=window,
                section=options.section, doc_type=doc, expand=options.expand,
                hybrid=options.hybrid, reranker=ask.rerank if options.rerank else None)
        else:
            exists = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='chunks_fts'").fetchone()
            if not exists:
                raise ValueError("Offline search requires the chunks_fts index. Run embed.py --help for store setup.")
            rows = ask.candidates(self.conn, window, options.section, doc)
            ranks = ask.lexical_ranks(self.conn, question, [r['content_hash'] for r in rows]) if rows else {}
            ranked = sorted((r for r in rows if r['content_hash'] in ranks),
                            key=lambda r: ranks[r['content_hash']])[:options.k]
            hits = [dict(row, label=f"S{i}", score=1 / ranks[row['content_hash']])
                    for i, row in enumerate(ranked, 1)]
            if options.expand:
                hits = ask.expand_hits(hits, self.conn, radius=options.expand)
            diagnostics = {"time_filter": window, "candidates": len(rows),
                           "corpus": self.summary['passages'], "expand": options.expand}
        diagnostics['doc_type'] = doc
        diagnostics['section'] = options.section
        if not hits:
            response = "No passages matched. Try a wider period or a different topic."
        elif options.answer_mode == "off":
            response = "Answer disabled; retrieved sources follow."
        elif options.answer_mode == "generated":
            model = os.environ.get("ANTHROPIC_MODEL")
            if not model:
                raise ValueError("Set ANTHROPIC_MODEL to a model available in your Anthropic account.")
            response = ask.answer(question, hits, model=model)
        else:
            selected = ask.extractive_answer(question, hits)
            response = "\n\n".join(
                f'“{sentence}”\n[{hit["label"]}] {hit["meeting_date"]} | {hit["section"]}'
                for _, _, sentence, hit in selected)
            if not response:
                response = "The retrieved passages do not provide a matching quoted answer. Inspect the sources or rephrase the question."
        return Result(question, response, hits, diagnostics, mode, options.answer_mode)
