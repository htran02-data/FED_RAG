"""Interactive terminal application. All answers retain their source labels."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import sys

from .config import data_root
from .service import Engine, Options
import ask

HELP = """Type a complete question, including a period when possible.
Each question is independent: repeat dates and topics for follow-up questions.

:help                       show commands
:status                     corpus coverage and current settings
:k 8                        number of passages (1–50)
:section participants       filter a section; :section any clears it
:sections                   list section names present in the corpus
:doc minutes|statement|any   document filter
:expand 1                   neighboring paragraphs (0–4)
:mode auto|semantic|offline  retrieval method
:answer quotes|generated|off answer style (quotes preserve original wording)
:filter on|off              automatic date/document filtering
:rerank on|off              optional API reranking
:full 1                     full source with its link
:last                       display the last answer again
:save "answer.md"            save last answer and all sources; .json also supported
:reset                      reset settings and clear the last answer
:quit                       leave (also Ctrl-D)
Ctrl-C cancels the current question. Questions are not saved unless you use :save.
"""


def render(result):
    d = result.diagnostics
    window = d.get('time_filter')
    period = f"{window.start} to {window.end}" if window else "all meetings (no date constraint)"
    print(f"\n{result.retrieval.upper()} retrieval | {result.answer_mode} answer | {period}")
    print(f"Searched {d['candidates']:,} of {d['corpus']:,} passages")
    if d.get('doc_type'):
        print(f"Document: {d['doc_type']}")
    if d.get('section'):
        print(f"Section: {d['section']}")
    print(f"\n{result.answer}\n")
    if result.hits:
        print("Sources (use :full N to read a passage):")
        for hit in result.hits:
            print(f"  [{hit['label']}] {hit['meeting_date']} | {hit['section']}")
            print(f"       {hit['source_url']}")
    print()


def save(result, filename):
    path = Path(filename).expanduser()
    text = (json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n"
            if path.suffix.lower() == '.json' else result.markdown())
    # Explicit export never silently replaces an existing file.
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)
    return path.resolve()


def status(engine, options):
    s = engine.summary
    print(f"\nAsk the Fed — {s['passages']:,} passages, {s['meetings']} meetings")
    print(f"Coverage: {s['first']} to {s['last']}")
    print(f"Retrieval: {options.retrieval}; answer: {options.answer_mode}; "
          f"passages: {options.k}; expand: {options.expand}")
    print(f"Section: {options.section or 'any'}; document: {options.doc_type or 'any'}; "
          f"filter: {'on' if options.use_filter else 'off'}; rerank: {'on' if options.rerank else 'off'}")


def interactive(engine, options):
    try:
        import readline  # in-memory history, no questions written to disk
    except ImportError:
        pass
    initial = replace(options)
    last = None
    status(engine, options)
    print("Type a question or :help. Ctrl-D or :quit exits.")
    if options.retrieval == 'offline' or (options.retrieval == 'auto' and not os.environ.get('VOYAGE_API_KEY')):
        print("Offline keyword search is active; no API calls are made with quoted answers.")
    while True:
        try:
            line = input('\nfed> ').strip()
            if not line:
                continue
            if not line.startswith(':'):
                last = engine.query(line, options)
                render(last)
                continue
            parts = shlex.split(line[1:])
            if not parts:
                continue
            command, args = parts[0].lower(), parts[1:]
            argument = ' '.join(args)
            if command in ('quit', 'q', 'exit'):
                return 0
            if command in ('help', 'h', '?'):
                print(HELP)
            elif command == 'status':
                status(engine, options)
            elif command == 'sections':
                for row in engine.conn.execute('SELECT DISTINCT section FROM chunks ORDER BY section'):
                    print(f"  {row[0]}")
            elif command == 'reset':
                options, last = replace(initial), None
                print('Settings reset; previous answer cleared.')
            elif command == 'last':
                if last is None:
                    print('Ask a question first.')
                else:
                    render(last)
            elif command == 'full':
                index = int(argument) - 1
                if last is None or not 0 <= index < len(last.hits):
                    raise ValueError('No source with that number. Ask a question first.')
                hit = last.hits[index]
                print(f"\n[{hit['label']}] {hit['meeting_date']} | {hit['section']}\n{hit['source_url']}\n\n{hit['text']}")
            elif command == 'save':
                if last is None:
                    raise ValueError('Ask a question before saving.')
                if len(args) != 1:
                    raise ValueError('Use :save "path/to/answer.md" or :save answer.json')
                print(f"Saved {save(last, args[0])}")
            else:
                updated = replace(options)
                if command in ('k', 'expand'):
                    setattr(updated, command, int(argument))
                elif command == 'doc':
                    updated.doc_type = None if argument == 'any' else argument
                elif command == 'section':
                    sections = [r[0] for r in engine.conn.execute('SELECT DISTINCT section FROM chunks')]
                    matches = [s for s in sections if argument.lower() in s.lower()]
                    if argument == 'any':
                        updated.section = None
                    elif not argument or len(matches) != 1:
                        raise ValueError('Section is missing, unknown, or ambiguous. Use :sections and enter a full name.')
                    else:
                        updated.section = matches[0]
                elif command == 'mode':
                    updated.retrieval = argument
                elif command == 'answer':
                    updated.answer_mode = 'quotes' if argument == 'on' else argument
                elif command in ('filter', 'rerank'):
                    if argument not in ('on', 'off'):
                        raise ValueError('Use on or off.')
                    setattr(updated, 'use_filter' if command == 'filter' else command, argument == 'on')
                else:
                    raise ValueError('Unknown command. Type :help.')
                updated.validate()
                options = updated
                status(engine, options)
        except EOFError:
            print('\nGoodbye.')
            return 0
        except KeyboardInterrupt:
            print('\nCancelled. Ask another question or type :quit.')
        except SystemExit as error:
            print(f'Cannot answer: {error}')
        except Exception as error:
            print(f'Error: {error}')


def doctor(engine):
    status(engine, Options())
    checks = []
    checks.append(('SQLite integrity', engine.conn.execute('PRAGMA quick_check').fetchone()[0] == 'ok'))
    checks.append(('Offline search index', bool(engine.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='chunks_fts'").fetchone())))
    try:
        engine.vectors()
        checks.append(('Vector/database alignment', True))
    except (ValueError, OSError) as error:
        checks.append((f'Vector/database alignment: {error}', False))
    for name, ok in checks:
        print(f"{'OK' if ok else 'FAIL'}  {name}")
    for key in ('VOYAGE_API_KEY', 'ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL'):
        print(f"{key}: {'configured' if os.environ.get(key) else 'not configured (optional for offline quoted answers)'}")
    print('This check does not contact API providers or verify account access.')
    return 0 if all(ok for _, ok in checks) else 1


def parser():
    p = argparse.ArgumentParser(description='Ask the Fed in your terminal. Omit the question to chat.')
    p.add_argument('question', nargs='?')
    p.add_argument('-i', '--interactive', action='store_true')
    p.add_argument('--doctor', action='store_true', help='check local corpus, vectors, and configuration')
    p.add_argument('--data-dir', type=Path, default=data_root())
    p.add_argument('--offline', action='store_true', help='keyword retrieval with no API calls')
    p.add_argument('--mode', choices=['auto', 'semantic', 'offline'], default='auto')
    p.add_argument('--answer', choices=['quotes', 'generated', 'off'], default='quotes')
    p.add_argument('--retrieve-only', action='store_true')
    p.add_argument('-k', type=int, default=8)
    p.add_argument('--expand', type=int, default=1)
    p.add_argument('--section')
    p.add_argument('--doc-type', choices=['minutes', 'statement'])
    p.add_argument('--no-filter', action='store_true')
    p.add_argument('--no-hybrid', action='store_true')
    p.add_argument('--rerank', action='store_true')
    p.add_argument('--json', action='store_true', help='machine-readable single-question output')
    p.add_argument('--output', type=Path, help='save answer and sources to a new .md or .json file')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.json or args.output) and (args.interactive or not args.question):
        p.error('--json and --output require a single question, without --interactive')
    if args.offline and (args.answer == 'generated' or args.rerank):
        p.error('--offline cannot be combined with generated answers or API reranking')
    options = Options(k=args.k, expand=args.expand, section=args.section,
                      doc_type=args.doc_type, use_filter=not args.no_filter,
                      hybrid=not args.no_hybrid, rerank=args.rerank,
                      retrieval='offline' if args.offline else args.mode,
                      answer_mode='off' if args.retrieve_only else args.answer)
    engine = None
    try:
        options.validate()
        engine = Engine(args.data_dir)
        if args.doctor:
            return doctor(engine)
        if args.interactive or not args.question:
            return interactive(engine, options)
        result = engine.query(args.question, options)
        if args.output:
            save(result, args.output)
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False))
        else:
            render(result)
        return 0
    except KeyboardInterrupt:
        print('\nCancelled.', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.close()
