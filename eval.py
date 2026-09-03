"""
Retrieval evaluation against the gold set in data/gold.jsonl.

Gold pairs are derived from passages that were read in data/raw/, and each one
is labelled with the meeting date and section it came from. None of them are
written from memory, so a pair can always be checked against its source.

Metrics:
    recall@k          the exact gold passage is in the top k
    meeting recall@k  some passage from the gold meeting is in the top k
    MRR               1/rank of the gold passage, averaged
    wrong-meeting     share of retrieved passages from a meeting other than the
                      gold one -- the dominant failure mode for this corpus,
                      where the right sentence is returned from the wrong year

Usage:
    python eval.py                 # evaluate with temporal filtering on
    python eval.py --compare       # with vs. without the date filter
    python eval.py --k 5 --verbose
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np

import ask
import rates

GOLD_PATH = pathlib.Path("data/gold.jsonl")


QUERY_CACHE = pathlib.Path("data/query_vectors.npz")


def cached_query_embedder(questions, cache_path=QUERY_CACHE):
    """
    Embed every gold question once and remember the result on disk.

    The eval runs each question twice -- filtered and unfiltered -- and the
    workflow calls for running it before and after every retrieval change. A
    free Voyage account allows 3 requests a minute, so re-embedding 22 questions
    on each run makes the eval too expensive to use the way it is meant to be
    used. Query vectors depend only on the question text and the model, neither
    of which changes when retrieval does, so they are cached and reused.
    """
    import numpy as np

    questions = list(questions)
    key = lambda q: hashlib.sha256(
        f"{ask.EMBED_MODEL}|{q}".encode("utf-8")).hexdigest()[:16]

    cache = {}
    if cache_path.exists():
        with np.load(cache_path) as stored:
            cache = {name: stored[name] for name in stored.files}

    missing = [q for q in questions if key(q) not in cache]
    if missing:
        import voyageai

        client = voyageai.Client()
        result = client.embed(missing, model=ask.EMBED_MODEL, input_type="query")
        matrix = np.asarray(result.embeddings, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        for question, vector in zip(missing, matrix):
            cache[key(question)] = vector
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **cache)
        print(f"  embedded {len(missing)} new question(s); "
              f"{len(questions) - len(missing)} served from cache")
    else:
        print(f"  all {len(questions)} question vectors served from cache")

    def embed(question):
        return cache[key(question)]

    return embed


def load_gold(path=GOLD_PATH):
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(gold, conn, matrix, k=8, use_filter=True, era_names=None,
             embedder=None, verbose=False, hybrid=True):
    """Run every gold question and score the retrieved passages."""
    results = []
    for item in gold:
        time_filter = (ask.parse_temporal(item["question"], era_names=era_names)
                       if use_filter else None)
        doc_type = ask.parse_doc_type(item["question"]) if use_filter else None
        hits, diagnostics = ask.search(item["question"], conn, matrix, k=k,
                                       time_filter=time_filter, doc_type=doc_type,
                                       embedder=embedder, hybrid=hybrid)

        hashes = [h["content_hash"] for h in hits]
        rank = hashes.index(item["content_hash"]) + 1 if item["content_hash"] in hashes else None
        meeting_hit = any(h["meeting_date"] == item["meeting_date"] for h in hits)
        wrong = sum(1 for h in hits if h["meeting_date"] != item["meeting_date"])

        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold_date": item["meeting_date"],
            "gold_section": item["section"],
            "rank": rank,
            "meeting_hit": meeting_hit,
            "wrong_meeting": wrong,
            "retrieved": len(hits),
            "candidates": diagnostics["candidates"],
            "filter": time_filter.label if time_filter else None,
            "top_date": hits[0]["meeting_date"] if hits else None,
            "top_section": hits[0]["section"] if hits else None,
        })

        if verbose:
            status = f"rank {rank}" if rank else "MISS"
            print(f"  {item['id']}  {status:<8} gold {item['meeting_date']}  "
                  f"top {results[-1]['top_date']}  filter={results[-1]['filter']}")
    return results


def summarize(results, k):
    total = len(results)
    retrieved = sum(r["retrieved"] for r in results) or 1
    return {
        "n": total,
        f"recall@{k}": sum(1 for r in results if r["rank"]) / total,
        f"meeting_recall@{k}": sum(1 for r in results if r["meeting_hit"]) / total,
        "mrr": sum(1 / r["rank"] for r in results if r["rank"]) / total,
        "wrong_meeting_rate": sum(r["wrong_meeting"] for r in results) / retrieved,
        "mean_candidates": sum(r["candidates"] for r in results) / total,
    }


def print_summary(title, stats):
    print(f"\n{title}")
    print("  " + "-" * 52)
    for key, value in stats.items():
        if key in ("n",):
            print(f"  {key:<22} {value}")
        elif key == "mean_candidates":
            print(f"  {key:<22} {value:.0f}")
        else:
            print(f"  {key:<22} {value:.3f}")


def print_comparison(before, after, k):
    print(f"\nRetrieval with and without the temporal filter (k={k})")
    print("  " + "-" * 66)
    print(f"  {'metric':<22}{'no filter':>12}{'filtered':>12}{'change':>14}")
    print("  " + "-" * 66)
    for key in (f"recall@{k}", f"meeting_recall@{k}", "mrr",
                "wrong_meeting_rate", "mean_candidates"):
        low, high = before[key], after[key]
        if key == "mean_candidates":
            print(f"  {key:<22}{low:>12.0f}{high:>12.0f}{high - low:>+14.0f}")
        else:
            print(f"  {key:<22}{low:>12.3f}{high:>12.3f}{high - low:>+14.3f}")
    print("\n  wrong_meeting_rate is the one to watch: lower is better.")


def print_failures(results):
    misses = [r for r in results if not r["rank"]]
    if not misses:
        print("\n  every gold passage was retrieved")
        return
    print(f"\n  {len(misses)} gold passage(s) not retrieved")
    print("  " + "-" * 66)
    for miss in misses:
        print(f"  {miss['id']}  gold {miss['gold_date']} {miss['gold_section'][:32]}")
        print(f"        top hit {miss['top_date']} {str(miss['top_section'])[:32]}"
              f"  (meeting hit: {miss['meeting_hit']})")
        print(f"        {miss['question'][:88]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--compare", action="store_true",
                        help="report retrieval with and without the date filter")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="rank by embedding similarity alone, without BM25")
    parser.add_argument("--compare-hybrid", action="store_true",
                        help="dense-only vs hybrid, both with the date filter")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gold = load_gold()
    conn, matrix = ask.load_store()
    era_names = rates.named_eras(rates.load_path())
    embedder = cached_query_embedder([g["question"] for g in gold])

    print(f"{len(gold)} gold pairs | "
          f"{conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]} chunks in the store")

    if args.compare_hybrid:
        dense = evaluate(gold, conn, matrix, args.k, use_filter=True,
                         era_names=era_names, embedder=embedder, hybrid=False)
        hybrid = evaluate(gold, conn, matrix, args.k, use_filter=True,
                          era_names=era_names, embedder=embedder, hybrid=True)
        print(f"\nDense-only vs hybrid (BM25 fused), both date-filtered (k={args.k})")
        print("  " + "-" * 66)
        print(f"  {'metric':<22}{'dense':>12}{'hybrid':>12}{'change':>14}")
        print("  " + "-" * 66)
        before, after = summarize(dense, args.k), summarize(hybrid, args.k)
        for key in (f"recall@{args.k}", f"meeting_recall@{args.k}", "mrr",
                    "wrong_meeting_rate"):
            print(f"  {key:<22}{before[key]:>12.3f}{after[key]:>12.3f}"
                  f"{after[key] - before[key]:>+14.3f}")
        for label, rows in (("dense", dense), ("hybrid", hybrid)):
            first = sum(1 for r in rows if r["rank"] == 1)
            top3 = sum(1 for r in rows if r["rank"] and r["rank"] <= 3)
            print(f"  {label:<8} rank-1 {first}/{len(rows)}   top-3 {top3}/{len(rows)}")
        print_failures(hybrid)
        return

    if args.compare:
        unfiltered = evaluate(gold, conn, matrix, args.k, use_filter=False,
                              era_names=era_names, embedder=embedder,
                              verbose=args.verbose, hybrid=not args.no_hybrid)
        filtered = evaluate(gold, conn, matrix, args.k, use_filter=True,
                            era_names=era_names, embedder=embedder,
                            verbose=args.verbose, hybrid=not args.no_hybrid)
        print_comparison(summarize(unfiltered, args.k),
                         summarize(filtered, args.k), args.k)
        print_failures(filtered)
        return

    results = evaluate(gold, conn, matrix, args.k, use_filter=not args.no_filter,
                       era_names=era_names, embedder=embedder,
                       verbose=args.verbose, hybrid=not args.no_hybrid)
    label = "without the temporal filter" if args.no_filter else "with the temporal filter"
    print_summary(f"Retrieval {label} (k={args.k})", summarize(results, args.k))
    print_failures(results)


if __name__ == "__main__":
    main()
