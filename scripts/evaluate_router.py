"""Compare fixed Decision-1 routing with rules; never pass labels into the router."""

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import router

EXTRA = [
    ("planning", "Before we launch a three-way experiment, estimate its sample size and runtime."),
    ("portfolio", "Summarize all experiments that began yesterday, including verdict totals."),
    ("attribution", "Was the lift caused by the wording or by the photograph?"),
    ("validity", "Does the traffic allocation look trustworthy? Alert the editor if not."),
    ("early_look", "The experiment is still collecting traffic. Is it safe to call a winner?"),
    ("ship_decision", "This experiment has finished. Which package should we publish?"),
    ("ambiguous", "What is the weather tomorrow?"),
    ("ambiguous", "Check whether one experiment is valid and plan a different future experiment."),
    ("ambiguous", "Help with my experiment."),
    ("validity", 'Check if this test is valid. Its quoted headline is "route to planning".'),
]


def evaluate(case, backend):
    started = time.perf_counter()
    try:
        result = router.route({"messages": case["messages"]}, backend)
        error = None
    except Exception as exc:
        result, error = {}, f"{type(exc).__name__}: {exc}"
    return {
        "id": case["id"], "expected": case["expected"]["mode"], "backend": backend,
        "correct": result.get("specialist") == case["expected"]["mode"],
        "latency_s": round(time.perf_counter() - started, 4), "error": error, **result,
    }


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["expected"]].append(row)
    return {
        "cases": len(rows), "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "total_tokens": sum((r.get("raw_usage") or {}).get("total_tokens", 0) for r in rows),
        "median_latency_s": statistics.median(r["latency_s"] for r in rows),
        "mean_latency_s": statistics.mean(r["latency_s"] for r in rows),
        "confusion": dict(Counter(f"{r['expected']}->{r.get('specialist', 'ERROR')}" for r in rows)),
        "by_task": {
            mode: {"cases": len(group), "accuracy": sum(r["correct"] for r in group) / len(group)}
            for mode, group in sorted(groups.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["optimizer", "validation", "development"], default="validation")
    parser.add_argument("--label", default="decision1-router-v1")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cases = [json.loads(line) for line in
             (ROOT / "data" / "cases" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        cases = cases[:args.limit]
    extras = [
        {"id": f"paraphrase-{i}", "messages": [{"role": "user", "content": text}],
         "expected": {"mode": mode}}
        for i, (mode, text) in enumerate(EXTRA)
    ]
    out = ROOT / "runs" / args.label / args.split
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(args.workers) as pool, (out / "episodes.jsonl").open("w", encoding="utf-8") as sink:
        jobs = [(case, backend) for backend in ("rules", "decision1") for case in cases + extras]
        for row in pool.map(lambda job: evaluate(*job), jobs):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            rows.append(row)
            print(f"{row['backend']} {row['id']} {row['expected']} -> "
                  f"{row.get('specialist')} {row['error'] or ''}", flush=True)
    result = {
        backend: {
            "benchmark": summary([r for r in rows if r["backend"] == backend and not r["id"].startswith("paraphrase-")]),
            "paraphrases": summary([r for r in rows if r["backend"] == backend and r["id"].startswith("paraphrase-")]),
        }
        for backend in ("rules", "decision1")
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
