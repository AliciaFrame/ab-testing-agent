"""Run the agent locally against a model deployment and grade every episode.

Example:
  $env:MAI_OPENAI_BASE_URL = "https://<resource>.openai.azure.com/openai/v1/"
  $env:MAI_MODEL_DEPLOYMENT_NAME = "<deployment>"
  python scripts/run_eval.py --split development --runs 3 --label baseline
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

import agent  # noqa: E402
import grader  # noqa: E402

CHECKS = list(grader.WEIGHTS)


def run_one(case, run):
    started = time.time()
    record = {"case_id": case["id"], "run": run, "mode": case["expected"]["mode"],
              "expected_verdict": case["expected"].get("verdict", "-")}
    try:
        result = agent.run_episode({"messages": case["messages"]})
        record.update(output_text=result["output_text"], output_tools=result["output_tools"],
                      calls=result["calls"], error=None)
    except Exception as error:  # noqa: BLE001 -- runtime errors count as zero, never retried
        record.update(output_text=None, output_tools=[], calls=getattr(error, "calls", []),
                      error=f"{type(error).__name__}: {error}")
    record["latency_s"] = round(time.time() - started, 2)
    checks = grader.check(record, case)
    record["checks"] = checks
    record["strict_pass"] = record["error"] is None and all(checks.values())
    record["reward"] = 0.0 if record["error"] else grader.grade(record, case)
    submit_tool = grader.SUBMIT_TOOL[record["mode"]]
    submits = [c for c in grader.parse_trace(record) if c["name"] == submit_tool]
    record["submitted_verdict"] = (
        submits[-1]["arguments"].get("verdict", "submitted") if submits else None
    )
    usage = [c.get("raw_usage") or {} for c in record["calls"]]
    record["input_tokens"] = sum(u.get("input_tokens", 0) for u in usage)
    record["output_tokens"] = sum(u.get("output_tokens", 0) for u in usage)
    return record


def summarize(records):
    n = len(records)
    out = {"episodes": n,
           "strict_pass": sum(r["strict_pass"] for r in records) / n,
           "mean_reward": sum(r["reward"] for r in records) / n,
           "errors": sum(r["error"] is not None for r in records),
           "mean_latency_s": sum(r["latency_s"] for r in records) / n,
           "mean_input_tokens": sum(r["input_tokens"] for r in records) / n,
           "mean_output_tokens": sum(r["output_tokens"] for r in records) / n}
    for check in CHECKS:
        out[f"check_{check}"] = sum(r["checks"][check] for r in records) / n
    return out


TRANSPORT_ERRORS = ("RateLimitError", "APIConnectionError", "APITimeoutError",
                    "InternalServerError")


def is_transport_error(record):
    return bool(record["error"]) and any(name in record["error"] for name in TRANSPORT_ERRORS)


def write_summary(out_dir, args, records):
    summary = {"label": args.label, "split": args.split, "runs": args.runs,
               "model": agent.policy_contract()["model"], "overall": summarize(records),
               "transport_reruns": sum(r.get("transport_reruns", 0) for r in records)}
    for mode in sorted({r["mode"] for r in records}):
        summary[mode] = summarize([r for r in records if r["mode"] == mode])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def rerun_transport_errors(args, cases, out_dir):
    """Re-run only episodes that failed on transport (429s, timeouts, 5xx), at most
    --max-reruns times each. Model failures are never re-run."""
    path = out_dir / "episodes.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_id = {c["id"]: c for c in cases}
    for _ in range(args.max_reruns):
        todo = [i for i, r in enumerate(records) if is_transport_error(r)]
        if not todo:
            break
        print(f"re-running {len(todo)} transport failures", flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            fresh = pool.map(lambda i: run_one(by_id[records[i]["case_id"]], records[i]["run"]),
                             todo)
            for i, record in zip(todo, fresh):
                record["transport_reruns"] = records[i].get("transport_reruns", 0) + 1
                records[i] = record
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    write_summary(out_dir, args, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun-transport-errors", action="store_true",
                        help="Only re-run episodes in an existing run that failed on transport")
    parser.add_argument("--max-reruns", type=int, default=3)
    args = parser.parse_args()
    path = ROOT / "data" / "cases" / f"{args.split}.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    cases = cases[: args.limit] if args.limit else cases
    out_dir = ROOT / "runs" / args.label / args.split
    if args.rerun_transport_errors:
        rerun_transport_errors(args, cases, out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(case, run) for run in range(args.runs) for case in cases]
    records = []
    with ThreadPoolExecutor(args.workers) as pool, open(
        out_dir / "episodes.jsonl", "w", encoding="utf-8"
    ) as sink:
        for record in pool.map(lambda job: run_one(*job), jobs):
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            records.append(record)
            mark = "PASS" if record["strict_pass"] else "fail"
            print(f"{mark} {record['case_id']} r{record['run']} "
                  f"{record['expected_verdict']}->{record['submitted_verdict']} "
                  f"{record['error'] or ''}", flush=True)
    write_summary(out_dir, args, records)


if __name__ == "__main__":
    main()
