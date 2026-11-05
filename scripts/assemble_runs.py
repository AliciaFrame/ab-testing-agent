"""Assemble a 270-episode result set from a base run plus per-task reruns.

Example:
  python scripts/assemble_runs.py --base routed-gpt6-sol --label routed-gpt6-sol-r4 \
      --task planning=routed-gpt6-sol-r4-planning --task validity=routed-gpt6-sol-r4-validity
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_runs
import run_eval


def assemble(base_label, label, tasks):
    base = compare_runs.load(base_label, "development")
    records = []
    for mode in compare_runs.MODES:
        if mode in tasks:
            rows = compare_runs.load(tasks[mode], "development")
            if {r["mode"] for r in rows} != {mode} or len(rows) != 45:
                raise ValueError(f"{tasks[mode]} is not a complete 45-episode {mode} run")
            for row in rows:
                row["measurement_source"] = f"rerun:{tasks[mode]}"
        else:
            rows = [dict(r, measurement_source=f"reused:{base_label}") for r in base if r["mode"] == mode]
        records += rows
    if len(records) != 270 or {(r["case_id"], r["run"]) for r in records} != {
        (r["case_id"], r["run"]) for r in base
    }:
        raise ValueError("Assembled episodes do not match the base case/run set")
    if any(r["error"] for r in records):
        raise ValueError("Assembled episodes contain unresolved runtime errors")
    out = ROOT / "runs" / label / "development"
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    summary = {
        "label": label, "split": "development", "runs": 3, "assembled_from": {
            "base": base_label, "tasks": tasks,
        },
        "overall": run_eval.summarize(records),
        **{mode: run_eval.summarize([r for r in records if r["mode"] == mode]) for mode in compare_runs.MODES},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{label}: {summary['overall']['strict_pass']:.1%}, {summary['overall']['total_tokens']:,} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--task", action="append", default=[], help="mode=label")
    args = parser.parse_args()
    assemble(args.base, args.label, dict(item.split("=", 1) for item in args.task))
