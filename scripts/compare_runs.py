"""Compare a candidate run with the baseline under docs/acceptance.md.

Usage: python scripts/compare_runs.py --baseline baseline --candidate opt-r3-cand1
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ["validity", "ship_decision", "early_look", "attribution", "portfolio", "planning"]
MAX_MODE_DROP = 0.05
MAX_LOST_STABLE = 3
TOKEN_FLAG = 1.5


def load(label, split):
    path = ROOT / "runs" / label / split / "episodes.jsonl"
    if not path.exists() and path.with_suffix(".jsonl.gz").exists():  # committed runs are gzipped
        import gzip

        with gzip.open(path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rate(episodes):
    return sum(e["strict_pass"] for e in episodes) / len(episodes) if episodes else float("nan")


def tokens(episodes):
    return sum(e["input_tokens"] + e["output_tokens"] for e in episodes) / len(episodes)


def run_spread(episodes):
    by_run = defaultdict(list)
    for e in episodes:
        by_run[e["run"]].append(e)
    rates = [rate(v) for v in by_run.values()]
    return statistics.pstdev(rates) if len(rates) > 1 else 0.0


def stable_passes(episodes):
    by_case = defaultdict(list)
    for e in episodes:
        by_case[e["case_id"]].append(e["strict_pass"])
    return {case for case, passes in by_case.items() if all(passes)}


def compare(baseline, candidate):
    errors = [e for e in candidate if e["error"]]
    base_spread = run_spread(baseline)
    gain = rate(candidate) - rate(baseline)
    rows, checks = [], {}
    for mode in MODES:
        b = [e for e in baseline if e["mode"] == mode]
        c = [e for e in candidate if e["mode"] == mode]
        lost = stable_passes(b) - {e["case_id"] for e in c if e["strict_pass"]}
        drop = rate(b) - rate(c)
        ok = drop <= MAX_MODE_DROP and len(lost) <= MAX_LOST_STABLE
        rows.append({"mode": mode, "baseline": rate(b), "candidate": rate(c), "delta": -drop,
                     "lost_stable_cases": sorted(lost), "baseline_tokens": tokens(b),
                     "candidate_tokens": tokens(c), "ok": ok})
    side_b = sum(e["checks"]["side_effects"] for e in baseline) / len(baseline)
    side_c = sum(e["checks"]["side_effects"] for e in candidate) / len(candidate)
    checks["overall_gain_exceeds_spread"] = gain > base_spread
    checks["no_scenario_regression"] = all(r["ok"] for r in rows)
    checks["no_new_side_effect_violations"] = side_c >= side_b
    checks["clean_evaluation"] = not errors and len(candidate) == len(baseline)
    flags = [r["mode"] for r in rows if r["candidate_tokens"] > TOKEN_FLAG * r["baseline_tokens"]]
    return {
        "overall": {"baseline": rate(baseline), "candidate": rate(candidate), "gain": gain,
                    "baseline_run_spread": base_spread,
                    "baseline_tokens": tokens(baseline), "candidate_tokens": tokens(candidate)},
        "scenarios": rows, "checks": checks, "accept": all(checks.values()),
        "token_flags": flags, "runtime_errors": len(errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--split", default="development")
    args = parser.parse_args()
    report = compare(load(args.baseline, args.split), load(args.candidate, args.split))
    out = ROOT / "runs" / args.candidate / args.split / "acceptance.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    o = report["overall"]
    print(f"overall {o['baseline']:.1%} -> {o['candidate']:.1%} ({o['gain']:+.1%}; baseline run "
          f"spread {o['baseline_run_spread']:.1%}); tokens {o['baseline_tokens']:,.0f} -> "
          f"{o['candidate_tokens']:,.0f}")
    for r in report["scenarios"]:
        lost = f" lost {len(r['lost_stable_cases'])} stable" if r["lost_stable_cases"] else ""
        print(f"  {r['mode']:14} {r['baseline']:.1%} -> {r['candidate']:.1%} ({r['delta']:+.1%})"
              f"{lost}  tokens {r['baseline_tokens']:,.0f} -> {r['candidate_tokens']:,.0f}"
              f"  {'ok' if r['ok'] else 'REGRESSION'}")
    for name, passed in report["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    if report["token_flags"]:
        print(f"  token flag (>{TOKEN_FLAG:.0%} of baseline): {', '.join(report['token_flags'])}")
    print("ACCEPT" if report["accept"] else "REJECT")


if __name__ == "__main__":
    main()
