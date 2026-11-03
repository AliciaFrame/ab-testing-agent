"""Add per-case judge criteria and freeze the prepared data for Agent Optimizer and RFT.

The optimizer's judge sees only the agent's final answer, so criteria state what a correct
answer must say for that case. Candidates are then re-scored with the canonical grader
(agent/grader.py) on the development split; the judge only nominates.

Usage: python scripts/prepare_optimizer.py --skill <mai-agent-hillclimb skill dir>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["optimizer", "train", "validation", "development", "final"]
RULES = (
    "The house rules are in the agent instructions: traffic-split check first (INVALID if "
    "p < 0.001 or the test started 2013-06-25 through 2014-01-10), leader vs every other arm "
    "with Holm at 0.05 (0.005 for running tests), and a power check before NO_DIFFERENCE."
)


def criterion(name, instruction):
    return {"name": name, "instruction": instruction}


def criteria(expected):
    mode = expected["mode"]
    items = []
    if mode in ("validity", "ship_decision", "early_look", "attribution"):
        verdict = expected["verdict"]
        items.append(criterion("verdict", f"The answer's verdict for test {expected['test_id']} "
                                          f"is {verdict}. Any other verdict fails."))
        if expected.get("label") == "INVALID:incident":
            items.append(criterion("incident", "The test started during the 2013-06-25 to "
                                   "2014-01-10 randomization incident, so it is INVALID even "
                                   "though its traffic split passes. Analysing its outcome fails."))
        if verdict == "SHIP":
            items.append(criterion("winner", f"The winning arm is {expected['winning_arm']}."))
        elif verdict != "INVALID" and mode in ("ship_decision", "early_look"):
            items.append(criterion("no_winner", "No arm is declared a winner or shipped."))
        if expected.get("ship_invited") and verdict == "SHIP":
            items.append(criterion("action", "The requester asked to ship a winner, so the "
                                   "answer confirms the winning arm was shipped."))
        elif mode in ("ship_decision", "early_look"):
            items.append(criterion("action", "Nothing is shipped."))
        if expected.get("notify_invited") and verdict == "INVALID":
            items.append(criterion("action", "The requester asked to alert the editor, so the "
                                   "answer confirms the editor was notified."))
        if mode == "early_look":
            items.append(criterion("peeking", "The test is still running: a winner requires "
                                   "significance at the stricter 0.005 family-wise level, and "
                                   "NO_DIFFERENCE is never allowed."))
    elif mode == "portfolio":
        counts = ", ".join(f"{k} {v}" for k, v in expected["counts"].items())
        ships = ", ".join(f"{s['test_id']} arm {s['arm_id']}" for s in expected["ship"]) or "none"
        items += [
            criterion("counts", f"For tests started in [{expected['start']}, {expected['end']}) "
                      f"the verdict counts are exactly: {counts}."),
            criterion("ship_list", f"The shippable tests and winning arms are exactly: {ships}."),
            criterion("action", "Nothing is shipped and no editor is notified."),
        ]
    else:
        excluded = ", ".join(expected["excluded_tests"]) or "none"
        items += [
            criterion("sample_size", f"Impressions per arm is {expected['impressions_per_arm']:,} "
                      "(within 2%)."),
            criterion("duration", f"Duration is {expected['days']} days."),
            criterion("baseline", f"Baseline click-through rate is {expected['baseline_ctr']:.5f} "
                      "(within 2%), from the 12 hours of tests before the planned start."),
            criterion("exclusions", f"Excluded look-back tests are exactly: {excluded}."),
        ]
    items.append(criterion("house_rules", RULES))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "prepared")
    args = parser.parse_args()
    rows = []
    for split in SPLITS:
        path = ROOT / "data" / "cases" / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["criteria"] = criteria(row["expected"])
            rows.append(row)
    combined = ROOT / "data" / "cases.all.jsonl"
    combined.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(args.skill / "scripts" / "journey.py"), "prepare",
         "--data", str(combined), "--output", str(args.output),
         "--tools", str(ROOT / "agent" / "tools.json"),
         "--instructions", str(ROOT / "agent" / "instructions.md")],
        check=True,
    )
    combined.unlink()
    print(f"prepared -> {args.output}")


if __name__ == "__main__":
    main()
