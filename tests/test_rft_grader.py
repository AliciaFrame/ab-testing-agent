import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rft import grader  # noqa: E402
from rft.common import load_cases, row  # noqa: E402


def _episode(strict: bool):
    episodes = [json.loads(line) for line in
                (ROOT / "runs" / "baseline" / "development" / "episodes.jsonl").read_text().splitlines()]
    return next(e for e in episodes if e["strict_pass"] is strict and e["mode"] == "ship_decision" and not e["error"])


def _sample(episode, output_tokens=None):
    sample = {"output_text": episode["output_text"], "output_tools": [
        {"type": "function_call", "name": t["name"],
         "arguments": t["arguments"] if isinstance(t["arguments"], str) else json.dumps(t["arguments"])}
        for t in episode["output_tools"]]}
    if output_tokens is not None:
        sample["usage"] = {"output_tokens": output_tokens}
    return sample


def test_accuracy_reward_reexecutes_calls_and_matches_canonical_grader():
    cases = {c["id"]: c for c in load_cases("development")}
    for strict in (True, False):
        episode = _episode(strict)
        result = grader.grade(_sample(episode), row(cases[episode["case_id"]]), None, "acc")
        assert abs(result["score"] - episode["reward"]) < 1e-9
        assert result["strict_pass"] is strict


def test_efficiency_reward_never_lets_wrong_answers_beat_correct_ones():
    cases = {c["id"]: c for c in load_cases("development")}
    good, bad = _episode(True), _episode(False)
    expensive = grader.grade(_sample(good, 10 ** 6), row(cases[good["case_id"]]), None, "eff")["score"]
    cheap_wrong = grader.grade(_sample(bad, 1), row(cases[bad["case_id"]]), None, "eff")["score"]
    assert expensive >= grader.MIN_EFFICIENCY > cheap_wrong
    frugal = grader.grade(_sample(good, 100), row(cases[good["case_id"]]), None, "eff")["score"]
    assert frugal > expensive
