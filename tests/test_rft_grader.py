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


def _trace(episode, gaps):
    calls = [{"call_id": f"c{i}", "name": t["name"], "output": t["output"],
              "arguments": t["arguments"] if isinstance(t["arguments"], dict) else json.loads(t["arguments"] or "{}")}
             for i, t in enumerate(episode["output_tools"])]
    times = [0.0]
    for i in range(len(calls) - 1):
        times.append(times[-1] + gaps(i))
    return [{**c, "created": t} for c, t in zip(calls, times)]


def test_turn_estimate_separates_batched_calls_from_one_call_turns():
    batched = [0, 2.0, 2.1, 1.9, 2.2, 30.0, 2.0, 2.4, 2.9, 3.4, 15.0, 2.0]  # 3 turns; spacing drifts 2 -> 3.4 s
    assert grader.estimate_turns([sum(batched[:i + 1]) for i in range(len(batched))]) == 3
    one_per_turn = [0] + [2.0, 6.0] * 10  # alternating in-batch call and a turn with ~4 s of generation
    assert grader.estimate_turns([sum(one_per_turn[:i + 1]) for i in range(len(one_per_turn))]) == 11
    never_batched = [6.0 * i for i in range(12)]  # every call its own turn: no in-batch gaps of its own
    assert grader.estimate_turns(never_batched) == 1
    assert grader.estimate_turns(never_batched, reference_gap=2.0) == 12


def test_turn_reward_prefers_fewer_turns_and_never_lets_wrong_answers_win():
    cases = {c["id"]: c for c in load_cases("development")}
    good, bad = _episode(True), _episode(False)
    item = row(cases[good["case_id"]])
    one_turn = grader.grade(_sample(good), item, _trace(good, lambda i: 2.0), "turns", 2.0)
    per_call = grader.grade(_sample(good), item, _trace(good, lambda i: 6.0), "turns", 2.0)
    assert one_turn["efficiency"]["turns"] == 1 and one_turn["score"] > per_call["score"]
    wrong = grader.grade(_sample(bad), row(cases[bad["case_id"]]), _trace(bad, lambda i: 2.0), "turns", 2.0)["score"]
    assert per_call["score"] >= grader.MIN_TURN_EFFICIENCY > wrong
