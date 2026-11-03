import json
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

import abstats  # noqa: E402
import domain  # noqa: E402
import grader  # noqa: E402

PATTERN_TIME = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"


def load_cases(*splits):
    cases = []
    for split in splits:
        path = ROOT / "data" / "cases" / f"{split}.jsonl"
        cases += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return cases


def call(trace, name, **arguments):
    result = domain.execute_tool(name, arguments)
    assert "error" not in result, result
    trace.append({"name": name, "arguments": arguments, "output": result})
    return result


def ship_decision(trace, test_id, arms, interim=False):
    """House rules for one valid test, using only tool outputs."""
    leader = max(arms, key=lambda a: a["clicks"] / a["impressions"])
    others = [a for a in arms if a is not leader]
    p_values = [
        call(trace, "compare_arms", test_id=test_id, arm_a=leader["arm_id"],
             arm_b=o["arm_id"])["p_value"]
        for o in others
    ]
    alpha = abstats.INTERIM_ALPHA if interim else abstats.FAMILY_ALPHA
    if abstats.holm_all_rejected(p_values, alpha):
        return "SHIP", leader["arm_id"]
    if interim:
        return "KEEP_TESTING", None
    pooled = sum(a["clicks"] for a in arms) / sum(a["impressions"] for a in arms)
    needed = call(trace, "power_calc", baseline_ctr=pooled, relative_lift=0.5,
                  alpha=0.05 / len(others), power=0.8)["impressions_per_arm"]
    smallest = min(a["impressions"] for a in arms)
    return ("NO_DIFFERENCE" if smallest >= needed else "KEEP_TESTING"), None


def single_test(case, trace):
    request = case["messages"][0]["content"].lower()
    mode = case["expected"]["mode"]
    test_id = re.search(r"test ([0-9a-f]{24}(?:@\d+)?)", request).group(1)
    invited = "ship it" in request or "ship the winner" in request
    notify = "editor know" in request or "flag it to the editor" in request
    test = call(trace, "get_test", test_id=test_id)
    srm = call(trace, "check_sample_ratio", test_id=test_id)["p_value"]
    winner = None
    if srm < abstats.SRM_ALPHA or abstats.in_incident(test["started"]):
        verdict = "INVALID"
        if notify:
            call(trace, "notify_editor", test_id=test_id, message="Sample ratio mismatch.")
    elif mode == "validity":
        verdict = "VALID"
    elif mode == "attribution":
        pairs = abstats.single_factor_pairs(test["arms"])
        results = []
        for factor, factor_pairs in pairs.items():
            for a, b in factor_pairs:
                p = call(trace, "compare_arms", test_id=test_id, arm_a=a, arm_b=b)["p_value"]
                results.append((factor, p))
        results.sort(key=lambda r: r[1])
        significant = set()
        for rank, (factor, p) in enumerate(results):
            if p > 0.05 / (len(results) - rank):
                break
            significant.add(factor)
        verdict = {0: "NEITHER", 2: "BOTH"}.get(len(significant)) or (
            "HEADLINE" if "headline" in significant else "IMAGE"
        )
    else:
        verdict, winner = ship_decision(trace, test_id, test["arms"], mode == "early_look")
        if verdict == "SHIP" and invited:
            call(trace, "ship_package", test_id=test_id, arm_id=winner)
    call(trace, "submit_verdict", test_id=test_id, verdict=verdict, winning_arm=winner,
         srm_p_value=srm, summary="reference")


def portfolio(case, trace):
    start, end = re.findall(PATTERN_TIME, case["messages"][0]["content"])
    tests = call(trace, "list_tests", start=start, end=end)["tests"]
    counts = {k: 0 for k in ("SHIP", "KEEP_TESTING", "NO_DIFFERENCE", "INVALID")}
    ships = []
    for t in tests:
        srm = call(trace, "check_sample_ratio", test_id=t["test_id"])["p_value"]
        if srm < abstats.SRM_ALPHA or abstats.in_incident(t["started"]):
            counts["INVALID"] += 1
            continue
        verdict, winner = ship_decision(trace, t["test_id"], t["arms"])
        counts[verdict] += 1
        if winner:
            ships.append({"test_id": t["test_id"], "arm_id": winner})
    call(trace, "submit_scorecard", start=start, end=end, counts=counts, ship=ships,
         summary="reference")


def planning(case, trace):
    request = case["messages"][0]["content"]
    planned = re.search(PATTERN_TIME, request).group(0)
    k = int(re.search(r"(\d)-arm", request).group(1))
    lift = int(re.search(r"(\d+)%", request).group(1)) / 100
    daily = int(re.search(r"about ([\d,]+) impressions", request).group(1).replace(",", ""))
    from datetime import datetime, timedelta

    start = (datetime.fromisoformat(planned) - timedelta(hours=abstats.PLANNING_LOOKBACK_HOURS)).isoformat(timespec="minutes")
    tests = call(trace, "list_tests", start=start, end=planned)["tests"]
    clicks = impressions = 0
    excluded = []
    for t in tests:
        srm = call(trace, "check_sample_ratio", test_id=t["test_id"])["p_value"]
        if srm < abstats.SRM_ALPHA or abstats.in_incident(t["started"]):
            excluded.append(t["test_id"])
            continue
        clicks += sum(a["clicks"] for a in t["arms"])
        impressions += sum(a["impressions"] for a in t["arms"])
    baseline = clicks / impressions
    needed = call(trace, "power_calc", baseline_ctr=baseline, relative_lift=lift,
                  alpha=0.05 / (k - 1), power=0.8)["impressions_per_arm"]
    call(trace, "submit_plan", baseline_ctr=baseline, impressions_per_arm=needed,
         days=math.ceil(k * needed / daily), excluded_tests=excluded, summary="reference")


def reference_agent(case):
    """Follows the house rules from the request and tool outputs, never the expected labels."""
    trace = []
    mode = case["expected"]["mode"]
    {"portfolio": portfolio, "planning": planning}.get(mode, single_test)(case, trace)
    return {"output_tools": trace, "output_text": "done"}


CASES = load_cases("development", "final", "optimizer", "validation", "train")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_reference_agent_passes(case):
    sample = reference_agent(case)
    assert grader.check(sample, case) == {name: True for name in grader.WEIGHTS}
    assert grader.grade(sample, case) == pytest.approx(1.0)


def first(verdict, mode="ship_decision"):
    return next(
        c for c in CASES if c["expected"].get("verdict") == verdict
        and c["expected"]["mode"] == mode
    )


def test_uninvited_ship_zeroes_reward():
    case = next(c for c in CASES if c["expected"].get("verdict") == "SHIP"
                and not c["expected"]["ship_invited"])
    sample = reference_agent(case)
    sample["output_tools"].insert(-1, {"name": "ship_package", "arguments": {
        "test_id": case["expected"]["test_id"], "arm_id": case["expected"]["winning_arm"]}})
    assert not grader.check(sample, case)["side_effects"]
    assert grader.grade(sample, case) == 0.0


def test_skipping_sample_ratio_fails_workflow():
    case = first("KEEP_TESTING")
    sample = reference_agent(case)
    sample["output_tools"] = [c for c in sample["output_tools"]
                              if c["name"] != "check_sample_ratio"]
    assert not grader.check(sample, case)["workflow"]


def test_analysing_invalid_test_fails_workflow():
    case = first("INVALID")
    sample = reference_agent(case)
    test_id = case["expected"]["test_id"]
    arms = [a["arm_id"] for a in domain.get_test(test_id)["arms"]]
    sample["output_tools"].insert(-1, {"name": "compare_arms", "arguments": {
        "test_id": test_id, "arm_a": arms[0], "arm_b": arms[1]}})
    assert not grader.check(sample, case)["workflow"]


def test_early_look_at_final_alpha_is_wrong():
    """Some running tests would be shipped at alpha 0.05 but not at the interim 0.005."""
    tempting = 0
    for case in CASES:
        e = case["expected"]
        if e["mode"] != "early_look" or e["verdict"] != "KEEP_TESTING":
            continue
        arms = domain.get_test(e["test_id"])["arms"]
        _, _, comparisons = abstats.leader_comparisons(arms)
        tempting += abstats.holm_all_rejected(list(comparisons.values()), 0.05)
    assert tempting > 0


def test_planning_without_exclusions_fails():
    case = next(c for c in CASES if c["expected"]["mode"] == "planning"
                and c["expected"]["excluded_tests"])
    sample = reference_agent(case)
    sample["output_tools"][-1]["arguments"]["excluded_tests"] = []
    assert not grader.check(sample, case)["workflow"]


def test_dashboard_first_place_is_not_the_label():
    disagreements = 0
    for case in CASES:
        if case["expected"]["mode"] != "ship_decision":
            continue
        arms = domain.get_test(case["expected"]["test_id"])["arms"]
        dashboard = next((a["arm_id"] for a in arms if a["dashboard_first_place"]), None)
        disagreements += dashboard != case["expected"]["winning_arm"]
    assert disagreements > 0


def test_snapshot_is_deterministic_subsample():
    case = first("SHIP", "early_look")
    test_id = case["expected"]["test_id"]
    base, percent = domain.split_test_id(test_id)
    full = {a["arm_id"]: a for a in domain.get_test(base)["arms"]}
    snap = domain.get_test(test_id)
    assert snap["status"] == "running"
    assert snap == domain.get_test(test_id)
    for arm in snap["arms"]:
        assert arm["impressions"] == full[arm["arm_id"]]["impressions"] * percent // 100
        assert arm["clicks"] <= full[arm["arm_id"]]["clicks"]


def test_calculate_is_restricted():
    assert domain.execute_tool("calculate", {"expression": "ceil(3 * 1000 / 7)"}) == {
        "result": 429
    }
    assert "error" in domain.execute_tool("calculate", {"expression": "__import__('os')"})


def test_bad_tool_calls_return_errors_to_the_model():
    assert "error" in domain.execute_tool("no_such_tool", {})
    assert "error" in domain.execute_tool("get_test", {"test_id": "nope"})
    assert "error" in domain.execute_tool("get_test", {"wrong_argument": 1})


def test_failed_calls_do_not_satisfy_required_steps():
    case = first("KEEP_TESTING")
    sample = reference_agent(case)
    for c in sample["output_tools"]:
        if c["name"] == "check_sample_ratio":
            c["output"] = {"error": "boom"}
    assert not grader.check(sample, case)["workflow"]


def test_failed_submit_then_retry_counts_once():
    case = first("SHIP")
    sample = reference_agent(case)
    final = sample["output_tools"][-1]
    sample["output_tools"].insert(-1, {"name": final["name"], "arguments": {"test_id": "x"},
                                       "output": {"error": "missing arguments"}})
    assert grader.check(sample, case)["submitted"]


def test_incident_trap_is_invalid_even_when_split_passes():
    traps = [c for c in CASES if c["expected"].get("label") == "INVALID:incident"]
    assert traps
    for case in traps:
        assert case["expected"]["srm_p_value"] >= abstats.SRM_ALPHA
        assert case["expected"]["verdict"] == "INVALID"
        base, _ = domain.split_test_id(case["expected"]["test_id"])
        assert abstats.in_incident(domain.get_test(base)["started"])


def test_incident_window_matches_author_correction():
    assert abstats.in_incident("2013-06-25T00:00:00")
    assert abstats.in_incident("2014-01-10T23:59:59")
    assert not abstats.in_incident("2013-06-24T23:59:59")
    assert not abstats.in_incident("2014-01-11T00:00:00")


def test_no_test_is_shared_across_splits():
    owner = {}
    for case in load_cases("development", "final", "optimizer", "validation", "train"):
        e = case["expected"]
        if e["mode"] == "portfolio":
            ids = e["test_ids"]
        elif e["mode"] == "planning":
            ids = e["window_test_ids"]
        else:
            ids = [domain.split_test_id(e["test_id"])[0]]
        for test_id in ids:
            assert owner.setdefault(test_id, case["split"]) == case["split"], test_id


def test_single_test_cases_never_repeat_a_test():
    seen = set()
    for case in load_cases("development", "final", "optimizer", "validation", "train"):
        e = case["expected"]
        if e["mode"] in ("portfolio", "planning"):
            continue
        base = domain.split_test_id(e["test_id"])[0]
        assert base not in seen
        seen.add(base)


def test_holm_step_down():
    assert abstats.holm_all_rejected([0.01, 0.04], alpha=0.05)
    assert not abstats.holm_all_rejected([0.03, 0.04], alpha=0.05)
    assert not abstats.holm_all_rejected([0.001, 0.06], alpha=0.05)


def test_required_per_arm_matches_textbook():
    # Two-sided alpha 0.05, power 0.8, 10% -> 15%: 686 per arm (Fleiss, no continuity correction).
    assert abstats.required_per_arm(0.10, 0.5, 0.05, 0.8) == 686
