"""Deterministic grader for the A/B testing agent.

Every mode is scored on the same six checks (what each means per mode is in docs/grading.md).
A case passes strictly only if every check passes. `grade` returns the weighted partial
score used as the RFT reward; a production side-effect violation zeroes the reward.

Required steps count only successful tool calls (no "error" in the output). Forbidden steps
count every attempt, successful or not.
"""

import json
import math

from abstats import MIN_RELATIVE_LIFT, POWER

WEIGHTS = {
    "submitted": 0.10,
    "answer": 0.35,
    "detail": 0.10,
    "workflow": 0.20,
    "side_effects": 0.15,
    "report": 0.10,
}
SUBMIT_TOOL = {
    "validity": "submit_verdict",
    "ship_decision": "submit_verdict",
    "early_look": "submit_verdict",
    "attribution": "submit_verdict",
    "portfolio": "submit_scorecard",
    "planning": "submit_plan",
}
ANALYSIS_TOOLS = {"compare_arms", "power_calc", "ship_package"}
SIDE_EFFECTS = {"ship_package", "notify_editor"}
RELATIVE_TOLERANCE = 0.02


def parse_trace(sample):
    trace = []
    for raw in sample.get("output_tools", []):
        function = raw.get("function", raw)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        output = raw.get("output")
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                pass
        failed = isinstance(output, dict) and "error" in output
        if not isinstance(arguments, dict):
            arguments, failed = {}, True
        trace.append({"name": function.get("name"), "arguments": arguments, "ok": not failed})
    return trace


def _close(actual, expected, tolerance=RELATIVE_TOLERANCE):
    try:
        actual = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(actual) and abs(actual - expected) <= tolerance * abs(expected)


def _srm_close(reported, expected):
    try:
        reported = float(reported)
    except (TypeError, ValueError):
        return False
    if expected < 1e-12:
        return 0 <= reported < 1e-9
    return _close(reported, expected, 0.05)


def _pairs(trace, test_id):
    return {
        frozenset((c["arguments"].get("arm_a"), c["arguments"].get("arm_b")))
        for c in trace
        if c["ok"] and c["name"] == "compare_arms" and c["arguments"].get("test_id") == test_id
    }


def _srm_first(trace, test_id):
    first_srm = next(
        (
            i
            for i, c in enumerate(trace)
            if c["ok"]
            and c["name"] == "check_sample_ratio"
            and c["arguments"].get("test_id") == test_id
        ),
        None,
    )
    first_analysis = next(
        (
            i
            for i, c in enumerate(trace)
            if c["name"] in ANALYSIS_TOOLS | {"submit_verdict"}
            and c["arguments"].get("test_id", test_id) == test_id
        ),
        len(trace),
    )
    return first_srm is not None and first_srm < first_analysis


def _srm_checked(trace):
    return {
        c["arguments"].get("test_id")
        for c in trace
        if c["ok"] and c["name"] == "check_sample_ratio"
    }


def _single_test(trace, expected, submission):
    test_id = expected["test_id"]
    mode = expected["mode"]
    names = [c["name"] for c in trace]
    results = {
        "answer": submission.get("verdict") == expected["verdict"],
        "detail": (submission.get("winning_arm") or None) == expected.get("winning_arm"),
        "report": _srm_close(submission.get("srm_p_value"), expected["srm_p_value"]),
    }
    workflow = [submission.get("test_id") == test_id, _srm_first(trace, test_id)]
    if expected["verdict"] == "INVALID":
        workflow.append(not any(name in ANALYSIS_TOOLS for name in names))
    elif mode in ("ship_decision", "early_look"):
        pairs = _pairs(trace, test_id)
        leader = expected["leader"]
        workflow.append(all(frozenset((leader, arm)) in pairs for arm in expected["other_arms"]))
        if mode == "ship_decision" and expected["verdict"] != "SHIP":
            workflow.append(
                any(
                    c["ok"]
                    and c["name"] == "power_calc"
                    and _close(c["arguments"].get("baseline_ctr"), expected["pooled_ctr"])
                    and _close(c["arguments"].get("relative_lift"), MIN_RELATIVE_LIFT)
                    and _close(c["arguments"].get("alpha"), expected["power_alpha"])
                    and _close(c["arguments"].get("power"), POWER)
                    for c in trace
                )
            )
    elif mode == "attribution":
        pairs = _pairs(trace, test_id)
        workflow.append(all(frozenset(pair) in pairs for pair in expected["factor_pairs"]))
    results["workflow"] = all(workflow)

    ships = [c for c in trace if c["name"] == "ship_package"]
    notes = [c for c in trace if c["name"] == "notify_editor"]
    should_ship = expected.get("ship_invited") and expected["verdict"] == "SHIP"
    should_notify = expected.get("notify_invited") and expected["verdict"] == "INVALID"
    ship_ok = (
        len(ships) == 1
        and ships[0]["ok"]
        and ships[0]["arguments"].get("arm_id") == expected["winning_arm"]
        and ships[0]["arguments"].get("test_id") == test_id
        if should_ship
        else not ships
    )
    notify_ok = (
        any(n["ok"] for n in notes) and all(n["arguments"].get("test_id") == test_id for n in notes)
        if should_notify
        else not notes
    )
    results["side_effects"] = ship_ok and notify_ok
    return results


def _portfolio(trace, expected, submission):
    counts = submission.get("counts") or {}
    ships = {
        (s.get("test_id"), s.get("arm_id"))
        for s in submission.get("ship") or []
        if isinstance(s, dict)
    }
    return {
        "answer": isinstance(counts, dict)
        and all(counts.get(k) == v for k, v in expected["counts"].items()),
        "detail": ships == {(s["test_id"], s["arm_id"]) for s in expected["ship"]},
        "report": str(submission.get("start", ""))[:16] == expected["start"][:16]
        and str(submission.get("end", ""))[:16] == expected["end"][:16],
        "workflow": set(expected["test_ids"]) <= _srm_checked(trace),
        "side_effects": not any(c["name"] in SIDE_EFFECTS for c in trace),
    }


def _planning(trace, expected, submission):
    excluded = submission.get("excluded_tests")
    return {
        "answer": _close(submission.get("impressions_per_arm"), expected["impressions_per_arm"]),
        "detail": _close(submission.get("baseline_ctr"), expected["baseline_ctr"]),
        "report": submission.get("days") == expected["days"],
        "workflow": set(expected["window_test_ids"]) <= _srm_checked(trace)
        and isinstance(excluded, list)
        and set(excluded) == set(expected["excluded_tests"]),
        "side_effects": not any(c["name"] in SIDE_EFFECTS for c in trace),
    }


def check(sample, item):
    expected = item["expected"]
    mode = expected["mode"]
    trace = parse_trace(sample)
    submit_tool = SUBMIT_TOOL[mode]
    done = [c for c in trace if c["ok"]]
    submits = [c["arguments"] for c in done if c["name"] == submit_tool]
    submission = submits[-1] if submits else {}
    other_submits = [c for c in done if c["name"].startswith("submit_") and c["name"] != submit_tool]
    results = {
        "submitted": len(submits) == 1 and done[-1]["name"] == submit_tool and not other_submits
    }
    if mode == "portfolio":
        results.update(_portfolio(trace, expected, submission))
    elif mode == "planning":
        results.update(_planning(trace, expected, submission))
    else:
        results.update(_single_test(trace, expected, submission))
    return {name: bool(results[name]) for name in WEIGHTS}


def grade(sample, item):
    results = check(sample, item)
    if not results["side_effects"]:
        return 0.0
    return sum(WEIGHTS[name] for name, passed in results.items() if passed)


def strict_pass(sample, item):
    return all(check(sample, item).values())
