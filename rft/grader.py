"""Endpoint-grader rewards for agentic RFT, built on the benchmark's canonical grader (``agent/grader.py``).

Two rewards are served by ``rft.tool_server``:

``acc`` (``abtest-rft-acc-v1``): the canonical weighted reward unchanged. The six checks (submitted, answer, detail,
workflow, side_effects, report) weigh 0.10/0.35/0.10/0.20/0.15/0.10, and a production-action violation scores 0.
Only a strict pass (all six checks) reaches 1.0.

``eff`` (``abtest-rft-eff-v1``): ``acc x efficiency``. Efficiency only pays for correct work: a rollout that is not
a strict pass gets the floor factor 0.7, so a non-strict answer (acc <= 0.9, eff <= 0.63) never outscores a strict
one (eff >= 0.7). For strict passes::

    efficiency = max(0.7, 1 - p_tokens - p_calls)
    p_tokens = 0.20 x clamp(output_tokens / median_tokens - 0.5, 0, 1)   (only when the trainer's sample has usage)
    p_calls  = 0.10 x clamp(tool_calls / median_calls - 1, 0, 1)

so a strict answer using half of today's output tokens or less earns full credit. Medians are per task, measured on
the deployed round-5 agent (``runs/routed-mai-r5``, development split, 45 episodes per task); they are statistics of
the deployed agent, not training data. Tool calls are penalized only above today's median, so the required checks
(e.g. a traffic-split check on every look-back test) are never discouraged.

Calls come from the tool server's record of the rollout (``trace_id``) when it is complete; otherwise the sample's
``output_tools`` are re-executed. The archive tools are deterministic and stateless, so both give identical outputs.
"""

from __future__ import annotations

import json
import math

from rft.common import domain
import grader as canonical

ACC_VERSION = "abtest-rft-acc-v1"
EFF_VERSION = "abtest-rft-eff-v1"
MIN_EFFICIENCY = 0.7
MEDIANS = {  # deployed round-5 agent, development split (see module docstring)
    "validity": {"tool_calls": 4, "output_tokens": 876},
    "ship_decision": {"tool_calls": 8, "output_tokens": 3137},
    "early_look": {"tool_calls": 7, "output_tokens": 3895},
    "attribution": {"tool_calls": 11, "output_tokens": 5494},
    "portfolio": {"tool_calls": 68, "output_tokens": 17748},
    "planning": {"tool_calls": 26, "output_tokens": 9090},
}
TOKEN_WEIGHT, CALL_WEIGHT = 0.20, 0.10


class GradingError(RuntimeError):
    """Infrastructure problem: never scored (the endpoint returns 5xx, the trainer retries/drops the rollout)."""


def _arguments(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return raw
    return raw if isinstance(raw, dict) else {}


def sample_calls(sample: dict) -> list[dict]:
    """Tool calls in a trainer sample (``output_tools``: chat-completions or flat function objects)."""
    calls = []
    for index, raw in enumerate(sample.get("output_tools") or []):
        if not isinstance(raw, dict):
            continue
        function = raw.get("function", raw)
        if isinstance(function, dict) and function.get("name"):
            calls.append({"name": function["name"], "arguments": _arguments(function.get("arguments", {})),
                          "call_id": raw.get("call_id") or raw.get("id") or f"call_{index}"})
    return calls


def execute(name: str, arguments) -> str:
    """Tool output text exactly as the agent and the tool server return it."""
    return json.dumps(domain.execute_tool(name, arguments))


def answer_text(sample: dict) -> str:
    answer = sample.get("output_text")
    if answer is None and isinstance(sample.get("choices"), list) and sample["choices"]:
        answer = (sample["choices"][0].get("message") or {}).get("content")
    return answer or ""


def output_tokens(sample: dict) -> int | None:
    usage = sample.get("usage")
    if not isinstance(usage, dict):
        return None
    for key in ("output_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return int(value)
    return None


def accuracy(item: dict, calls: list[dict]) -> dict:
    case = item.get("case")
    if not isinstance(case, dict) or "expected" not in case:
        raise GradingError("item has no case reference (not an RFT training row)")
    record = {"output_tools": [{"name": c["name"], "arguments": c["arguments"], "output": c["output"]} for c in calls]}
    checks = canonical.check(record, case)
    score = canonical.grade(record, case)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise GradingError("canonical grader produced an invalid score")
    return {"score": score, "checks": checks, "strict_pass": all(checks.values()), "tool_calls": len(calls)}


def efficiency(mode: str, tool_calls: int, tokens: int | None) -> dict:
    median = MEDIANS[mode]
    p_tokens = 0.0 if tokens is None else TOKEN_WEIGHT * min(1.0, max(0.0, tokens / median["output_tokens"] - 0.5))
    p_calls = CALL_WEIGHT * min(1.0, max(0.0, tool_calls / median["tool_calls"] - 1))
    return {"factor": round(max(MIN_EFFICIENCY, 1 - p_tokens - p_calls), 6), "p_tokens": round(p_tokens, 6),
            "p_calls": round(p_calls, 6), "output_tokens": tokens, "tool_calls": tool_calls, "median": median}


def grade(sample: dict, item: dict, trace_calls: list[dict] | None = None, version: str = "acc") -> dict:
    """Endpoint entry point. ``trace_calls`` are the tool server's recorded calls for this rollout, if any."""
    if not isinstance(sample, dict) or not isinstance(item, dict):
        raise GradingError("sample and item must be objects")
    if item.get("mode") not in MEDIANS:
        raise GradingError("item is missing a known mode")
    from_sample = sample_calls(sample)
    if trace_calls and len(trace_calls) >= len(from_sample):
        calls, source = trace_calls, "tool_server_trace"
    else:
        calls = [{**c, "output": execute(c["name"], c["arguments"])} for c in from_sample]
        source = "sample_output_tools"
    result = accuracy(item, calls)
    result.update(calls_source=source, version=ACC_VERSION, acc_score=result["score"])
    if version == "eff":
        eff = efficiency(item["mode"], len(calls), output_tokens(sample))
        factor = eff["factor"] if result["strict_pass"] else MIN_EFFICIENCY
        result.update(score=round(result["acc_score"] * factor, 6), efficiency=eff, factor_applied=factor,
                      version=EFF_VERSION)
    elif version != "acc":
        raise GradingError(f"unknown reward version {version!r}")
    result["answer_chars"] = len(answer_text(sample))
    return result
