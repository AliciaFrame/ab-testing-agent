"""Independently configurable task policies with fixed tool boundaries."""

import json
import os
import re
from pathlib import Path

from domain import TOOL_DEFINITIONS

ROOT = Path(__file__).resolve().parent
TOOLS = {
    "validity": {"get_test", "check_sample_ratio", "calculate", "notify_editor", "submit_verdict"},
    "ship_decision": {
        "get_test", "check_sample_ratio", "compare_arms", "power_calc", "calculate",
        "ship_package", "submit_verdict",
    },
    "early_look": {
        "get_test", "check_sample_ratio", "compare_arms", "calculate",
        "ship_package", "submit_verdict",
    },
    "attribution": {
        "get_test", "check_sample_ratio", "compare_arms", "calculate", "submit_verdict",
    },
    "portfolio": {
        "list_tests", "get_test", "check_sample_ratio", "compare_arms", "power_calc",
        "calculate", "submit_scorecard",
    },
    "planning": {
        "list_tests", "get_test", "check_sample_ratio", "power_calc", "calculate", "submit_plan",
    },
}


EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def directory():
    profile = os.getenv("MAI_SPECIALIST_PROFILE", "baseline")
    if not re.fullmatch(r"baseline|optimized(?:-r\d+)?", profile):
        raise ValueError(f"Unknown specialist profile: {profile}")
    default = ROOT / "specialists"
    if profile != "baseline":
        default = default / profile
    return Path(os.getenv("MAI_SPECIALIST_INSTRUCTIONS_DIR", default))


def instructions(mode):
    if mode not in TOOLS:
        raise ValueError(f"Unknown specialist: {mode}")
    return (directory() / f"{mode}.md").read_text(encoding="utf-8")


def all_settings():
    path = directory() / "settings.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for mode, values in data.items():
        if mode not in TOOLS or not isinstance(values, dict) or set(values) - {"reasoning_effort"}:
            raise ValueError(f"Invalid specialist settings for {mode}")
        if values.get("reasoning_effort") not in EFFORTS | {None}:
            raise ValueError(f"Invalid reasoning effort for {mode}")
    return data


def settings(mode):
    return all_settings().get(mode, {})


def common_instructions():
    return (ROOT / "specialists" / "common.md").read_text(encoding="utf-8")


def tools(mode):
    return [tool for tool in TOOL_DEFINITIONS if tool["function"]["name"] in TOOLS[mode]]
