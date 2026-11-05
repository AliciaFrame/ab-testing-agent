"""Shared paths and helpers for agentic RFT on the routed A/B-testing specialists."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

import domain  # noqa: E402

RFT_DIR = ROOT / "rft"
DATA_DIR = RFT_DIR / "data"
JOBS_DIR = RFT_DIR / "jobs"
PROFILE_DIR = ROOT / "agent" / "specialists" / "optimized-r5"  # the deployed round-5 specialist prompts
COMMON_PROMPT = ROOT / "agent" / "specialists" / "common.md"
MODES = ["validity", "ship_decision", "early_look", "attribution", "portfolio", "planning"]
TARGET_MODES = ["planning", "ship_decision", "portfolio"]
RETENTION_MODES = ["validity", "early_look", "attribution"]
# One tool list for every row: RFT tools are defined per job, so rows share the union. At serving time each
# specialist keeps its own subset, which the model already sees as a smaller choice.
TOOLS = domain.TOOL_DEFINITIONS
TOOL_NAMES = [tool["function"]["name"] for tool in TOOLS]


def load_cases(split: str) -> list[dict]:
    path = ROOT / "data" / "cases" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def developer_prompt(mode: str) -> str:
    """Exactly the instructions the deployed specialist sends (common constraints + specialist prompt)."""
    return COMMON_PROMPT.read_text(encoding="utf-8") + "\n\n" + (PROFILE_DIR / f"{mode}.md").read_text(encoding="utf-8")


def row(case: dict) -> dict:
    """One RFT row. Only ``messages`` and ``tools`` are model-visible; the rest is grader reference data."""
    mode = case["expected"]["mode"]
    return {
        "messages": [{"role": "developer", "content": developer_prompt(mode)}]
        + [{"role": m["role"], "content": m["content"]} for m in case["messages"]],
        "tools": TOOLS,
        "case_id": case["id"],
        "mode": mode,
        "case": {"id": case["id"], "expected": case["expected"]},
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
