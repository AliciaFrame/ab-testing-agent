"""Package only canonically accepted specialist policies, preserving baseline prompts."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import compare_runs

ROOT = Path(__file__).resolve().parents[1]


def publish(label, profile="optimized"):
    if not re.fullmatch(r"optimized(?:-r\d+)?", profile):
        raise ValueError(f"Invalid profile name: {profile}")
    source = ROOT / "runs" / label
    manifest = json.loads((source / "optimization.json").read_text(encoding="utf-8"))
    baseline = compare_runs.load(manifest["baseline"], "development")
    selected = compare_runs.load(label, "development")
    if len(baseline) != 270 or len(selected) != 270:
        raise ValueError("Publishing requires complete 270-episode measurements")
    baseline_keys = {(r["case_id"], r["run"]) for r in baseline}
    selected_keys = {(r["case_id"], r["run"]) for r in selected}
    acceptance = compare_runs.compare(baseline, selected)
    if len(baseline_keys) != 270 or baseline_keys != selected_keys or not acceptance["accept"]:
        raise ValueError("Candidate did not pass canonical development acceptance")
    common = (ROOT / "agent" / "specialists" / "common.md").read_text(encoding="utf-8")
    if hashlib.sha256(common.encode()).hexdigest() != manifest["common_instruction_sha256"]:
        raise ValueError("Shared instructions changed after evaluation")
    for mode in compare_runs.MODES:
        path = source / "instructions" / f"{mode}.md"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["instruction_sha256"][mode]:
            raise ValueError(f"Selected {mode} instructions changed after evaluation")
        policy_hash = hashlib.sha256(
            (common + "\n\n" + path.read_text(encoding="utf-8")).encode()
        ).hexdigest()
        if any(
            r["policy_contract"]["policy_sha256"] != policy_hash
            for r in selected if r["mode"] == mode
        ):
            raise ValueError(f"Selected {mode} policy does not match measured episodes")
    settings_path = source / "instructions" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    if settings != manifest.get("specialist_settings", {}):
        raise ValueError("Selected specialist settings changed after evaluation")
    for mode in compare_runs.MODES:
        effort = settings.get(mode, {}).get("reasoning_effort")
        if any(
            r["policy_contract"].get("reasoning_effort") != effort
            for r in selected if r["mode"] == mode
        ):
            raise ValueError(f"Selected {mode} reasoning effort does not match measured episodes")
    target = ROOT / "agent" / "specialists" / profile
    target.mkdir(exist_ok=True)
    for mode in compare_runs.MODES:
        shutil.copyfile(source / "instructions" / f"{mode}.md", target / f"{mode}.md")
    if settings:
        (target / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps({
        "source": label, "baseline": manifest["baseline"],
        "acceptance": acceptance,
        "instruction_sha256": manifest["instruction_sha256"],
        "common_instruction_sha256": manifest["common_instruction_sha256"],
        "specialist_settings": settings,
    }, indent=2), encoding="utf-8")
    print(f"Published {target}; activate with MAI_SPECIALIST_PROFILE={profile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="routed-mai-optimized")
    parser.add_argument("--profile", default="optimized")
    args = parser.parse_args()
    publish(args.label, args.profile)
