"""Canonical-grader prompt search, isolated by task, followed by development acceptance."""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import specialists
import compare_runs
import run_eval

MODES = ["planning", "portfolio", "validity", "early_look", "ship_decision", "attribution"]


def save_candidates(mode, current, result, path):
    name = "numeric_incident_check"
    if mode in ("validity", "early_look") and "Mandatory incident-date verification" not in current and not any(
        candidate["name"] == name for candidate in result["candidates"]
    ):
        result["candidates"].append({
            "name": name,
            "source": "Optimizer-split failure analysis; arithmetic date-membership check",
            "instructions": current + (
                "\n\nMandatory incident-date verification before any verdict or outcome analysis:\n"
                "Read the actual started timestamp returned by get_test. Convert its YYYY-MM-DD "
                "date to the eight-digit integer YYYYMMDD by removing the hyphens. Call "
                "calculate with min(DATE-20130625, 20140110-DATE), replacing BOTH DATE "
                "placeholders with that actual eight-digit date integer. Use only arithmetic "
                "in the expression, not comparison operators. If the returned minimum is "
                "zero or positive, this is an incident-window test: verdict INVALID, even "
                "when sample-ratio p >= 0.001. A negative result means outside the incident "
                "window, but the sample-ratio rule still applies. Never claim a test started "
                "outside the incident without this check. Include the observed start date "
                "and incident-check result in the submission summary. Do not compare arm "
                "outcomes or ship an INVALID test. Still perform check_sample_ratio and "
                "any explicitly requested INVALID notification before the final submission.\n"
            ),
        })
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_cases(split, mode):
    return [
        row for line in (ROOT / "data" / "cases" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))["expected"]["mode"] == mode
    ]


def evaluate(cases, runs, workers, path):
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    else:
        records = []
    done = {(r["case_id"], r["run"]) for r in records}
    jobs = [(case, run) for run in range(runs) for case in cases if (case["id"], run) not in done]
    with ThreadPoolExecutor(workers) as pool, path.open("a", encoding="utf-8") as sink:
        for record in pool.map(lambda job: run_eval.run_one(*job), jobs):
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            records.append(record)
            print(f"{path.parent.name} {record['case_id']} r{record['run']} "
                  f"{'PASS' if record['strict_pass'] else 'fail'} {record['error'] or ''}", flush=True)
    by_id = {case["id"]: case for case in cases}
    for _ in range(3):
        indexes = [i for i, record in enumerate(records) if run_eval.is_transport_error(record)]
        if not indexes:
            break
        with ThreadPoolExecutor(workers) as pool:
            fresh = pool.map(
                lambda i: run_eval.run_one(by_id[records[i]["case_id"]], records[i]["run"]), indexes
            )
            for index, record in zip(indexes, fresh):
                with path.with_name("transport-failures.jsonl").open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(records[index]) + "\n")
                records[index] = run_eval.retain_transport_history(records[index], record)
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return records


def manual_candidates(mode, current, experiments):
    texts = experiments.get("_texts", {})
    result = []
    for spec in experiments.get(mode, []):
        append = spec.get("append", "")
        for key, value in texts.items():
            append = append.replace("{" + key + "}", value)
        if re.search(r"\{[a-z_]+\}", append):
            raise ValueError(f"Unresolved placeholder in {mode}/{spec['name']}")
        result.append({
            "name": spec["name"], "source": "manual", "hypothesis": spec.get("hypothesis"),
            "instructions": current + ("\n\n" + append if append else ""),
            "settings": {"reasoning_effort": spec["reasoning_effort"]}
            if spec.get("reasoning_effort") else {},
        })
    return result


def development_decision(base, development, min_token_reduction):
    gain = compare_runs.rate(development) - compare_runs.rate(base)
    lost = compare_runs.stable_passes(base) - {r["case_id"] for r in development if r["strict_pass"]}
    safe = sum(r["checks"]["side_effects"] for r in development) >= sum(
        r["checks"]["side_effects"] for r in base
    )
    tokens = sum(r["input_tokens"] + r["output_tokens"] for r in development)
    base_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in base)
    cheaper = tokens <= (1 - min_token_reduction) * base_tokens
    accept = (
        (gain > 0 or (gain >= 0 and cheaper))
        and len(lost) <= compare_runs.MAX_LOST_STABLE and safe
        and not any(r["error"] for r in development)
    )
    return {"accept": accept, "gain": gain, "lost_stable_cases": sorted(lost),
            "token_change": tokens / base_tokens - 1, "side_effects_ok": safe}


def candidate_instructions(mode, current, records, cases, count, out):
    path = out / "generated.json"
    if path.exists():
        return save_candidates(mode, current, json.loads(path.read_text(encoding="utf-8")), path)
    failures = [
        {
            "request": next(c["messages"] for c in cases if c["id"] == r["case_id"]),
            "expected": next(c["expected"] for c in cases if c["id"] == r["case_id"]),
            "failed_checks": [k for k, v in r["checks"].items() if not v],
            "tool_calls": r["output_tools"], "final_answer": r["output_text"],
        }
        for r in records if not r["strict_pass"] and not r["error"]
    ]
    prompt = {
        "task": mode, "current_instructions": current,
        "immutable_shared_instructions": specialists.common_instructions(),
        "available_tools": specialists.tools(mode),
        "optimizer_split_failures": failures,
        "request": (
            f"Produce {count} distinct improved specialist instruction candidates. "
            "Optimize strict passing of ALL canonical checks, not just the answer. "
            "Keep every statistical rule, tolerance and requested-action requirement. "
            "Do not hardcode test IDs, reference answers, dates other than the fixed incident "
            "window, or benchmark cases. Generalize the workflow. No new tools or changes "
            "to the router. Instructions must work without the expected labels shown in "
            "these diagnostic examples. Avoid expanding irrelevant instructions. "
            'Return JSON only: {"candidates":[{"name":"...","instructions":"..."}]}.'
        ),
    }
    with DefaultAzureCredential(process_timeout=60) as credential:
        token = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        with OpenAI(
            base_url=os.environ["OPTIMIZER_OPENAI_BASE_URL"], api_key=token,
            max_retries=8, timeout=900,
        ) as api:
            response = api.responses.create(
                model=os.getenv("OPTIMIZER_MODEL", "abtest-gpt-5-2"),
                instructions="Improve only the specified task policy using the supplied search diagnostics.",
                input=json.dumps(prompt), max_output_tokens=16000,
                text={"format": {"type": "json_object"}},
            )
    result = json.loads(response.output_text)
    candidates = result["candidates"]
    if len(candidates) != count or any(
        not isinstance(c.get("instructions"), str) or not c["instructions"].strip() for c in candidates
    ):
        raise ValueError("Optimizer returned invalid candidates")
    save_candidates(mode, current, result, path)
    (out / "generation-usage.json").write_text(json.dumps({
        "returned_model": response.model,
        "usage": response.usage.model_dump() if response.usage else None,
    }, indent=2), encoding="utf-8")
    return result


def prepare_search(args):
    out = ROOT / "runs" / args.label
    selected = out / "instructions"
    selected.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        path = selected / f"{mode}.md"
        if not path.exists():
            path.write_text(specialists.instructions(mode), encoding="utf-8")
    (out / "search-execution.json").write_text(json.dumps({
        "endpoint": os.environ["MAI_OPENAI_BASE_URL"],
        "model": os.environ["MAI_MODEL_DEPLOYMENT_NAME"],
        "max_output_tokens": os.getenv("MAI_MAX_OUTPUT_TOKENS"),
        "workers": args.workers, "split": "optimizer", "modes": args.modes,
        "note": "Search uses separate MAI capacity; development uses matched baseline settings.",
    }, indent=2), encoding="utf-8")
    for mode in args.modes:
        task_out = out / mode
        task_out.mkdir(exist_ok=True)
        if (task_out / "search-ready.json").exists():
            continue
        cases = load_cases("optimizer", mode)[:args.search_cases]
        os.environ["MAI_SPECIALIST_INSTRUCTIONS_DIR"] = str(selected)
        base = evaluate(cases, 1, args.workers, task_out / "baseline-search.jsonl")
        generated = candidate_instructions(
            mode, (selected / f"{mode}.md").read_text(encoding="utf-8"),
            base, cases, args.candidates, task_out,
        )
        for index, candidate in enumerate(generated["candidates"], 1):
            candidate_dir = task_out / f"candidate-{index}"
            candidate_dir.mkdir(exist_ok=True)
            for other in MODES:
                text = candidate["instructions"] if other == mode else (
                    selected / f"{other}.md"
                ).read_text(encoding="utf-8")
                (candidate_dir / f"{other}.md").write_text(text, encoding="utf-8")
            os.environ["MAI_SPECIALIST_INSTRUCTIONS_DIR"] = str(candidate_dir)
            evaluate(cases, 1, args.workers, candidate_dir / "search.jsonl")
        (task_out / "search-ready.json").write_text(
            json.dumps({"cases": len(cases), "candidates": len(generated["candidates"])}),
            encoding="utf-8",
        )
        print(f"SEARCH_READY {mode}", flush=True)


def run(args):
    baseline_path = ROOT / "runs" / args.baseline / "development" / "summary.json"
    deadline = time.monotonic() + 24 * 3600
    while True:
        if baseline_path.exists():
            baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
            if baseline_summary["overall"]["episodes"] == 270 and not baseline_summary["overall"]["errors"]:
                break
        if time.monotonic() >= deadline:
            raise RuntimeError("Baseline did not complete cleanly; optimization was not started")
        time.sleep(30)
    baseline = compare_runs.load(args.baseline, "development")
    out = ROOT / "runs" / args.label
    out.mkdir(parents=True, exist_ok=True)
    selected = out / "instructions"
    selected.mkdir(exist_ok=True)
    for mode in MODES:
        path = selected / f"{mode}.md"
        if not path.exists():
            path.write_text(specialists.instructions(mode), encoding="utf-8")
    if not (selected / "settings.json").exists():
        (selected / "settings.json").write_text(
            json.dumps(specialists.all_settings(), indent=2), encoding="utf-8"
        )
    experiments = (
        json.loads(Path(args.experiments).read_text(encoding="utf-8")) if args.experiments else None
    )
    manifest = {
        "baseline": args.baseline, "optimizer_model": os.getenv("OPTIMIZER_MODEL", "abtest-gpt-5-2"),
        "method": (
            "Hypothesis-driven manual candidates (prompt fixes and reasoning effort) plus GPT-5.2 "
            "proposals where search failures exist; canonical scoring"
            if experiments else
            "Independent GPT-5.2 proposals plus optimizer-failure diagnostic candidates; canonical scoring"
        ),
        "experiments": args.experiments,
        "search_gate": (
            "No fewer search passes than the search baseline; when the search baseline already "
            "passes every search case it cannot discriminate, so development decides "
            "(added after attribution's search baseline saturated)" if experiments
            else "Strictly better search rank (passes, then fewer tokens)"
        ),
        "development_acceptance": (
            "Accuracy gain, or no accuracy loss with at least "
            f"{args.min_token_reduction:.0%} fewer tokens; plus stable-case, action and error checks"
            if experiments else "Accuracy gain plus stable-case, action and error checks"
        ),
        "development_selection": True,
        "unchanged_tasks_reuse_baseline_measurements": True,
        "search_cases_per_task": args.search_cases, "candidates_per_task": args.candidates,
        "additional_diagnostic_candidates": {
            "validity": "Numeric incident-date membership via calculate",
            "early_look": "Numeric incident-date membership via calculate",
        },
        "router_frozen": True, "final_split_opened": False, "tasks": {},
        "common_instruction_sha256": hashlib.sha256(
            specialists.common_instructions().encode()
        ).hexdigest(),
    }
    result_records = list(baseline)
    source_directory = os.getenv("MAI_SPECIALIST_INSTRUCTIONS_DIR")
    try:
        for mode in dict.fromkeys([*args.modes, *MODES]):
            base = [r for r in baseline if r["mode"] == mode]
            task = {"baseline_development": run_eval.summarize(base), "accepted": False}
            manifest["tasks"][mode] = task
            if mode not in args.modes:
                task["reason"] = "Frozen in this optimization cycle"
                continue
            if all(r["strict_pass"] for r in base) and not (experiments and experiments.get(mode)):
                task["reason"] = "Already 100% strict pass; frozen"
                continue
            task_out = out / mode
            task_out.mkdir(exist_ok=True)
            if args.prepared_search:
                while not (task_out / "search-ready.json").exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Prepared search did not complete for {mode}")
                    time.sleep(20)
            cases = load_cases("optimizer", mode)[:args.search_cases]
            os.environ["MAI_SPECIALIST_INSTRUCTIONS_DIR"] = str(selected)
            search_base = evaluate(cases, 1, args.workers, task_out / "baseline-search.jsonl")
            current = (selected / f"{mode}.md").read_text(encoding="utf-8")
            if experiments is None:
                candidates = candidate_instructions(
                    mode, current, search_base, cases, args.candidates, task_out,
                )["candidates"]
            else:
                candidates = manual_candidates(mode, current, experiments)
                if args.candidates and not all(r["strict_pass"] for r in search_base):
                    candidates += candidate_instructions(
                        mode, current, search_base, cases, args.candidates, task_out,
                    )["candidates"]
                if (task_out / "candidates.json").exists():
                    # Keep measured candidate numbering stable; append newly added experiments.
                    existing = json.loads((task_out / "candidates.json").read_text(encoding="utf-8"))
                    known = {c["name"] for c in existing}
                    candidates = existing + [c for c in candidates if c["name"] not in known]
                (task_out / "candidates.json").write_text(
                    json.dumps(candidates, indent=2), encoding="utf-8"
                )
            selected_settings = json.loads((selected / "settings.json").read_text(encoding="utf-8"))
            eligible = []
            base_passes = sum(r["strict_pass"] for r in search_base)
            base_rank = (base_passes,
                         -sum(r["input_tokens"] + r["output_tokens"] for r in search_base))
            task["search"] = [{"name": "baseline", **run_eval.summarize(search_base)}]
            for index, candidate in enumerate(candidates, 1):
                candidate_dir = task_out / f"candidate-{index}"
                candidate_dir.mkdir(exist_ok=True)
                for other in MODES:
                    text = candidate["instructions"] if other == mode else (
                        selected / f"{other}.md"
                    ).read_text(encoding="utf-8")
                    (candidate_dir / f"{other}.md").write_text(text, encoding="utf-8")
                candidate_settings = {k: v for k, v in selected_settings.items() if k != mode}
                if candidate.get("settings"):
                    candidate_settings[mode] = candidate["settings"]
                (candidate_dir / "settings.json").write_text(
                    json.dumps(candidate_settings, indent=2), encoding="utf-8"
                )
                os.environ["MAI_SPECIALIST_INSTRUCTIONS_DIR"] = str(candidate_dir)
                records = evaluate(cases, 1, args.workers, candidate_dir / "search.jsonl")
                task["search"].append({"name": candidate["name"], **run_eval.summarize(records)})
                clean = not any(r["error"] for r in records)
                safe = sum(r["checks"]["side_effects"] for r in records) >= sum(
                    r["checks"]["side_effects"] for r in search_base
                )
                rank = (sum(r["strict_pass"] for r in records),
                        -sum(r["input_tokens"] + r["output_tokens"] for r in records))
                passes_gate = (
                    (rank[0] >= base_passes or base_passes == len(search_base)) if experiments
                    else rank > base_rank
                )
                if clean and safe and passes_gate:
                    eligible.append((rank, candidate_dir))
            best_dir, best_development = None, base
            task["development_candidates"] = []
            ranked = sorted(eligible, key=lambda item: item[0], reverse=True)
            for _, candidate_dir in ranked[:args.max_development_candidates]:
                os.environ["MAI_SPECIALIST_INSTRUCTIONS_DIR"] = str(candidate_dir)
                development = evaluate(
                    load_cases("development", mode), 3, args.workers, candidate_dir / "development.jsonl"
                )
                decision = (
                    development_decision(base, development, args.min_token_reduction)
                    if experiments else development_decision(base, development, float("inf"))
                )
                task["development_candidates"].append({
                    "candidate": str(candidate_dir.relative_to(ROOT)), "decision": decision,
                    **run_eval.summarize(development),
                })
                if decision["accept"]:
                    rank = (sum(r["strict_pass"] for r in development),
                            -sum(r["input_tokens"] + r["output_tokens"] for r in development))
                    best_rank = (sum(r["strict_pass"] for r in best_development),
                                 -sum(r["input_tokens"] + r["output_tokens"] for r in best_development))
                    if rank > best_rank:
                        best_dir, best_development = candidate_dir, development
                    if all(r["strict_pass"] for r in development):
                        break
            if mode in (getattr(args, "override", None) or {}):
                candidate_dir = task_out / args.override[mode]
                development = [
                    json.loads(line) for line in
                    (candidate_dir / "development.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                decision = development_decision(base, development, args.min_token_reduction)
                drop = compare_runs.rate(base) - compare_runs.rate(development)
                if (
                    len(development) != 45 or any(r["error"] for r in development)
                    or not decision["side_effects_ok"]
                    or len(decision["lost_stable_cases"]) > compare_runs.MAX_LOST_STABLE
                    or drop > compare_runs.MAX_MODE_DROP
                ):
                    raise ValueError(f"Override for {mode} violates project-level acceptance limits")
                best_dir, best_development = candidate_dir, development
                task["override"] = {
                    "candidate": str(candidate_dir.relative_to(ROOT)), "approved_by": "user",
                    "decision": decision,
                    "reason": getattr(args, "override_reason", None) or (
                        "User accepted a small accuracy loss for the token saving; "
                        "within the per-task drop and stable-case limits in docs/acceptance.md"
                    ),
                }
            if best_dir:
                (selected / f"{mode}.md").write_text(
                    (best_dir / f"{mode}.md").read_text(encoding="utf-8"), encoding="utf-8"
                )
                chosen = json.loads((best_dir / "settings.json").read_text(encoding="utf-8")) if (
                    best_dir / "settings.json"
                ).exists() else {}
                current_settings = json.loads((selected / "settings.json").read_text(encoding="utf-8"))
                current_settings.pop(mode, None)
                if chosen.get(mode):
                    current_settings[mode] = chosen[mode]
                (selected / "settings.json").write_text(
                    json.dumps(current_settings, indent=2), encoding="utf-8"
                )
                for record in best_development:
                    record["measurement_source"] = "optimized_specialist_rerun"
                result_records = [r for r in result_records if r["mode"] != mode] + best_development
                task["accepted"] = True
                task["selected_candidate"] = str(best_dir.relative_to(ROOT))
                task["candidate_development"] = run_eval.summarize(best_development)
            elif eligible:
                task["reason"] = "Development gain/stable-case/action/error acceptance failed"
            else:
                task["reason"] = "No candidate beat the task's optimizer baseline"
            (out / "optimization.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        if source_directory is None:
            os.environ.pop("MAI_SPECIALIST_INSTRUCTIONS_DIR", None)
        else:
            os.environ["MAI_SPECIALIST_INSTRUCTIONS_DIR"] = source_directory
    for record in result_records:
        record.setdefault("measurement_source", "unchanged_baseline_specialist")
    dev_out = out / "development"
    dev_out.mkdir(exist_ok=True)
    (dev_out / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in result_records), encoding="utf-8"
    )
    summary = {
        "label": args.label, "split": "development", "runs": 3,
        "model": os.environ["MAI_MODEL_DEPLOYMENT_NAME"], "router": os.getenv("MAI_ROUTER"),
        "overall": run_eval.summarize(result_records),
        **{mode: run_eval.summarize([r for r in result_records if r["mode"] == mode]) for mode in MODES},
        "measurement_note": "Changed tasks rerun; unchanged tasks reuse frozen baseline episodes.",
    }
    (dev_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    acceptance = compare_runs.compare(baseline, result_records)
    (dev_out / "acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    manifest["acceptance"] = acceptance
    manifest["instruction_sha256"] = {
        mode: hashlib.sha256((selected / f"{mode}.md").read_bytes()).hexdigest() for mode in MODES
    }
    manifest["specialist_settings"] = json.loads((selected / "settings.json").read_text(encoding="utf-8"))
    (out / "optimization.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"overall": summary["overall"], "acceptance": acceptance}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="routed-mai-baseline")
    parser.add_argument("--label", default="routed-mai-optimized")
    parser.add_argument("--search-cases", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        default=["planning", "validity", "early_look"])
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--prepared-search", action="store_true")
    parser.add_argument("--experiments", help="JSON file of hypothesis-driven candidates")
    parser.add_argument("--max-development-candidates", type=int)
    parser.add_argument("--min-token-reduction", type=float, default=0.10)
    parser.add_argument("--override", action="append", default=[],
                        help="mode=candidate-N accepted by explicit user decision")
    parser.add_argument("--override-reason")
    args = parser.parse_args()
    args.override = dict(item.split("=", 1) for item in args.override)
    (prepare_search if args.search_only else run)(args)
