import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_comparison_report as report
import optimize_specialists
import publish_specialists
import run_eval


def test_diagnostic_search_candidate_is_resumable_and_keeps_other_candidates(tmp_path):
    path = tmp_path / "generated.json"
    result = {"candidates": [{"name": "generated", "instructions": "original"}]}
    first = optimize_specialists.save_candidates("validity", "baseline", result, path)
    resumed = optimize_specialists.save_candidates(
        "validity", "baseline", json.loads(path.read_text()), path
    )
    assert resumed == first
    assert len(resumed["candidates"]) == 2
    assert resumed["candidates"][0]["instructions"] == "original"
    frozen = optimize_specialists.save_candidates(
        "planning", "baseline", {"candidates": []}, tmp_path / "planning.json"
    )
    assert frozen["candidates"] == []


def test_search_ties_try_another_candidate_when_first_fails_development(monkeypatch, tmp_path):
    def record(mode, case, run, passed=True, tokens=10):
        return {
            "case_id": f"{mode}-{case}", "mode": mode, "run": run, "strict_pass": passed,
            "error": None, "reward": float(passed), "latency_s": 1,
            "input_tokens": tokens, "output_tokens": 0,
            "calls": [{"raw_usage": {"input_tokens": tokens}}],
            "checks": {name: passed if name == "answer" else True for name in run_eval.CHECKS},
        }

    baseline = [
        record(mode, case, run, mode != "validity" or case != 0)
        for mode in optimize_specialists.MODES for run in range(3) for case in range(15)
    ]
    source = tmp_path / "runs" / "reference" / "development"
    source.mkdir(parents=True)
    (source / "summary.json").write_text(json.dumps({"overall": {"episodes": 270, "errors": 0}}))
    monkeypatch.setattr(optimize_specialists, "ROOT", tmp_path)
    monkeypatch.setattr(optimize_specialists.compare_runs, "load", lambda *args: baseline)
    monkeypatch.setattr(optimize_specialists.specialists, "instructions", lambda mode: "baseline")
    monkeypatch.setattr(optimize_specialists.specialists, "common_instructions", lambda: "shared")
    monkeypatch.setattr(optimize_specialists, "load_cases", lambda *args: [])
    monkeypatch.setattr(optimize_specialists, "candidate_instructions", lambda *args: {
        "candidates": [
            {"name": "first", "instructions": "first"},
            {"name": "second", "instructions": "second"},
        ],
    })
    development_attempts = []

    def evaluate(cases, runs, workers, path):
        if path.name == "baseline-search.jsonl":
            return [record("validity", case, 0, case >= 2) for case in range(6)]
        if path.name == "search.jsonl":
            return [record("validity", case, 0, tokens=10 if path.parent.name == "candidate-1" else 20)
                    for case in range(6)]
        development_attempts.append(path.parent.name)
        if path.parent.name == "candidate-1":
            return [copy.deepcopy(r) for r in baseline if r["mode"] == "validity"]
        return [record("validity", case, run) for run in range(3) for case in range(15)]

    monkeypatch.setattr(optimize_specialists, "evaluate", evaluate)
    monkeypatch.setenv("MAI_MODEL_DEPLOYMENT_NAME", "test-model")
    optimize_specialists.run(SimpleNamespace(
        baseline="reference", label="candidate", modes=["validity"], prepared_search=False,
        search_cases=6, candidates=2, workers=1, experiments=None,
        max_development_candidates=None, min_token_reduction=0.1,
    ))
    manifest = json.loads((tmp_path / "runs" / "candidate" / "optimization.json").read_text())
    assert next(iter(manifest["tasks"])) == "validity"
    assert development_attempts == ["candidate-1", "candidate-2"]
    assert manifest["tasks"]["validity"]["accepted"]
    assert manifest["acceptance"]["accept"]
    assert (tmp_path / "runs" / "candidate" / "instructions" / "validity.md").read_text() == "second"


def test_transport_replacement_retains_cumulative_usage_without_retrying_model_failures():
    failed = {
        "error": "EpisodeFailure: RateLimitError: quota", "latency_s": 12,
        "input_tokens": 30, "output_tokens": 4,
    }
    replacement = run_eval.retain_transport_history(failed, {
        "error": "EpisodeFailure: APITimeoutError: timeout", "latency_s": 5,
        "input_tokens": 10, "output_tokens": 2,
    })
    successful = {
        "error": None, "latency_s": 3, "input_tokens": 20, "output_tokens": 1,
        "strict_pass": True, "reward": 1,
        "calls": [{"raw_usage": {"input_tokens": 20, "output_tokens": 1}}],
        "checks": {name: True for name in run_eval.CHECKS},
    }
    successful = run_eval.retain_transport_history(replacement, successful)
    summary = run_eval.summarize([successful])
    assert summary["total_tokens"] == 21
    assert summary["transport_overhead_tokens"] == 46
    assert summary["all_attempt_tokens"] == 67
    assert summary["transport_overhead_latency_s"] == 17
    assert successful["transport_reruns"] == 2
    assert run_eval.is_transport_error(failed)
    assert not run_eval.is_transport_error({"error": None, "strict_pass": False})


@pytest.mark.parametrize("error, retryable", [
    ("EpisodeFailure: HTTPStatusError: Client error '429 Too Many Requests' for url 'router'", True),
    ("EpisodeFailure: HTTPStatusError: Server error '503 Service Unavailable' for url 'router'", True),
    ("EpisodeFailure: HTTPStatusError: Client error '401 Unauthorized' for url 'router'", False),
    ("EpisodeFailure: HTTPStatusError: Client error '400 Bad Request' for url 'router'", False),
    ("EpisodeFailure: ReadTimeout: timeout", True),
    ("EpisodeFailure: RuntimeError: Tool-step limit exceeded", False),
])
def test_router_http_recovery_does_not_retry_authentication_or_model_failures(error, retryable):
    assert run_eval.is_transport_error({"error": error}) is retryable


@pytest.mark.parametrize("change, message", [
    ("limit", "runtime controls or tools differ"),
    ("model", ""),
    ("router", "Router deployment differs"),
])
def test_comparison_rejects_mismatched_runtime_or_model(monkeypatch, change, message):
    baseline = [{
        "case_id": mode, "run": 0, "mode": mode,
        "policy_contract": {"policy_sha256": "same", "max_output_tokens": 32768,
                            "endpoint": "https://example.openai.azure.com"},
        "routing": {"router_sha256": "same", "model": "decision", "backend": "decision1"},
        "calls": [{"purpose": "inference", "returned_model": "mai-code-flash-1-1"}],
    } for mode in report.compare_runs.MODES]
    sol = copy.deepcopy(baseline)
    for row in sol:
        row["calls"][0]["returned_model"] = "gpt-6-sol"
    if change == "limit":
        sol[0]["policy_contract"]["max_output_tokens"] = 8192
    elif change == "model":
        sol[0]["calls"][0]["returned_model"] = "gpt-6.1-sol"
    else:
        sol[0]["routing"]["model"] = "different-decision-deployment"
    monkeypatch.setattr(report, "load", lambda label: (
        sol if label == "routed-gpt6-sol" else baseline, {}
    ))
    with pytest.raises(AssertionError, match=message):
        report.build()


def test_publication_refuses_unaccepted_results(monkeypatch, tmp_path):
    source = tmp_path / "runs" / "candidate"
    source.mkdir(parents=True)
    (source / "optimization.json").write_text(json.dumps({"baseline": "reference"}))
    records = [{"case_id": str(index), "run": 0} for index in range(270)]
    monkeypatch.setattr(publish_specialists, "ROOT", tmp_path)
    monkeypatch.setattr(publish_specialists.compare_runs, "load", lambda *args: records)
    monkeypatch.setattr(publish_specialists.compare_runs, "compare", lambda *args: {"accept": False})
    with pytest.raises(ValueError, match="did not pass canonical"):
        publish_specialists.publish("candidate")
    assert not (tmp_path / "agent").exists()


def test_publication_checks_measured_policy_not_only_manifest_hash(monkeypatch, tmp_path):
    source = tmp_path / "runs" / "candidate"
    (source / "instructions").mkdir(parents=True)
    target = tmp_path / "agent" / "specialists"
    target.mkdir(parents=True)
    (target / "common.md").write_text("shared")
    hashes = {}
    for mode in publish_specialists.compare_runs.MODES:
        path = source / "instructions" / f"{mode}.md"
        path.write_text("changed after measurement")
        hashes[mode] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "optimization.json").write_text(json.dumps({
        "baseline": "reference", "instruction_sha256": hashes,
        "common_instruction_sha256": hashlib.sha256(b"shared").hexdigest(),
    }))
    records = [{
        "case_id": str(index), "run": 0, "mode": "validity",
        "policy_contract": {"policy_sha256": "original recorded policy"},
    } for index in range(270)]
    monkeypatch.setattr(publish_specialists, "ROOT", tmp_path)
    monkeypatch.setattr(publish_specialists.compare_runs, "load", lambda *args: records)
    monkeypatch.setattr(publish_specialists.compare_runs, "compare", lambda *args: {"accept": True})
    with pytest.raises(ValueError, match="does not match measured episodes"):
        publish_specialists.publish("candidate")
    assert not (target / "optimized").exists()


def _episodes(passes, tokens, mode="planning"):
    return [{
        "case_id": f"{mode}-{case}", "run": run, "mode": mode, "error": None,
        "strict_pass": case < passes, "input_tokens": tokens, "output_tokens": 0,
        "checks": {name: case < passes or name != "answer" for name in run_eval.CHECKS},
    } for run in range(3) for case in range(15)]


@pytest.mark.parametrize("passes, tokens, accept", [
    (13, 100, True),    # accuracy gain
    (12, 85, True),     # same accuracy, >=10% fewer tokens
    (12, 95, False),    # same accuracy, savings too small
    (11, 50, False),    # cheaper but less accurate
])
def test_round4_acceptance_allows_token_only_wins_without_accuracy_loss(passes, tokens, accept):
    decision = optimize_specialists.development_decision(
        _episodes(12, 100), _episodes(passes, tokens), 0.10
    )
    assert decision["accept"] is accept


def test_manual_experiments_resolve_placeholders_and_effort(tmp_path):
    experiments = {
        "_texts": {"fix": "Use calculate."},
        "planning": [
            {"name": "prompt", "append": "{fix}"},
            {"name": "cheap", "reasoning_effort": "low"},
        ],
    }
    built = optimize_specialists.manual_candidates("planning", "current", experiments)
    assert built[0]["instructions"] == "current\n\nUse calculate." and built[0]["settings"] == {}
    assert built[1]["instructions"] == "current" and built[1]["settings"] == {"reasoning_effort": "low"}
    with pytest.raises(ValueError, match="Unresolved placeholder"):
        optimize_specialists.manual_candidates(
            "planning", "current", {"planning": [{"name": "bad", "append": "{missing}"}]}
        )


def test_round4_experiment_file_resolves():
    experiments = json.loads((ROOT / "experiments" / "round4.json").read_text())
    built = {
        mode: optimize_specialists.manual_candidates(mode, "current", experiments)
        for mode in optimize_specialists.MODES
    }
    assert all(built.values())


def test_added_experiments_keep_measured_candidate_numbering(monkeypatch, tmp_path):
    monkeypatch.setattr(optimize_specialists, "ROOT", tmp_path)
    base = [{
        "case_id": f"portfolio-{case}", "run": run, "mode": mode, "error": None,
        "strict_pass": True, "reward": 1, "latency_s": 1, "input_tokens": 1, "output_tokens": 0,
        "calls": [{"raw_usage": {"input_tokens": 1}}],
        "checks": {name: True for name in run_eval.CHECKS},
    } for mode in optimize_specialists.MODES for run in range(3) for case in range(15)]
    source = tmp_path / "runs" / "reference" / "development"
    source.mkdir(parents=True)
    (source / "summary.json").write_text(json.dumps({"overall": {"episodes": 270, "errors": 0}}))
    monkeypatch.setattr(optimize_specialists.compare_runs, "load", lambda *args: base)
    monkeypatch.setattr(optimize_specialists.specialists, "instructions", lambda mode: "current")
    monkeypatch.setattr(optimize_specialists.specialists, "all_settings", lambda: {})
    monkeypatch.setattr(optimize_specialists.specialists, "common_instructions", lambda: "shared")
    monkeypatch.setattr(optimize_specialists, "load_cases", lambda *args: [])
    monkeypatch.setattr(optimize_specialists, "evaluate", lambda *args: [
        r for r in base if r["mode"] == "portfolio"
    ][:6])
    monkeypatch.setenv("MAI_MODEL_DEPLOYMENT_NAME", "test-model")
    experiments = tmp_path / "experiments.json"
    def run_with(specs):
        experiments.write_text(json.dumps({"portfolio": specs}))
        optimize_specialists.run(SimpleNamespace(
            baseline="reference", label="r", modes=["portfolio"], prepared_search=False,
            search_cases=6, candidates=0, workers=1, experiments=str(experiments),
            max_development_candidates=0, min_token_reduction=0.1,
        ))
        return json.loads((tmp_path / "runs" / "r" / "portfolio" / "candidates.json").read_text())
    run_with([{"name": "first", "append": "a"}])
    names = [c["name"] for c in run_with([{"name": "new", "append": "b"}, {"name": "first", "append": "a"}])]
    assert names == ["first", "new"]
