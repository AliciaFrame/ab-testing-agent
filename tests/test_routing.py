import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import agent
import router
import specialists


@pytest.fixture(autouse=True)
def runtime_environment(monkeypatch):
    monkeypatch.setenv("MAI_OPENAI_BASE_URL", "https://example.openai.azure.com/openai/v1")
    for name in ("MAI_MODEL_OVERRIDE", "OPTIMIZATION_CANDIDATE_ID", "OPTIMIZATION_CONFIG",
                 "MAI_SPECIALIST", "MAI_OPTIMIZATION_SPECIALIST", "MAI_ROUTER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("MAI_SPECIALIST_PROFILE", raising=False)
    monkeypatch.delenv("MAI_SPECIALIST_INSTRUCTIONS_DIR", raising=False)


def test_router_state_does_not_expose_labels_or_statistical_results():
    case = json.loads((ROOT / "data" / "cases" / "development.jsonl").read_text().splitlines()[0])
    state = router.state_for(case)
    assert set(state) == {"messages", "test_metadata"}
    assert "expected" not in json.dumps(state)
    assert all(set(test) == {"test_id", "status"} for test in state["test_metadata"])


@pytest.mark.parametrize("payload", [
    {}, {"answers": {"specialist": {"type": "choice", "choice": "new_agent"}}},
    {"answers": {"specialist": {"type": "choice", "choice": "planning",
                               "probabilities": {"planning": float("nan")}}}},
])
def test_invalid_decisions_fail_closed(payload):
    with pytest.raises(router.RoutingError):
        router.parse_choice(payload)


def test_running_status_does_not_override_validity_intent():
    state = {"messages": [{"role": "user", "content": "Is this running test trustworthy?"}],
             "test_metadata": [{"test_id": "x", "status": "running"}]}
    assert router.rules_choice(state) == "validity"


def test_authentication_diagnostic_excludes_token_and_unapproved_claims():
    claims = {
        "oid": "11111111-2222-3333-4444-555555555555",
        "appid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "aud": "https://cognitiveservices.azure.com",
        "credential": "must-not-appear", "name": "must-not-appear",
    }
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    token = f"header.{encoded}.signature"
    context = router.identity_context(token)
    assert set(context) == {"oid", "appid", "aud"}
    assert "must-not-appear" not in json.dumps(context)
    assert token not in json.dumps(context)
    assert router.identity_context("opaque") == {"identity": "unavailable"}


def test_optimized_profile_preserves_baseline_and_common_policy(monkeypatch, tmp_path):
    directory = tmp_path / "specialists"
    (directory / "optimized").mkdir(parents=True)
    (directory / "planning.md").write_text("baseline")
    (directory / "optimized" / "planning.md").write_text("selected")
    (directory / "common.md").write_text("immutable common")
    monkeypatch.setattr(specialists, "ROOT", tmp_path)
    assert specialists.instructions("planning") == "baseline"
    monkeypatch.setenv("MAI_SPECIALIST_PROFILE", "optimized")
    assert specialists.instructions("planning") == "selected"
    assert specialists.common_instructions() == "immutable common"
    monkeypatch.setenv("MAI_SPECIALIST_PROFILE", "typo")
    with pytest.raises(ValueError, match="Unknown specialist profile"):
        specialists.instructions("planning")


def test_specialist_effort_overrides_global_and_is_validated(monkeypatch, tmp_path):
    (tmp_path / "planning.md").write_text("plan")
    (tmp_path / "validity.md").write_text("check")
    (tmp_path / "settings.json").write_text(json.dumps({"planning": {"reasoning_effort": "low"}}))
    monkeypatch.setenv("MAI_SPECIALIST_INSTRUCTIONS_DIR", str(tmp_path))
    monkeypatch.setenv("MAI_MODEL_DEPLOYMENT_NAME", "test-model")
    monkeypatch.setenv("MAI_REASONING_EFFORT", "high")
    assert agent.policy_contract("planning")["reasoning_effort"] == "low"
    assert agent.policy_contract("validity")["reasoning_effort"] == "high"
    (tmp_path / "settings.json").write_text(json.dumps({"planning": {"reasoning_effort": "max"}}))
    with pytest.raises(ValueError, match="Invalid reasoning effort"):
        agent.policy_contract("planning")


def test_candidate_changes_only_its_target_specialist(monkeypatch):
    monkeypatch.setenv("MAI_MODEL_DEPLOYMENT_NAME", "test-model")
    monkeypatch.setenv("MAI_OPTIMIZATION_SPECIALIST", "planning")
    monkeypatch.setenv("OPTIMIZATION_CONFIG", json.dumps({
        "instructions": "candidate planning instructions", "model": "test-model",
    }))
    planning = agent.policy_contract("planning")
    validity = agent.policy_contract("validity")
    assert "candidate planning instructions" in planning["instructions"]
    assert "candidate planning instructions" not in validity["instructions"]
    assert specialists.common_instructions() in planning["instructions"]


def test_optimization_cannot_expand_specialist_tools(monkeypatch):
    monkeypatch.setenv("MAI_MODEL_DEPLOYMENT_NAME", "test-model")
    monkeypatch.setenv("MAI_OPTIMIZATION_SPECIALIST", "planning")
    monkeypatch.setenv("OPTIMIZATION_CONFIG", json.dumps({
        "instructions": "plan", "tools": [
            {"type": "function", "function": {"name": "ship_package", "parameters": {}}},
            {"type": "function", "function": {"name": "submit_plan", "parameters": {}}},
        ],
    }))
    assert [t["name"] for t in agent.policy_contract("planning")["tools"]] == ["submit_plan"]


def test_ambiguous_request_executes_no_specialist(monkeypatch):
    monkeypatch.delenv("MAI_SPECIALIST", raising=False)
    monkeypatch.setenv("MAI_ROUTER", "decision1")
    monkeypatch.setattr(router, "route", lambda item: {"specialist": "ambiguous"})
    result = agent._run_episode({"messages": [{"role": "user", "content": "help"}]}, [], None)
    assert result["output_tools"] == []
    assert "specify" in result["output_text"]


def test_disallowed_tool_is_not_executed(monkeypatch):
    monkeypatch.setenv("MAI_SPECIALIST", "planning")
    monkeypatch.setenv("MAI_MODEL_DEPLOYMENT_NAME", "test-model")
    monkeypatch.setenv("MAI_OPENAI_BASE_URL", "https://example.openai.azure.com/openai/v1")
    monkeypatch.delenv("MAI_OPTIMIZATION_SPECIALIST", raising=False)
    monkeypatch.delenv("OPTIMIZATION_CONFIG", raising=False)
    call = SimpleNamespace(type="function_call", name="ship_package", arguments="{}", call_id="c")
    call.model_dump = lambda **kwargs: {"type": "function_call", "name": "ship_package"}
    responses = iter([
        SimpleNamespace(usage=None, output=[call], output_text=""),
        SimpleNamespace(usage=None, output=[], output_text="done"),
    ])
    api = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: next(responses)))
    monkeypatch.setattr(agent, "execute_tool", lambda *args: pytest.fail("forbidden tool executed"))
    result = agent._run_episode({"messages": [{"role": "user", "content": "plan"}]}, [], api)
    assert "not available" in result["output_tools"][0]["output"]["error"]


def test_specialist_model_and_endpoint_override_agent_defaults(monkeypatch, tmp_path):
    for mode in ("portfolio", "validity"):
        (tmp_path / f"{mode}.md").write_text(mode)
    (tmp_path / "settings.json").write_text(json.dumps({"portfolio": {
        "model": "base-model", "endpoint": "https://other.openai.azure.com/openai/v1/"}}))
    monkeypatch.setenv("MAI_SPECIALIST_INSTRUCTIONS_DIR", str(tmp_path))
    monkeypatch.setenv("MAI_MODEL_DEPLOYMENT_NAME", "rft-model")
    monkeypatch.setenv("MAI_MODEL_OVERRIDE", "rft-model")
    portfolio, validity = agent.policy_contract("portfolio"), agent.policy_contract("validity")
    assert (portfolio["model"], portfolio["endpoint"]) == ("base-model", "https://other.openai.azure.com/openai/v1/")
    assert (validity["model"], validity["endpoint"]) == ("rft-model", "https://example.openai.azure.com/openai/v1")
    opened = []

    class Api:
        def __init__(self, endpoint):
            self.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
                usage=None, output=[], output_text=f"answered via {endpoint}"))

    from contextlib import contextmanager

    @contextmanager
    def fake_client(endpoint=None):
        opened.append(endpoint)
        yield Api(endpoint)

    monkeypatch.setattr(agent, "client", fake_client)
    monkeypatch.setenv("MAI_SPECIALIST", "portfolio")
    result = agent._run_episode({"messages": [{"role": "user", "content": "scorecard"}]}, [], Api("default"))
    assert opened == ["https://other.openai.azure.com/openai/v1/"]
    assert "other.openai.azure.com" in result["output_text"]
    (tmp_path / "settings.json").write_text(json.dumps({"portfolio": {"endpoint": "http://insecure/"}}))
    with pytest.raises(ValueError, match="Invalid endpoint"):
        agent.policy_contract("portfolio")
