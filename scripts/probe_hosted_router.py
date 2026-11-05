"""Verify an exact hosted version and grade its emitted tool trajectory."""

import argparse
import json
import sys
import time
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import VersionRefIndicator
from azure.identity import DefaultAzureCredential

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import grader


def main(version, case_id=None):
    cases = [json.loads(line) for line in
             (ROOT / "data" / "cases" / "development.jsonl").read_text(encoding="utf-8").splitlines()]
    case = next(
        c for c in cases if c["id"] == case_id
    ) if case_id else next(
        c for c in cases if c["expected"]["mode"] == "validity"
        and c["expected"]["verdict"] == "VALID"
    )
    endpoint = "https://mai-abtest-agent-resource.services.ai.azure.com/api/projects/mai-abtest-agent"
    with DefaultAzureCredential(process_timeout=60) as credential, AIProjectClient(
        endpoint=endpoint, credential=credential, allow_preview=True, retry_total=0,
    ) as project:
        session = project.agents.create_session(
            agent_name="ab-testing-agent", version_indicator=VersionRefIndicator(agent_version=version)
        )
        print("SESSION", session.agent_session_id, flush=True)
        try:
            deadline = time.monotonic() + 300
            while session.status != "active":
                if session.status == "failed" or time.monotonic() > deadline:
                    raise RuntimeError(f"Session status: {session.status}")
                time.sleep(5)
                session = project.agents.get_session(
                    agent_name="ab-testing-agent", session_id=session.agent_session_id
                )
            with project.get_openai_client(
                agent_name="ab-testing-agent", max_retries=0, timeout=600
            ) as api:
                response = api.responses.create(
                    input=case["messages"][0]["content"],
                    extra_body={"agent_session_id": session.agent_session_id},
                )
            data = response.model_dump(warnings=False)
            outputs = {
                item["call_id"]: item["output"] for item in data["output"]
                if item["type"] == "function_call_output"
            }
            trace = [
                {"name": item["name"], "arguments": item["arguments"],
                 "output": outputs.get(item["call_id"], {"error": "missing tool output"})}
                for item in data["output"] if item["type"] == "function_call"
            ]
            result = {
                "version": version, "case_id": case["id"], "response_status": response.status,
                "output_text": response.output_text, "output_tools": trace,
                "usage": data.get("usage"), "checks": grader.check({"output_tools": trace}, case),
            }
            out = ROOT / ".foundry" / "results"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"hosted-router-v{version}-{case['id']}-probe.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(json.dumps(result, indent=2))
            assert response.status == "completed" and all(result["checks"].values())
        except Exception:
            def read_logs(_pipeline, chunks, _headers):
                result = bytearray()
                try:
                    for index, chunk in enumerate(chunks):
                        result.extend(chunk[:20000 - len(result)])
                        if index >= 20 or len(result) >= 20000:
                            break
                finally:
                    _pipeline.http_response.close()
                return result.decode("utf-8", errors="replace")
            try:
                logs = project.agents.get_session_log_stream(
                    agent_name="ab-testing-agent", agent_version=version,
                    session_id=session.agent_session_id, connection_timeout=5,
                    read_timeout=10, retry_total=0, cls=read_logs,
                )
                print("LOGS", logs[-9000:])
            except Exception as error:
                print("LOGS_UNAVAILABLE", type(error).__name__)
            raise
        finally:
            project.agents.stop_session(
                agent_name="ab-testing-agent", session_id=session.agent_session_id
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="3")
    parser.add_argument("--case-id")
    args = parser.parse_args()
    main(args.version, args.case_id)
