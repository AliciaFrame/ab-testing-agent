"""Deploy a finished RFT model and evaluate it with the full routed agent on the development benchmark.

For each succeeded job: deploy the fine-tuned model (GlobalStandard) on the training resource, run the routed agent
(Decision-1 router + round-5 specialist prompts, ``MAI_SPECIALIST_PROFILE=optimized-r5``) on 90 development cases x 3
runs with that model for every specialist, re-run transport failures only, then compare with the deployed agent
(``routed-mai-r5``) under ``docs/acceptance.md``. Low-effort jobs are served at low reasoning effort.

    python -m rft.evaluate --job abtest-rft-low
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from rft.common import JOBS_DIR, ROOT
from rft.submit import JOBS, RESOURCE, client

DEPLOY_CAPACITY = 250  # four models may be deployed at once; 250K TPM still runs 270 episodes in ~40 minutes
WORKERS = 8


def az(*args: str) -> str:
    out = subprocess.run(["az", *args, "--subscription", RESOURCE["subscription_id"], "-o", "json"],
                         capture_output=True, text=True, shell=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-800:])
    return out.stdout


def deploy(model: str, name: str) -> None:
    existing = subprocess.run(["az", "cognitiveservices", "account", "deployment", "show", "--subscription",
                               RESOURCE["subscription_id"], "-g", RESOURCE["resource_group"], "-n",
                               RESOURCE["foundry_resource"], "--deployment-name", name, "-o", "json"],
                              capture_output=True, text=True, shell=True)
    if existing.returncode != 0:
        az("cognitiveservices", "account", "deployment", "create", "-g", RESOURCE["resource_group"],
           "-n", RESOURCE["foundry_resource"], "--deployment-name", name, "--model-name", model,
           "--model-version", "1", "--model-format", "OpenAI", "--sku-name", "GlobalStandard",
           "--sku-capacity", str(DEPLOY_CAPACITY))
    for _ in range(90):
        state = json.loads(az("cognitiveservices", "account", "deployment", "show", "-g", RESOURCE["resource_group"],
                              "-n", RESOURCE["foundry_resource"], "--deployment-name", name))["properties"]["provisioningState"]
        if state == "Succeeded":
            time.sleep(60)  # data plane can lag ARM; avoid DeploymentNotReady
            return
        if state in ("Failed", "Canceled"):
            raise RuntimeError(f"deployment {name} {state}")
        time.sleep(20)
    raise RuntimeError(f"deployment {name} did not finish")


def evaluate(job: str) -> dict:
    receipt = json.loads((JOBS_DIR / job / "submission.json").read_text(encoding="utf-8"))
    info = client().fine_tuning.jobs.retrieve(receipt["job_id"])
    if info.status != "succeeded" or not info.fine_tuned_model:
        raise SystemExit(f"{job} is {info.status}; nothing to evaluate")
    name = job  # deployment name = job name
    deploy(info.fine_tuned_model, name)
    label = f"rft-{job.removeprefix('abtest-rft-')}"
    env = {**os.environ, "MAI_OPENAI_BASE_URL": RESOURCE["base_url"], "MAI_MODEL_DEPLOYMENT_NAME": name,
           "MAI_SPECIALIST_PROFILE": "optimized-r5", "MAI_ROUTER": "decision1",
           "DECISION_ENDPOINT": "https://mai-abtest-agent-resource.services.ai.azure.com",
           "DECISION_DEPLOYMENT": "abtest-decision-1", "MAI_MAX_RETRIES": "8", "MAI_MAX_OUTPUT_TOKENS": "32768",
           "PYTHONWARNINGS": "ignore"}
    env.pop("MAI_SPECIALIST_INSTRUCTIONS_DIR", None)
    env.pop("MAI_INSTRUCTIONS_PATH", None)
    if JOBS[job]["effort"] == "low":
        env["MAI_REASONING_EFFORT"] = "low"
    else:
        env.pop("MAI_REASONING_EFFORT", None)
    run = [sys.executable, str(ROOT / "scripts" / "run_eval.py"), "--split", "development", "--runs", "3",
           "--label", label, "--workers", str(WORKERS)]
    summary_path = ROOT / "runs" / label / "development" / "summary.json"
    if not summary_path.exists():
        subprocess.run(run, env=env, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    subprocess.run(run + ["--rerun-transport-errors", "--max-reruns", "5"], env=env, check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(ROOT / "scripts"))
    import compare_runs

    acceptance = compare_runs.compare(compare_runs.load("routed-mai-r5", "development"),
                                      compare_runs.load(label, "development"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = {"job": job, "job_id": info.id, "fine_tuned_model": info.fine_tuned_model, "deployment": name,
              "label": label, "served_reasoning_effort": env.get("MAI_REASONING_EFFORT", "default"),
              "overall": summary["overall"],
              "by_task": {m: summary[m]["strict_pass"] for m in compare_runs.MODES if m in summary},
              "acceptance_vs_r5": acceptance}
    (JOBS_DIR / job / "evaluation.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    o = summary["overall"]
    print(f"{job}: strict {o['strict_pass']:.1%}, tokens/episode {o['total_tokens'] / o['episodes']:,.0f}, "
          f"errors {o['errors']}, accept vs r5: {acceptance['accept']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True, choices=sorted(JOBS))
    evaluate(parser.parse_args().job)


if __name__ == "__main__":
    main()
