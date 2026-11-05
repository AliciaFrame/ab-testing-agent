"""Submit and monitor the agentic RFT jobs (MAI-Code-1.1-Flash, developerTier, endpoint grader + 11 tools).

    python -m rft.submit preflight
    python -m rft.submit submit --jobs abtest-rft-acc abtest-rft-eff abtest-rft-low abtest-rft-low-eff --confirm-submit
    python -m rft.submit status

Every job trains the base model, served by the same tool server. Jobs differ in reward (``/grade/acc``, ``/grade/eff``
or ``/grade/turns``), training reasoning effort, epochs and rows (``rft/data`` = v1; the turns jobs use
``rft/data/v2``, portfolio doubled). Receipts (redacted payload, file and job IDs)
are written to ``rft/jobs/<name>/``. The bearer token is read from Key Vault and never written to disk.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import httpx
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import OpenAI

from rft import grader
from rft.common import DATA_DIR, JOBS_DIR, RFT_DIR, TOOL_NAMES, read_jsonl, sha256_file

RESOURCE = {"subscription_id": "507baf40-e783-4961-baf7-d7bc3109baab", "resource_group": "alicia-rl-demo",
            "foundry_resource": "amf-mai-shoppingbench-resource", "region": "swedencentral",
            "base_url": "https://amf-mai-shoppingbench-resource.openai.azure.com/openai/v1/"}
SERVICE = {"container_app": "abtest-rft-tools",
           "url": "https://abtest-rft-tools.reddune-8f0ff2f0.swedencentral.azurecontainerapps.io",
           "token_secret": {"vault": "kv-att-evalmai-75343", "name": "abtest-rft-endpoint-token"}}
MODEL = "mai-code-1.1-flash"  # resolves to MAI-Code-1.1-Flash-2026-09-15, the version of every benchmark run
TRAINING_TYPE = "developerTier"
MAX_EPISODE_STEPS = 160  # calibration: portfolio up to 130 tool calls / 15 turns; planning 48 / 12
_COMMON = {"n_epochs": 1, "batch_size": 8, "learning_rate_multiplier": 1.0, "eval_interval": 5, "eval_samples": 10,
           "compute_multiplier": 1.0}
JOBS = {
    "abtest-rft-acc": {"route": "acc", "threshold": 0.99, "effort": "medium"},
    "abtest-rft-eff": {"route": "eff", "threshold": 0.9, "effort": "medium"},
    "abtest-rft-low": {"route": "acc", "threshold": 0.99, "effort": "low"},
    "abtest-rft-low-eff": {"route": "eff", "threshold": 0.9, "effort": "low"},
    # Same configs on a non-spot tier; queued after the developerTier jobs waited over an hour for GPUs.
    "abtest-rft-acc-gs": {"route": "acc", "threshold": 0.99, "effort": "medium", "training_type": "GlobalStandard"},
    "abtest-rft-leff-gs": {"route": "eff", "threshold": 0.9, "effort": "low", "training_type": "GlobalStandard"},
    # Turn-aware reward (``rft/grader.py``) on dataset v2 (portfolio doubled); one vs two epochs.
    "abtest-rft-turns": {"route": "turns", "threshold": 0.9, "effort": "medium", "training_type": "GlobalStandard",
                         "dataset": "v2"},
    "abtest-rft-turns-2ep": {"route": "turns", "threshold": 0.9, "effort": "medium", "training_type": "GlobalStandard",
                             "dataset": "v2", "n_epochs": 2},
    # Same as abtest-rft-turns on developerTier, queued after the GlobalStandard jobs waited two hours for GPUs.
    "abtest-rft-turns-dt": {"route": "turns", "threshold": 0.9, "effort": "medium", "dataset": "v2"},
    # abtest-rft-turns-dt trained two steps on turns-v1, which counted drift in the trainer's call spacing as turns;
    # cancelled and resubmitted on turns-v2.
    "abtest-rft-turns-dt2": {"route": "turns", "threshold": 0.9, "effort": "medium", "dataset": "v2"},
}


def token() -> str:
    value = subprocess.run(
        f"az keyvault secret show --subscription {RESOURCE['subscription_id']} --vault-name "
        f"{SERVICE['token_secret']['vault']} -n {SERVICE['token_secret']['name']} --query value -o tsv",
        shell=True, capture_output=True, text=True, check=True).stdout.strip()
    if len(value) < 32:
        raise SystemExit("endpoint token missing or too short")
    return value


def client() -> OpenAI:
    return OpenAI(base_url=RESOURCE["base_url"], api_key=get_bearer_token_provider(
        AzureCliCredential(process_timeout=180), "https://cognitiveservices.azure.com/.default"))


def payload(job: str, train_id: str, valid_id: str, secret: str | None) -> dict:
    cfg = JOBS[job]
    headers = {"Authorization": f"Bearer {secret}" if secret else "Bearer <redacted>"}
    return {
        "model": MODEL, "suffix": job, "training_file": train_id, "validation_file": valid_id,
        "method": {"type": "reinforcement", "reinforcement": {
            "grader": {"type": "endpoint", "name": f"abtest_{cfg['route']}", "url": f"{SERVICE['url']}/grade/{cfg['route']}",
                       "headers": headers, "rate_limit": 20, "pass_threshold": cfg["threshold"]},
            "tools": [{"name": n, "server_url": f"{SERVICE['url']}/tools/{n}", "headers": headers} for n in TOOL_NAMES],
            "max_episode_steps": MAX_EPISODE_STEPS, "pass_threshold": cfg["threshold"],
            "hyperparameters": {**_COMMON, "n_epochs": cfg.get("n_epochs", _COMMON["n_epochs"]),
                                "reasoning_effort": cfg["effort"]}}},
        "extra_body": {"trainingType": cfg.get("training_type", TRAINING_TYPE)},
    }


def data_dir(dataset: str):
    return DATA_DIR if dataset == "v1" else DATA_DIR / dataset


def preflight(dataset: str = "v1") -> dict:
    data = data_dir(dataset)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name in ("train.jsonl", "valid.jsonl"):
        if sha256_file(data / name) != manifest["files"][name]["sha256"]:
            raise SystemExit(f"{name} does not match the manifest")
    rows = read_jsonl(data / "train.jsonl") + read_jsonl(data / "valid.jsonl")
    if len({json.dumps(r["tools"]) for r in rows}) != 1 or [t["function"]["name"] for t in rows[0]["tools"]] != TOOL_NAMES:
        raise SystemExit("rows must share the job's tool list")
    if manifest["heldout_test_overlap"]:
        raise SystemExit("rows overlap held-out tests")
    if manifest["max_row_bytes"] > 900_000:
        raise SystemExit("a row is too large")
    health = httpx.get(f"{SERVICE['url']}/health", timeout=30).json()
    if health.get("graders") != grader.VERSIONS or health.get("tools") != TOOL_NAMES:
        raise SystemExit(f"service does not serve the local grader/tool versions: {health}")
    calibration = []
    for path in sorted((RFT_DIR / "calibration").glob("train-*.jsonl")):
        calibration += read_jsonl(path)
    ok = [r for r in calibration if not r.get("error")]
    if not ok or any(not r["parity"] for r in ok):
        raise SystemExit("calibration missing or hosted/local score mismatch")
    return {"dataset": dataset, "health": health, "train_sha256": manifest["files"]["train.jsonl"]["sha256"],
            "valid_sha256": manifest["files"]["valid.jsonl"]["sha256"], "calibration_rollouts": len(ok),
            "calibration_errors": len(calibration) - len(ok)}


def write(path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")


def upload(c: OpenAI, checks: dict) -> dict:
    files_path = JOBS_DIR / ("files.json" if checks["dataset"] == "v1" else f"files-{checks['dataset']}.json")
    files = json.loads(files_path.read_text(encoding="utf-8")) if files_path.exists() else {}
    if files.get("train_sha256") != checks["train_sha256"] or files.get("valid_sha256") != checks["valid_sha256"]:
        files = {"train_sha256": checks["train_sha256"], "valid_sha256": checks["valid_sha256"]}
        for key, name in (("training_file", "train.jsonl"), ("validation_file", "valid.jsonl")):
            with (data_dir(checks["dataset"]) / name).open("rb") as handle:
                files[key] = c.files.create(file=(name, handle.read(), "application/jsonl"), purpose="fine-tune").id
        write(files_path, files)
    for key in ("training_file", "validation_file"):
        deadline = time.monotonic() + 1800
        while (status := c.files.retrieve(files[key]).status) not in ("processed", "completed"):
            if status in ("error", "failed") or time.monotonic() > deadline:
                raise SystemExit(f"file {files[key]} not processed: {status}")
            time.sleep(10)
    return files


def submit(jobs: list[str]) -> list[dict]:
    c, secret, receipts, uploaded = client(), token(), [], {}
    for job in jobs:
        dataset = JOBS[job].get("dataset", "v1")
        if dataset not in uploaded:
            checks = preflight(dataset)
            uploaded[dataset] = (checks, upload(c, checks))
        checks, files = uploaded[dataset]
        path = JOBS_DIR / job / "submission.json"
        receipt = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if receipt.get("job_id"):
            print(f"{job}: already submitted as {receipt['job_id']}")
            receipts.append(receipt)
            continue
        body = payload(job, files["training_file"], files["validation_file"], secret)
        receipt = {"resource": RESOURCE, "service": SERVICE, "model": MODEL, "config": JOBS[job],
                   "max_episode_steps": MAX_EPISODE_STEPS, "preflight": checks, "files": files,
                   "payload": payload(job, files["training_file"], files["validation_file"], None)}
        extra = body.pop("extra_body")
        created = c.fine_tuning.jobs.create(**body, extra_body=extra)
        receipt.update(job_id=created.id, status=created.status, created_at=created.created_at, base_model=created.model)
        write(path, receipt)
        receipts.append(receipt)
        print(f"{job}: submitted {created.id} ({created.status})")
    return receipts


def status() -> list[dict]:
    c, out = client(), []
    for job in JOBS:
        path = JOBS_DIR / job / "submission.json"
        if not path.exists():
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        info = c.fine_tuning.jobs.retrieve(receipt["job_id"])
        events = c.fine_tuning.jobs.list_events(receipt["job_id"], limit=100).data
        metrics = [e.data for e in events if isinstance(getattr(e, "data", None), dict) and e.data.get("step") is not None]
        checkpoints = c.fine_tuning.jobs.checkpoints.list(receipt["job_id"]).data if info.status == "succeeded" else []
        row = {"job": job, "job_id": info.id, "status": info.status, "fine_tuned_model": info.fine_tuned_model,
               "error": info.error.model_dump() if getattr(info, "error", None) else None,
               "latest_metrics": metrics[0] if metrics else None,
               "checkpoints": [{"id": k.id, "step": k.step_number, "model": k.fine_tuned_model_checkpoint,
                                "metrics": k.metrics.model_dump() if k.metrics else None} for k in checkpoints],
               "events": [{"created_at": e.created_at, "message": e.message} for e in events[:15]]}
        write(JOBS_DIR / job / "status.json", row)
        out.append(row)
        print(f"{job}: {info.status} step={(row['latest_metrics'] or {}).get('step')} "
              f"model={info.fine_tuned_model or '-'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["preflight", "submit", "status"])
    parser.add_argument("--jobs", nargs="+", choices=sorted(JOBS), default=sorted(JOBS))
    parser.add_argument("--confirm-submit", action="store_true")
    parser.add_argument("--dataset", default="v1", help="preflight only: v1 (rft/data) or v2 (rft/data/v2)")
    args = parser.parse_args()
    if args.action == "preflight":
        print(json.dumps(preflight(args.dataset), indent=1, default=str))
    elif args.action == "submit":
        if not args.confirm_submit:
            raise SystemExit("submission requires --confirm-submit")
        submit(args.jobs)
    else:
        status()


if __name__ == "__main__":
    main()
