"""Unattended RFT queue: race duplicate jobs, stop collapsing runs, evaluate finished models one at a time.

    python -m rft.orchestrate --jobs abtest-rft-port-dt abtest-rft-port-gs ... [--interval 300]

Every interval:

* **Race duplicates.** Jobs named ``<x>-dt`` / ``<x>-gs`` are the same configuration on two tiers. When one starts
  training (a ``Training started`` event) while its twin has not, the twin is cancelled.
* **Stop collapsing runs.** A job whose latest training mean reward is below ``--min-reward`` at step 3 or later is
  cancelled (the base model scores about 0.85 on these rewards, so this only catches a collapse).
* **Evaluate.** Succeeded jobs are evaluated with ``rft.evaluate`` one at a time, only when no other fine-tuned
  candidate is deployed and no other evaluation is running (fine-tune serving quota fits one candidate next to the
  deployed ``abtest-rft-eff``). The candidate deployment is deleted afterwards; a winner is redeployed for release.

Decisions are appended to ``rft/jobs/orchestrator.log``; state lives in ``rft/jobs/orchestrator-state.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

from rft.common import JOBS_DIR
from rft.submit import JOBS, RESOURCE, client

LOG = JOBS_DIR / "orchestrator.log"
STATE = JOBS_DIR / "orchestrator-state.json"
KEEP = {"abtest-rft-eff"}  # serves hosted v10


def log(message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def az(*args: str) -> str:
    return subprocess.run(["az", *args, "--subscription", RESOURCE["subscription_id"], "-o", "tsv"],
                          capture_output=True, text=True, shell=sys.platform == "win32").stdout.strip()


def candidate_deployments() -> list[str]:
    names = az("cognitiveservices", "account", "deployment", "list", "-g", RESOURCE["resource_group"],
               "-n", RESOURCE["foundry_resource"], "--query", "[].name").split()
    return [n for n in names if n.startswith("abtest-rft-") and n not in KEEP]


def evaluation_running() -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "Where-Object { $_.CommandLine -match 'rft\\.evaluate' } | Measure-Object | "
                          "Select-Object -ExpandProperty Count"], capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


def snapshot(c, job: str) -> dict:
    receipt = json.loads((JOBS_DIR / job / "submission.json").read_text(encoding="utf-8"))
    info = c.fine_tuning.jobs.retrieve(receipt["job_id"])
    events = c.fine_tuning.jobs.list_events(receipt["job_id"], limit=100).data
    rewards = []
    for e in events:
        if e.message.startswith("Step ") and "training mean reward=" in e.message:
            step, value = e.message[5:].split(":", 1)
            value = value.split("=", 1)[1].strip()
            if value:
                rewards.append((int(step), float(value)))
    return {"id": receipt["job_id"], "status": info.status, "started": any(e.message == "Training started."
                                                                          for e in events),
            "rewards": sorted(rewards)}


def twin(job: str) -> str | None:
    for a, b in (("-dt", "-gs"), ("-gs", "-dt")):
        if job.endswith(a) and job[: -len(a)] + b in JOBS:
            return job[: -len(a)] + b
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", nargs="+", required=True, choices=sorted(JOBS))
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--min-reward", type=float, default=0.5)
    args = parser.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    c = client()
    while True:
        snaps = {}
        for job in args.jobs:
            try:
                snaps[job] = snapshot(c, job)
            except Exception as error:  # noqa: BLE001 - transient API errors: retry next round
                log(f"{job}: status error {error}")
        for job, s in snaps.items():
            last = s["rewards"][-1] if s["rewards"] else None
            key = f"{s['status']}|{last}"
            if state.get(job, {}).get("seen") != key:
                log(f"{job}: {s['status']} started={s['started']} reward={last}")
                state.setdefault(job, {})["seen"] = key
            other = twin(job)
            if (s["started"] and other in snaps and not snaps[other]["started"]
                    and snaps[other]["status"] in ("pending", "queued", "running", "validating_files")):
                c.fine_tuning.jobs.cancel(snaps[other]["id"])
                snaps[other]["status"] = "cancelled"
                log(f"{other}: cancelled (twin {job} started training first)")
            if s["status"] == "running" and last and last[0] >= 3 and last[1] < args.min_reward:
                c.fine_tuning.jobs.cancel(s["id"])
                s["status"] = "cancelled"
                log(f"{job}: cancelled (training mean reward {last[1]:.3f} at step {last[0]})")
        ready = [j for j, s in snaps.items() if s["status"] == "succeeded" and not state.get(j, {}).get("evaluated")]
        if ready and not candidate_deployments() and not evaluation_running():
            job = ready[0]
            log(f"{job}: evaluating")
            out = subprocess.run([sys.executable, "-m", "rft.evaluate", "--job", job], capture_output=True, text=True)
            (JOBS_DIR / job / "evaluate.log").write_text(out.stdout + out.stderr, encoding="utf-8")
            lines = out.stdout.strip().splitlines()
            log(f"{job}: evaluation exit {out.returncode}: {lines[-1] if lines else out.stderr.strip()[-300:]}")
            az("cognitiveservices", "account", "deployment", "delete", "-g", RESOURCE["resource_group"],
               "-n", RESOURCE["foundry_resource"], "--deployment-name", job)
            state[job]["evaluated"] = out.returncode == 0
            if out.returncode != 0:
                state[job]["failures"] = state[job].get("failures", 0) + 1
                state[job]["evaluated"] = state[job]["failures"] >= 2
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        if all(s["status"] in ("succeeded", "failed", "cancelled") for s in snaps.values()) and \
                all(state.get(j, {}).get("evaluated") for j, s in snaps.items() if s["status"] == "succeeded"):
            log("all jobs finished and evaluated")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
