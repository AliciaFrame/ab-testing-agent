"""Parallel load + parity test of the hosted RFT service against recorded agent episodes.

Replays recorded development episodes (tool calls in order, one trace per episode) through ``/tools/{name}``,
compares every output with local execution, then grades each episode with ``/grade/acc`` and checks the score
equals the benchmark's canonical reward.

    python -m rft.loadtest --server https://<tool server> --episodes 60 --concurrency 32
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx

from rft import grader
from rft.common import ROOT, load_cases, read_jsonl, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True)
    parser.add_argument("--label", default="routed-mai-r5")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--out", default=str(ROOT / "rft" / "calibration" / "loadtest.json"))
    args = parser.parse_args()
    tok = subprocess.run("az keyvault secret show --subscription 507baf40-e783-4961-baf7-d7bc3109baab --vault-name "
                         "kv-att-evalmai-75343 -n abtest-rft-endpoint-token --query value -o tsv", shell=True,
                         capture_output=True, text=True, check=True).stdout.strip()
    auth = {"Authorization": f"Bearer {tok}"}
    episodes = read_jsonl(ROOT / "runs" / args.label / "development" / "episodes.jsonl")
    random.Random(7).shuffle(episodes)
    episodes = episodes[:args.episodes]
    cases = {c["id"]: c for c in load_cases("development")}

    def replay(ep):
        trace_id, mismatches, failures = str(uuid.uuid4()), 0, 0
        with httpx.Client(timeout=120) as http:
            for i, t in enumerate(ep["output_tools"]):
                arguments = t["arguments"] if isinstance(t["arguments"], dict) else json.loads(t["arguments"])
                reply = http.post(f"{args.server}/tools/{t['name']}", headers=auth, json={
                    "type": "function_call", "call_id": f"c{i}", "arguments": json.dumps(arguments), "trace_id": trace_id})
                if reply.status_code != 200:
                    failures += 1
                    continue
                mismatches += reply.json()["output"] != grader.execute(t["name"], arguments)
            sample = {"output_text": ep["output_text"], "output_tools": [
                {"type": "function_call", "name": t["name"], "arguments": json.dumps(
                    t["arguments"] if isinstance(t["arguments"], dict) else json.loads(t["arguments"]))}
                for t in ep["output_tools"]]}
            reply = http.post(f"{args.server}/grade/acc", headers=auth,
                              json={"sample": sample, "item": row(cases[ep["case_id"]]), "trace_id": trace_id})
            score = reply.json()["score"] if reply.status_code == 200 else None
        return {"calls": len(ep["output_tools"]), "mismatches": mismatches, "failures": failures,
                "grade_ok": score is not None and abs(score - ep["reward"]) < 1e-9}

    started = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(replay, episodes))
    seconds = time.time() - started
    calls = sum(r["calls"] for r in results)
    summary = {"episodes": len(results), "tool_calls": calls, "requests_per_second": round((calls + len(results)) / seconds, 1),
               "tool_failures": sum(r["failures"] for r in results), "output_mismatches": sum(r["mismatches"] for r in results),
               "grade_mismatches": sum(not r["grade_ok"] for r in results), "concurrency": args.concurrency}
    (ROOT / "rft" / "calibration").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
