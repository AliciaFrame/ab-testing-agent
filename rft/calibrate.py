"""Calibrate RFT rewards on the base model through the hosted tool server and grader.

Each rollout replays what the trainer does: the row's developer + user messages and the shared 11-tool list, tool
calls executed by ``POST /tools/{name}`` on the hosted service with one ``trace_id`` per rollout, then
``POST /grade/acc`` and ``/grade/eff``. Every hosted score is re-computed locally (parity check).

    python -m rft.calibrate --split train --modes planning portfolio --items 30 --efforts medium low \
        --base-url https://<resource>.openai.azure.com/openai/v1/ --deployment <base MAI deployment> \
        --server https://<tool server> --out rft/calibration/<name>.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

from rft import grader
from rft.common import TOOLS, load_cases, read_jsonl, row

SEED = 20261104
_lock = threading.Lock()


def token(vault: str, name: str, subscription: str) -> str:
    out = subprocess.run(["az", "keyvault", "secret", "show", "--subscription", subscription, "--vault-name", vault,
                          "-n", name, "--query", "value", "-o", "tsv"], capture_output=True, text=True, check=True,
                         shell=True)
    return out.stdout.strip()


def response_tools() -> list[dict]:
    return [{"type": "function", **tool["function"]} for tool in TOOLS]


def rollout(api: OpenAI, http: httpx.Client, server: str, auth: dict, deployment: str, item: dict, effort: str,
            max_steps: int) -> dict:
    trace_id = str(uuid.uuid4())
    inputs = [{"role": m["role"], "content": m["content"]} for m in item["messages"]]
    calls, usage_in, usage_out, turns, started = [], 0, 0, 0, time.time()
    answer = ""
    for _ in range(max_steps):
        response = None
        for attempt in range(8):
            try:
                response = api.responses.create(model=deployment, input=inputs, tools=response_tools(),
                                                reasoning={"effort": effort}, max_output_tokens=32768)
                break
            except Exception as error:  # noqa: BLE001 - provider throttling/transient errors are retried
                if attempt == 7:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt) if "429" in str(error) or "timeout" in str(error).lower() else 5)
        turns += 1
        usage_in += getattr(response.usage, "input_tokens", 0) or 0
        usage_out += getattr(response.usage, "output_tokens", 0) or 0
        inputs.extend(o.model_dump(exclude_none=True) for o in response.output)
        function_calls = [o for o in response.output if o.type == "function_call"]
        if not function_calls:
            answer = response.output_text
            break
        for call in function_calls:
            reply = http.post(f"{server}/tools/{call.name}", headers=auth, timeout=120, json={
                "type": "function_call", "call_id": call.call_id, "id": call.id, "name": call.name,
                "arguments": call.arguments, "trace_id": trace_id})
            reply.raise_for_status()
            output = reply.json()["output"]
            calls.append({"name": call.name, "arguments": call.arguments, "call_id": call.call_id, "output": output})
            inputs.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
    else:
        answer = ""
    sample = {"output_text": answer, "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
              "output_tools": [{"type": "function_call", "name": c["name"], "arguments": c["arguments"],
                                "call_id": c["call_id"]} for c in calls]}
    hosted = {}
    for version in ("acc", "eff"):
        reply = http.post(f"{server}/grade/{version}", headers=auth, timeout=180,
                          json={"sample": sample, "item": item, "trace_id": trace_id})
        reply.raise_for_status()
        hosted[version] = reply.json()["score"]
    local_calls = [{"call_id": c["call_id"], "name": c["name"], "arguments": grader._arguments(c["arguments"]),
                    "output": c["output"]} for c in calls]
    local = {v: grader.grade(sample, item, local_calls, v) for v in ("acc", "eff")}
    # ``turns`` needs the server's call timestamps, so it has no local twin: record the hosted score and the hosted
    # turn estimate next to the true number of tool-calling turns.
    reply = http.post(f"{server}/grade/turns", headers=auth, timeout=180,
                      json={"sample": sample, "item": item, "trace_id": trace_id})
    reply.raise_for_status()
    turns_score = reply.json()["score"]
    details = http.get(f"{server}/grades/{trace_id}", headers=auth, timeout=60).json()["grades"]
    estimated = next((g["details"]["efficiency"]["turns"] for g in details if g["version"] == "turns"), None)
    return {"case_id": item["case_id"], "mode": item["mode"], "effort": effort, "trace_id": trace_id,
            "acc": hosted["acc"], "eff": hosted["eff"], "turns_reward": turns_score,
            "tool_turns": turns - (1 if answer else 0), "estimated_tool_turns": estimated, "strict_pass": local["acc"]["strict_pass"],
            "checks": local["acc"]["checks"],
            "parity": all(abs(hosted[v] - local[v]["score"]) < 1e-9 for v in hosted),
            "tool_calls": len(calls), "turns": turns, "input_tokens": usage_in, "output_tokens": usage_out,
            "truncated": not answer and len(calls) > 0, "seconds": round(time.time() - started, 1), "error": None}


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if not r["error"]]

    def block(rows):
        if not rows:
            return {}
        timed = [r for r in rows if r.get("estimated_tool_turns") is not None]
        return {"episodes": len(rows), "strict_pass": sum(r["strict_pass"] for r in rows) / len(rows),
                "acc_fail_rate_at_1": sum(r["acc"] < 0.999 for r in rows) / len(rows),
                "eff_fail_rate_at_0.9": sum(r["eff"] < 0.9 for r in rows) / len(rows),
                "mean_acc": statistics.mean(r["acc"] for r in rows), "mean_eff": statistics.mean(r["eff"] for r in rows),
                "tool_calls_p50": statistics.median(r["tool_calls"] for r in rows),
                "tool_calls_max": max(r["tool_calls"] for r in rows), "turns_max": max(r["turns"] for r in rows),
                "output_tokens_mean": statistics.mean(r["output_tokens"] for r in rows),
                "input_tokens_mean": statistics.mean(r["input_tokens"] for r in rows),
                **({"tool_turns_mean": statistics.mean(r["tool_turns"] for r in timed),
                    "estimated_tool_turns_mean": statistics.mean(r["estimated_tool_turns"] for r in timed),
                    "turn_estimate_exact": sum(r["estimated_tool_turns"] == r["tool_turns"] for r in timed) / len(timed),
                    "mean_turns_reward": statistics.mean(r["turns_reward"] for r in timed)} if timed else {})}

    return {"rollouts": len(results), "errors": len(results) - len(ok),
            "parity_mismatches": sum(not r["parity"] for r in ok), "overall": block(ok),
            "by_mode_effort": {f"{m}/{e}": block([r for r in ok if r["mode"] == m and r["effort"] == e])
                               for m in sorted({r["mode"] for r in ok}) for e in sorted({r["effort"] for r in ok})}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train")
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--items", type=int, default=30, help="items per mode (seeded sample)")
    parser.add_argument("--offset", type=int, default=0, help="skip this many seeded items per mode")
    parser.add_argument("--efforts", nargs="+", default=["medium"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--vault", default="kv-att-evalmai-75343")
    parser.add_argument("--secret", default="abtest-rft-endpoint-token")
    parser.add_argument("--subscription", default="507baf40-e783-4961-baf7-d7bc3109baab")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    by_mode = {}
    for case in load_cases(args.split):
        by_mode.setdefault(case["expected"]["mode"], []).append(case)
    items = []
    for mode in args.modes:
        pool = sorted(by_mode[mode], key=lambda c: c["id"])
        random.Random(f"{SEED}-{mode}").shuffle(pool)
        items += [row(c) for c in pool[args.offset:args.offset + args.items]]
    done = {(r["case_id"], r["effort"]) for r in read_jsonl(args.out) if not r["error"]} if args.out.exists() else set()
    jobs = [(i, e) for e in args.efforts for i in items if (i["case_id"], e) not in done]
    auth = {"Authorization": "Bearer " + token(args.vault, args.secret, args.subscription)}
    provider = get_bearer_token_provider(DefaultAzureCredential(process_timeout=120),
                                         "https://cognitiveservices.azure.com/.default")
    api = OpenAI(base_url=args.base_url, api_key=provider, max_retries=0, timeout=900)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def work(job):
        item, effort = job
        with httpx.Client() as http:
            try:
                result = rollout(api, http, args.server, auth, args.deployment, item, effort, args.max_steps)
            except Exception as error:  # noqa: BLE001 - recorded, never scored
                result = {"case_id": item["case_id"], "mode": item["mode"], "effort": effort,
                          "error": f"{type(error).__name__}: {error}"[:500]}
        with _lock, args.out.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(result) + "\n")
        print(f"{item['case_id']} {effort} acc={result.get('acc')} eff={result.get('eff')} "
              f"calls={result.get('tool_calls')} err={result['error']}", flush=True)

    with ThreadPoolExecutor(args.concurrency) as pool:
        list(pool.map(work, jobs))
    results = read_jsonl(args.out)
    summary = summarize(results)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=1))


if __name__ == "__main__":
    main()
