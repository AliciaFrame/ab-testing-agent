"""Build a standalone comparison from recorded, matched development episodes."""

import argparse
import html
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import compare_runs
import run_eval

LABELS = {
    "routed-mai-baseline": "Routed MAI baseline",
    "routed-mai-optimized": "Optimized MAI · round 3",
    "routed-mai-r4": "Optimized MAI · round 4",
    "routed-mai-r5": "Optimized MAI · round 5",
    "routed-gpt6-sol": "Routed GPT-6 Sol",
    "routed-gpt6-sol-r5": "GPT-6 Sol · final prompts",
}
MAI_LABELS = ("routed-mai-baseline", "routed-mai-optimized", "routed-mai-r4", "routed-mai-r5")
FINAL_MAI, FINAL_SOL = "routed-mai-r5", "routed-gpt6-sol-r5"


def load(label):
    directory = ROOT / "runs" / label / "development"
    path = directory / "episodes.jsonl"
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        import gzip

        with gzip.open(path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as handle:
            text = handle.read()
    records = [json.loads(line) for line in text.splitlines()]
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert len(records) == 270, (label, len(records))
    assert len({(r["case_id"], r["run"]) for r in records}) == 270
    assert not any(r["error"] for r in records), f"{label} has unresolved runtime failures"
    assert run_eval.summarize(records)["usage_complete"], f"{label} has missing token usage"
    assert all(r.get("routing") and r.get("policy_contract") for r in records)
    return records, summary


def evidence(records):
    policies = {}
    for mode in compare_runs.MODES:
        rows = [r for r in records if r["mode"] == mode]
        policies[mode] = sorted({
            r.get("policy_contract", {}).get("policy_sha256", "unrecorded") for r in rows
        })
    return {
        "actual_inference_models": sorted({
            c.get("returned_model", "unrecorded")
            for r in records for c in r["calls"] if c.get("purpose") == "inference"
        }),
        "router_hashes": sorted({
            r["routing"]["router_sha256"] for r in records if r.get("routing")
        }),
        "router_deployments": sorted({r["routing"]["model"] for r in records}),
        "router_backends": sorted({r["routing"]["backend"] for r in records}),
        "specialist_endpoints": sorted({r["policy_contract"]["endpoint"] for r in records}),
        "task_policy_hashes": policies,
        "reasoning_effort_by_task": {
            mode: sorted({
                str(r["policy_contract"].get("reasoning_effort") or "default")
                for r in records if r["mode"] == mode
            })
            for mode in compare_runs.MODES
        },
        "reused_baseline_episodes": sum(
            r.get("measurement_source") == "unchanged_baseline_specialist" for r in records
        ),
    }


def metrics(records):
    result = run_eval.summarize(records)
    result["total_routing_tokens"] = sum(
        (call.get("raw_usage") or {}).get("total_tokens", 0)
        for record in records for call in record["calls"] if call.get("purpose") == "routing"
    )
    result["total_specialist_tokens"] = result["total_tokens"] - result["total_routing_tokens"]
    result["total_reasoning_tokens"] = sum(
        ((call.get("raw_usage") or {}).get("output_tokens_details") or {}).get("reasoning_tokens") or 0
        for record in records for call in record["calls"] if call.get("purpose") != "routing"
    )
    result["reasoning_share_of_output"] = (
        result["total_reasoning_tokens"] / result["total_output_tokens"]
        if result["total_output_tokens"] else 0.0
    )
    result["mean_routing_latency_s"] = statistics.mean(
        r["routing"]["latency_s"] for r in records if r.get("routing")
    )
    return result


def table(rows, overall=False):
    head = (
        "<tr><th>Configuration</th><th>Strict accuracy</th><th>Episodes</th>"
        "<th>Total tokens</th><th>Tokens / episode</th><th>Mean latency</th>"
        "<th>Median latency</th><th>P95 latency</th></tr>"
    )
    body = []
    for name, result in rows:
        body.append(
            f"<tr><td>{html.escape(name)}</td><td><strong>{result['strict_pass']:.1%}</strong></td>"
            f"<td>{result['episodes']}</td><td>{result['total_tokens']:,}</td>"
            f"<td>{result['total_tokens']/result['episodes']:,.0f}</td>"
            f"<td>{result['mean_latency_s']:.1f}s</td><td>{result['median_latency_s']:.1f}s</td>"
            f"<td>{result['p95_latency_s']:.1f}s</td></tr>"
        )
    return "<div class='scroll'><table>" + head + "".join(body) + "</table></div>"


def build():
    loaded = {label: load(label) for label in LABELS}
    keys = [{(r["case_id"], r["run"]) for r in records} for records, _ in loaded.values()]
    assert all(key == keys[0] for key in keys), "Episode sets differ"
    traces = {label: evidence(records) for label, (records, _) in loaded.items()}
    assert traces["routed-mai-baseline"]["task_policy_hashes"] == (
        traces["routed-gpt6-sol"]["task_policy_hashes"]
    ), "Sol did not receive the same specialist instructions"
    def contracts(records):
        return {
            mode: {
                json.dumps({
                    key: value for key, value in r["policy_contract"].items()
                    if key not in ("model", "model_override", "endpoint")
                }, sort_keys=True)
                for r in records if r["mode"] == mode
            }
            for mode in compare_runs.MODES
        }
    assert contracts(loaded["routed-mai-baseline"][0]) == contracts(
        loaded["routed-gpt6-sol"][0]
    ), "Baseline runtime controls or tools differ"
    assert traces[FINAL_MAI]["task_policy_hashes"] == (
        traces[FINAL_SOL]["task_policy_hashes"]
    ), "Sol final run did not receive the final specialist instructions"
    assert contracts(loaded[FINAL_MAI][0]) == contracts(
        loaded[FINAL_SOL][0]
    ), "Final runtime controls or tools differ between MAI and Sol"
    assert all(
        item["router_hashes"] == traces["routed-mai-baseline"]["router_hashes"]
        for item in traces.values()
    ), "Router changed during comparison"
    assert all(
        trace["router_backends"] == ["decision1"]
        and trace["router_deployments"] == traces["routed-mai-baseline"]["router_deployments"]
        for trace in traces.values()
    ), "Router deployment differs"
    assert traces["routed-gpt6-sol"]["actual_inference_models"] == ["gpt-6-sol"]
    assert traces[FINAL_SOL]["actual_inference_models"] == ["gpt-6-sol"]
    # The isolated deployment returns its deployment name; ARM confirms it is the same
    # MAI-Code-1.1-Flash 2026-09-15 model as the primary deployment.
    assert all(
        set(traces[label]["actual_inference_models"]) <= {"mai-code-flash-1-1", "abtest-mai-flash"}
        for label in MAI_LABELS
    ), "MAI model identity differs"
    results = {
        label: {
            "overall": metrics(records),
            "tasks": {
                mode: metrics([r for r in records if r["mode"] == mode])
                for mode in compare_runs.MODES
            },
            "evidence": traces[label],
        }
        for label, (records, _) in loaded.items()
    }
    results["acceptance"] = compare_runs.compare(
        loaded["routed-mai-baseline"][0], loaded["routed-mai-optimized"][0]
    )
    results["acceptance_round4"] = compare_runs.compare(
        loaded["routed-mai-optimized"][0], loaded["routed-mai-r4"][0]
    )
    results["acceptance_round5"] = compare_runs.compare(
        loaded["routed-mai-r4"][0], loaded["routed-mai-r5"][0]
    )
    results["acceptance_final_vs_baseline"] = compare_runs.compare(
        loaded["routed-mai-baseline"][0], loaded[FINAL_MAI][0]
    )
    return results


def render(results):
    base = results["routed-mai-baseline"]["overall"]
    sol = results["routed-gpt6-sol"]["overall"]
    sol4 = results[FINAL_SOL]["overall"]
    optimized = results[FINAL_MAI]["overall"]
    cards = [
        ("MAI baseline", f"{base['strict_pass']:.1%}", "Strict pass, routed specialists"),
        ("Optimized MAI", f"{optimized['strict_pass']:.1%}",
         f"After round 5 · {(optimized['strict_pass']-base['strict_pass'])*100:+.1f} pp vs baseline"),
        ("GPT-6 Sol", f"{sol4['strict_pass']:.1%}",
         f"Same final prompts · {sol['strict_pass']:.1%} on baseline prompts"),
        ("MAI tokens", f"{optimized['total_tokens']/1e6:.2f}M",
         f"vs {base['total_tokens']/1e6:.2f}M baseline · {sol4['total_tokens']/1e6:.2f}M Sol"),
    ]
    hero_cards = "".join(
        f"<div class='card'><span>{html.escape(title)}</span><b>{value}</b>"
        f"<small>{html.escape(note)}</small></div>" for title, value, note in cards
    )
    overall = table([(name, results[label]["overall"]) for label, name in LABELS.items()])
    task_tables = "".join(
        f"<article><h3>{mode.replace('_', ' ').title()}</h3>"
        + table([(name, results[label]["tasks"][mode]) for label, name in LABELS.items()])
        + "</article>"
        for mode in compare_runs.MODES
    )
    router_rows = "".join(
        f"<tr><td>{name}</td><td>{results[label]['overall']['routing_accuracy']:.1%}</td>"
        f"<td>{results[label]['overall']['total_routing_tokens']:,}</td>"
        f"<td>{results[label]['overall']['total_specialist_tokens']:,}</td>"
        f"<td>{results[label]['overall']['total_reasoning_tokens']:,}"
        f" ({results[label]['overall']['reasoning_share_of_output']:.0%} of output)</td>"
        f"<td>{results[label]['overall']['mean_routing_latency_s']:.2f}s</td></tr>"
        for label, name in LABELS.items()
    )
    overhead_rows = "".join(
        f"<tr><td>{name}</td><td>{results[label]['overall']['transport_overhead_tokens']:,}</td>"
        f"<td>{results[label]['overall']['transport_overhead_latency_s']:,.1f}s</td>"
        f"<td>{results[label]['overall']['all_attempt_tokens']:,}</td></tr>"
        for label, name in LABELS.items()
    )
    accepted = results["acceptance"]["accept"]
    state = "ACCEPTED" if accepted else "NOT ACCEPTED"
    state4 = "ACCEPTED" if results["acceptance_round4"]["accept"] else "NOT ACCEPTED"
    state5 = "ACCEPTED" if results["acceptance_round5"]["accept"] else "NOT ACCEPTED"
    evidence_html = html.escape(json.dumps(
        {label: results[label]["evidence"] for label in LABELS}, indent=2
    ))
    checks_html = html.escape(json.dumps({
        "round_3_vs_baseline": results["acceptance"],
        "round_4_vs_round_3": results["acceptance_round4"],
        "round_5_vs_round_4": results["acceptance_round5"],
        "final_vs_baseline": results["acceptance_final_vs_baseline"],
    }, indent=2))
    links = "".join(
        f"<a class='button' href='../runs/{label}/development/summary.json'>{name}: summary</a>"
        for label, name in LABELS.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Routed agent: MAI, GPT-6 Sol and specialist optimization</title>
<style>
:root{{--ink:#17243a;--muted:#607089;--line:#dce4ef;--accent:#5a42cf}}
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5fb;color:var(--ink);font:15px/1.6 system-ui,sans-serif}}
header{{background:linear-gradient(125deg,#142c53,#4a39a4 65%,#16879b);color:white;padding:48px max(5vw,24px)}}
header p{{max-width:900px;color:#e3e7ff}}h1{{font-size:clamp(28px,4vw,46px);line-height:1.15;margin:12px 0}}
main{{max-width:1280px;margin:auto;padding:28px 24px}}h2{{margin:30px 0 12px}}h3{{margin:0 0 12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin-top:28px}}
.card{{background:#ffffff18;border:1px solid #ffffff38;border-radius:16px;padding:20px}}
.card span,.card small{{display:block}}.card b{{display:block;font-size:38px;line-height:1.4}}
.card small{{color:#e0e6ff}}nav{{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 24px}}
a{{color:var(--accent)}}.button,nav a{{display:inline-block;background:white;border:1px solid var(--line);border-radius:8px;
padding:9px 14px;text-decoration:none;margin:4px 5px 4px 0}}
article,section.panel,details{{background:white;border:1px solid var(--line);border-radius:14px;padding:22px;margin:16px 0}}
.callout{{padding:18px 22px;background:#e8e4ff;border-left:5px solid var(--accent);border-radius:8px}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:var(--muted);font-size:12px}}
td:first-child{{white-space:normal;min-width:170px}}summary{{cursor:pointer;font-weight:650}}
pre{{overflow:auto;background:#f4f6fb;padding:16px;border-radius:8px;font-size:12px}}
.muted{{color:var(--muted)}}footer{{padding:25px 0;color:var(--muted)}}
</style></head><body>
<header><div>MAI AGENT BENCHMARK · MEASURED DEVELOPMENT RESULTS</div>
<h1>One router. Six specialists.<br>MAI versus GPT-6 Sol.</h1>
<p>Decision-1 selects the task; the selected model executes it. All accuracy scores come from
the same deterministic six-check grader, not an LLM judge.</p><div class="cards">{hero_cards}</div></header>
<main><nav><a href="#overall">Overall</a><a href="#tasks">By task</a><a href="#routing">Router & tokens</a>
<a href="#method">Methodology</a><a href="#evidence">Evidence</a></nav>
<div class="callout"><strong>Round 5 acceptance (vs round 4): {state5}.</strong>
Round 4 (vs round 3): {state4}. Round 3 (vs baseline): {state}. Each decision uses improvement beyond baseline run-to-run spread,
per-task regression checks, production-action checks, and zero unresolved runtime errors.</div>
<section id="overall" class="panel"><h2>Matched overall comparison</h2>
<p class="muted">90 development cases × 3 runs = 270 episodes per configuration.
Total tokens include Decision-1 plus the specialist. Latency is observed end-to-end episode
latency, including routing, tool execution, authentication and service retries.</p>{overall}</section>
<section id="tasks"><h2>Accuracy, tokens and latency by task</h2>{task_tables}</section>
<section id="routing" class="panel"><h2>Router accuracy and token attribution</h2><div class="scroll"><table>
<tr><th>Configuration</th><th>Router accuracy</th><th>Router tokens</th><th>Specialist tokens</th>
<th>Reasoning tokens</th><th>Mean routing latency</th></tr>{router_rows}</table></div></section>
<section class="panel"><h2>Transport rerun overhead</h2><p class="muted">
Only failed infrastructure attempts are rerun; incorrect model answers are never rerun.
The benchmark tables use retained episode attempts. This table adds the recorded discarded
transport attempts, separate from prompt search and candidate-generation traffic.</p>
<div class="scroll"><table><tr><th>Configuration</th><th>Extra transport tokens</th>
<th>Cumulative failed-attempt latency</th><th>Tokens including failed attempts</th></tr>
{overhead_rows}</table></div></section>
<section id="method" class="panel"><h2>What this comparison does—and does not—show</h2>
<p><strong>Controlled:</strong> identical development case/run keys, frozen Decision-1 router,
same baseline specialist prompts and tool boundaries for MAI and GPT-6 Sol, four-worker initial
baseline evaluations, max output 32,768 tokens, same tool-step limit and canonical grader.
Actual provider-returned model names and policy hashes are checked before this report builds.</p>
<p><strong>Optimization:</strong> task-specific prompt proposals from GPT-5.2, plus diagnostic
variants derived from optimizer-split failures, are scored with the canonical grader.
One diagnostic variant checks incident-window membership arithmetically through the existing
calculate tool; no statistical tool or grader is changed. Search uses separate MAI capacity
and an 8,192-token output limit for both its baseline and candidates. Selected prompts are
checked at the matched 32,768-token limit on all 45 development
episodes for that task. Changed specialists are rerun; unchanged specialists reuse their
frozen baseline measurements, identified in the evidence below. Development data is used for
selection, so optimized results are selection-set performance—not an unbiased final-test estimate.</p>
<p>Validity candidate development was prefetched on isolated MAI capacity at two workers,
overlapping baseline transport recovery. Candidate records retain their inference endpoint.
Other changed-task development checks use the primary deployment. This further limits
causal interpretation of latency differences; accuracy uses the same model version,
prompts, tools, output limit and canonical checks.</p>
<p><strong>Round 4:</strong> hypothesis-driven candidates from development failure analysis
(<code>experiments/round4.json</code>): numeric checks via <code>calculate</code> for the
0.001 traffic-split threshold, incident window, Holm and power inputs; batched tool calls for
portfolio; and per-specialist <code>low</code> reasoning effort. GPT-5.2 also proposed one
candidate where search failures existed. Candidates needed no fewer optimizer-split passes
than their search baseline (or, when that baseline already passed every search case, went
straight to development; this rule was added mid-round after attribution saturated), then were
checked on all 45 development episodes for that task.
A task change was kept for an accuracy gain, or for no accuracy loss with at least 10% fewer
tokens. Ship decision and attribution ran on the isolated MAI deployment at two workers; other
tasks on the primary deployment. Azure Resource Manager confirms both deployments serve
MAI-Code-1.1-Flash version 2026-09-15; the isolated one reports its deployment name
(<code>abtest-mai-flash</code>) as the returned model. Round-4 measurements reuse round-3 episodes for unchanged tasks.
Selecting on development twice compounds selection bias; the final split remains sealed.</p>
<p><strong>Round-4 portfolio override:</strong> batched portfolio tool calls cut portfolio tokens
40% for one extra failed episode (84.4% vs 86.7%). The pre-set round-4 rule rejected it; the user
explicitly accepted the trade. It is recorded as an override in the optimization manifest and
stays within the documented per-task limits (drop at most 5 points, at most 3 stable cases lost).</p>
<p><strong>Round 5:</strong> targeted fixes from a review of round-4 development failures
(<code>experiments/round5.json</code>): a scientific-notation rule for the 0.001 threshold in
planning, the proven incident date check for ship decision, and a list-valued sample-sufficiency
check for portfolio, all at default reasoning effort. GPT-6 Sol replaced GPT-5.2 as the prompt
proposer, seeing optimizer-split failures only. Same search gate and development acceptance as round 4.</p>
<p><strong>Sol on final prompts:</strong> to keep the setup identical, GPT-6 Sol was rerun with
the exact final specialist instructions and settings for every task whose prompt or
settings changed. Unchanged tasks reuse Sol's baseline-prompt episodes. Policy hashes and runtime
contracts are asserted equal between optimized MAI and this Sol configuration.</p>
<p><strong>Operational latency:</strong> MAI and Sol have different deployment capacities.
These are observed timings under the available service conditions, not capacity-normalized
claims about intrinsic model speed. Total tokens describe the final benchmark episodes;
SDK retries within an attempt are included in its latency. Discarded transport attempts
are accounted for separately above. Search/generation and pilot traffic are separate
experimental overhead. Unchanged-task retry overhead is reused with its baseline measurement.</p>
<p><strong>Final split:</strong> sealed and not used for model selection or this comparison.
No scenario labels or expected answers enter the router or specialist during execution.</p></section>
<section id="evidence"><h2>Reproducible evidence</h2>{links}
<a class="button" href="../runs/routed-mai-optimized/optimization.json">Optimization trace</a>
<a class="button" href="../runs/decision1-router-v1/validation/summary.json">Router validation</a>
<details><summary>Model identities, router hashes, policy hashes and reused measurements</summary>
<pre>{evidence_html}</pre></details>
<details><summary>Canonical acceptance checks and per-task changes</summary><pre>{checks_html}</pre></details>
</section><footer>Sources: recorded local episode trajectories and provider token usage.
No projected accuracy, invented measurements, or cross-dataset score substitutions.</footer>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + 24 * 3600
    while True:
        ready = all(
            (ROOT / "runs" / label / "development" / "summary.json").exists() for label in LABELS
        )
        if ready:
            try:
                results = build()
                break
            except AssertionError:
                if not args.wait or time.monotonic() >= deadline:
                    raise
        if not args.wait or time.monotonic() >= deadline:
            raise RuntimeError("Comparison runs are not all complete")
        time.sleep(30)
    out = ROOT / "reports" / "2026-11-04-routed-comparison.html"
    out.write_text(render(results), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(str(out))
    for label, name in LABELS.items():
        result = results[label]["overall"]
        print(f"{name}: {result['strict_pass']:.1%}; {result['total_tokens']:,} tokens; "
              f"median {result['median_latency_s']:.1f}s")


if __name__ == "__main__":
    main()
