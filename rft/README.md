# Agentic RFT for the routed A/B-testing specialists

Reinforcement fine-tuning of **MAI-Code-1.1-Flash** on the deployed agent's own task: the routed specialists with
their round-5 prompts (`agent/specialists/optimized-r5`) and the benchmark's archive tools. Rewards are the
benchmark's deterministic six-check grader (`agent/grader.py`), never an LLM judge.

| File | Purpose |
|---|---|
| `common.py` | Rows: the specialist's exact developer prompt (common constraints + task prompt) and the user request; the shared 11-tool list; grader reference data. |
| `grader.py` | Endpoint rewards `acc` (canonical weighted reward), `eff` (accuracy x efficiency) and `turns` (accuracy x turn efficiency). |
| `tool_server.py` | Starlette service: `POST /tools/{name}`, `POST /grade/{acc,eff,turns}`, `GET /health`, `/traces/{id}`, `/grades/{id}`, `/stats`. |
| `calibrate.py` | Replays the base model through the hosted service; reward distribution and hosted/local parity. |
| `loadtest.py` | Parallel replay of recorded episodes: every tool output and grade compared with local execution. |
| `build_data.py` | Hard-filtered train rows and validation rows from calibration, with a manifest. |
| `submit.py` | Preflight, upload, one job per configuration, receipts and status. |
| `Dockerfile`, `requirements.txt` | Container for Azure Container Apps (library versions pinned to the evaluation environment). |

## Why RFT, and what it targets

After five rounds of prompt optimization the deployed agent scores 94.8% strict pass versus 97.4% for GPT-6 Sol on the
same prompts. The remaining gaps are ship decision (88.9% vs 100%), planning (95.6% vs 100%) and portfolio (84.4%,
tied with Sol), and MAI uses more tokens: about 85% of its output is reasoning (Sol about 12%). Prompt-level low
reasoning effort cut tokens but cost accuracy, so the jobs test whether training can keep accuracy at lower cost.

## Data (`data/`)

Rows come only from the **train** split (600 cases, 100 per task) and the **validation** split; development,
optimizer and final cases are never used, and splits share no test (whole calendar days are assigned to one split).
Train rows are the planning, ship-decision and portfolio items the base model failed in calibration plus a seeded 30%
of their clean passes, and the calibrated validity, early-look and attribution items for retention.

Only `messages` (developer prompt + request) and `tools` are model-visible; `case` is grader reference data.
RFT tools are defined per job, so every row carries the same 11 tools. At serving time each specialist still gets its
own subset.

## Rewards (`grader.py`)

`acc` is the canonical weighted reward unchanged: checks submitted / answer / detail / workflow / side effects /
report weigh 0.10 / 0.35 / 0.10 / 0.20 / 0.15 / 0.10, a production-action violation scores 0, and only a strict pass
reaches 1.0.

`eff` multiplies `acc` by an efficiency factor that only pays for correct work. A non-strict rollout gets the floor
0.7, so it can never outscore a strict one. For strict passes,
`factor = max(0.7, 1 - 0.20 x clamp(output_tokens / median - 0.5, 0, 1) - 0.10 x clamp(tool_calls / median - 1, 0, 1))`,
with per-task medians from the deployed agent's development episodes. Full credit needs at most half of today's
output tokens. Tool calls are penalized only above today's median, so required checks are never discouraged.

**Observed trainer contract (Nov 4):** the trainer's grader samples carry `choices`, `output_text`, `output_tools`
(chat-completions function calls), `output_json` and `output_reasoning_summary` (both empty). They carry no token
usage and no reasoning text, so the token term never applies in training. In practice `eff` rewards correct answers
with fewer tool calls. Token savings rely mainly on the low-reasoning-effort jobs. Calibration samples do carry usage,
so calibration `eff` scores include the token term and overstate how often training rollouts fall below 0.9.

Calls come from the server's record for the rollout's `trace_id` when complete; otherwise the sample's `output_tools`
are re-executed. The archive tools are deterministic and stateless, so both give the same outputs, and no per-rollout
scenario binding is needed. Tool problems return HTTP 200 with the agent's `{"error": ...}` text; only
infrastructure failures return 5xx, which makes the trainer retry and then drop the rollout.

### Turn-aware reward (`turns`)

`eff` produced the most accurate model (99.3%) but it unlearned batching: on 29% of portfolio episodes it makes one
`compare_arms` call per turn (13 turns vs 6), so it re-reads a growing context each time and portfolio input tokens
nearly tripled (181k per episode). `eff` could not see this because the trainer sends no usage or turn boundaries.
Continued fine-tuning of `abtest-rft-eff` is not supported ("Continuous fine-tuning is not currently supported for
Blossom execution config", [probe](jobs/continued-rft-probe.json)), so `turns` retrains from base.

`turns` estimates tool-calling turns from the server's call arrival times. The trainer executes a turn's calls back to
back (2-3.5 s apart in training, drifting up within long batches), and a new turn adds the model's generation time
(about 4 s per one-call turn on the deployed RFT model), so a gap above max(1.75 x spacing, spacing + 1.5 s) starts a
new turn (`turns-v2`). `turns-v1` used spacing + 0.6 s: exact on calibration rollouts, but in training it counted
spacing drift as turns (base-like portfolio rollouts at step 0-1: median 18 estimated turns instead of about 5), so
`abtest-rft-turns-dt` was cancelled after two steps and resubmitted as `abtest-rft-turns-dt2`. On the same step 0-1
traces v2 gives portfolio 4 (p90 6), ship decision 5, planning 6, attribution 4: at or below the true counts, so
batched rollouts are not penalized. The in-batch spacing is the smaller of the rollout's own 25th-percentile gap and
the pooled 25th percentile over the last 15 minutes, so a rollout that never batches cannot hide one-call turns. For
strict passes, `factor = max(0.6, 1 - 0.30 x clamp(turns / (1.5 x median) - 1, 0, 1) - p_calls)` with the r5 agent's
median tool turns (portfolio 5, planning 9, ship decision 5, early look 4, attribution 4, validity 3); non-strict
rollouts get the 0.6 floor, so a strict answer always outscores a wrong one. Calibration on the 15 new portfolio rows
([`train-portfolio-v2.jsonl`](calibration/train-portfolio-v2.jsonl)): the estimate equals the true turn count on
15/15 base rollouts; 5-7 turn passes score 0.97-1.0, a 19-turn pass scores 0.7, the one failure 0.33.

## Service

Container App `abtest-rft-tools` in environment `azce34tnx7umq74fw` (Sweden Central), image in `azcr34tnx7umq74fw`,
2 vCPU / 4 GiB, one always-on replica so the trace store is shared. Bearer token: Container Apps secret, copy in Key
Vault `kv-att-evalmai-75343/abtest-rft-endpoint-token`. Load test (`calibration/loadtest.json`): 60 recorded episodes
(1,114 tool calls) at 32 parallel clients, 0 failures, 0 grade mismatches. Some p-values differ from Windows in the
last floating-point digit (Linux libm); no score changes because the grader's tolerances are 2-5%.

## Calibration and jobs

`calibration/` holds base-model rollouts at medium and low reasoning effort through the hosted service. `jobs/<name>/`
holds the redacted payload, file and job IDs, and status. Four jobs train the base model on the same rows:

| Job | Reward | Training reasoning effort |
|---|---|---|
| `abtest-rft-acc` | `acc` | medium |
| `abtest-rft-eff` | `eff` | medium |
| `abtest-rft-low` | `acc` | low |
| `abtest-rft-low-eff` | `eff` | low |
| `abtest-rft-turns` | `turns`, data v2 | medium, GlobalStandard, 1 epoch |
| `abtest-rft-turns-2ep` | `turns`, data v2 | medium, GlobalStandard, 2 epochs |
| `abtest-rft-turns-dt` | `turns-v1`, data v2 | cancelled at step 1 (v1 turn estimate overcounted in training) |
| `abtest-rft-turns-dt2` | `turns`, data v2 | medium, developerTier, 1 epoch (same as `abtest-rft-turns`; whichever gets GPUs first) |
| `abtest-rft-strict-dt` / `-gs` | `turns-strict`, data v2 | medium, 1 epoch, developerTier / GlobalStandard |
| `abtest-rft-noport-dt` / `-gs` | `turns-strict`, data v3 (no portfolio) | medium, 1 epoch, developerTier / GlobalStandard |

`turns` detected unbatching in training but did not reverse it: estimated portfolio turns rose from 4 to 11-13 and partly recovered to 8, and `abtest-rft-turns-dt2` still uses about 12 portfolio turns. `turns-strict` penalizes from 1.2x the base median turns with weight 0.6 and floor 0.4 (non-strict answers stay below: `acc` <= 0.9 x 0.4). On recorded calibration rollouts it scores batched base portfolio answers 0.88-1.0 and one-call-per-turn answers 0.4-0.5. Data v3 drops portfolio rows: the deployed agent keeps portfolio on the base model.

Data v2 (`data/v2/`) is v1 plus the next 15 seeded portfolio items (30 portfolio rows, 105 total).

Common settings: `mai-code-1.1-flash` (MAI-Code-1.1-Flash-2026-09-15), developerTier, 2 epochs, batch 8, LR 1.0,
compute 1.0, eval every 5 steps on 10 samples. Checkpoints are evaluated on the 270-episode development benchmark
with the full routed agent before any adoption.

## Results (development, 270 episodes; composites = v10 with the candidate's tasks swapped in)

| Model | Configuration | Strict pass | Tokens / episode | Accepted vs v10 |
|---|---|---:|---:|:-:|
| `abtest-rft-eff` | every task (v9) | 99.3% | 54.2k | - |
| `abtest-rft-eff` | every task except portfolio (v10, deployed) | 96.7% | 35.1k | - |
| `abtest-rft-port-dt` | portfolio specialist in v10 | 98.5% | 55.9k | yes, but dominated by v9 |
| `abtest-rft-noport-dt` | every task except portfolio | 96.3% | 35.7k | no |
| `abtest-rft-turns` | every task | 98.1% | 94.9k | no |
| `abtest-rft-turns-dt2` | every task | 97.0% | 68.9k | no |
| `abtest-rft-strict-dt` | every task | 94.8% | 52.1k | no |
| `abtest-rft-turns-2ep` | every task | 76.3% (58 unresolved 429s) | 56.2k | no |
| `abtest-rft-port2-dt` | portfolio specialist, 2 epochs | portfolio 27/27 so far (evaluation retrying 429s) | portfolio 254k | no (tokens) |

**What we learned.** Every RFT run that trains on portfolio makes it accurate (95-100%) and unbatches it (12-13 model
turns vs 6, 180-190k tokens vs 67k), whatever the reward: the turn penalties (`turns`, `turns-strict`) detected the
extra turns in training but did not reverse them, and neither did an explicit per-response batching prompt
([experiment](jobs/prompt-batching-experiment.json)). The portfolio specialist (`port-dt`, 60 portfolio rows, strict
turn penalty) scores 95.6% at 192k tokens: no better than `abtest-rft-eff` on portfolio. Without portfolio rows
(`noport-dt`), RFT matches the `eff` model on the other tasks (96.3% vs 96.7% in the v10 configuration). So the
choice is accuracy (v9: 99.3% at 54.2k) or tokens (v10: 96.7% at 35.1k).

**Why the turn penalties did not bite.** The training-time turn estimate undercounts one-call turns under the
trainer's timing: for `abtest-rft-port2-dt` the tool server estimated a median of 5 portfolio turns in training,
while the same model takes 16 model turns at evaluation. The trainer's short one-call turns arrive at nearly the
same spacing as calls within a batch, so `turns-v2`/`turns-strict` rarely applied a penalty and those runs were
effectively accuracy training. (`turns-v1` had the opposite problem: it counted spacing drift as turns.) The
estimator was exact on calibration rollouts, which use our own client's timing, not the trainer's. The tool server
now records the trainer's job ID per rollout (`x-rft-ftjobid`), so training behaviour can be compared per job.

**Open hypothesis: `parallel_tool_calls`.** The rows never set `parallel_tool_calls`, so the trainer may sample one
tool call per response. `abtest-rft-ptc-eff` repeats `abtest-rft-eff` with only `parallel_tool_calls: true` added
(data `v1p`); `abtest-rft-ptc-strict` uses `v2p` with `turns-strict`. Batching is judged at evaluation, not from
training-time turn estimates.
