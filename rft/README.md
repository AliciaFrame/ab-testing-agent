# Agentic RFT for the routed A/B-testing specialists

Reinforcement fine-tuning of **MAI-Code-1.1-Flash** on the deployed agent's own task: the routed specialists with
their round-5 prompts (`agent/specialists/optimized-r5`) and the benchmark's archive tools. Rewards are the
benchmark's deterministic six-check grader (`agent/grader.py`), never an LLM judge.

| File | Purpose |
|---|---|
| `common.py` | Rows: the specialist's exact developer prompt (common constraints + task prompt) and the user request; the shared 11-tool list; grader reference data. |
| `grader.py` | Endpoint rewards `acc` (canonical weighted reward) and `eff` (accuracy x efficiency). |
| `tool_server.py` | Starlette service: `POST /tools/{name}`, `POST /grade/{acc,eff}`, `GET /health`, `/traces/{id}`, `/grades/{id}`, `/stats`. |
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

Common settings: `mai-code-1.1-flash` (MAI-Code-1.1-Flash-2026-09-15), developerTier, 2 epochs, batch 8, LR 1.0,
compute 1.0, eval every 5 steps on 10 samples. Checkpoints are evaluated on the 270-episode development benchmark
with the full routed agent before any adoption.
