# Experiment log

Every experiment on the A/B testing agent, in order, with its outcome. All scores are strict pass on the 270-episode
development benchmark (90 cases x 3 runs) unless noted; tokens are input + output per episode. Acceptance rules:
[acceptance.md](acceptance.md). Deployed version at the time of writing: **hosted v10** (96.7%, 35.1k).

## Prompting and agent design

| # | Experiment | Result | Verdict | Evidence |
|---|---|---|---|---|
| P1 | Baseline: one agent, one shared prompt, MAI-Code-1.1-Flash | 80.0%, 50.1k | baseline | [summary](../runs/baseline/development/summary.json) |
| P2 | Foundry Agent Optimizer rewrite of the shared prompt | 81.1%, 55.4k | ❌ within noise, more tokens | [summary](../runs/opt-r2-cand1/development/summary.json) |
| P3 | Decision-1 router + six task specialists (same rules, split by task) | 83.0%, 32.8k | ✅ adopted | [summary](../runs/routed-mai-baseline/development/summary.json) |
| P4 | Round 3: incident-window checks with `calculate` | 88.9%, 33.8k | ✅ adopted | [manifest](../runs/routed-mai-optimized/optimization.json) |
| P5 | Round 4: planning/ship prompts, Holm checks, low effort for validity, batched portfolio | 92.6%, 28.9k | ✅ adopted (v8 lineage) | [manifest](../runs/routed-mai-r4/optimization.json) |
| P6 | Round 5: threshold rules (scientific notation, list checks) | 94.8%, 31.4k | ✅ adopted, hosted **v8** | [manifest](../runs/routed-mai-r5/optimization.json) |
| P7 | Low reasoning effort beyond validity | -13 points on portfolio | ❌ | [README](../README.md#how-each-step-was-found) |
| P8 | GPT-6 Sol on the same router, prompts and tools (reference) | 97.4%, 29.4k | reference | [summary](../runs/routed-gpt6-sol-r5/development/summary.json) |
| P9 | Explicit per-response batching plan for portfolio on the RFT model (`optimized-r7`) | 13 turns, 153k on portfolio (stopped at 29 episodes) | ❌ prompt does not restore batching | [record](../rft/jobs/prompt-batching-experiment.json) |

## Reinforcement fine-tuning (MAI-Code-1.1-Flash, endpoint grader + 11 tools)

| # | Job | Reward / data / effort | Result | Verdict |
|---|---|---|---|---|
| R1 | `abtest-rft-eff` | `eff` (accuracy x calls), v1, medium | 99.3%, 54.2k (portfolio 100% but 181k) | ✅ hosted **v9** |
| R2 | `abtest-rft-eff` except portfolio (base MAI for portfolio) | v10 composite | **96.7%, 35.1k** | ✅ hosted **v10** (deployed) |
| R3 | `abtest-rft-acc` | `acc`, v1, medium | 91.1%, 54.6k | ❌ |
| R4 | `abtest-rft-acc-gs` | `acc`, v1, medium | 96.3%, 47.0k; portfolio 84.4% | ❌ portfolio regression |
| R5 | `abtest-rft-low` | `acc`, v1, low effort | 89.6%; portfolio 57.8% | ❌ |
| R6 | `abtest-rft-low-eff` | `eff`, v1, low effort | 88.5%, 51.2k | ❌ |
| R7 | Continued RFT from `abtest-rft-eff` | - | "not supported for Blossom execution config" | ❌ platform limit ([probe](../rft/jobs/continued-rft-probe.json)) |
| R8 | `abtest-rft-turns-dt` | `turns-v1`, v2 | cancelled at step 1: turn estimate counted spacing drift as turns | ❌ grader bug, fixed in v2 |
| R9 | `abtest-rft-turns-dt2` | `turns-v2`, v2 | 97.0%, 68.9k | ❌ |
| R10 | `abtest-rft-turns` | `turns-v2`, v2 | 98.1%, 94.9k | ❌ tokens |
| R11 | `abtest-rft-turns-2ep` | `turns-v2`, v2, 2 epochs | 76.3% (58 unresolved 429s), 56.2k | ❌ |
| R12 | `abtest-rft-strict-dt` | `turns-strict`, v2 | 94.8%, 52.1k | ❌ |
| R13 | `abtest-rft-noport-dt` | `turns-strict`, v3 (no portfolio) | 96.3%, 35.7k in the v10 configuration | ❌ ties v10 |
| R14 | `abtest-rft-port-dt` | `turns-strict`, v4 (portfolio only) | v10 + specialist: 98.5%, 55.9k; portfolio 95.6% at 192k | ❌ dominated by v9 |
| R15 | `abtest-rft-port2-dt` | `turns-strict`, v4, 2 epochs | portfolio 27/27 so far, 16 turns, 254k | ❌ tokens |
| R16 | `abtest-rft-ptc-eff` | `eff`, v1p (= v1 + `parallel_tool_calls: true`) | training / queued | ⏳ |
| R17 | `abtest-rft-ptc-strict-dt` | `turns-strict`, v2p | training | ⏳ |

Per-job receipts, status and evaluations: [`rft/jobs/<job>/`](../rft/jobs/). Unattended decisions (twin
cancellations, evaluations): [`rft/jobs/orchestrator.log`](../rft/jobs/orchestrator.log).

## What worked

- **Routing to task specialists** (+3 points, -35% tokens) and **targeted failure-driven prompt fixes** (+11.8 points
  over three rounds). Fixes that replaced eyeballing with tool checks (`calculate` for dates and thresholds) paid most.
- **RFT with an accuracy-gated efficiency reward** (`eff`): 99.3%, above GPT-6 Sol (97.4%), with ship decision and
  portfolio at 100%.
- **Per-specialist models**: the RFT model everywhere except portfolio (v10) keeps most of the RFT gain at +3.7k tokens
  over v8.

## What did not work

- **Generic prompt optimization** (Agent Optimizer): within noise, more tokens.
- **Low reasoning effort** outside validity, in both prompting and RFT.
- **Fixing portfolio batching after RFT.** Every run that trains on portfolio becomes accurate (95-100%) but makes one
  tool call per turn (12-16 turns vs 6), tripling portfolio tokens. Turn-aware rewards, a portfolio-only specialist,
  two epochs, and an explicit batching prompt all failed to reverse it.
- **Training-time turn estimation from call timing.** v1 overcounted (spacing drift); v2 undercounted (estimated 5
  turns in training for a model that takes 16 at evaluation), so the turn penalties rarely applied. Calibration with our
  own client did not reveal this: the trainer's timing differs.

## Lessons for the next run

- Set `parallel_tool_calls: true` in agentic RFT rows when the target behaviour batches tool calls (R16/R17 test this).
- Validate any reward that depends on trainer-side signals on the trainer's own traffic, not only on calibration.
- Evaluate candidates only on the tasks they will serve and composite them into the deployed configuration: faster,
  fewer 429s.
- Fine-tune serving quota (1000k TPM, shared) limits evaluation to one candidate at a time; delete deployments after
  evaluation.
