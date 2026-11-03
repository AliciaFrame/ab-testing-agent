# A/B testing agent: hill climbing MAI on real headline experiments

An experimentation-analyst agent built on **MAI-Code-1.1-Flash**, evaluated on 32,487 real A/B
tests from the [Upworthy Research Archive](https://upworthy.natematias.com/) (Matias et al.,
*Scientific Data*, 2021). Every answer is graded deterministically against the archive's
recorded impressions and clicks, with no LLM judge. That makes the benchmark suitable for
hill climbing with Foundry Agent Optimizer and reinforcement fine-tuning (RFT), following the
same pattern as [ShoppingBench](https://github.com/AliciaFrame/shoppingbench-foundry).

## What the agent does

Editors and producers ask one agent six kinds of question:

| Mode | Request | What makes it hard |
|---|---|---|
| **Validity** | "Can we trust test X?" | Must run the sample-ratio (traffic-split) check and read its p-value against 0.001. The archive has a real randomisation fault from June 2013 to January 2014. |
| **Ship decision** | "Should we ship a headline?" | Leader vs every other arm, Holm correction, and a power check before saying "no difference". Must ignore the legacy dashboard `significance` column. |
| **Early look** | "It's still running, can we call it?" | Running tests need a stricter alpha (0.005) and can never be called "no difference". |
| **Headline vs image** | "Was it the headline or the image?" | Must find every pair that differs in exactly one factor, and apply one correction across them. |
| **Portfolio** | "Scorecard for tests started 09:00–15:00" | Classify 8–15 tests in one episode (50–100 tool calls); errors compound. |
| **Planning** | "How big and how long should our next test be?" | Baseline from the last 12 hours of tests, excluding INVALID ones (failed traffic split or the known incident), then sample size and duration. |

Every mode starts with the same validity rule. A test is INVALID if its traffic-split check
gives p < 0.001, **or** if it started during the randomization incident that the archive's
authors reported in their [2024 correction](https://doi.org/10.1038/s41597-024-03600-w)
(June 25, 2013 through January 10, 2014), even when its split looks fine.

## Tools

The tools only fetch data and compute statistics. The agent has to apply the house rules:
the incident window, the leader, which pairs to compare, the Holm correction, the power
inputs, the exclusions and the verdict. That is what the grader scores. A bad call returns
`{"error": ...}` to the model rather than ending the episode.

| Tool | Inputs | Returns / does | Used in |
|---|---|---|---|
| `list_tests` | `start`, `end` | Tests (2+ arms) started in the window, with per-arm impressions and clicks (at most 40) | Portfolio, planning |
| `get_test` | `test_id` (`<id>@25` = running test) | Start time, status, per arm: headline, excerpt, image ID, impressions, clicks, and the legacy dashboard fields (a trap) | Single-test modes |
| `check_sample_ratio` | `test_id` | Chi-square against an equal split: p-value and expected impressions per arm | Every mode |
| `compare_arms` | `test_id`, `arm_a`, `arm_b` | Pooled two-proportion z-test; uncorrected p-value | Ship decision, early look, attribution, portfolio |
| `power_calc` | `baseline_ctr`, `relative_lift`, `alpha`, `power` | Impressions needed per arm (Fleiss) | Ship decision, portfolio, planning |
| `calculate` | `expression` | Safe arithmetic (+ − × ÷ **, `sum`, `min`, `max`, `ceil`, `round`) | Any |
| `ship_package` ⚠️ | `test_id`, `arm_id` | Production action, recorded and never executed. Allowed only if requested and the verdict is SHIP. | Ship decision, early look |
| `notify_editor` ⚠️ | `test_id`, `message` | Production action, recorded and never executed. Allowed only if requested and the test is INVALID. | Validity |
| `submit_verdict` | `test_id`, `verdict`, `winning_arm`, `srm_p_value`, `summary` | Final answer for one test | Single-test modes |
| `submit_scorecard` | `start`, `end`, `counts`, `ship`, `summary` | Final answer for a window | Portfolio |
| `submit_plan` | `baseline_ctr`, `impressions_per_arm`, `days`, `excluded_tests`, `summary` | Final answer for a plan | Planning |

The full house rules are in [`agent/instructions.md`](agent/instructions.md), and the exact
graded checks are in [docs/grading.md](docs/grading.md). Required steps count only successful
tool calls; a forbidden action counts even if it errored.

## Baseline: MAI-Code-1.1-Flash off the shelf

90 development cases (15 per mode) × 3 runs = 270 episodes. Score = **strict pass**: all six
graded checks pass. Runtime errors count as failures.

| Mode | Strict pass | Mean reward | Median latency | Tokens / episode | Cost / 1k episodes |
|---|---:|---:|---:|---:|---:|
| Validity | 84.4% | 0.864 | 11s | 7,852 | $2.65 |
| Ship decision | 86.7% | 0.942 | 20s | 16,477 | $6.25 |
| Early look | 77.8% | 0.888 | 15s | 12,542 | $4.52 |
| Headline vs image | 88.9% | 0.943 | 22s | 14,742 | $6.59 |
| Portfolio | 80.0% | 0.933 | 139s | 202,928 | $64.30 |
| Planning | **62.2%** | 0.833 | 52s | 45,823 | $17.89 |
| **All** | **80.0%** | 0.901 | 21s | 50,061 | $17.03 |

Cost uses MAI-Code-1.1-Flash list prices of $0.20 input and $1.20 output per 1M tokens, with
input priced as uncached. 59 of 90 cases pass in all three runs, and 7 fail in all three.
Failure analysis: [reports/2026-11-03-baseline](reports/2026-11-03-baseline/README.md).

**Where the headroom is:**
- **The incident rule**, the biggest single cause (17 of 54 failures): MAI checks the traffic
  split but analyses incident-window tests as if they were valid.
- **Planning** (62%): skipping the traffic-split check on look-back tests, wrong exclusions,
  and answering in prose without calling `submit_plan`.
- **Portfolio** bookkeeping, and requested `notify_editor` actions not sent.

## Data and splits

| Split | Source of single-test cases | Window days | Cases | Use |
|---|---|---:|---:|---|
| development | Holdout sample | 10% | 90 | Configuration and checkpoint selection (3 runs each) |
| final | Confirmatory sample | 15% | 150 | Sealed; opened once after selection |
| optimizer | Exploratory + spare Confirmatory | 10% | 90 | Agent Optimizer search |
| validation | Exploratory + spare Confirmatory | 10% | 90 | RFT validation |
| train | Exploratory + spare Confirmatory | 55% | 600 | RFT training |

The archive authors published three random samples of tests: Exploratory, Holdout and
Confirmatory. Portfolio and planning cases cover time windows, so whole calendar days are
assigned to one split. **No test appears in more than one split**, and no single-test case
reuses a test. Labels come only from [`agent/abstats.py`](agent/abstats.py), never from
Upworthy's `winner`, `significance` or `first_place` columns. All labels were cross-checked
against statsmodels. The "running test" snapshots are deterministic random subsamples of each
arm's final traffic; the archive itself holds only final totals. See the
[audit](reports/2026-11-03-audit/README.md) for data checks and known limitations.

## Layout

```
agent/            hosted agent (scaffolded from the mai-agent-hillclimb skill)
  abstats.py      house statistical rules: the reference for labels and tools
  domain.py       tools over agent/data/arms.parquet
  grader.py       deterministic grader and RFT reward
  instructions.md baseline instructions; tools.json is generated from domain.py
scripts/          build_data.py, build_cases.py, run_eval.py, prepare_optimizer.py
data/cases/       generated cases per split
runs/             recorded episodes and summaries
tests/            reference agent passes 100%; negative grader tests
```

## Reproduce

```powershell
# 1. Data: download the three package CSVs from https://osf.io/jd64p/ (upworthy-archive-datasets)
#    to data\raw\upworthy-{exploratory,holdout,confirmatory}.csv
python scripts\build_data.py      # -> agent\data\arms.parquet
python scripts\build_cases.py     # -> data\cases\*.jsonl (seeded, deterministic)
python -m pytest tests -q         # reference agent must pass every case

# 2. Baseline on a MAI deployment (Entra auth)
$env:MAI_OPENAI_BASE_URL = "https://<resource>.openai.azure.com/openai/v1/"
$env:MAI_MODEL_DEPLOYMENT_NAME = "<mai-code-1.1-flash deployment>"
$env:MAI_MAX_RETRIES = "8"        # transport retries for 429s only
python scripts\run_eval.py --split development --runs 3 --label baseline --workers 3
python scripts\run_eval.py --split development --runs 3 --label baseline --rerun-transport-errors
```

## Current state and next steps

- ✅ Data checked against the paper, six-mode case set (1,110 cases), deterministic grader
  (1,037 passing tests, cross-checked against statsmodels) and the baseline.
- 🟡 **Agent Optimizer: prepared, waiting on quota.** `scripts/prepare_optimizer.py` freezes
  the splits (hash manifest in `data/prepared/manifest.json`) and writes per-case judge
  criteria for the 90 optimizer cases. Plans for the MAI deployment, hosted agent and a
  5-candidate built-in `task_adherence` search have been validated as dry runs. Launch is
  blocked only by quota: the subscription has no free MAI-Code-1.1-Flash or GPT-5.4 capacity
  in the target region. Candidates will be re-scored with the canonical grader on development.
- ⏭️ **RFT** for whichever modes the optimizer leaves short, once the reward has been
  calibrated on the train split.
- ⏭️ Frontier reference (for example GPT-6 Sol) on the same development cases.

**Infrastructure note:** MAI-Code-1.1-Flash quota in our subscription is shared with other
projects, so the baseline ran on a shared deployment at 4 workers with transport retries.
Episodes that still hit rate limits were re-run with `--rerun-transport-errors`; model failures
are never re-run.

## Data licence and citation

The test data comes from the Upworthy Research Archive, released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `agent/data/arms.parquet` and
`data/cases/` are derived from it. Please cite:

> Matias, J. N., Munger, K., Le Quere, M. A., & Ebersole, C. (2021). The Upworthy Research
> Archive, a time series of 32,487 experiments in U.S. media. *Scientific Data*, 8, 195.
> https://doi.org/10.1038/s41597-021-00934-7 (author correction:
> https://doi.org/10.1038/s41597-024-03600-w)

