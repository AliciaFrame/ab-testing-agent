# Baseline: MAI-Code-1.1-Flash, November 3, 2026

Run: `runs/baseline/development` covers 90 cases (15 per mode) × 3 runs = 270 episodes.
Deployment `mai-code-flash-1-1` (MAI-Code-1.1-Flash 2026-09-15), with the audited case set and
grader ([audit](../2026-11-03-audit/README.md)). Limits were 131,072 output tokens, 200 tool
steps and a 900-second request timeout. 21 episodes hit rate limits on the shared deployment
and were re-run (`--rerun-transport-errors`). No model failure was re-run, and no runtime
errors remain.

**Strict pass: 80.0% (216/270).** 59 cases pass in all three runs; 7 fail in all three.

| Mode | Strict pass | Mean reward | Median latency | Tokens / episode | Largest output | Cost / 1k episodes |
|---|---:|---:|---:|---:|---:|---:|
| Validity | 84.4% | 0.864 | 11s | 7,852 | 2,258 | $2.65 |
| Ship decision | 86.7% | 0.942 | 20s | 16,477 | 6,979 | $6.25 |
| Early look | 77.8% | 0.888 | 15s | 12,542 | 4,345 | $4.52 |
| Headline vs image | 88.9% | 0.943 | 22s | 14,742 | 8,755 | $6.59 |
| Portfolio | 80.0% | 0.933 | 139s | 202,928 | 46,046 | $64.30 |
| Planning | **62.2%** | 0.833 | 52s | 45,823 | 23,865 | $17.89 |
| **All** | **80.0%** | 0.901 | 21s | 50,061 | 46,046 | $17.03 |

Cost uses MAI-Code-1.1-Flash list prices of $0.20 input and $1.20 output per 1M tokens, with
input priced as uncached. The largest output, 46,046 tokens in one portfolio episode, confirms
that the earlier 8,192-token cap was a harness artifact.

## The 54 failures, by cause

Each failure was classified once, from its trace against the reference labels.

| Cause | Episodes | Modes |
|---|---:|---|
| **Missed the incident rule**: a test from June 25, 2013 to January 10, 2014 whose split looked fine was analysed as valid | 17 | Early look 8, ship decision 4, attribution 4, validity 1 |
| **Portfolio: wrong verdict counts** | 8 | Portfolio |
| **Answered in prose, never called the submit tool** | 7 | Planning 6, validity 1 |
| **Planning: skipped the traffic-split check** on some or all look-back tests | 6 | Planning |
| **Planning: wrong exclusions** (included invalid tests, or excluded valid ones) | 5 | Planning |
| **Requested `notify_editor` not sent** for an INVALID test | 5 | Validity |
| Right verdict, but a required comparison or power check was missing | 3 | Ship decision 2, early look 1 |
| Wrong verdict on a valid test | 2 | Early look (SHIP called KEEP_TESTING), attribution (IMAGE called HEADLINE) |
| Malformed submission | 1 | Portfolio |

Only 4% of episodes contained any failed tool call (16% in portfolio, mostly `calculate`
syntax errors the model then recovered from). Tool errors are not a meaningful cause of failure.

## What to hill-climb

1. **The incident rule (17).** The single biggest cause: MAI checks the traffic split but
   doesn't apply the date rule. This is a good fit for an instruction rewrite or a skill.
2. **Planning workflow (17 of 45 planning episodes):** check the split on every look-back
   test, exclude the right ones, and always finish with `submit_plan`.
3. **Portfolio bookkeeping (8):** apply the rules consistently across 8–15 tests.
4. **Requested production actions (5):** send `notify_editor` when it was asked for.
5. Whatever the optimizer leaves becomes the RFT target, with reward from
   [`grader.grade`](../../agent/grader.py).
