# Baseline failure analysis: MAI-Code-1.1-Flash, November 2, 2026

> **Superseded** by [the November 3 audit](../2026-11-03-audit/README.md) and its rerun. That
> audit found that the 4 "output cap" failures below were our harness's 8,192-token limit,
> not the model's. Incident-window tests were also mislabelled, and the case set has since
> been rebuilt and enlarged. Its recorded episodes are not included in this repository.

Run: `runs/baseline/development` covers 48 cases × 3 runs = 144 episodes. Deployment
`mai-code-flash-1-1` (MAI-Code-1.1-Flash 2026-09-15) on a shared deployment, with baseline
[`instructions.md`](../../agent/instructions.md), at most 60 tool steps and 8,192 output tokens
per turn. Strict pass was **87.5%** (126/144).

## All 18 failures, by cause

Each failure was checked against its trace (the tool calls and arguments) and the reference
labels.

| Cause | Episodes | Evidence |
|---|---:|---|
| **Traffic-split threshold misread**: valid tests treated as failing | 6 | `ship_decision-002` (all 3 runs): p = 0.0038 called "below the 0.001 threshold", so the verdict is INVALID instead of KEEP_TESTING. `planning-001` (2 runs) excludes tests with p = 0.0013, 0.0023 and 0.0040. `planning-002` excludes one with p = 0.011, so per-arm impressions are 107,124 against the correct 95,082, and the duration is 3 days against 2. |
| **Threshold misread plus exclusions from outside the window** | 2 | `planning-000`, `planning-004`: first query a 24-hour window, then list excluded tests from outside the 12-hour look-back, along with tests at p = 0.0018–0.039. |
| **Output cap hit mid-scorecard** | 4 | `portfolio-000/002/003`: one turn spent all 8,192 output tokens (inputs were 11–27K, so this is not context length). Runtime errors count as failures. |
| **Scorecard miscount** | 3 | `portfolio-000`: `power_calc` alpha = 0.05 for every test instead of 0.05/(arms−1), so one test moves from KEEP_TESTING to NO_DIFFERENCE. `portfolio-004`: one INVALID test counted as KEEP_TESTING; the window has tests at p = 0.00068 and 0.00079. `portfolio-007`: the window's one SHIP test is missed and an INVALID is reported instead. |
| **Missing power check** | 1 | `ship_decision-006`: correct verdict, but no `power_calc` call with the house inputs. |
| **Headline vs image over-call** | 1 | `attribution-001`: IMAGE called BOTH. |
| **No submission** | 1 | `validity-000`: correct answer in the text, but `submit_verdict` never called. |

Rate-limited episodes from an earlier attempt (6 workers, no transport retries; 91 HTTP 429
errors on the shared deployment) were discarded. The baseline was rerun with 3 workers and
bounded transport retries.

## Why the first design was too easy

The first version had only the validity and ship-decision modes, and scored **98.9%**
(89/90). Once the rules are explicit and the tools do the statistics, MAI applies them very
reliably to a single test. The difficulty now comes from:

- long-horizon bookkeeping (portfolio and planning)
- a stricter threshold for running tests
- finding the right pairs (headline vs image)
- traps: p-values just above 0.001, the misleading legacy dashboard fields, and production
  actions requested only some of the time

## What hill climbing should target

1. **Threshold discipline (8 of 18 failures).** "Below 0.001" is not "below 0.05", and
   0.0038 is not below 0.001. This is a likely fix for an instruction rewrite or a skill.
2. **Per-test parameters in loops.** Recompute alpha = 0.05/(arms−1) for each test in a
   scorecard.
3. **Output economy in portfolio.** Process tests incrementally instead of reasoning over
   all of them in one turn. `list_tests` already returns arm counts, so `get_test` per test
   is unnecessary.
4. **Window discipline in planning.** Query exactly the 12-hour look-back, and report
   exclusions only from it.
5. Whatever the optimizer leaves becomes the RFT target. The reward is
   [`grader.grade`](../../agent/grader.py), and it must first be calibrated for within-case
   variation on the train split.
