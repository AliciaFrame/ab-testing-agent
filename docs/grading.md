# Grading

Every case is graded deterministically by [`agent/grader.py`](../agent/grader.py) against
labels computed by [`agent/abstats.py`](../agent/abstats.py). No LLM judge is involved, and no
label comes from Upworthy's own `winner`, `significance` or `first_place` columns. The
`significance` column, for example, wasn't a statistical test (Matias et al., 2021).

## Checks

Every mode is scored on the same six checks. A case **strictly passes** only if all six pass.

| Check | Validity / ship decision / early look / attribution | Portfolio | Planning |
|---|---|---|---|
| **submitted** | `submit_verdict` called exactly once, last, with no other submit tool | `submit_scorecard` likewise | `submit_plan` likewise |
| **answer** | Verdict matches the house rules | Count of every verdict matches | Impressions per arm within 2% |
| **detail** | `winning_arm` matches (null unless SHIP) | Set of (test, winning arm) pairs matches exactly | Baseline click-through rate within 2% |
| **workflow** | Traffic-split check on this test before any analysis; nothing analysed once the test is INVALID; leader compared with every other arm (ship decision, early look) or every single-factor pair compared (attribution); `power_calc` called with the house inputs before NO_DIFFERENCE or KEEP_TESTING (ship decision) | Traffic-split check run on every test in the window | Traffic-split check run on every look-back test, and `excluded_tests` equals the INVALID set (traffic split or incident) |
| **side_effects** | `ship_package` only if invited and the verdict is SHIP (with the right arm); `notify_editor` only if invited and the test is INVALID | No `ship_package` or `notify_editor` | No `ship_package` or `notify_editor` |
| **report** | Reported traffic-split p-value matches the tool's (5%, or below 1e-9 when the true value is below 1e-12) | Window start and end echoed correctly | Days exactly right |

## Reward for RFT

`grade()` returns a weighted partial score in [0, 1]: submitted 0.10, answer 0.35, detail 0.10,
workflow 0.20, side_effects 0.15, report 0.10. An uninvited production action (`ship_package`
or `notify_editor`) zeroes the reward. The headline benchmark number is the strict pass rate,
not the reward.

## House rules (the reference)

| Rule | Value |
|---|---|
| Traffic-split check | Chi-square goodness of fit against an equal split; INVALID if p < 0.001 |
| Known incident | INVALID if the test started from 2013-06-25T00:00 up to, but not including, 2014-01-11T00:00 (UTC), even if the split passes. Source: [author correction, 2024](https://doi.org/10.1038/s41597-024-03600-w) |
| Comparisons | Pooled two-sided two-proportion z-test, leader vs every other arm |
| Multiple comparisons | Holm step-down at family-wise alpha 0.05; SHIP only if every comparison rejects |
| Enough data | Fleiss sample size per arm for a 50% relative lift, alpha 0.05 / (arms - 1), power 0.8 |
| Running tests | Same comparisons at family-wise alpha 0.005; SHIP or KEEP_TESTING only |
| Headline vs image | All pairs differing in exactly one of headline or image; one Holm correction at 0.05 across them |
| Planning baseline | Pooled click-through rate of tests started in the 12 hours before the planned start that are not INVALID (traffic split or incident) |

The 50% minimum lift is deliberately large. Upworthy tests have a median click-through rate
around 1.2% and roughly 15,000 impressions in total, so at a 20% lift no completed test in the
archive would ever count as NO_DIFFERENCE.

## Validation

[`tests/test_grader.py`](../tests/test_grader.py) runs a scripted reference agent that sees only
the request and tool outputs, never the labels. It must strictly pass every development,
final, optimizer and validation case. Targeted negative tests confirm that the grader catches:

- skipped or failed traffic-split checks
- analysis of invalid tests
- uninvited shipping
- wrong power inputs
- planning without exclusions
- analysis of incident tests whose traffic split looks fine
- tests shared across splits, or single-test cases that reuse a test

Failed tool calls (output with an `error`) never satisfy a required step. Every attempt
counts against a forbidden step.
