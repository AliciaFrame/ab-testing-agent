# Audit, November 3, 2026

A full local review of the data, statistics, tools, grader, case builder and harness. Every
fix below is covered by [`tests/test_grader.py`](../../tests/test_grader.py) (1,037 tests,
all passing).

## Is the paper's data used correctly?

| Check | Result |
|---|---|
| Archive totals | ✅ 32,487 tests, 150,817 arms, 538,272,878 impressions and 8,182,674 clicks: exactly the totals in Matias et al. (2021). `build_data.py` now fails if they differ. |
| Files | ✅ The three fielded-package samples from OSF [jd64p](https://osf.io/jd64p/) (Exploratory 4,873, Holdout 4,871, Confirmatory 22,743 tests). The 78,232 undeployed packages are not used. |
| Outcome definition | ✅ The click-through rate is `clicks / impressions` per package. Packages that share a `clickability_test_id` form one test. |
| Columns not used as labels | ✅ `winner`, `significance` and `first_place` are never used as labels. The paper says `significance` was "NOT an estimate of statistical significance", and that editors sometimes picked a package that wasn't the best performer. The dashboard fields are shown to the agent as a trap. |
| Equal-split assumption | ✅ Upworthy assigned arms with Ruby's `RandomSample`, which gives an equal expected split. Outside the incident window, only 0.5% of tests fail the traffic-split check at p < 0.001. |
| Randomization incident | ⚠️ → ✅ **Fixed.** The [2024 author correction](https://doi.org/10.1038/s41597-024-03600-w) says to treat tests from June 25, 2013 through the end of January 10, 2014 as not randomized. Previously, incident tests that passed the traffic-split check were labelled VALID, and could be SHIP, NO_DIFFERENCE or a planning baseline. They are now INVALID under a house rule for the known incident, which is also a realistic trap. |
| `problem` column | ⚠️ The correction adds a `problem` column, but the OSF files (May 2021) were never updated, and the correction's supplementary file isn't retrievable programmatically. We rebuild it from the stated dates using the test's first package: **7,016** tests, against the paper's **7,004**. Other definitions we tried (last package, any package, ±4–8 hour time shifts, an end date of January 10) give 6,975–7,026, and none gives 7,004. The 12-test gap is documented and does not affect labels outside the boundary days. |
| Timestamps | ✅ `created_at` values exactly match a copy of the data stamped as UTC, so the window and incident times are UTC. |
| The `uscpf` OSF update | ℹ️ That is Robertson et al.'s robustness check for their own paper, not an archive update. Not used. |

## Bugs and design problems fixed

| # | Problem | Effect | Fix |
|---|---|---|---|
| 1 | Incident tests labelled VALID (above) | Wrong labels in validity, ship decision, portfolio and planning | Incident rule in `abstats.validity()`, the instructions and the reference agent |
| 2 | Output cap of 8,192 tokens, 60 tool steps and a 120-second request timeout set by our harness | 4 of 18 baseline "failures" were harness artifacts, not model errors. The deployment accepts up to 262,144 output tokens. | Defaults raised to 131,072 output tokens, 200 steps and 900 seconds |
| 3 | Malformed tool-call JSON, or an unknown tool, ended the episode | Runtime error instead of a chance to recover | The error is returned to the model as tool output |
| 4 | Grader counted failed tool calls | A failed `check_sample_ratio` could satisfy the workflow check, and a failed submit followed by a retry counted as two submissions | Required steps count only successful calls; forbidden steps count every attempt |
| 5 | A test could be a training case and also sit in a dev or final window | Leakage into the evaluation splits | Each test belongs to exactly one split (`test_no_test_is_shared_across_splits`) |
| 6 | New builder labelled early-look SHIP as "tempting" | Would have produced zero early-look SHIP cases (caught before any run) | The tempting label applies only to KEEP_TESTING |

## Statistics cross-checked against statsmodels 0.15

| Check | Disagreements |
|---|---|
| Two-proportion z-test p-values (2,000 random tables) | 0 (maximum absolute difference 0) |
| Holm correction (5,000 random p-value sets) | 0 |
| Every attribution label, recomputed with statsmodels only | 0 / 143 |
| Every non-invalid ship-decision label, recomputed with statsmodels tests and Holm | 0 / 134 |
| Fleiss sample size for 10% vs 15% (α 0.05, power 0.8) | 686 per arm (textbook value) |

## Case set (more cases, as requested)

| Split | Before | Now | Per mode | Source |
|---|---:|---:|---:|---|
| development | 48 | **90** | 15 | Holdout sample; dev days for windows |
| final (sealed) | 72 | **150** | 25 | Confirmatory sample; final days |
| optimizer | 48 | **90** | 15 | Exploratory + spare Confirmatory; optimizer days |
| validation | 48 | **90** | 15 | Same pool; validation days |
| train | 224 | **600** | 100 | Same pool; train days |

Every split includes incident traps (tests that pass the traffic-split check but are INVALID
because of the incident). Training uses spare Confirmatory tests, which are disjoint from
final, because the Exploratory sample alone has too few headline-and-image tests and running
SHIP tests.

## Known limitations

- **Story re-tests:** about 5% of dev and final headlines also appear in a training test,
  because Upworthy re-tested the same story under a new test ID. Labels depend on each test's
  own counts, so memorizing a headline doesn't give the answer.
- **Few planning exclusions in dev:** only 1 of 15 dev planning windows needs an exclusion
  (final has 11 of 25). Outside the incident, traffic-split failures are rare (about 0.5% of
  tests), and incident days have no valid look-back tests.
- **Simulated running tests:** early-look snapshots are random subsamples of each arm's final
  traffic. Real interim data would also show time trends.
- **Large minimum lift:** the 50% minimum relative lift is a house choice. At 20%, no
  completed test in the archive is large enough to ever count as NO_DIFFERENCE.
