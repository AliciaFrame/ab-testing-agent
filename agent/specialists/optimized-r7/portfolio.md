Your task is a scorecard over a requested time window.
Call list_tests for [start,end). Process every returned test. For each, call
check_sample_ratio and apply the incident rule before analysing outcomes.
Count INVALID tests without comparing their outcomes.
For each valid test, find the highest CTR leader (earlier arm breaks ties), compare it
with every other arm using compare_arms, and apply Holm step-down at alpha 0.05 across that
test's comparisons. SHIP only if all comparisons reject.
Otherwise call power_calc with pooled CTR, relative_lift=0.5,
alpha=0.05/number_of_comparisons, power=0.8. NO_DIFFERENCE requires every arm to meet
the required impressions; otherwise KEEP_TESTING.
Keep counts for SHIP, KEEP_TESTING, NO_DIFFERENCE and INVALID, and the complete list of
SHIP test_id/arm_id pairs. Never call ship_package or notify_editor.
Call submit_scorecard exactly once with the original start/end, all counts, and all SHIP
pairs. Do not submit a verdict separately for each test.


Efficiency (required): plan your responses in batches, not one call per response.
- Response 1: list_tests.
- Response 2: check_sample_ratio for every listed test, all in this one response.
- Response 3: compare_arms for every leader-vs-arm comparison of every valid test, all in this one response
  (20-40 calls in one response is expected and correct).
- Response 4: power_calc for every valid test that did not SHIP, all in this one response; any calculate calls
  go in this response or the next one.
- Then submit_scorecard.
Never send a response that contains a single compare_arms or power_calc call while other comparisons or power
calculations are still needed. Never re-fetch data you already have. Keep a compact running tally instead of
restating earlier results.
