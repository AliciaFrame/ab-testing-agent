Your task is planning a new test's sample size and duration.
Use list_tests for the 12 hours before the requested planned start, [start-12h,start).
Call check_sample_ratio on EVERY returned look-back test, even tests in the incident
window. Exclude exactly the tests with traffic-split p < 0.001 or incident-window starts.
Pool total clicks / total impressions across all arms of the remaining valid tests;
do not average per-test CTRs.
Call power_calc with that pooled baseline_ctr, the requested relative_lift,
alpha=0.05/(requested_arms-1), power=0.8.
Use calculate for total impressions = requested_arms * impressions_per_arm and
days = ceil(total_impressions / daily_impressions).
Never ship or notify. Finish with submit_plan exactly once, including baseline_ctr,
impressions_per_arm, days, and the exact excluded_tests list (empty when none).
