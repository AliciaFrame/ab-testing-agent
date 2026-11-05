Your task is an early look at a running test.
Load the exact test identifier, including its @snapshot suffix. Check validity first;
submit INVALID without outcome analysis when invalid.
For a valid running test, find the highest CTR arm (ties go to the earlier arm). Call
compare_arms between this leader and every other arm. Apply Holm step-down to all these
p-values with family-wise alpha 0.005, not 0.05. Sort ascending, compare p[i] with
0.005/(number_of_comparisons-i), and stop at the first non-rejection.
SHIP the leader only if every comparison rejects; otherwise KEEP_TESTING with
winning_arm=null. Never use NO_DIFFERENCE for a running test. No power check is needed.
Call ship_package exactly once only when shipping was requested and the verdict is SHIP.
Finish with submit_verdict, including the traffic-split p-value.


Mandatory incident-date verification before any verdict or outcome analysis:
Read the actual started timestamp returned by get_test. Convert its YYYY-MM-DD date to the eight-digit integer YYYYMMDD by removing the hyphens. Call calculate with min(DATE-20130625, 20140110-DATE), replacing BOTH DATE placeholders with that actual eight-digit date integer. Use only arithmetic in the expression, not comparison operators. If the returned minimum is zero or positive, this is an incident-window test: verdict INVALID, even when sample-ratio p >= 0.001. A negative result means outside the incident window, but the sample-ratio rule still applies. Never claim a test started outside the incident without this check. Include the observed start date and incident-check result in the submission summary. Do not compare arm outcomes or ship an INVALID test. Still perform check_sample_ratio and any explicitly requested INVALID notification before the final submission.
