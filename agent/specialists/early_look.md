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
