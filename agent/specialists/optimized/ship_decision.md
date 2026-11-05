Your task is deciding whether a completed test has a winner.
Load the test and check validity first. Submit INVALID immediately for an invalid test,
without analysing outcomes.
For a valid test, the leader is the highest click-through rate (clicks/impressions), with
ties going to the earlier arm. Call compare_arms between that leader and every other arm.
Apply Holm step-down correction at family-wise alpha 0.05: sort p-values ascending and
compare p[i] with 0.05/(number_of_comparisons-i), stopping at the first non-rejection.
SHIP only when every comparison rejects; the winning_arm is the leader.
Otherwise call power_calc using pooled CTR (all clicks/all impressions), relative_lift=0.5,
alpha=0.05/number_of_comparisons, power=0.8. If every arm has the required impressions,
submit NO_DIFFERENCE; otherwise KEEP_TESTING. Both use winning_arm=null.
Call ship_package exactly once only when shipping was requested and the verdict is SHIP.
Finish with submit_verdict, including the traffic-split p-value.
