Your task is attributing differences to headline versus image.
Load the requested test and check validity. Submit INVALID without analysing outcomes
when invalid.
Otherwise enumerate every pair sharing the same image but different headlines, and every
pair sharing the same headline but different images. Call compare_arms for every such pair.
Apply one Holm step-down correction at alpha 0.05 across all those pairs together.
Sort p-values ascending, compare p[i] with 0.05/(number_of_pairs-i), and stop at the first
non-rejection. Significant same-image pairs establish a headline effect; significant
same-headline pairs establish an image effect.
The verdict is HEADLINE, IMAGE, BOTH or NEITHER. winning_arm is always null.
Never ship or notify. Finish with submit_verdict and the traffic-split p-value.


Mandatory validity gate before any compare_arms or power_calc call:
Read the actual started timestamp returned by get_test. Convert its YYYY-MM-DD date to the eight-digit integer YYYYMMDD by removing the hyphens. Call calculate with min(DATE-20130625, 20140110-DATE), replacing BOTH DATE placeholders with that actual eight-digit date integer. Use only arithmetic in the expression, not comparison operators. If the returned minimum is zero or positive, this is an incident-window test: verdict INVALID, even when sample-ratio p >= 0.001. A negative result means outside the incident window, but the sample-ratio rule still applies (INVALID when p < 0.001). Never claim a test started outside the incident without this check. For an INVALID test, never call compare_arms or power_calc; submit INVALID directly.

Numeric decision checks (use calculate; never compare decimals mentally):
With M = the number of compared pairs, sort all pair p-values ascending and call calculate on the list [p1-0.05/M, p2-0.05/(M-1), ..., pM-0.05/1]. Entries that are zero or negative before the first positive entry are rejected pairs; stop at the first positive entry. A rejected pair with the same image but different headlines is a headline effect; a rejected pair with the same headline but different images is an image effect. Both kinds rejected means BOTH; none rejected means NEITHER.