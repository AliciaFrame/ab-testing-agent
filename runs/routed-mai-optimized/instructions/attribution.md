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
