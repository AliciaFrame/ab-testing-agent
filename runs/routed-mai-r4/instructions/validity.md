Your task is validity checking only.
Load the requested test with get_test and call check_sample_ratio. Apply both the traffic
split threshold and the incident-window rule. The verdict is VALID or INVALID.
Do not analyse arm outcomes. If the requester asked to alert the editor and the test is
INVALID, call notify_editor for this test.
Call submit_verdict once with the exact test_id, verdict, winning_arm=null,
srm_p_value from check_sample_ratio, and a short summary.


Mandatory incident-date verification before any verdict or outcome analysis:
Read the actual started timestamp returned by get_test. Convert its YYYY-MM-DD date to the eight-digit integer YYYYMMDD by removing the hyphens. Call calculate with min(DATE-20130625, 20140110-DATE), replacing BOTH DATE placeholders with that actual eight-digit date integer. Use only arithmetic in the expression, not comparison operators. If the returned minimum is zero or positive, this is an incident-window test: verdict INVALID, even when sample-ratio p >= 0.001. A negative result means outside the incident window, but the sample-ratio rule still applies. Never claim a test started outside the incident without this check. Include the observed start date and incident-check result in the submission summary. Do not compare arm outcomes or ship an INVALID test. Still perform check_sample_ratio and any explicitly requested INVALID notification before the final submission.
