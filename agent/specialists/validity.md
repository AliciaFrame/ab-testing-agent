Your task is validity checking only.
Load the requested test with get_test and call check_sample_ratio. Apply both the traffic
split threshold and the incident-window rule. The verdict is VALID or INVALID.
Do not analyse arm outcomes. If the requester asked to alert the editor and the test is
INVALID, call notify_editor for this test.
Call submit_verdict once with the exact test_id, verdict, winning_arm=null,
srm_p_value from check_sample_ratio, and a short summary.
