# Acceptance policy

Fixed before any optimized or fine-tuned candidate is evaluated.

## Data

| Split | Cases | Role |
|---|---:|---|
| optimizer | 90 (searches use a subset) | Agent Optimizer search only |
| development | 90 (15 per scenario) × 3 runs | Accept or reject candidates and checkpoints |
| final | 150 (25 per scenario) | Sealed; opened once, after selection |

Optimizer search sets can be weighted toward weak scenarios. Round 3 uses 40 cases: planning
10, early look 8, portfolio 6, validity 6, ship decision 5, headline vs image 5. Acceptance
always uses the balanced development split.

## Accept a candidate only if all of these hold

1. **Overall strict pass improves** on development (270 episodes) by more than the baseline's
   run-to-run spread, measured as the standard deviation of per-run pass rates across its
   3 runs.
2. **No scenario regresses.** For each of the six scenarios (45 episodes each), the
   candidate's strict pass rate may not fall more than 5 points below the baseline's, and
   no scenario may lose more than 3 cases that passed in all three baseline runs.
3. **No new production-action violations.** The side-effects check pass rate is at least the
   baseline's.
4. **Clean evaluation:** zero runtime errors after re-running transport failures (rate limits,
   timeouts) only. Model failures are never re-run.

Cost and latency are reported per scenario. A candidate that raises tokens per episode by
more than 50% without a matching quality gain is flagged for review.

## When to switch to per-scenario instructions or skills (option 2)

Try per-scenario instructions or skills if, after a weighted optimizer round, any of these
is true:

- planning or early look stays below 80% strict pass on development
- the gain comes from some scenarios while another regresses (rule 2 fails)
- average tokens per episode rise more than 50% over the baseline, or portfolio exceeds about
  250K tokens per episode

## Scores in use

- **Acceptance:** canonical deterministic grader ([`agent/grader.py`](../agent/grader.py)),
  strict pass over six checks, overall and per scenario.
- **Optimizer search:** GPT-5.2 judge `abtest-answer-and-process`, which sees the final answer,
  the tool calls the agent made and the reference. It is calibrated against
  canonical-grader labels before use. It nominates candidates; it never decides acceptance.
