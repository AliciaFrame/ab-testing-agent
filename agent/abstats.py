"""House statistical rules for headline A/B tests. Shared by tools, grader and case builder."""

import hashlib
import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

SRM_ALPHA = 0.001
FAMILY_ALPHA = 0.05
INTERIM_ALPHA = 0.005
MIN_RELATIVE_LIFT = 0.50
POWER = 0.80
SNAPSHOT_PERCENTS = (10, 25, 50)
PLANNING_LOOKBACK_HOURS = 12
# Matias et al., Sci Data 2021, author correction 2024 (doi:10.1038/s41597-024-03600-w):
# randomization is unreliable for tests from June 25, 2013 through the end of January 10, 2014.
INCIDENT_START = pd.Timestamp("2013-06-25")
INCIDENT_END = pd.Timestamp("2014-01-11")  # exclusive


def in_incident(started):
    started = pd.Timestamp(started)
    return INCIDENT_START <= started < INCIDENT_END


def sample_ratio(impressions):
    """Chi-square goodness of fit against an equal split."""
    total = sum(impressions)
    expected = total / len(impressions)
    chi2, p = stats.chisquare(impressions)
    return {
        "chi2": float(chi2),
        "df": len(impressions) - 1,
        "p_value": float(p),
        "expected_per_arm": expected,
    }


def validity(arms, started):
    """Shared first step of every rule: the traffic-split check plus the known incident."""
    srm = sample_ratio([a["impressions"] for a in arms])["p_value"]
    incident = in_incident(started)
    return {
        "srm_p_value": srm,
        "srm_failed": srm < SRM_ALPHA,
        "incident": incident,
        "valid": srm >= SRM_ALPHA and not incident,
    }


def two_proportion(clicks_a, imps_a, clicks_b, imps_b):
    """Pooled two-sided z-test for a difference in click-through rate."""
    ctr_a, ctr_b = clicks_a / imps_a, clicks_b / imps_b
    pooled = (clicks_a + clicks_b) / (imps_a + imps_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / imps_a + 1 / imps_b))
    z = (ctr_a - ctr_b) / se if se > 0 else 0.0
    return {
        "ctr_a": ctr_a,
        "ctr_b": ctr_b,
        "absolute_difference": ctr_a - ctr_b,
        "relative_lift": (ctr_a - ctr_b) / ctr_b if ctr_b > 0 else None,
        "z": z,
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


def holm_all_rejected(p_values, alpha=FAMILY_ALPHA):
    """True when Holm's step-down procedure rejects every hypothesis."""
    m = len(p_values)
    return all(p <= alpha / (m - i) for i, p in enumerate(sorted(p_values)))


def required_per_arm(baseline_ctr, relative_lift, alpha, power):
    """Impressions per arm for a two-sided two-proportion test (Fleiss, no continuity term)."""
    p0 = baseline_ctr
    p1 = baseline_ctr * (1 + relative_lift)
    p_bar = (p0 + p1) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    numerator = z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(
        p0 * (1 - p0) + p1 * (1 - p1)
    )
    return math.ceil(numerator**2 / (p1 - p0) ** 2)


def leader_comparisons(arms):
    """Leader (highest CTR, ties to the earliest arm) and its uncorrected p-values."""
    leader = max(arms, key=lambda a: a["clicks"] / a["impressions"])
    others = [a for a in arms if a is not leader]
    comparisons = {
        o["arm_id"]: two_proportion(
            leader["clicks"], leader["impressions"], o["clicks"], o["impressions"]
        )["p_value"]
        for o in others
    }
    return leader, others, comparisons


def decide(arms, started):
    """Reference ship decision (Mode 2) for a completed test."""
    check = validity(arms, started)
    result = {**check, "arms": len(arms)}
    if not check["valid"]:
        return {**result, "verdict": "INVALID", "winning_arm": None}
    leader, others, comparisons = leader_comparisons(arms)
    result.update(leader=leader["arm_id"], comparison_p_values=comparisons)
    if holm_all_rejected(list(comparisons.values())):
        return {**result, "verdict": "SHIP", "winning_arm": leader["arm_id"]}
    pooled = sum(a["clicks"] for a in arms) / sum(a["impressions"] for a in arms)
    alpha = FAMILY_ALPHA / len(others)
    needed = required_per_arm(pooled, MIN_RELATIVE_LIFT, alpha, POWER)
    smallest = min(a["impressions"] for a in arms)
    result.update(
        pooled_ctr=pooled, power_alpha=alpha, required_per_arm=needed, smallest_arm=smallest
    )
    verdict = "NO_DIFFERENCE" if smallest >= needed else "KEEP_TESTING"
    return {**result, "verdict": verdict, "winning_arm": None}


def decide_interim(arms, started):
    """Reference early-look decision (Mode 3): stricter alpha, never NO_DIFFERENCE."""
    check = validity(arms, started)
    result = {**check, "arms": len(arms)}
    if not check["valid"]:
        return {**result, "verdict": "INVALID", "winning_arm": None}
    leader, _others, comparisons = leader_comparisons(arms)
    p_values = list(comparisons.values())
    result.update(
        leader=leader["arm_id"],
        comparison_p_values=comparisons,
        tempting=holm_all_rejected(p_values, FAMILY_ALPHA),
    )
    if holm_all_rejected(p_values, INTERIM_ALPHA):
        return {**result, "verdict": "SHIP", "winning_arm": leader["arm_id"]}
    return {**result, "verdict": "KEEP_TESTING", "winning_arm": None}


def single_factor_pairs(arms):
    """Pairs differing only in headline (same image) or only in image (same headline)."""
    pairs = {"headline": [], "image": []}
    for a, b in itertools.combinations(arms, 2):
        same_headline = a["headline"] == b["headline"]
        same_image = a["image_id"] == b["image_id"]
        if same_image and not same_headline:
            pairs["headline"].append((a["arm_id"], b["arm_id"]))
        elif same_headline and not same_image:
            pairs["image"].append((a["arm_id"], b["arm_id"]))
    return pairs


def attribute(arms, started):
    """Reference headline-vs-image attribution (Mode 4)."""
    check = validity(arms, started)
    result = {**check, "arms": len(arms)}
    if not check["valid"]:
        return {**result, "verdict": "INVALID"}
    by_id = {a["arm_id"]: a for a in arms}
    pairs = single_factor_pairs(arms)
    tests = []
    for factor, factor_pairs in pairs.items():
        for x, y in factor_pairs:
            a, b = by_id[x], by_id[y]
            p = two_proportion(a["clicks"], a["impressions"], b["clicks"], b["impressions"])
            tests.append((factor, (x, y), p["p_value"]))
    order = sorted(range(len(tests)), key=lambda i: tests[i][2])
    rejected = set()
    for rank, index in enumerate(order):
        if tests[index][2] > FAMILY_ALPHA / (len(tests) - rank):
            break
        rejected.add(index)
    significant = {tests[i][0] for i in rejected}
    verdict = {
        frozenset(): "NEITHER",
        frozenset({"headline"}): "HEADLINE",
        frozenset({"image"}): "IMAGE",
        frozenset({"headline", "image"}): "BOTH",
    }[frozenset(significant)]
    return {**result, "verdict": verdict, "pairs": pairs, "pair_count": len(tests)}


def snapshot(arms, test_id, percent):
    """Deterministic interim snapshot: a random subsample of each arm's final traffic."""
    seed = int(hashlib.sha256(f"{test_id}@{percent}".encode()).hexdigest()[:16], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    for arm in arms:
        shown = int(arm["impressions"] * percent // 100)
        clicks = int(
            rng.hypergeometric(arm["clicks"], arm["impressions"] - arm["clicks"], shown)
        ) if shown else 0
        result.append({**arm, "impressions": shown, "clicks": clicks})
    return result


def plan(window_tests, arms_planned, relative_lift, daily_impressions):
    """Reference next-test plan (Mode 6) from tests in the look-back window.

    window_tests: list of {test_id, started, arms: [{impressions, clicks}]}.
    Returns None when no look-back test is valid.
    """
    valid, excluded = [], []
    for test in window_tests:
        (valid if validity(test["arms"], test["started"])["valid"] else excluded).append(test)
    if not valid:
        return None
    clicks = sum(a["clicks"] for t in valid for a in t["arms"])
    impressions = sum(a["impressions"] for t in valid for a in t["arms"])
    baseline = clicks / impressions
    alpha = FAMILY_ALPHA / (arms_planned - 1)
    needed = required_per_arm(baseline, relative_lift, alpha, POWER)
    return {
        "valid_tests": [t["test_id"] for t in valid],
        "excluded_tests": [t["test_id"] for t in excluded],
        "baseline_ctr": baseline,
        "alpha": alpha,
        "impressions_per_arm": needed,
        "days": math.ceil(arms_planned * needed / daily_impressions),
    }
