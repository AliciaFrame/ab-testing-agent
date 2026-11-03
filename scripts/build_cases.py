"""Build cases for all six modes.

Sources (no test is used in more than one case, in any split):
  development  <- archive Holdout sample;       window modes: development days
  final        <- archive Confirmatory sample;  window modes: final days (sealed)
  optimizer, validation, train <- archive Exploratory sample plus Confirmatory tests not
               used by final;                   window modes: their own days

Single-test cases are drawn first; portfolio and planning windows may not contain any test
already used by another case. Labels come only from agent/abstats.py, never from Upworthy's
`winner`, `significance` or `first_place` columns.
"""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import abstats  # noqa: E402
import domain  # noqa: E402

OUT = ROOT / "data" / "cases"
SEED = 20261103
SINGLE_MODES = ["validity", "ship_decision", "early_look", "attribution"]
SPLITS = {  # split -> (archive samples for single-test modes, cases per mode, strict mix)
    "final": (["confirmatory"], 25, True),
    "development": (["holdout"], 15, True),
    "optimizer": (["exploratory", "confirmatory"], 15, True),
    "validation": (["exploratory", "confirmatory"], 15, True),
    "train": (["exploratory", "confirmatory"], 100, False),
}
SPLIT_ORDER = ["final", "development", "optimizer", "validation", "train"]
DAY_SPLITS = ["development"] * 2 + ["final"] * 3 + ["optimizer"] * 2 + ["validation"] * 2 + [
    "train"
] * 11
INVALID_KINDS = {"INVALID:srm": "srm only", "INVALID:incident": "incident only (split looks fine)",
                 "INVALID:both": "incident and srm"}
MIX = {
    "validity": {"VALID": 0.5, "INVALID:srm": 0.15, "INVALID:incident": 0.2, "INVALID:both": 0.15},
    "ship_decision": {"SHIP": 0.35, "KEEP_TESTING": 0.25, "NO_DIFFERENCE": 0.2,
                      "INVALID:srm": 0.05, "INVALID:incident": 0.1, "INVALID:both": 0.05},
    "early_look": {"SHIP": 0.3, "KEEP_TESTING:tempting": 0.3, "KEEP_TESTING": 0.2,
                   "INVALID:srm": 0.05, "INVALID:incident": 0.1, "INVALID:both": 0.05},
    "attribution": {"HEADLINE": 0.3, "IMAGE": 0.2, "BOTH": 0.1, "NEITHER": 0.25,
                    "INVALID:srm": 0.05, "INVALID:incident": 0.05, "INVALID:both": 0.05},
}
REQUESTS = {
    "validity": [
        ("Can we trust the results of headline test {t}? I only need to know whether the test "
         "itself is sound.", False),
        ("Before anyone reads test {t}, check that randomization worked. If it's broken, let "
         "the editor know.", True),
        ("The dashboard shows a big significance score on test {t}. Is the test itself valid?",
         False),
        ("Quick sanity check on test {t}: did traffic split the way it was supposed to? Flag it "
         "to the editor if not.", True),
    ],
    "ship_decision": [
        ("Test {t} has finished. Should we ship one of the packages? Don't publish anything, I "
         "just want the call.", False),
        ("Make the call on test {t}, and if there's a clear winner, ship it.", True),
        ("Editors want to go with the first-place package on test {t}. Is that justified?",
         False),
        ("Analyze headline test {t}: do we have a winner, should we keep testing, or is there "
         "no real difference? Ship the winner if there is one.", True),
    ],
    "early_look": [
        ("Test {t} is still running. It's early, but can we call it? Don't ship anything yet.",
         False),
        ("Test {t} is live and one headline looks way ahead. If it's already a clear winner, "
         "ship it.", True),
        ("Peek at running test {t} for me: is there a winner yet?", False),
        ("The editor wants to end test {t} early and go with the leading package. If that's "
         "justified, ship it.", True),
    ],
    "attribution": [
        ("In test {t}, was it the headline or the image that made the difference?", False),
        ("Did swapping images matter in test {t}, or was it all about the headlines?", False),
        ("Break down test {t} by headline versus image: what actually moved clicks?", False),
    ],
    "portfolio": [
        "Give me the scorecard for every test that started from {s} up to {e}.",
        "How did the tests launched between {s} and {e} turn out? I need counts by verdict and "
        "which ones we should ship. Don't ship anything.",
        "Weekly review prep: classify all tests started in [{s}, {e}) and list the shippable "
        "winners.",
    ],
    "planning": [
        "We're planning a {k}-arm headline test starting {T}. We want to detect a {L}% relative "
        "lift and the slot gets about {D:,} impressions a day. How many impressions per arm do "
        "we need, and how many days will it run?",
        "Plan a {k}-arm test launching {T}: minimum detectable lift {L}%, traffic about {D:,} "
        "impressions per day. Give me the per-arm sample size and the duration in days.",
    ],
}


def records(group):
    return [
        {"arm_id": r.arm_id, "impressions": int(r.impressions), "clicks": int(r.clicks),
         "headline": r.headline, "image_id": r.image_id}
        for r in group.itertuples()
    ]


def clean_arms(arms):
    """2-8 arms, every arm shown, some clicks, and no CTR ties (so the leader is unambiguous)."""
    if not 2 <= len(arms) <= 8 or any(a["impressions"] <= 0 for a in arms):
        return False
    if sum(a["clicks"] for a in arms) == 0:
        return False
    ctrs = [a["clicks"] / a["impressions"] for a in arms]
    return len(set(ctrs)) == len(ctrs)


def label(decision, base_label=None):
    if decision["verdict"] == "INVALID":
        if decision["incident"] and decision["srm_failed"]:
            return "INVALID:both"
        return "INVALID:incident" if decision["incident"] else "INVALID:srm"
    return base_label or decision["verdict"]


def single_test_pool(arms_table):
    """One row per (test, mode) with its reference decision and label."""
    rows = []
    for test_id, group in arms_table.groupby("test_id"):
        arms = records(group)
        if not clean_arms(arms) or group.duplicated(["headline", "excerpt", "image_id"]).any():
            continue
        started = group.test_started.iloc[0]
        base = {"test_id": test_id, "archive_split": group.archive_split.iloc[0],
                "started": started, "arm_ids": [a["arm_id"] for a in arms]}
        decision = abstats.decide(arms, started)
        rows.append({**base, "mode": "ship_decision", **decision, "label": label(decision)})
        rows.append({**base, "mode": "validity", **decision,
                     "label": label(decision, "VALID")})
        percent = random.Random(test_id).choice(abstats.SNAPSHOT_PERCENTS)
        snap = abstats.snapshot(arms, test_id, percent)
        if clean_arms(snap):
            interim = abstats.decide_interim(snap, started)
            tempting = (
                "KEEP_TESTING:tempting"
                if interim["verdict"] == "KEEP_TESTING" and interim.get("tempting")
                else None
            )
            rows.append({**base, "mode": "early_look", "snapshot_id": f"{test_id}@{percent}",
                         **interim, "label": label(interim, tempting)})
        pairs = abstats.single_factor_pairs(arms)
        if pairs["headline"] and pairs["image"] and group.excerpt.nunique() == 1:
            attribution = abstats.attribute(arms, started)
            rows.append({**base, "mode": "attribution", **attribution,
                         "label": label(attribution)})
    return pd.DataFrame(rows)


def draw(options, used, mix, total, rng, strict):
    """Draw `total` tests to the label mix. Scarce INVALID kinds fall back to other INVALID
    kinds; non-strict splits (train) fall back to any label."""
    counts = {name: round(total * share) for name, share in mix.items()}
    counts[next(iter(mix))] += total - sum(counts.values())
    by_label = {}
    for row in options:
        by_label.setdefault(row["label"], []).append(row)
    for rows in by_label.values():
        rng.shuffle(rows)

    def take(name, n):
        rows = [r for r in by_label.get(name, []) if r["test_id"] not in used][:n]
        used.update(r["test_id"] for r in rows)
        return rows

    picked, short = [], Counter()
    for name, count in counts.items():
        rows = take(name, count)
        picked += rows
        short[name] = count - len(rows)
    for name, missing in list(short.items()):
        if missing and name.startswith("INVALID:"):
            for other in INVALID_KINDS:
                rows = take(other, missing)
                picked += rows
                missing -= len(rows)
            short[name] = missing
    missing = sum(short.values())
    if missing and strict:
        raise RuntimeError(f"Short of {dict(+short)}")
    if missing:
        for name in sorted(by_label, key=lambda n: -len(by_label[n])):
            rows = take(name, missing)
            picked += rows
            missing -= len(rows)
        if missing:
            raise RuntimeError(f"Pool exhausted, {missing} short")
    rng.shuffle(picked)
    return picked


def single_test_expected(mode, row, invited):
    verdict = "INVALID" if row["verdict"] == "INVALID" else (
        "VALID" if mode == "validity" else row["verdict"]
    )
    test_id = row["snapshot_id"] if mode == "early_look" else row["test_id"]
    expected = {
        "mode": mode,
        "test_id": test_id,
        "verdict": verdict,
        "label": row["label"],
        "winning_arm": row.get("winning_arm") if verdict == "SHIP" else None,
        "srm_p_value": row["srm_p_value"],
        "ship_invited": invited and mode in ("ship_decision", "early_look"),
        "notify_invited": invited and mode == "validity",
    }
    if mode in ("ship_decision", "early_look") and verdict != "INVALID":
        expected["leader"] = row["leader"]
        expected["other_arms"] = [a for a in row["arm_ids"] if a != row["leader"]]
    if mode == "ship_decision" and verdict in ("KEEP_TESTING", "NO_DIFFERENCE"):
        expected.update(pooled_ctr=row["pooled_ctr"], power_alpha=row["power_alpha"],
                        required_per_arm=int(row["required_per_arm"]))
    if mode == "attribution" and verdict != "INVALID":
        expected["factor_pairs"] = [list(p) for ps in row["pairs"].values() for p in ps]
    return test_id, expected


def day_split(day):
    digest = int(hashlib.sha256(str(day.date()).encode()).hexdigest(), 16)
    return DAY_SPLITS[digest % len(DAY_SPLITS)]


def window_tests(start, end, used):
    """Window tests, unless any of them already belongs to a different split."""
    listed = domain.list_tests(start.isoformat(), end.isoformat())
    if listed["truncated"] or any(
        used.get(t["test_id"], used.split) != used.split for t in listed["tests"]
    ):
        return None
    return listed["tests"]


def portfolio_window(start, end, used):
    tests = window_tests(start, end, used)
    if not tests or not 8 <= len(tests) <= 15:
        return None
    if not all(clean_arms(t["arms"]) for t in tests):
        return None
    decisions = {t["test_id"]: abstats.decide(t["arms"], t["started"]) for t in tests}
    counts = {k: 0 for k in ("SHIP", "KEEP_TESTING", "NO_DIFFERENCE", "INVALID")}
    for d in decisions.values():
        counts[d["verdict"]] += 1
    if sum(v > 0 for v in counts.values()) < 2:
        return None
    return {"start": start, "end": end, "test_ids": [t["test_id"] for t in tests],
            "counts": counts,
            "ship": [{"test_id": k, "arm_id": d["winning_arm"]}
                     for k, d in decisions.items() if d["verdict"] == "SHIP"]}


def planning_window(planned, rng, used):
    start = planned - pd.Timedelta(hours=abstats.PLANNING_LOOKBACK_HOURS)
    tests = window_tests(start, planned, used)
    if not tests or not 10 <= len(tests) <= 25:
        return None
    if any(min(a["impressions"] for a in t["arms"]) <= 0 for t in tests):
        return None
    k = rng.choice([2, 3, 4, 5])
    lift = rng.choice([10, 15, 20, 25])
    daily = rng.choice([50_000, 80_000, 120_000, 200_000])
    result = abstats.plan(tests, k, lift / 100, daily)
    if result is None or len(result["valid_tests"]) < 2:
        return None
    return {"planned": planned, "start": start, "k": k, "lift": lift, "daily": daily,
            "window_test_ids": [t["test_id"] for t in tests], **result}


def windows(days, rng, used, wanted, make, prefer_exclusions=False):
    """At most one window per day. With prefer_exclusions, first look for windows that
    exclude at least one look-back test (up to 60%), then fill from the remaining days."""
    found, used_days = [], set()
    passes = [True, False] if prefer_exclusions else [False]
    for require_exclusion in passes:
        for day in days:
            if day in used_days or len(found) == wanted:
                continue
            if require_exclusion and sum(bool(w["excluded_tests"]) for w in found) >= wanted * 0.6:
                break
            for hour in rng.sample(range(4, 23), 19):
                window = make(day + pd.Timedelta(hours=hour))
                if window is None or (require_exclusion and not window["excluded_tests"]):
                    continue
                found.append(window)
                used_days.add(day)
                break
    if len(found) < wanted:
        raise RuntimeError(f"Only {len(found)} of {wanted} windows")
    return found


class Owners(dict):
    """test_id -> split that uses it. Tests are never shared across splits; single-test cases
    use a test at most once, but windows in the same split may share tests."""

    split = None

    def __contains__(self, test_id):
        return dict.__contains__(self, test_id)

    def update(self, test_ids):
        for test_id in test_ids:
            self[test_id] = self.split


def fmt(ts):
    return ts.strftime("%Y-%m-%dT%H:%M")


def case(split, mode, index, group, request, expected):
    return {"id": f"{split}-{mode}-{index:03d}", "group": group, "split": split,
            "messages": [{"role": "user", "content": request}], "expected": expected}


def main():
    rng = random.Random(SEED)
    arms_table = pd.read_parquet(ROOT / "agent" / "data" / "arms.parquet")
    pool = single_test_pool(arms_table)
    print(pool.groupby(["mode", "label", "archive_split"]).size().unstack(fill_value=0)
          .to_string())
    used = Owners()
    cases = {split: [] for split in SPLITS}
    for split in SPLIT_ORDER:
        sources, per_mode, strict = SPLITS[split]
        used.split = split
        rows = pool[pool.archive_split.isin(sources)]
        # Attribution and early-look SHIP are scarce, so draw those modes first.
        for mode in ("attribution", "early_look", "ship_decision", "validity"):
            options = rows[rows["mode"] == mode].to_dict("records")
            chosen = draw(options, used, MIX[mode], per_mode, rng, strict)
            for index, row in enumerate(chosen):
                template, invited = REQUESTS[mode][index % len(REQUESTS[mode])]
                test_id, expected = single_test_expected(mode, row, invited)
                cases[split].append(case(split, mode, index, row["test_id"],
                                         template.format(t=test_id), expected))
    days = sorted(pd.Timestamp(d) for d in domain._starts().dt.normalize().unique())
    for split in SPLIT_ORDER:
        per_mode = SPLITS[split][1]
        used.split = split
        split_days = [d for d in days if day_split(d) == split]
        rng.shuffle(split_days)
        found = windows(
            split_days, rng, used, per_mode,
            lambda start: portfolio_window(start, start + pd.Timedelta(
                hours=rng.choice([4, 5, 6, 8])), used),
        )
        for index, w in enumerate(found):
            used.update(w["test_ids"])
            template = REQUESTS["portfolio"][index % len(REQUESTS["portfolio"])]
            expected = {"mode": "portfolio", "start": fmt(w["start"]), "end": fmt(w["end"]),
                        "test_ids": w["test_ids"], "counts": w["counts"], "ship": w["ship"]}
            cases[split].append(case(split, "portfolio", index, f"window:{fmt(w['start'])}",
                                     template.format(s=fmt(w["start"]), e=fmt(w["end"])),
                                     expected))
        rng.shuffle(split_days)
        found = windows(split_days, rng, used, per_mode,
                        lambda planned: planning_window(planned, rng, used),
                        prefer_exclusions=True)
        for index, p in enumerate(found):
            used.update(p["window_test_ids"])
            template = REQUESTS["planning"][index % len(REQUESTS["planning"])]
            expected = {"mode": "planning", "planned_start": fmt(p["planned"]),
                        "window_start": fmt(p["start"]), "arms": p["k"],
                        "relative_lift": p["lift"] / 100, "daily_impressions": p["daily"],
                        "window_test_ids": p["window_test_ids"],
                        "excluded_tests": p["excluded_tests"], "baseline_ctr": p["baseline_ctr"],
                        "alpha": p["alpha"], "impressions_per_arm": p["impressions_per_arm"],
                        "days": p["days"]}
            request = template.format(k=p["k"], T=fmt(p["planned"]), L=p["lift"], D=p["daily"])
            cases[split].append(case(split, "planning", index, f"window:{fmt(p['start'])}",
                                     request, expected))
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for split, items in cases.items():
        path = OUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(c) + "\n" for c in items), encoding="utf-8")
        print(f"{split}: {len(items)} cases -> {path.relative_to(ROOT)}")
        for c in items:
            e = c["expected"]
            kind = e.get("label", "-")
            if e["mode"] == "planning":
                kind = "with exclusions" if e["excluded_tests"] else "no exclusions"
            summary.append({"split": split, "mode": e["mode"], "label": kind})
    table = pd.DataFrame(summary).groupby(["mode", "label", "split"]).size().unstack(fill_value=0)
    print(table[list(SPLITS)].to_string())


if __name__ == "__main__":
    main()
