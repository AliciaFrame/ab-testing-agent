"""Build the arm table the agent tools read from the raw Upworthy Research Archive CSVs.

Download the three package files from https://osf.io/jd64p/ (folder upworthy-archive-datasets)
into data/raw/ first; see README.
"""

import string
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import abstats  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "agent" / "data" / "arms.parquet"
SOURCES = {
    "exploratory": "upworthy-exploratory.csv",
    "holdout": "upworthy-holdout.csv",
    "confirmatory": "upworthy-confirmatory.csv",
}
# Archive totals reported by Matias et al. (2021).
EXPECTED = {"tests": 32_487, "arms": 150_817, "impressions": 538_272_878, "clicks": 8_182_674}


def main():
    frames = []
    for archive_split, name in SOURCES.items():
        frame = pd.read_csv(RAW / name, low_memory=False)
        frame["archive_split"] = archive_split
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")
    df = df.rename(columns={"clickability_test_id": "test_id", "eyecatcher_id": "image_id"})
    df = df.sort_values(["test_id", "created_at", "Unnamed: 0"]).reset_index(drop=True)
    df["arm_index"] = df.groupby("test_id").cumcount()
    letters = list(string.ascii_uppercase)
    df["arm_id"] = df["arm_index"].map(lambda i: letters[i] if i < 26 else f"Z{i}")
    start = df.groupby("test_id")["created_at"].transform("min")
    df["test_started"] = start
    df["problem_period"] = (start >= abstats.INCIDENT_START) & (start < abstats.INCIDENT_END)
    for column in ("headline", "excerpt"):
        df[column] = df[column].fillna("").astype(str)
    columns = [
        "test_id",
        "arm_id",
        "archive_split",
        "test_started",
        "problem_period",
        "headline",
        "excerpt",
        "image_id",
        "impressions",
        "clicks",
        "significance",
        "first_place",
        "winner",
    ]
    actual = {"tests": df.test_id.nunique(), "arms": len(df),
              "impressions": int(df.impressions.sum()), "clicks": int(df.clicks.sum())}
    if actual != EXPECTED:
        raise SystemExit(f"Archive totals {actual} differ from the paper's {EXPECTED}")
    incident_tests = df.loc[df.problem_period, "test_id"].nunique()
    # The 2024 correction reports 7,004 affected tests; the stated date rule gives 7,016 here.
    print(f"incident-window tests: {incident_tests:,} (paper: 7,004)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_parquet(OUT, index=False)
    print(f"{len(df):,} arms, {df.test_id.nunique():,} tests -> {OUT}")


if __name__ == "__main__":
    main()
