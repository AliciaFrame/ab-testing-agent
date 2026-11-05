"""Build RFT train/valid rows and a manifest.

Train rows are the seeded calibration sample (``rft.calibrate``, same seed and order) from the train split: 30
planning, 30 ship-decision and 15 portfolio items (the targets) plus 5 each of validity, early look and attribution
(retention). Calibration showed the base model already passes most of these at the accuracy reward, so every
calibrated item is kept: clean passes still carry the efficiency signal. Valid rows are a seeded sample of the
validation split. Rows never come from the development, optimizer or final splits.

Dataset ``v2`` (``rft/data/v2/``, the ``turns`` jobs) is ``v1`` with portfolio doubled to 30 items: the next 15
seeded portfolio items are appended, so the first 15 are the ``v1`` items.

    python -m rft.build_data [--dataset v2]
"""

from __future__ import annotations

import argparse
import json
import random

from rft.calibrate import SEED
from rft.common import DATA_DIR, MODES, RFT_DIR, load_cases, read_jsonl, row, sha256_file, write_jsonl

TRAIN_PER_MODE = {"planning": 30, "ship_decision": 30, "portfolio": 15, "validity": 5, "early_look": 5, "attribution": 5}
VALID_PER_MODE = {"planning": 3, "ship_decision": 3, "portfolio": 3, "validity": 2, "early_look": 2, "attribution": 2}
DATASETS = {"v1": {"dir": DATA_DIR, "train_per_mode": TRAIN_PER_MODE},
            "v2": {"dir": DATA_DIR / "v2", "train_per_mode": {**TRAIN_PER_MODE, "portfolio": 30}}}


def seeded(split: str, mode: str, salt: str = "") -> list[dict]:
    pool = sorted([c for c in load_cases(split) if c["expected"]["mode"] == mode], key=lambda c: c["id"])
    random.Random(f"{SEED}-{mode}{salt}").shuffle(pool)
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="v1")
    spec = DATASETS[parser.parse_args().dataset]
    out, per_mode = spec["dir"], spec["train_per_mode"]
    out.mkdir(parents=True, exist_ok=True)
    train = [row(c) for m in MODES for c in seeded("train", m)[:per_mode[m]]]
    valid = [row(c) for m in MODES for c in seeded("validation", m, "-valid")[:VALID_PER_MODE[m]]]
    random.Random(SEED).shuffle(train)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid.jsonl", valid)
    calibration = []
    for path in sorted((RFT_DIR / "calibration").glob("train-*.jsonl")):
        calibration += [r for r in read_jsonl(path) if not r.get("error")]
    chosen = {r["case_id"] for r in train}
    stats = {}
    for r in calibration:
        if r["case_id"] in chosen:
            s = stats.setdefault(f"{r['mode']}/{r['effort']}", {"rollouts": 0, "acc_below_1": 0, "eff_below_0.9": 0})
            s["rollouts"] += 1
            s["acc_below_1"] += r["acc"] < 0.999
            s["eff_below_0.9"] += r["eff"] < 0.9
    heldout = set()
    for split in ("development", "optimizer", "final"):
        heldout |= {c["expected"]["test_id"] for c in load_cases(split) if "test_id" in c["expected"]}
    used = {r["case"]["expected"]["test_id"] for r in train + valid if "test_id" in r["case"]["expected"]}
    manifest = {
        "seed": SEED, "train_per_mode": per_mode, "valid_per_mode": VALID_PER_MODE,
        "train_rows": len(train), "valid_rows": len(valid),
        "calibration_on_train_rows": stats,
        "files": {name: {"sha256": sha256_file(out / name)} for name in ("train.jsonl", "valid.jsonl")},
        "heldout_test_overlap": sorted(used & heldout),
        "max_row_bytes": max(len(json.dumps(r).encode()) for r in train + valid),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
