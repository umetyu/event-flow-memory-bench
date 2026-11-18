#!/usr/bin/env python3
"""Audit existing author-preprocessed files and create recording-grouped manifests.

Reads HDF5 metadata and endpoint samples; does not copy, extract or download data.
The input recording IDs are the author archive filename prefixes before crop IDs.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from event_flow_bench.config import write_json
from event_flow_bench.data import split_manifest


def audit(path, resolution):
    with h5py.File(path, "r") as f:
        counts = {key: len(f[f"events/{key}"]) for key in ("xs", "ys", "ts", "ps")}
        if len(set(counts.values())) != 1 or counts["ts"] < 2:
            raise ValueError(f"Invalid event columns: {path}")
        if "t0" not in f.attrs:
            raise ValueError(f"Missing upstream timestamp origin: {path}")
        for sl in (slice(0, 1000), slice(-1000, None)):
            x, y, t, p = [f[f"events/{key}"][sl] for key in ("xs", "ys", "ts", "ps")]
            if (not np.isfinite(t).all() or np.any(np.diff(t) < 0)
                    or np.any(x < 0) or np.any(x >= resolution[1])
                    or np.any(y < 0) or np.any(y >= resolution[0])
                    or not np.isin(p, [0, 1]).all()):
                raise ValueError(f"Invalid endpoint sample: {path}")
        row = {"file": path.name, "events": counts["ts"], "bytes": path.stat().st_size,
               "first_timestamp": float(f["events/ts"][0]),
               "last_timestamp": float(f["events/ts"][-1]), "flow_maps": {}}
        if row["last_timestamp"] <= row["first_timestamp"]:
            raise ValueError(f"Nonpositive sequence duration: {path}")
        for dt in (1, 4):
            key = f"flow_dt{dt}"
            if key in f:
                names = sorted(f[key])
                if len(names) < 2:
                    raise ValueError(f"Insufficient GT maps: {path}/{key}")
                stamps = [float(f[key][name].attrs["timestamp"]) for name in names]
                if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
                    raise ValueError(f"Unordered GT timestamps: {path}/{key}")
                for name in (names[0], names[-1]):
                    if f[key][name].shape != (2, *resolution):
                        raise ValueError(f"Unexpected flow dimensions: {path}/{key}/{name}")
                row["flow_maps"][key] = len(names)
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uzh", required=True)
    parser.add_argument("--mvsec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    train, training_rows = [], []
    for path in sorted(Path(args.uzh).resolve().glob("*.h5")):
        match = re.fullmatch(r"(.+_davis)(?:_with_gt)?_([0-9]+)\.h5", path.name)
        if not match:
            raise ValueError(f"Unrecognized author recording/crop name: {path.name}; supply explicit groups instead")
        row = audit(path, (128, 128))
        if row["events"] != 500_000:
            raise ValueError(f"Author training crop must contain 500k events: {path}")
        training_rows.append(row)
        train.append({"path": str(path), "group": match[1], "id": path.stem})
    if len(train) != 140:
        raise ValueError(f"Exact author training archive requires 140 crops; found {len(train)}")
    test, test_rows = [], []
    for name in ("outdoor_day1", "indoor_flying1", "indoor_flying2", "indoor_flying3"):
        path = Path(args.mvsec).resolve() / f"{name}_data.h5"
        row = audit(path, (256, 256))
        if set(row["flow_maps"]) != {"flow_dt1", "flow_dt4"}:
            raise ValueError(f"Both dt1 and dt4 ground truth required: {path}")
        test_rows.append(row)
        test.append({"path": str(path), "group": name, "id": name})
    write_json(root / "uzh-all.json", {"version": 1, "dataset": "uzh_fpv", "train": train})
    matched = split_manifest(root / "uzh-all.json", root / "uzh-matched.json", seed=args.seed)
    write_json(root / "mvsec.json", {"version": 1, "dataset": "mvsec", "test": test})
    report = {"training": training_rows, "mvsec": test_rows,
              "recording_groups": dict(Counter(e["group"] for e in train)),
              "events_per_epoch_all": sum(r["events"] for r in training_rows),
              "matched_split_counts": {s: len(matched[s]) for s in ("train", "val", "test")},
              "matched_validation_groups": sorted({e["group"] for e in matched["val"]}),
              "checks": "All event-column lengths, 1k endpoint samples, all GT timestamps, endpoint GT shapes",
              "limits": "Not a full archive checksum or full event/pixel scan",
              "protocols": {"paper_reproduction": "all 140 crops; no held-out validation",
                            "matched_comparison": "recording-level validation; MVSEC remains test-only"}}
    write_json(root / "audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("training", "mvsec")}, indent=2))


if __name__ == "__main__":
    main()
