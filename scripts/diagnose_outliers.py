#!/usr/bin/env python3
"""Compare outlier conventions on identical predictions, without changing scores.

Use one author-preprocessed recording per invocation. The primary AEE/outlier
values remain the pinned upstream values; alternatives are separate diagnostics.
"""
import argparse
import importlib
import json
from pathlib import Path
import statistics

import torch
from event_flow_bench.config import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dt", type=int, choices=[1, 4], default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trust-legacy-pickle", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(list(Path(args.data).rglob("*.h5"))) != 1:
        parser.error("Use a directory containing exactly one recording")
    torch.set_num_threads(args.threads)
    reproduction = importlib.import_module("event_flow_bench.reproduce")
    original = reproduction.AEE
    rows = []

    class DiagnosticAEE(original):
        def forward(self):
            primary = super().forward()
            flow = self._flow_map[-1] * self.flow_scaling
            flow = flow * self._dt_gt.to(self.device) / self._dt_input.to(self.device)
            error = (flow - self._gtflow).pow(2).sum(1).sqrt()
            pred_mag = flow.pow(2).sum(1).sqrt()
            gt_mag = self._gtflow.pow(2).sum(1).sqrt()
            valid = self._event_mask[:, -1].bool() & (self._gtflow != 0).any(1)
            n = int(valid.sum())
            if n:
                three = (error > 3) & valid
                rows.append({
                    "upstream_prediction_relative_pct": float(primary[1][0]) * 100,
                    "gt_relative_pct": float((three & (error > .05 * gt_mag)).sum()) * 100 / n,
                    "absolute_3px_only_pct": float(three.sum()) * 100 / n,
                    "prediction_formula_pct": float((three & (error > .05 * pred_mag)).sum()) * 100 / n,
                })
            return primary

    reproduction.AEE = DiagnosticAEE
    try:
        config = reproduction.read_training_config(args.run_dir)
        result = reproduction.reproduce(args.data, Path(args.run_dir) / "artifacts/model/data/model.pth",
                                        config, args.dt, args.device,
                                        trust_legacy_pickle=args.trust_legacy_pickle)
    finally:
        reproduction.AEE = original
    if not rows:
        raise ValueError("No valid diagnostic windows")
    result["outlier_diagnostics"] = {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
    result["outlier_diagnostics"]["windows"] = len(rows)
    result["notes"].append("Alternative thresholds are diagnostics only; original result fields and paper references are unchanged")
    write_json(args.output, result)
    print(json.dumps(result["outlier_diagnostics"], indent=2))


if __name__ == "__main__":
    main()
