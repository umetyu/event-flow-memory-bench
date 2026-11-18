"""Causal CMax evaluation on the existing preprocessed MVSEC manifest."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from event_flow_bench.cmax import PatchCMax2018, SecretsCMax2022, SECRETS_SHA
from event_flow_bench.config import digest, write_json
from event_flow_bench.data import EventFile, load_manifest
from event_flow_bench.metrics import flow_metrics, summarize

SEQUENCES = ("outdoor_day1", "indoor_flying1", "indoor_flying2", "indoor_flying3")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def input_bounds(end_event, history):
    return max(0, end_event-history), end_event


def score_velocity(velocity, gt, valid, events, duration):
    pred = torch.from_numpy(np.asarray(velocity, np.float32))[None] * duration
    mask = torch.zeros_like(valid, dtype=torch.bool)
    x, y, _, _ = events
    mask[0, y.astype(np.int64), x.astype(np.int64)] = True
    metrics = flow_metrics(pred, gt, valid, mask, "gt")
    # Publish both conventions; use the GT-relative convention for matched comparisons.
    selected = valid & mask & torch.isfinite(gt).all(1)
    error = torch.linalg.vector_norm(pred-torch.nan_to_num(gt), dim=1)
    metrics["outlier_3px_pct"] = float((error[selected] > 3).float().mean()*100) if selected.any() else None
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("patch2018", "secrets2022"), required=True)
    p.add_argument("--profile", choices=("matched1k", "long"), required=True)
    p.add_argument("--sequence", choices=SEQUENCES, required=True)
    p.add_argument("--dt", type=int, choices=(1, 4), required=True)
    p.add_argument("--config", default="configs/cmax.json")
    p.add_argument("--manifest", default="runs/setup/mvsec.json")
    p.add_argument("--checkout", default="external/secrets-event-flow")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0, help="Smoke only: first N eligible intervals; 0 means full")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.limit < 0 or args.threads < 1: p.error("limit >= 0 and threads >= 1 required")
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.ERROR)
    c = json.loads(Path(args.config).read_text())
    manifest = load_manifest(args.manifest)
    if manifest["dataset"] != "mvsec": raise ValueError("MVSEC manifest required")
    entries = [e for e in manifest["test"] if e.get("id", e["group"]) == args.sequence]
    if len(entries) != 1: raise ValueError("Expected exactly one requested sequence")
    n_events = c["windows"][args.profile]
    if isinstance(n_events, dict): n_events = n_events["outdoor" if args.sequence.startswith("outdoor") else "indoor"]
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume: raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    root = Path(__file__).resolve().parents[1]
    sources = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
               ("src/event_flow_bench/cmax.py", "scripts/run_cmax.py", "src/event_flow_bench/data.py", "src/event_flow_bench/metrics.py")}
    identity = {"config": c, "config_sha256": digest(c), "manifest_sha256": digest(manifest),
                "method": args.method, "profile": args.profile, "sequence": args.sequence, "dt": args.dt,
                "input_events": n_events, "limit": args.limit, "git_revision": revision, "source_sha256": sources,
                "official_revision": SECRETS_SHA if args.method == "secrets2022" else None,
                "protocol": c["protocol"], "ground_truth_used_for_optimization": False,
                "future_events": False, "previous_window_initialization": False,
                "flow_units": "velocity px/s multiplied by GT interval seconds",
                "score_mask": "GT valid AND at least one event in GT interval", "hot_pixel_filter": False}
    if (output/"identity.json").exists():
        if json.loads((output/"identity.json").read_text()) != identity: raise ValueError("Resume identity mismatch")
    else: atomic_json(output/"identity.json", identity)
    versions = {}
    for name in ("torch", "numpy", "scipy", "torchvision", "optuna", "scikit-image"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    write_json(output/"environment.json", {"versions": versions, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"})
    rows_path = output/"windows.jsonl"
    rows = [json.loads(p.read_text()) for p in sorted(output.glob("window-*.json"))]
    rows_path.write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows))
    completed = {r["interval_index"] for r in rows}
    if len(completed) != len(rows): raise ValueError("Duplicate rows in resume log")
    solver = PatchCMax2018(c["resolution"], c[args.method], "cuda" if torch.cuda.is_available() else "cpu") if args.method == "patch2018" else SecretsCMax2022(c["resolution"], c[args.method], args.checkout)
    print(json.dumps({"phase": "start", **{k: identity[k] for k in ("method", "profile", "sequence", "dt", "input_events", "limit")}, "resumed_windows": len(rows)}), flush=True)
    started = time.monotonic()
    eligible = 0
    with EventFile(entries[0]["path"], c["resolution"]) as stream:
        for index, (t0, t1, gt, valid) in enumerate(stream.gt_intervals(f"gtflow_dt{args.dt}")):
            a, b = stream.index(t0), stream.index(t1)
            if a < c["warmup_events"]: continue
            if args.limit and eligible >= args.limit: break
            eligible += 1
            if index in completed: continue
            lo, hi = input_bounds(b, n_events)
            events = stream.read(lo, hi)
            if len(events[2]) and events[2][-1] + stream.origin >= t1: raise AssertionError("Future event leak")
            stamp = time.monotonic()
            seed = int.from_bytes(hashlib.sha256(f"{c['seed']}:{args.sequence}:{args.dt}:{index}".encode()).digest()[:4], "little") % (2**31-1)
            velocity, diagnostics = solver.estimate(events, seed)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            seconds = time.monotonic()-stamp
            # GT arrays become available to scoring only after optimization.
            metrics = score_velocity(velocity, gt, valid, stream.read(a, b), t1-t0)
            row = {"interval_index": index, "timestamp_start": t0, "timestamp_end": t1,
                   "start_event": a, "end_event": b, "input_start_event": lo, "input_end_event": hi,
                   "input_duration_s": float(events[2][-1]-events[2][0]) if len(events[2]) > 1 else 0.,
                   "optimization_seconds": seconds, "seed": seed, **metrics, "diagnostics": diagnostics}
            # One atomic file per interval makes resume safe even after a wall-time kill.
            atomic_json(output/f"window-{index:05}.json", row)
            rows.append(row)
            with rows_path.open("a") as f: f.write(json.dumps(row, allow_nan=False)+"\n"); f.flush()
            if len(rows) <= 3 or len(rows) % 25 == 0:
                print(json.dumps({"phase": "progress", "windows": len(rows), "last_seconds": seconds,
                                  "aee": metrics["aee"], "elapsed_seconds": time.monotonic()-started}), flush=True)
    if len(rows) != eligible: raise RuntimeError("Unexpected completed interval count")
    summary = summarize(rows)
    for key in ("outlier_3px_pct", "optimization_seconds"):
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary["optimization_seconds_total"] = sum(r["optimization_seconds"] for r in rows)
    summary["estimated_pixel_fraction"] = float(np.mean([r["diagnostics"]["estimated_pixel_fraction"] for r in rows])) if rows else None
    result = {**identity, "status": "smoke_completed" if args.limit else "completed", "summary": summary,
              "seconds_this_invocation": time.monotonic()-started, "windows": rows}
    atomic_json(output/("smoke.json" if args.limit else "completed.json"), result)
    print(json.dumps({"phase": result["status"], "summary": summary}), flush=True)


if __name__ == "__main__": main()
