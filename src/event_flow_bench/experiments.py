"""Generate ordered experiment configurations and aggregate comparable results."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import statistics
import yaml
from .config import DEFAULT, digest, validate, write_json

VARIANTS = {
    "pilot": [("FireFlowNet", "cnt"), ("FireFlowNet", "voxel"), ("LeakyFireNet", "cnt"),
              ("RNNFireNet", "cnt"), ("FireNet", "cnt")],
    "main": [("EVFlowNet", "cnt"), ("EVFlowNet", "voxel"), ("LeakyRecEVFlowNet", "cnt"),
             ("RNNRecEVFlowNet", "cnt"), ("RecEVFlowNet", "cnt")],
    "snn": [(n, "cnt") for n in ("LIFFireNet", "ALIFFireNet", "PLIFFireNet", "XLIFFireNet",
                                    "SpikingRecEVFlowNet", "ALIFRecEVFlowNet", "PLIFRecEVFlowNet", "XLIFRecEVFlowNet")],
    "layerseg": [("EVLayerSegNet", "voxel")],
}

def generate_grid(output, manifest, stage="pilot", dataset="uzh_fpv", seeds=(0, 1, 2), histories=(1000,), device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model, encoding in VARIANTS[stage]:
        for history in histories:
            for seed in seeds:
                c = deepcopy(DEFAULT)
                c.update({"seed": seed, "device": device})
                c["data"].update({"dataset": dataset, "manifest": str(manifest)})
                if dataset == "affineobjects":
                    c["data"]["resolution"] = [480, 640]
                c["model"].update({"name": model, "encoding": encoding, "num_bins": 5 if encoding == "voxel" else 2})
                if stage == "snn":
                    c["model"].update({"activations": ["arctanspike", "arctanspike"],
                                      "spiking_neuron": {"leak": [-4., .1], "thresh": [.8, .1],
                                                         "learn_leak": True, "learn_thresh": True, "hard_reset": True}})
                c["window"]["history_events"] = history
                c["window"]["warmup_events"] = max(50000, max(histories))
                validate(c)
                name = f"{dataset}-{model}-{encoding}-h{history}-s{seed}"
                path = output / f"{name}.yml"
                path.write_text(yaml.safe_dump(c, sort_keys=False))
                jobs.append({"name": name, "config": str(path), "run_dir": f"runs/{name}"})
    write_json(output / "jobs.json", jobs)
    return jobs

def collect(paths, output):
    """Separate groups by full scoring policy; never mix dt, datasets or mask conventions."""
    grouped, seen = {}, set()
    for path in paths:
        result = json.loads(Path(path).read_text())
        if result["protocol"] != "matched-v1":
            raise ValueError("Use compare-paper for original-protocol results")
        c = result["provenance"]["config"]
        training = result.get("training_config", c)
        model_controls = {k: v for k, v in c["model"].items() if k not in ("name", "encoding", "num_bins", "activations", "spiking_neuron", "source")}
        policy = {"dataset": c["data"]["dataset"], "split": result["split"], "eval": c["eval"],
                  "train": training["train"], "training_dataset": training["data"]["dataset"],
                  "training_manifest": result.get("training_manifest_sha256", result["provenance"]["manifest_sha256"]),
                  "model_controls": model_controls, "loss_events": training["window"]["loss_events"],
                  "stride": c["window"]["stride_events"], "eval_events": c["window"]["eval_events"],
                  "warmup": c["window"]["warmup_events"], "resolution": c["data"]["resolution"],
                  "manifest": result["provenance"]["manifest_sha256"]}
        for sequence, values in result["sequences"].items():
            # Score anchors must match too, including sequence warmup and tail exclusions.
            anchors = [(r["start_event"], r["end_event"]) for r in result["rows"] if r["sequence"] == sequence]
            policy_id = digest({**policy, "anchors": anchors})[:16]
            base = (policy_id, sequence, c["model"]["name"], c["model"]["encoding"], c["window"]["history_events"])
            key_seed = (*base, c["seed"])
            if key_seed in seen:
                raise ValueError("Duplicate seed/result in aggregation")
            seen.add(key_seed)
            for metric in ("aee", "outlier_pct", "fwl", "rsat", "iou", "detected", "warped_event_fraction"):
                if metric in values:
                    if values.get(metric + "_invalid", 0) or values[metric] is None:
                        raise ValueError(f"Invalid {metric} windows in {path}; resolve before aggregation")
                    grouped.setdefault((*base, metric), []).append(values[metric])
    rows = []
    for key, values in sorted(grouped.items()):
        rows.append(dict(zip(("policy_id", "sequence", "model", "encoding", "history_events", "metric"), key),
                         n_seeds=len(values), mean=statistics.mean(values),
                         std=statistics.stdev(values) if len(values) > 1 else None))
    if not rows:
        raise ValueError("No results supplied")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows

def compare_paper(result_path, reference, output, aee_tol=0.02, outlier_tol=0.2):
    result = json.loads(Path(result_path).read_text())
    if result["protocol"] != "hagenaars2021-upstream":
        raise ValueError("Paper comparison requires the upstream protocol")
    with open(reference) as f:
        refs = [r for r in csv.DictReader(f) if r["model"] == result["model"] and int(r["dt"]) == result["dt"]]
    if not refs:
        raise ValueError("No reference rows for this model/dt")
    rows = []
    for ref in refs:
        matches = [v for name, v in result["sequences"].items() if ref["sequence"] in name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one sequence matching {ref['sequence']}")
        actual = matches[0]
        if actual.get("aee_invalid", 0) or actual.get("outlier_pct_invalid", 0):
            raise ValueError("Invalid evaluation intervals")
        delta_aee = actual["aee"] - float(ref["aee"])
        delta_out = actual["outlier_pct"] - float(ref["outlier_pct"])
        rows.append({"sequence": ref["sequence"], "aee_delta": delta_aee, "outlier_pct_delta": delta_out,
                     "within_tolerance": abs(delta_aee) <= aee_tol and abs(delta_out) <= outlier_tol})
    report = {"passed": all(r["within_tolerance"] for r in rows), "rows": rows,
              "aee_tolerance": aee_tol, "outlier_percentage_point_tolerance": outlier_tol}
    write_json(output, report)
    return report
