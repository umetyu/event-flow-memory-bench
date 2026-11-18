"""Configuration and protocol invariants; no implicit dataset-dependent defaults."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import yaml

DEFAULT = {
    "protocol": "matched-v1", "seed": 0, "device": "cpu",
    "data": {"dataset": "uzh_fpv", "manifest": "data/uzh.json", "resolution": [128, 128]},
    "model": {"name": "FireNet", "encoding": "cnt", "num_bins": 2,
              "base_num_channels": 32, "kernel_size": 3, "norm_input": False,
              "mask_output": True, "activations": ["relu", None], "spiking_neuron": None},
    "window": {"stride_events": 1000, "history_events": 1000, "loss_events": 10000,
               "eval_events": 50000, "warmup_events": 50000},
    "train": {"epochs": 100, "batch_size": 8, "lr": 0.0002, "clip_grad": 100.0,
              "smooth_weight": 0.001, "augment": True},
    "eval": {"mode": "events", "flow_scale": 128.0, "outlier_reference": "gt",
             "mask_channel": 0},
}

def merge(a, b):
    result = deepcopy(a)
    for key, value in b.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result

def validate(c):
    if c["protocol"] != "matched-v1":
        raise ValueError("Use the reproduce command for the upstream protocol")
    w = c["window"]
    if any(not isinstance(v, int) or v < (0 if k == "warmup_events" else 1) for k, v in w.items()):
        raise ValueError("Event counts must be positive integers (warmup may be zero)")
    if w["history_events"] < w["stride_events"]:
        raise ValueError("history_events must cover the current stride")
    for key in ("history_events", "loss_events", "eval_events", "warmup_events"):
        if w[key] % w["stride_events"]:
            raise ValueError(f"{key} must be divisible by stride_events")
    if len(c["data"]["resolution"]) != 2 or min(c["data"]["resolution"]) < 16:
        raise ValueError("resolution must be [height, width], both >=16")
    model = c["model"]
    if model["encoding"] not in ("cnt", "voxel"):
        raise ValueError("encoding must be cnt or voxel")
    if model["encoding"] == "cnt" and model["num_bins"] != 2:
        raise ValueError("cnt requires two polarity channels")
    if model["num_bins"] < 2:
        raise ValueError("At least two voxel bins are required")
    if model["norm_input"]:
        raise ValueError("norm_input is disabled: sparse/empty upstream normalization is unsafe")
    if min(c["train"]["epochs"], c["train"]["batch_size"], c["train"]["lr"]) <= 0:
        raise ValueError("epochs, batch_size and lr must be positive")
    if c["eval"]["flow_scale"] <= 0:
        raise ValueError("flow_scale must be positive")
    if c["eval"]["mode"] not in ("events", "gtflow_dt1", "gtflow_dt4", "masks"):
        raise ValueError("Unknown evaluation mode")
    if c["eval"]["outlier_reference"] not in ("gt", "prediction"):
        raise ValueError("outlier_reference must be gt or prediction")
    if c["data"]["dataset"] == "mvsec" and not c["eval"]["mode"].startswith("gtflow"):
        raise ValueError("MVSEC requires a GT interval mode")
    if model["name"] == "EVLayerSegNet" and c["data"]["dataset"] != "affineobjects":
        raise ValueError("EVLayerSegNet is restricted to AffineObjects")
    return c

def read_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    c = merge(DEFAULT, raw)
    return validate(c)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

def upstream_config(c):
    return {"loader": {"resolution": c["data"]["resolution"]}, "model": {"mask_output": True},
            "loss": {"flow_regul_weight": c["train"]["smooth_weight"], "overwrite_intermediate": False}}
