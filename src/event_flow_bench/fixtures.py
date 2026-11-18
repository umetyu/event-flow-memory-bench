"""Small synthetic fixtures for engineering checks, never scientific benchmark scores."""
from copy import deepcopy
from pathlib import Path
import h5py
import numpy as np
import yaml
from .config import DEFAULT, write_json

def write_fixture(path, seed=0, events=512, resolution=(16, 16), epoch_timestamp=False):
    rng = np.random.default_rng(seed)
    h, w = resolution
    idx = np.arange(events)
    # Moving, textured edge with both polarities and a nonconstant unwarped image.
    x = (2 + idx // 32) % (w - 4) + 2
    y = rng.integers(2, h - 2, events)
    ts = idx.astype(np.float64) * 0.0001 + (1.7e9 if epoch_timestamp else 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, values in {"xs": x.astype(np.int16), "ys": y.astype(np.int16), "ts": ts,
                            "ps": rng.integers(0, 2, events).astype(np.uint8)}.items():
            f.create_dataset(f"events/{key}", data=values)
        f.attrs["t0"] = ts[0]
        f.attrs["duration"] = ts[-1] - ts[0]
        for dt in (1, 4):
            # Fixture GT is a known displacement, not a claim about the synthetic scene.
            for i, index in enumerate(range(0, events, 64 * dt)):
                gt = np.zeros((2, h, w), np.float32)
                gt[0] = float(dt)
                ds = f.create_dataset(f"flow_dt{dt}/{i:06}", data=gt)
                ds.attrs["timestamp"] = ts[index]
    return path

def make_demo(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "dataset": "synthetic", "train": [], "val": [], "test": []}
    for i, split in enumerate(("train", "train", "val", "test")):
        path = write_fixture(output / f"sequence-{i}.h5", seed=i)
        manifest[split].append({"path": path.name, "group": f"recording-{i}", "id": f"sequence-{i}"})
    write_json(output / "manifest.json", manifest)
    c = deepcopy(DEFAULT)
    c["data"].update({"dataset": "synthetic", "resolution": [16, 16], "manifest": str(output / "manifest.json")})
    c["model"]["base_num_channels"] = 2
    c["window"].update({"stride_events": 16, "history_events": 32, "loss_events": 64,
                         "eval_events": 64, "warmup_events": 32})
    c["train"].update({"epochs": 1, "batch_size": 2})
    c["eval"]["flow_scale"] = 16.0
    (output / "config.yml").write_text(yaml.safe_dump(c, sort_keys=False))
    return c
