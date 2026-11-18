"""Causal HDF5 slices. Timestamps stay float64 until after subtracting the origin."""
from pathlib import Path
import json
import random
import h5py
import hdf5plugin  # noqa: F401 -- registers compression filters used in upstream packs
import numpy as np
import torch
from .config import write_json
from ._vendor.event_flow.dataloader.encodings import events_to_channels, events_to_voxel

KEYS = ("xs", "ys", "ts", "ps")

def load_manifest(path):
    path = Path(path).resolve()
    m = json.loads(path.read_text())
    if m.get("version") != 1:
        raise ValueError("Expected manifest version 1")
    paths, groups = {}, {}
    for split in ("train", "val", "test"):
        m.setdefault(split, [])
        for entry in m[split]:
            p = (path.parent / entry["path"]).resolve()
            if not entry.get("group"):
                raise ValueError("Every entry needs an original-recording group")
            group = entry["group"]
            if p in paths or (group in groups and groups[group] != split):
                raise ValueError(f"Split leakage or duplicate path: {entry}")
            paths[p], groups[group] = split, split
            entry["path"] = str(p)
            for annotation in entry.get("masks", []):
                annotation["path"] = str((path.parent / annotation["path"]).resolve())
    return m

def split_manifest(inventory, output, val_fraction=0.1, seed=0):
    """Split ONLY an explicitly grouped training inventory; never mix test data in."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    m = load_manifest(inventory)
    if m["val"] or not m["train"]:
        raise ValueError("Inventory must contain train entries and no existing validation split")
    groups = sorted({e["group"] for e in m["train"]})
    if len(groups) < 2:
        raise ValueError("At least two original recordings are needed for a held-out validation split")
    random.Random(seed).shuffle(groups)
    n = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
    val = set(groups[:n])
    entries = m["train"]
    m["train"] = [e for e in entries if e["group"] not in val]
    m["val"] = [e for e in entries if e["group"] in val]
    m["split_seed"] = seed
    write_json(output, m)
    return m

class EventFile:
    def __init__(self, path, resolution):
        self.path, self.resolution = str(path), tuple(resolution)
        self.file = h5py.File(path, "r")
        try:
            lengths = [len(self.file[f"events/{k}"]) for k in KEYS]
            if len(set(lengths)) != 1 or lengths[0] < 2:
                raise ValueError(f"Invalid event columns: {path}")
            self.size = lengths[0]
            self.origin = float(self.file["events/ts"][0])
        except Exception:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def index(self, timestamp):
        """Left search, including repeated timestamps, without loading the entire stream."""
        lo, hi = 0, self.size
        ts = self.file["events/ts"]
        while lo < hi:
            mid = (lo + hi) // 2
            if ts[mid] < timestamp:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def read(self, start, end, flips=(False, False, False)):
        if not 0 <= start <= end <= self.size:
            raise IndexError((start, end, self.size))
        x, y, t, p = [np.asarray(self.file[f"events/{k}"][start:end]) for k in KEYS]
        x, y = x.astype(np.float32), y.astype(np.float32)
        t = t.astype(np.float64) - self.origin
        if not np.all(np.isin(p, [-1, 0, 1])):
            raise ValueError("Polarity must use {0,1} or {-1,1}")
        p = np.where(p > 0, 1.0, -1.0).astype(np.float32)
        h, w = self.resolution
        if len(t) and (not np.isfinite(t).all() or np.any(np.diff(t) < 0)
                       or np.any(x < 0) or np.any(x >= w) or np.any(y < 0) or np.any(y >= h)
                       or not np.isfinite(x).all() or not np.isfinite(y).all()
                       or np.any(x != np.floor(x)) or np.any(y != np.floor(y))):
            raise ValueError(f"Invalid timestamps or coordinates for resolution {self.resolution}: {self.path}")
        if flips[0]: x = w - 1 - x
        if flips[1]: y = h - 1 - y
        if flips[2]: p = -p
        return x, y, t, p

    def gt_intervals(self, mode):
        key = {"gtflow_dt1": "flow_dt1", "gtflow_dt4": "flow_dt4"}[mode]
        if key not in self.file:
            raise ValueError(f"Missing {key} in {self.path}; supply the preprocessed MVSEC pack")
        records = []
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                records.append((float(obj.attrs["timestamp"]), name))
        self.file[key].visititems(visit)
        records.sort()
        for i in range(1, len(records)):
            end, name = records[i]
            obj = self.file[f"{key}/{name}"]
            start = float(obj.attrs.get("timestamp_start", records[i - 1][0]))
            if end <= start:
                raise ValueError("GT intervals must have positive duration")
            gt = np.asarray(obj, dtype=np.float32)
            if gt.shape != (2, *self.resolution):
                raise ValueError(f"GT must be (2,H,W), got {gt.shape}")
            # The upstream pack encodes invalid GT as (0,0). Explicit masks override this.
            mask_key = f"{key}_valid/{name}"
            valid = np.asarray(self.file[mask_key], dtype=bool) if mask_key in self.file else np.any(gt != 0, axis=0)
            yield start, end, torch.from_numpy(gt)[None], torch.from_numpy(valid)[None]

def encode(events, resolution, bins):
    x, y, t, p = events
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    normalized = (t - t[0]) / duration if duration > 0 else np.zeros_like(t)
    x, y, ts, p = [torch.from_numpy(np.asarray(v, dtype=np.float32)) for v in (x, y, normalized, p)]
    cnt = events_to_channels(x, y, p, sensor_size=resolution)
    voxel = events_to_voxel(x, y, ts, p, bins, sensor_size=resolution)
    return {"event_cnt": cnt, "event_voxel": voxel,
            "event_mask": (cnt.sum(0, keepdim=True) > 0).float(),
            "event_list": torch.stack((ts, y, x, p), dim=-1),
            "event_list_pol_mask": torch.stack((p > 0, p < 0), dim=-1).float(),
            "dt_input": torch.tensor(duration, dtype=torch.float64), "dt_gt": torch.tensor(0.0)}

def sample(stream, start, end, c, flips=(False, False, False)):
    """Input history is [end-history,end); targets contain only [start,end)."""
    bins, res = c["model"]["num_bins"], c["data"]["resolution"]
    now = encode(stream.read(start, end, flips), res, bins)
    history_start = max(0, end - c["window"]["history_events"])
    history = encode(stream.read(history_start, end, flips), res, bins)
    now["event_cnt"], now["event_voxel"] = history["event_cnt"], history["event_voxel"]
    return now

def batch(samples, device):
    return {k: torch.stack([x[k] for x in samples]).to(device) for k in samples[0]}

def inspect_file(path, resolution):
    with EventFile(path, resolution) as stream:
        previous = None
        for a in range(0, stream.size, 1000000):
            t = stream.read(a, min(stream.size, a + 1000000))[2]
            if previous is not None and len(t) and t[0] < previous:
                raise ValueError("Timestamps decrease across chunks")
            if len(t): previous = t[-1]
        return {"path": str(path), "events": stream.size, "duration_s": float(previous),
                "groups": sorted(stream.file.keys()), "resolution": list(resolution)}
