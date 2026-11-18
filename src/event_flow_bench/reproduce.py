"""NeurIPS 2021 MVSEC evaluation using the pinned upstream loader and AEE."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import torch
import yaml
from .config import merge, write_json
from .engine import sha256
from .metrics import summarize
from .models import EVENT_FLOW_SHA, MODEL_NAMES
from ._vendor.event_flow.models import model as upstream
from ._vendor.event_flow.loss.flow import AEE

def read_training_config(run_dir=None, train_config=None):
    if (run_dir is None) == (train_config is None):
        raise ValueError("Provide exactly one of run_dir or train_config")
    if train_config:
        return yaml.safe_load(Path(train_config).read_text())
    params = Path(run_dir) / "params"
    if not params.is_dir():
        raise FileNotFoundError(params)
    config = {}
    for p in params.iterdir():
        if p.is_file():
            text = p.read_text()
            try:
                config[p.name] = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                config[p.name] = text
    return config

def load_weights(checkpoint, trust_legacy_pickle=False):
    """Missing/incompatible checkpoints are fatal; never fall back to random weights."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Historical MLflow artifacts contain a full models.model Python object.
    # Explicit opt-in is required for torch's executable legacy pickle format.
    if trust_legacy_pickle:
        vendor = Path(__file__).parent / "_vendor/event_flow"
        sys.path.insert(0, str(vendor))
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        finally:
            sys.path.remove(str(vendor))
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, torch.nn.Module):
        return state.state_dict()
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state

@torch.no_grad()
def reproduce(data_path, checkpoint, training, dt=1, device="cpu", resolution=(256, 256),
              trust_legacy_pickle=False, output=None):
    from ._vendor.event_flow.dataloader.h5 import H5Loader
    if dt not in (1, 4):
        raise ValueError("dt must be 1 or 4")
    if not list(Path(data_path).rglob("*.h5")):
        raise FileNotFoundError(f"No HDF5 sequences in {data_path}")
    c = merge(training, {
        "data": {"path": str(data_path), "mode": f"gtflow_dt{dt}", "window": 1.0 / dt},
        "loader": {"batch_size": 1, "resolution": list(resolution), "augment": [], "seed": 0},
        "loss": {"overwrite_intermediate": False},
        "vis": {"enabled": False, "bars": False},
        "hot_filter": {"enabled": True, "max_px": 100, "min_obvs": 5, "max_rate": 0.8}})
    kwargs = deepcopy(c["model"])
    kwargs["spiking_neuron"] = c.get("spiking_neuron", kwargs.get("spiking_neuron"))
    if kwargs["name"] not in MODEL_NAMES:
        raise ValueError("Unknown checkpoint architecture")
    net = getattr(upstream, kwargs["name"])(deepcopy(kwargs)).to(device)
    net.load_state_dict(load_weights(checkpoint, trust_legacy_pickle), strict=True)
    net.eval()
    loader = H5Loader(c, kwargs["num_bins"], kwargs.get("round_encoding", False))
    # Bound iteration explicitly, and reject malformed files before the loader can wrap forever.
    import h5py
    max_calls = 0
    for p in loader.files:
        with h5py.File(p) as f:
            if f"flow_dt{dt}" not in f or len(f[f"flow_dt{dt}"]) < 2:
                raise ValueError(f"No valid GT intervals: {p}")
            max_calls += (len(f[f"flow_dt{dt}"]) - 1) * dt
    metric, passes, rows = AEE(c, device, flow_scaling=128), 0, []
    try:
        for _ in range(max_calls + len(loader.files) + 1):
            inputs = loader.custom_collate([loader[0]])
            if loader.new_seq:
                loader.new_seq = False
                net.reset_states()
                metric.reset()
                passes = 0
            if loader.seq_num >= len(loader.files):
                break
            out = net(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device))
            metric.event_flow_association(out["flow"], inputs)
            passes += 1
            if passes == dt:
                if float(inputs["dt_input"]) <= 0:
                    row = {"aee": None, "outlier_pct": None, "status": "zero_duration"}
                else:
                    aee, fraction = metric()
                    if not torch.isfinite(aee).all() or not torch.isfinite(fraction).all():
                        row = {"aee": None, "outlier_pct": None, "status": "nonfinite"}
                    else:
                        row = {"aee": float(aee[0]), "outlier_pct": 100 * float(fraction[0])}
                row["sequence"] = Path(loader.files[loader.batch_idx[0] % len(loader.files)]).stem
                rows.append(row)
                metric.reset()
                passes = 0
        else:
            raise RuntimeError("Upstream loader exceeded expected GT interval count")
    finally:
        for f in loader.open_files:
            f.close()
    if not rows:
        raise ValueError("No scored upstream evaluation intervals")
    result = {"protocol": "hagenaars2021-upstream", "model": kwargs["name"], "dt": dt,
              "summary": summarize(rows),
              "sequences": {name: summarize([r for r in rows if r["sequence"] == name]) for name in sorted({r["sequence"] for r in rows})},
              "rows": rows, "checkpoint_sha256": sha256(checkpoint), "upstream_revision": EVENT_FLOW_SHA,
              "config": c, "notes": ["Original last-partition event mask and prediction-magnitude outlier threshold",
                                     "Metric buffers reset at sequence boundaries; no inference warmup exclusion"]}
    if output:
        write_json(output, result)
    return result
