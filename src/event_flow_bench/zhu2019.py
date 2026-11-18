"""Paper-based Zhu 2019 optical-flow reimplementation, not an author checkpoint.

See docs/ZHU2019.md for architectural choices and deviations. This separate
protocol intentionally does not reuse Hagenaars' training loss or flow scaling.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .config import digest, write_json
from .data import EventFile, batch, encode, load_manifest
from .engine import seed_all, sha256
from .metrics import flow_metrics, summarize
from ._vendor.event_flow.models.unet import MultiResUNet
from ._vendor.event_flow.loss.flow import RSAT

PROTOCOL = "zhu2019-reimplementation-v1"


def read_config(path):
    c = yaml.safe_load(Path(path).read_text())
    if not isinstance(c, dict) or c.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected protocol {PROTOCOL}")
    if c["data"]["dataset"] not in ("uzh_fpv", "synthetic"):
        raise ValueError("Retraining is restricted to UZH or synthetic tests")
    if len(c["data"]["resolution"]) != 2 or any(x < 16 or x % 16 for x in c["data"]["resolution"]):
        raise ValueError("Training resolution must be divisible by 16")
    m, t = c["model"], c["train"]
    if m["num_bins"] != 9 or m["base_num_channels"] < 1 or m["kernel_size"] not in (3, 5):
        raise ValueError("Zhu protocol requires nine bins and a valid encoder-decoder")
    for key in ("epochs", "batch_size", "events_per_input"):
        if not isinstance(t[key], int) or t[key] < (2 if key == "events_per_input" else 1):
            raise ValueError(f"Invalid {key}")
    if min(t["lr"], t["clip_grad"], t["charbonnier_eps"], t["splat_eps"]) <= 0 or t["smooth_weight"] < 0:
        raise ValueError("Invalid optimizer or loss settings")
    return c


class ZhuFlowNet(nn.Module):
    """4 encoders, 2 residual blocks, 4 flow-fed decoders (paper Fig. 2).

    Native outputs are pixels per temporal bin in each decoder's spatial grid.
    Widths, kernels, upsampling and linear heads are documented local choices.
    """
    def __init__(self, c):
        super().__init__()
        self.net = MultiResUNet(dict(
            num_bins=9, base_num_channels=c["model"]["base_num_channels"],
            num_encoders=4, num_residual_blocks=2, num_output_channels=2,
            skip_type="concat", norm=None, use_upsample_conv=True,
            kernel_size=c["model"]["kernel_size"], channel_multiplier=2,
            activations=["relu", None], final_activation=None,
        ))
        # A small nonzero head keeps initial warps in view and encoder gradients alive.
        for head in self.net.preds:
            for layer in head.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=1e-3)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, voxel):
        if voxel.shape[1] != 9 or any(n % 16 for n in voxel.shape[-2:]):
            raise ValueError("Expected nine-bin input with spatial dimensions divisible by 16")
        return self.net(voxel)


def resize_flow(flow, resolution):
    """Resize a native spatial velocity field, scaling x and y independently."""
    h, w = flow.shape[-2:]
    height, width = resolution
    scale = flow.new_tensor([width / w, height / h]).view(1, 2, 1, 1)
    return F.interpolate(flow, size=resolution, mode="bilinear", align_corners=False) * scale


def sample_flow(flow, events):
    """Bilinearly sample flow at floating (y,x) event positions."""
    h, w = flow.shape[-2:]
    grid = torch.stack((2 * events[..., 2] / (w - 1) - 1,
                        2 * events[..., 1] / (h - 1) - 1), dim=-1).unsqueeze(2)
    return F.grid_sample(flow, grid, mode="bilinear", padding_mode="border",
                         align_corners=True).squeeze(-1).transpose(1, 2)


def timestamp_image(events, positions, resolution, eps=1e-6):
    """Eq. 5: bilinear count-normalized timestamps, separately by polarity.

    events=(t_bin,y,x,polarity), positions=(x,y). Out-of-image splat neighbors
    contribute zero. The original timestamp is used for BOTH warp directions.
    """
    height, width = resolution
    b, n, _ = events.shape
    x, y = positions.unbind(-1)
    x0, y0 = torch.floor(x), torch.floor(y)
    xx = torch.stack((x0, x0 + 1, x0, x0 + 1), dim=-1)
    yy = torch.stack((y0, y0, y0 + 1, y0 + 1), dim=-1)
    weights = (1 - (x.unsqueeze(-1) - xx).abs()).clamp_min(0)
    weights = weights * (1 - (y.unsqueeze(-1) - yy).abs()).clamp_min(0)
    valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    weights = weights * valid
    pol = (events[..., 3] < 0).long().unsqueeze(-1)
    index = (pol * height * width + yy.long().clamp(0, height - 1) * width
             + xx.long().clamp(0, width - 1)).reshape(b, n * 4)
    count = events.new_zeros((b, 2 * height * width)).scatter_add(1, index, weights.reshape(b, -1))
    weighted_time = weights * events[..., 0].unsqueeze(-1)
    total = events.new_zeros((b, 2 * height * width)).scatter_add(1, index, weighted_time.reshape(b, -1))
    return (total / (count + eps)).reshape(b, 2, height, width), count.reshape(b, 2, height, width)


def temporal_loss(flow, events, resolution, num_bins=9, eps=1e-6):
    """Eqs. 4-7 at one decoder scale, without active-pixel normalization."""
    h, w = flow.shape[-2:]
    e = events.clone()
    e[..., 0] *= num_bins - 1
    e[..., 1] *= h / resolution[0]
    e[..., 2] *= w / resolution[1]
    velocity = sample_flow(flow, e)
    xy = torch.stack((e[..., 2], e[..., 1]), dim=-1)
    loss = flow.new_zeros(())
    for reference in (0., float(num_bins - 1)):
        warped = xy + (reference - e[..., 0]).unsqueeze(-1) * velocity
        image, _ = timestamp_image(e, warped, (h, w), eps)
        loss = loss + image.square().sum((1, 2, 3)).mean()
    return loss


def smoothness_loss(flow, eps=1e-3):
    """Eq. 8: sum over directed four-connected neighbors and u,v channels."""
    dx = flow[..., 1:] - flow[..., :-1]
    dy = flow[..., 1:, :] - flow[..., :-1, :]
    return 2 * (torch.sqrt(dx.square() + eps ** 2).sum((1, 2, 3))
                + torch.sqrt(dy.square() + eps ** 2).sum((1, 2, 3))).mean()


def loss_terms(flows, inputs, c):
    t = c["train"]
    temporal = sum(temporal_loss(f, inputs["event_list"], c["data"]["resolution"],
                                 eps=t["splat_eps"]) for f in flows)
    smooth = sum(smoothness_loss(f, t["charbonnier_eps"]) for f in flows)
    return temporal + t["smooth_weight"] * smooth, temporal, smooth


def window_batches(entries, c, augment=False, rng=None):
    """Fixed nonoverlapping windows, equal-length streams per batch; report tails."""
    rng = rng or random.Random(0)
    n, bs = c["train"]["events_per_input"], c["train"]["batch_size"]
    for offset in range(0, len(entries), bs):
        with ExitStack() as stack:
            streams = [stack.enter_context(EventFile(e["path"], c["data"]["resolution"]))
                       for e in entries[offset:offset + bs]]
            if len({s.size for s in streams}) != 1:
                raise ValueError("Use equal-length crops to avoid silent event truncation")
            flips = [tuple(augment and rng.random() < .5 for _ in range(3)) for s in streams]
            for a in range(0, streams[0].size - n + 1, n):
                inputs = batch([encode(s.read(a, a + n, f), c["data"]["resolution"], 9)
                                for s, f in zip(streams, flips)], c["device"])
                if torch.any(inputs["dt_input"] <= 0):
                    raise ValueError("Cannot train a zero-duration event window")
                yield inputs


@torch.no_grad()
def validation(model, entries, c):
    """Select only on held-out UZH RSAT, using upstream single-window RSAT."""
    model.eval()
    scores, kept = [], []
    for inputs in window_batches(entries, c):
        flow = model(inputs["event_voxel"])[-1]
        if not torch.isfinite(flow).all():
            raise FloatingPointError("Nonfinite validation flow")
        metric = RSAT({"loader": {"resolution": c["data"]["resolution"]},
                       "model": {"mask_output": True}, "loss": {}}, c["device"], flow_scaling=8.)
        metric.event_flow_association([flow], inputs)
        values = metric()
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite validation RSAT")
        scores.extend(values.cpu().tolist())
        e = inputs["event_list"]
        xy = torch.stack((e[..., 2], e[..., 1]), -1) + (1 - e[..., 0:1]) * 8 * sample_flow(flow, e)
        h, w = c["data"]["resolution"]
        in_view = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
        kept.extend(in_view.float().mean(1).cpu().tolist())
    if not scores:
        raise ValueError("No complete validation windows")
    return {"rsat": float(np.mean(scores)), "windows": len(scores),
            "forward_events_in_view_fraction": float(np.mean(kept))}


def save_state(path, model, optimizer, epoch, best, c, manifest_hash):
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": c, "epoch": epoch, "best_rsat": best, "manifest_sha256": manifest_hash,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, tmp)
    tmp.replace(path)


def train(c, output, resume=None, max_updates=None):
    from .models import EVENT_FLOW_SHA
    import subprocess
    manifest = load_manifest(c["data"]["manifest"])
    if manifest["dataset"] != c["data"]["dataset"] or not manifest["train"] or not manifest["val"]:
        raise ValueError("Expected explicit UZH training and validation groups")
    if max_updates is not None and (max_updates < 1 or resume is not None):
        raise ValueError("Smoke limit must be positive and cannot be resumed")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Choose a new output directory or resume explicitly")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(c["seed"])
    model = ZhuFlowNet(c).to(c["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=c["train"]["lr"])
    start_epoch, best, updates = 0, float("inf"), 0
    if resume:
        state = torch.load(resume, map_location=c["device"], weights_only=True)
        old = deepcopy(state["config"])
        old["train"]["epochs"], old["device"] = c["train"]["epochs"], c["device"]
        if old != c or state["manifest_sha256"] != digest(manifest):
            raise ValueError("Resume requires identical experiment and split")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = state["epoch"] + 1, state["best_rsat"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, timeout=10).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    source_hashes = {str(p.relative_to(Path(__file__).parent)): sha256(p)
                     for p in Path(__file__).parent.rglob("*.py")}
    write_json(output / "run.json", {"protocol": PROTOCOL, "result_kind": "paper_based_reimplementation",
        "config": c, "config_sha256": digest(c), "manifest": manifest, "manifest_sha256": digest(manifest),
        "git_revision": revision, "git_dirty": dirty, "source_sha256": source_hashes,
        "architecture_components_revision": EVENT_FLOW_SHA, "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name() if c["device"].startswith("cuda") else None,
        "parameters": sum(p.numel() for p in model.parameters()), "smoke_max_updates": max_updates})
    print(json.dumps({"status": "started", "protocol": PROTOCOL, "output": str(output),
                      "epochs": c["train"]["epochs"], "smoke_max_updates": max_updates}), flush=True)
    wall_start = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, c["train"]["epochs"]):
        started = time.monotonic()
        entries = list(manifest["train"])
        rng = random.Random(c["seed"] + epoch)
        rng.shuffle(entries)
        model.train()
        losses, samples = [], 0
        for inputs in window_batches(entries, c, c["train"]["augment"], rng):
            optimizer.zero_grad(set_to_none=True)
            flows = model(inputs["event_voxel"])
            loss, temporal, smooth = loss_terms(flows, inputs, c)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Zhu loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), c["train"]["clip_grad"], error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            samples += len(inputs["event_voxel"])
            losses.append(float(loss.detach()))
            if updates % 20 == 0 or updates == 1:
                print(json.dumps({"epoch": epoch, "update": updates, "loss": losses[-1],
                                  "temporal": float(temporal.detach()), "smooth": float(smooth.detach()),
                                  "gradient_norm_before_clip": float(grad),
                                  "elapsed_s": time.monotonic() - wall_start}), flush=True)
            if max_updates is not None and updates >= max_updates:
                probe = validation(model, manifest["val"][:1], c)
                result = {"status": "smoke_complete_not_full_training", "updates": updates,
                          "first_loss": losses[0], "last_loss": losses[-1], "validation": probe,
                          "seconds_including_probe": time.monotonic() - wall_start,
                          "cuda_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
                write_json(output / "smoke.json", result)
                print(json.dumps(result), flush=True)
                return result
        if not losses:
            raise ValueError("No training windows")
        train_seconds = time.monotonic() - started
        val = validation(model, manifest["val"], c)
        improved = val["rsat"] < best
        best = min(best, val["rsat"])
        save_state(output / "latest.pt", model, optimizer, epoch, best, c, digest(manifest))
        if improved:
            save_state(output / "best.pt", model, optimizer, epoch, best, c, digest(manifest))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "updates": len(losses),
                  "trained_events": samples * c["train"]["events_per_input"], "validation": val,
                  "best_rsat": best, "train_seconds": train_seconds, "seconds": time.monotonic() - started}
        with (output / "epochs.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
    write_json(output / "completed.json", {"status": "completed", "epochs": c["train"]["epochs"],
                                          "best_rsat": best, "seconds": time.monotonic() - wall_start})
    return str(output / "best.pt")


def displacement(flow, input_seconds, gt_seconds):
    if input_seconds <= 0 or gt_seconds <= 0:
        raise ValueError("Input and GT durations must be positive")
    return flow * (8. * gt_seconds / input_seconds)


@torch.no_grad()
def evaluate(checkpoint, manifest_path, output, device="cuda", dt=1):
    """Causal fixed-count history at each GT endpoint, with explicit GT-interval mask."""
    if dt not in (1, 4):
        raise ValueError("dt must be 1 or 4")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    c = state["config"]
    if c["protocol"] != PROTOCOL:
        raise ValueError("Wrong checkpoint protocol")
    manifest = load_manifest(manifest_path)
    if manifest["dataset"] != "mvsec" or not manifest["test"]:
        raise ValueError("Expected MVSEC test manifest")
    model = ZhuFlowNet(c).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    rows, sequences = [], {}
    for entry in manifest["test"]:
        name = entry.get("id", Path(entry["path"]).stem)
        if "outdoor_day1" in name:
            n = 30000
        elif any(f"indoor_flying{i}" in name for i in (1, 2, 3)):
            n = 15000
        else:
            raise ValueError(f"Unexpected MVSEC sequence: {name}")
        current, skipped = [], 0
        with EventFile(entry["path"], [256, 256]) as stream:
            for t0, t1, gt, valid in stream.gt_intervals(f"gtflow_dt{dt}"):
                a, b = stream.index(t0), stream.index(t1)
                if b < n or b <= a:
                    skipped += 1
                    continue
                inputs = batch([encode(stream.read(b - n, b), [256, 256], 9)], device)
                flow = model(inputs["event_voxel"])[-1]
                pred = displacement(flow, float(inputs["dt_input"]), t1 - t0)
                interval = encode(stream.read(a, b), [256, 256], 9)
                mask = interval["event_mask"].to(device).bool()
                gt, valid = gt.to(device), valid.to(device)
                score = flow_metrics(pred, gt, valid, mask, "gt")
                active = valid & mask & torch.isfinite(gt).all(1)
                err = torch.linalg.vector_norm(pred - torch.nan_to_num(gt), dim=1)
                score["outlier_pct_3px"] = float((err[active] > 3).float().mean() * 100) if active.any() else None
                current.append({"sequence": name, "timestamp_start": t0, "timestamp_end": t1,
                                "start_event": a, "end_event": b, "input_start_event": b - n, **score})
        summary = summarize(current)
        values = [r["outlier_pct_3px"] for r in current if r["outlier_pct_3px"] is not None]
        summary["outlier_pct_3px"] = float(np.mean(values)) if values else None
        summary["skipped_incomplete_history_or_empty_interval"] = skipped
        sequences[name] = summary
        rows.extend(current)
        print(json.dumps({"sequence": name, **summary}), flush=True)
    if not rows:
        raise ValueError("No valid evaluation windows")
    result = {"protocol": PROTOCOL, "result_kind": "paper_based_reimplementation",
              "training_dataset": c["data"]["dataset"], "training_config": c,
              "training_manifest_sha256": state["manifest_sha256"], "checkpoint_sha256": sha256(checkpoint),
              "dt": dt, "summary": summarize(rows), "sequences": sequences, "rows": rows,
              "evaluation": {"resolution": [256, 256], "flow_unit": "pixels_per_bin",
                  "history_outdoor": 30000, "history_indoor": 15000, "causal": True,
                  "mask": "valid GT AND event in GT interval", "warmup": "require complete input history",
                  "outlier_pct": "EPE>3px AND EPE>5% of GT", "outlier_pct_3px": "EPE>3px",
                  "aggregation": "unweighted mean of valid GT intervals",
                  "matches_hagenaars_upstream_protocol": False}}
    write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--resume"); p.add_argument("--max-updates", type=int)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--dt", type=int, choices=[1, 4], required=True)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "train":
        train(read_config(args.config), args.output, args.resume, args.max_updates)
    else:
        evaluate(args.checkpoint, args.manifest, args.output, args.device, args.dt)


if __name__ == "__main__":
    main()
