"""Sequence-safe training and evaluation for the matched experimental protocol."""
from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from PIL import Image
from .config import digest, upstream_config, validate, write_json
from .data import EventFile, batch, load_manifest, sample
from .metrics import WarpMetrics, flow_metrics, mask_metrics, summarize
from .models import EVENT_FLOW_SHA, LAYERSEG_SHA, build_model
from ._vendor.event_flow.loss.flow import EventWarping

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()

def provenance(c, manifest):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain"], text=True, timeout=10).strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        revision, dirty = None, None
    return {"config": c, "config_sha256": digest(c), "manifest": manifest,
            "manifest_sha256": digest(manifest), "git_revision": revision, "git_dirty": dirty,
            "torch": str(torch.__version__), "numpy": np.__version__,
            "event_flow_revision": EVENT_FLOW_SHA, "layerseg_revision": LAYERSEG_SHA,
            "flow_units": "output * flow_scale = displacement over CURRENT stride, not input history",
            "hot_pixel_filter": False, "time_axis": "per-stride normalized timestamps (upstream convention)"}

def _forward(model, stream, start, end, c):
    inputs = batch([sample(stream, start, end, c)], c["device"])
    out = model(inputs)
    if any(not torch.isfinite(f).all() for f in out["flow"]):
        raise FloatingPointError(f"Nonfinite prediction: {stream.path} at event {end}")
    return out, inputs

def _event_rows(model, stream, c):
    w = c["window"]
    stride = w["stride_events"]
    metrics, count, row_start = None, 0, None
    for start in range(0, stream.size - stride + 1, stride):
        end = start + stride
        out, inputs = _forward(model, stream, start, end, c)
        if start < w["warmup_events"]:
            continue
        if float(inputs["dt_input"]) <= 0:
            raise ValueError("A stride has no positive timestamp span; cannot define displacement")
        if metrics is None:
            metrics, row_start = WarpMetrics(c), start
        metrics.add(out["flow"], inputs)
        count += stride
        if count == w["eval_events"]:
            yield {"start_event": row_start, "end_event": end, **metrics.compute()}
            metrics, count = None, 0
    # All models use the same anchors. Incomplete score windows are deliberately omitted.

def _gt_rows(model, stream, c):
    stride, warmup = c["window"]["stride_events"], c["window"]["warmup_events"]
    cursor = 0
    for ts0, ts1, gt, valid in stream.gt_intervals(c["eval"]["mode"]):
        a, b = stream.index(ts0), stream.index(ts1)
        if a < cursor:
            raise ValueError("Overlapping GT intervals need an explicit evaluation policy")
        # Maintain state on unscored prefix/gaps, without ever reading future events.
        while cursor < a:
            end = min(a, cursor + stride)
            _forward(model, stream, cursor, end, c)
            cursor = end
        mask = torch.zeros((1, *stream.resolution), dtype=torch.bool, device=c["device"])
        last, last_inputs = None, None
        while cursor < b:
            end = min(b, cursor + stride)
            last, last_inputs = _forward(model, stream, cursor, end, c)
            mask |= last_inputs["event_mask"][:, 0].bool()
            cursor = end
        if a < warmup:
            continue
        row = {"start_event": a, "end_event": b, "timestamp_start": ts0, "timestamp_end": ts1}
        if last is None or float(last_inputs["dt_input"]) <= 0:
            yield {**row, "aee": None, "outlier_pct": None, "valid_pixels": 0, "status": "empty_or_zero_duration"}
            continue
        displacement = last["flow"][-1] * c["eval"]["flow_scale"] * ((ts1 - ts0) / float(last_inputs["dt_input"]))
        yield {**row, **flow_metrics(displacement, gt.to(c["device"]), valid.to(c["device"]), mask,
                                    c["eval"]["outlier_reference"])}

def _mask_rows(model, stream, c, entry):
    if not model.layerseg:
        raise ValueError("Mask scores require EVLayerSegNet; flow-only networks have no segmentation head")
    if not entry.get("masks"):
        raise ValueError("masks mode requires explicit timestamp/path annotations in the manifest")
    w = c["window"]
    for annotation in sorted(entry["masks"], key=lambda x: x["timestamp"]):
        end = stream.index(annotation["timestamp"])
        start = end - w["eval_events"] - w["warmup_events"]
        if start < 0:
            continue
        model.reset()
        metrics = WarpMetrics(c)
        for a in range(start, end, w["stride_events"]):
            out, inputs = _forward(model, stream, a, a + w["stride_events"], c)
            if a >= start + w["warmup_events"]:
                metrics.add(out["flow"], inputs)
        with Image.open(annotation["path"]) as image:
            gt = np.asarray(image.convert("L")) > 128
        channel = c["eval"]["mask_channel"]
        if channel not in (0, 1):
            raise ValueError("mask_channel must be fixed to 0 or 1 using validation data")
        pred = out["alpha_masks"][0, channel].cpu().numpy() > 0
        yield {"start_event": end - w["eval_events"], "end_event": end,
               "timestamp_end": annotation["timestamp"], **metrics.compute(), **mask_metrics(pred, gt)}

@torch.no_grad()
def evaluate(c, model, split="test", output=None):
    validate(c)
    manifest = load_manifest(c["data"]["manifest"])
    if manifest.get("dataset") != c["data"]["dataset"]:
        raise ValueError("Config dataset and manifest dataset differ")
    if not manifest[split]:
        raise ValueError(f"No {split} entries in manifest")
    model.eval()
    rows, by_sequence = [], {}
    for entry in manifest[split]:
        model.reset()
        with EventFile(entry["path"], c["data"]["resolution"]) as stream:
            mode = c["eval"]["mode"]
            if mode.startswith("gtflow"):
                current = list(_gt_rows(model, stream, c))
            elif mode == "masks":
                current = list(_mask_rows(model, stream, c, entry))
            else:
                current = list(_event_rows(model, stream, c))
            sequence = entry.get("id", str(Path(entry["path"]).resolve()))
            if sequence in by_sequence:
                raise ValueError("Sequence IDs must be unique")
            for row in current:
                row["sequence"] = sequence
            info = summarize(current)
            info.update({"total_events": stream.size, "scored_events": sum(r["end_event"] - r["start_event"] for r in current),
                         "warmup_events": c["window"]["warmup_events"]})
            by_sequence[sequence] = info
            rows.extend(current)
    model.reset()
    if not rows:
        raise ValueError("No complete evaluation windows after warmup; check interval and sequence lengths")
    result = {"protocol": c["protocol"], "split": split, "aggregation": "unweighted mean of scored windows",
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "summary": summarize(rows), "sequences": by_sequence, "rows": rows,
              "provenance": provenance(c, manifest)}
    if output:
        write_json(output, result)
    return result

def save_checkpoint(path, model, optimizer, epoch, c, best_rsat, manifest_hash):
    state = {"state_dict": model.net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
             "config": c, "best_rsat": best_rsat, "manifest_sha256": manifest_hash,
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)

def load_checkpoint(path, c):
    state = torch.load(path, map_location=c["device"], weights_only=True)
    if state["config"]["protocol"] != c["protocol"] or state["config"]["eval"]["flow_scale"] != c["eval"]["flow_scale"]:
        raise ValueError("Checkpoint protocol or flow units differ from evaluation config")
    if state["config"]["model"] != c["model"]:
        raise ValueError("Checkpoint model/encoding differs from evaluation config")
    if state["config"]["window"]["history_events"] != c["window"]["history_events"]:
        raise ValueError("Checkpoint history differs; history sweeps require matched training runs")
    if state["config"]["window"]["stride_events"] != c["window"]["stride_events"]:
        raise ValueError("Checkpoint stride differs")
    model = build_model(c)
    model.net.load_state_dict(state["state_dict"], strict=True)
    return model, state

def train(c, output, resume=None):
    validate(c)
    if c["data"]["dataset"] not in ("uzh_fpv", "affineobjects", "synthetic"):
        raise ValueError("Only UZH-FPV, AffineObjects or synthetic fixtures can be used for training")
    if c["eval"]["mode"] != "events":
        raise ValueError("Model selection must use held-out event-only RSAT")
    manifest = load_manifest(c["data"]["manifest"])
    if not manifest["train"] or not manifest["val"]:
        raise ValueError("Training requires explicit, disjoint train and val groups")
    if manifest.get("dataset") != c["data"]["dataset"]:
        raise ValueError("Config dataset and manifest dataset differ")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError("Run directory is nonempty; choose a new directory or resume")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(c["seed"])
    model = build_model(c)
    optimizer = torch.optim.Adam(model.parameters(), lr=c["train"]["lr"])
    start_epoch, best = 0, float("inf")
    if resume:
        model, state = load_checkpoint(resume, c)
        old = deepcopy(state["config"])
        old["train"]["epochs"] = c["train"]["epochs"]
        old["device"] = c["device"]
        if old != c or state["manifest_sha256"] != digest(manifest):
            raise ValueError("Resume requires unchanged experiment and data (epochs/device may change)")
        optimizer = torch.optim.Adam(model.parameters(), lr=c["train"]["lr"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = state["epoch"] + 1, state["best_rsat"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    write_json(output / "run.json", provenance(c, manifest))
    w = c["window"]
    stride, steps_per_loss = w["stride_events"], w["loss_events"] // w["stride_events"]
    for epoch in range(start_epoch, c["train"]["epochs"]):
        started = time.monotonic()
        rng = random.Random(c["seed"] + epoch)
        entries = list(manifest["train"])
        rng.shuffle(entries)
        model.train()
        losses, trained_events, dropped_events = [], 0, 0
        for offset in range(0, len(entries), c["train"]["batch_size"]):
            group = entries[offset:offset + c["train"]["batch_size"]]
            with ExitStack() as stack:
                streams = [stack.enter_context(EventFile(e["path"], c["data"]["resolution"])) for e in group]
                length = min(s.size for s in streams)
                flips = [tuple(c["train"]["augment"] and rng.random() < 0.5 for _ in range(3)) for _ in streams]
                model.reset()
                optimizer.zero_grad(set_to_none=True)
                criterion = EventWarping(upstream_config(c), c["device"], flow_scaling=c["eval"]["flow_scale"])
                steps = 0
                usable = max(0, (length - w["warmup_events"]) // w["loss_events"]) * w["loss_events"]
                stop = min(length // stride * stride, w["warmup_events"] + usable)
                dropped_events += sum(s.size - min(s.size, w["warmup_events"] + usable) for s in streams)
                for a in range(0, stop, stride):
                    inputs = batch([sample(s, a, a + stride, c, f) for s, f in zip(streams, flips)], c["device"])
                    if a < w["warmup_events"]:
                        with torch.no_grad():
                            model(inputs)
                        continue
                    if torch.any(inputs["dt_input"] <= 0):
                        raise ValueError("Zero-duration training stride")
                    out = model(inputs)
                    criterion.event_flow_association(out["flow"], inputs["event_list"].clone(),
                                                     inputs["event_list_pol_mask"], inputs["event_mask"])
                    steps += 1
                    if steps == steps_per_loss:
                        # Upstream sums over batch. Divide to make a short final batch comparable.
                        loss = criterion() / len(streams)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite contrast loss; no checkpoint written for this update")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), c["train"]["clip_grad"], error_if_nonfinite=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        model.detach()
                        criterion.reset()
                        steps = 0
                        losses.append(float(loss.detach()))
                        trained_events += len(streams) * w["loss_events"]
                        if len(losses) == 1 or len(losses) % 100 == 0:
                            print(json.dumps({"phase": "training", "epoch": epoch, "updates": len(losses),
                                              "last_loss": losses[-1], "trained_events": trained_events,
                                              "seconds": time.monotonic() - started}), flush=True)
        if not losses:
            raise ValueError("No training updates; sequences must exceed warmup + loss_events")
        print(json.dumps({"phase": "validation", "epoch": epoch, "updates": len(losses),
                          "train_seconds": time.monotonic() - started}), flush=True)
        validation = evaluate(c, model, "val", output / f"validation-{epoch:03}.json")
        score = validation["summary"].get("rsat")
        if score is None or validation["summary"].get("rsat_invalid", 0):
            raise FloatingPointError("Validation RSAT contains invalid windows; checkpoint selection aborted")
        improved = score < best
        best = min(best, score)
        save_checkpoint(output / "latest.pt", model, optimizer, epoch, c, best, digest(manifest))
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, epoch, c, best, digest(manifest))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_rsat": score,
                  "best_rsat": best, "updates": len(losses), "trained_events": trained_events,
                  "dropped_tail_events": dropped_events, "seconds": time.monotonic() - started}
        with (output / "epochs.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    return output / "best.pt"
