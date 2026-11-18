#!/usr/bin/env python3
"""Time real-data training updates with the pinned Hagenaars loader/model/loss.

This is a bounded throughput probe, not a training run or accuracy experiment.
No checkpoint from this probe is used for evaluation.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time

import h5py
import numpy as np
import torch
import yaml

from event_flow_bench._vendor.event_flow.dataloader.h5 import H5Loader
from event_flow_bench._vendor.event_flow.loss.flow import EventWarping
from event_flow_bench._vendor.event_flow.models import model as architectures
from event_flow_bench.config import write_json
from event_flow_bench.models import EVENT_FLOW_SHA, MODEL_NAMES


def benchmark(config, model_name, data_path, warmup, updates, device, threads):
    c = deepcopy(config)
    c["data"]["path"] = str(Path(data_path).resolve())
    c["model"]["name"] = model_name
    c["model"]["spiking_neuron"] = c.get("spiking_neuron")
    c["loader"].setdefault("seed", 0)
    c["vis"].update(bars=False, enabled=False, verbose=False, store_grads=False)
    if c["data"]["mode"] != "events":
        raise ValueError("Throughput probe requires fixed event-count training")
    files = sorted(Path(data_path).rglob("*.h5"))
    batch = c["loader"]["batch_size"]
    if len(files) < batch:
        raise ValueError("Data must contain at least one full batch of sequences")
    total_events = 0
    for path in files:
        with h5py.File(path, "r") as f:
            total_events += len(f["events/ts"])
    torch.set_num_threads(threads)
    random.seed(c["loader"]["seed"])
    np.random.seed(c["loader"]["seed"])
    torch.manual_seed(c["loader"]["seed"])
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run in a GPU allocation")

    def synchronize():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    net = getattr(architectures, model_name)(deepcopy(c["model"])).to(dev).train()
    data = H5Loader(c, c["model"]["num_bins"], c["model"].get("round_encoding", False))
    # The upstream loader mutates stream state; workers must stay at zero.
    loader = torch.utils.data.DataLoader(data, batch_size=batch, drop_last=True,
                                        collate_fn=data.custom_collate, num_workers=0,
                                        pin_memory=dev.type == "cuda")
    objective = EventWarping(c, dev)
    opt = getattr(torch.optim, c["optimizer"]["name"])(net.parameters(), lr=c["optimizer"]["lr"])
    opt.zero_grad()
    data.shuffle()
    elapsed, losses = [], []
    completed, forwards, resets = 0, 0, 0
    iterator = iter(loader)
    synchronize()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    try:
        while completed < warmup + updates:
            try:
                inputs = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs = next(iterator)
            if data.new_seq:
                data.new_seq = False
                objective.reset()
                net.reset_states()
                opt.zero_grad()
                resets += 1
            output = net(inputs["event_voxel"].to(dev), inputs["event_cnt"].to(dev))
            forwards += 1
            objective.event_flow_association(output["flow"], inputs["event_list"].to(dev),
                                             inputs["event_list_pol_mask"].to(dev),
                                             inputs["event_mask"].to(dev))
            if objective.num_events < c["data"]["window_loss"]:
                continue
            if c["loss"]["overwrite_intermediate"]:
                objective.overwrite_intermediate_flow(output["flow"])
            loss = objective()
            value = loss.item()  # same synchronization point as upstream training
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite probe loss")
            loss.backward()
            if c["loss"]["clip_grad"] is not None:
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), c["loss"]["clip_grad"])
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm")
            opt.step()
            opt.zero_grad()
            net.detach_states()
            objective.reset()
            synchronize()
            ended = time.perf_counter()
            if completed >= warmup:
                elapsed.append(ended - started)
                losses.append(value)
            completed += 1
            if completed % 20 == 0 or completed == warmup + updates:
                print(f"{model_name}: {completed}/{warmup + updates} updates", flush=True)
            started = time.perf_counter()
        mean = statistics.mean(elapsed)
        nominal_updates = total_events * c["loader"]["n_epochs"] / (batch * c["data"]["window_loss"])
        result = {
            "kind": "training_throughput_probe", "model": model_name, "config": c,
            "upstream_revision": EVENT_FLOW_SHA, "python": platform.python_version(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cpu_threads": threads,
            "dtype": "float32", "warmup_updates": warmup, "measured_updates": updates,
            "forward_passes": forwards, "sequence_resets": resets,
            "sequence_count": len(files), "events_per_epoch": total_events,
            "seconds_per_update_mean": mean, "seconds_per_update_median": statistics.median(elapsed),
            "seconds_per_update_min": min(elapsed), "seconds_per_update_max": max(elapsed),
            "seconds_per_update": elapsed, "losses": losses,
            "nominal_full_training_updates": nominal_updates,
            "estimated_full_training_hours": nominal_updates * mean / 3600,
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else None,
            "notes": ["Includes HDF5 reads, encodings, CPU-to-GPU copies, forward, loss, backward and optimizer",
                      "Excludes startup, discarded warmup, checkpoint writing, validation and queue time",
                      "Nominal update count ignores sequence-boundary tail/reset effects",
                      "Random initialization; probe weights are discarded; no accuracy claim"],
        }
        return result
    finally:
        for f in data.open_files:
            f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/upstream/train_ANN.yml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["FireNet", "RecEVFlowNet"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.updates <= 0 or args.threads <= 0:
        parser.error("warmup >= 0, updates > 0 and threads > 0 required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    for name in args.models:
        result = benchmark(config, name, args.data, args.warmup, args.updates, args.device, args.threads)
        result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_json(output / f"{name}.json", result)
        print(json.dumps({k: result[k] for k in ("model", "seconds_per_update_mean", "estimated_full_training_hours", "peak_gpu_allocated_gib")}))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
