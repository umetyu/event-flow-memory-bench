"""Run one member of the matched feedforward count/voxel5 pair on Izar."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from event_flow_bench.config import digest, read_config, validate, write_json
from event_flow_bench.data import load_manifest
from event_flow_bench.engine import evaluate, load_checkpoint, sha256, train

CONFIGS = ("configs/uzh_evflownet_count.yml", "configs/uzh_evflownet_voxel5.yml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume latest.pt in the same output directory")
    parser.add_argument("--mvsec-manifest", default="runs/setup/mvsec.json")
    args = parser.parse_args()
    if args.threads < 1 or (args.smoke and args.resume):
        parser.error("threads must be positive; smoke and resume cannot be combined")
    torch.set_num_threads(args.threads)
    pair = [read_config(p) for p in CONFIGS]
    controls = deepcopy(pair)
    for c in controls:
        del c["model"]["encoding"], c["model"]["num_bins"]
    if controls[0] != controls[1]:
        raise ValueError("Count and voxel5 settings differ beyond encoding/channel count")
    c = pair[args.index]
    if not torch.cuda.is_available():
        raise RuntimeError("This launch requires a CUDA allocation")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Nonempty output directory: {output}")
    manifest = load_manifest(c["data"]["manifest"])
    if args.smoke:
        if len(manifest["train"]) < 8 or not manifest["val"]:
            raise ValueError("Smoke requires eight training crops and a held-out validation crop")
        small = {**manifest, "train": manifest["train"][:8], "val": manifest["val"][:1], "test": []}
        smoke_manifest = output.parent / f"{output.name}-smoke-manifest.json"
        write_json(smoke_manifest, small)
        c["data"]["manifest"] = str(smoke_manifest)
        c["train"]["epochs"] = 1
        manifest = load_manifest(smoke_manifest)
    validate(c)
    print(json.dumps({"phase": "start", "kind": "smoke_only" if args.smoke else "full_training",
                      "config": CONFIGS[args.index], "config_sha256": digest(c),
                      "manifest_sha256": digest(manifest), "train_crops": len(manifest["train"]),
                      "val_crops": len(manifest["val"]), "epochs": c["train"]["epochs"],
                      "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__)}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    best = train(c, output, output / "latest.pt" if args.resume else None)
    training_seconds = time.monotonic() - started
    epochs = [json.loads(line) for line in (output / "epochs.jsonl").read_text().splitlines()]
    if epochs[-1]["epoch"] != c["train"]["epochs"] - 1:
        raise RuntimeError("Training did not reach the requested last epoch")
    record = {"status": "completed", "kind": "smoke_only" if args.smoke else "full_training_and_mvsec",
              "model": c["model"], "seed": c["seed"], "epochs": c["train"]["epochs"],
              "config_sha256": digest(c), "training_manifest_sha256": digest(manifest),
              "best_checkpoint_sha256": sha256(best), "training_seconds_this_invocation": training_seconds,
              "peak_training_gpu_bytes": torch.cuda.max_memory_allocated(), "last_epoch": epochs[-1]}
    if args.smoke:
        write_json(output / "smoke.json", record)
        print(json.dumps(record), flush=True)
        return
    print(json.dumps({"phase": "training_completed", "epochs": c["train"]["epochs"],
                      "best_rsat": epochs[-1]["best_rsat"], "training_seconds": training_seconds}), flush=True)
    summaries = {}
    for dt in (1, 4):
        evaluation = deepcopy(c)
        evaluation["data"].update({"dataset": "mvsec", "manifest": args.mvsec_manifest,
                                   "resolution": [256, 256]})
        evaluation["eval"]["mode"] = f"gtflow_dt{dt}"
        validate(evaluation)
        write_json(output / f"mvsec-dt{dt}-config.json", evaluation)
        model, state = load_checkpoint(best, evaluation)
        result = evaluate(evaluation, model, "test")
        result.update({"checkpoint_sha256": record["best_checkpoint_sha256"],
                       "training_config": state["config"], "training_manifest_sha256": state["manifest_sha256"]})
        write_json(output / f"mvsec-dt{dt}.json", result)
        summaries[f"dt{dt}"] = result["summary"]
        print(json.dumps({"phase": "evaluation_completed", "dt": dt, "summary": result["summary"]}), flush=True)
        del model, state
    record.update({"seconds_this_invocation": time.monotonic() - started, "mvsec": summaries})
    write_json(output / "completed.json", record)


if __name__ == "__main__":
    main()
