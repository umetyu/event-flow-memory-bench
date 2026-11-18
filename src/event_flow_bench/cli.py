import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
import yaml
from .config import read_config, write_json, validate
from .data import inspect_file, split_manifest, load_manifest
from .engine import evaluate, load_checkpoint, train, sha256
from .experiments import generate_grid, collect, compare_paper
from .fixtures import make_demo
from .models import MODEL_NAMES

def main(argv=None):
    parser = argparse.ArgumentParser(description="Event-flow memory baseline benchmark")
    parser.add_argument("--threads", type=int, default=4, help="CPU worker threads, including CPU ops feeding CUDA")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("models"); p.set_defaults(command="models")
    p = sub.add_parser("inspect")
    p.add_argument("paths", nargs="+"); p.add_argument("--resolution", nargs=2, type=int, required=True)
    p = sub.add_parser("inventory")
    p.add_argument("root"); p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--groups", help="JSON mapping relative HDF5 filenames to original recording IDs; required for train")
    p.add_argument("--output", required=True)
    p = sub.add_parser("split")
    p.add_argument("inventory"); p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--val-fraction", type=float, default=.1)
    p = sub.add_parser("grid")
    p.add_argument("--stage", choices=["pilot", "main", "snn", "layerseg"], default="pilot")
    p.add_argument("--dataset", choices=["uzh_fpv", "affineobjects"], default="uzh_fpv")
    p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    p.add_argument("--histories", nargs="+", type=int, default=[1000])
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("train")
    p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p = sub.add_parser("evaluate")
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True); p.add_argument("--split", choices=["val", "test"], default="test")
    p = sub.add_parser("eval-config")
    p.add_argument("--train-config", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--dataset", choices=["mvsec", "ecd", "hqf", "affineobjects"], required=True)
    p.add_argument("--mode", choices=["events", "gtflow_dt1", "gtflow_dt4", "masks"], required=True)
    p.add_argument("--resolution", nargs=2, type=int, required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("collect")
    p.add_argument("results", nargs="+"); p.add_argument("--output", required=True)
    p = sub.add_parser("compare-paper")
    p.add_argument("result"); p.add_argument("--reference", default="configs/paper_reference.csv")
    p.add_argument("--output", required=True); p.add_argument("--aee-tol", type=float, default=.02)
    p.add_argument("--outlier-tol", type=float, default=.2)
    p = sub.add_parser("reproduce")
    p.add_argument("--data", required=True); p.add_argument("--checkpoint")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir"); group.add_argument("--train-config")
    p.add_argument("--dt", type=int, choices=[1,4], default=1); p.add_argument("--device", default="cpu")
    p.add_argument("--resolution", nargs=2, type=int, default=[256,256])
    p.add_argument("--trust-legacy-pickle", action="store_true"); p.add_argument("--output", required=True)
    p = sub.add_parser("demo")
    p.add_argument("--output", default="runs/demo")
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    command = args.command
    if command == "models":
        print("\n".join((*MODEL_NAMES, "EVLayerSegNet (optional pinned checkout)")))
    elif command == "inspect":
        print(json.dumps([inspect_file(p, args.resolution) for p in args.paths], indent=2))
    elif command == "inventory":
        root = Path(args.root).resolve()
        groups = json.loads(Path(args.groups).read_text()) if args.groups else None
        if args.split == "train" and groups is None:
            parser.error("Training inventory requires --groups; crop filenames do not prove recording independence")
        entries = []
        for p in sorted(root.rglob("*.h5")):
            relative = p.relative_to(root).as_posix()
            entries.append({"path": str(p), "group": groups[relative] if groups else relative, "id": relative})
        if not entries:
            parser.error("No .h5 files found")
        write_json(args.output, {"version": 1, "dataset": args.dataset, args.split: entries})
        load_manifest(args.output)
    elif command == "split":
        result = split_manifest(args.inventory, args.output, args.val_fraction, args.seed)
        print(json.dumps({k: len(result[k]) for k in ("train", "val", "test")}))
    elif command == "grid":
        jobs = generate_grid(args.output, args.manifest, args.stage, args.dataset, args.seeds, args.histories, args.device)
        print(f"Generated {len(jobs)} jobs in {args.output}; no training submitted")
    elif command == "train":
        print(train(read_config(args.config), args.output, args.resume))
    elif command == "evaluate":
        c = read_config(args.config)
        model, state = load_checkpoint(args.checkpoint, c)
        result = evaluate(c, model, args.split)
        result["checkpoint_sha256"] = sha256(args.checkpoint)
        result["training_config"] = state["config"]
        result["training_manifest_sha256"] = state["manifest_sha256"]
        write_json(args.output, result)
        print(json.dumps(result["summary"], indent=2))
    elif command == "eval-config":
        c = read_config(args.train_config)
        c["data"].update({"manifest": args.manifest, "dataset": args.dataset, "resolution": args.resolution})
        c["eval"]["mode"] = args.mode
        validate(c)
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(c, sort_keys=False))
    elif command == "collect":
        print(f"Wrote {len(collect(args.results, args.output))} rows")
    elif command == "compare-paper":
        result = compare_paper(args.result, args.reference, args.output, args.aee_tol, args.outlier_tol)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    elif command == "reproduce":
        from .reproduce import read_training_config, reproduce
        training = read_training_config(args.run_dir, args.train_config)
        checkpoint = args.checkpoint or (Path(args.run_dir) / "artifacts/model/data/model.pth" if args.run_dir else None)
        if checkpoint is None:
            parser.error("--checkpoint is required with --train-config")
        result = reproduce(args.data, checkpoint, training, args.dt, args.device, args.resolution,
                           args.trust_legacy_pickle, args.output)
        print(json.dumps(result["summary"], indent=2))
    elif command == "demo":
        root = Path(args.output)
        c = make_demo(root / "fixtures")
        checkpoint = train(c, root / "training")
        model, _ = load_checkpoint(checkpoint, c)
        result = evaluate(c, model, "test", root / "test.json")
        print("SYNTHETIC ENGINEERING CHECK ONLY: " + json.dumps(result["summary"]))

if __name__ == "__main__":
    main()
