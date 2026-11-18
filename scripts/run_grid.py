"""Run one ordered stage, or one Slurm array item. Does not submit jobs itself."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("jobs")
p.add_argument("--index", type=int)
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()
jobs = json.loads(Path(args.jobs).read_text())
if args.index is not None:
    if not 0 <= args.index < len(jobs):
        raise SystemExit("index out of range")
    jobs = [jobs[args.index]]
for job in jobs:
    command = [sys.executable, "-m", "event_flow_bench", "train", "--config", job["config"], "--output", job["run_dir"]]
    print(command, flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)
