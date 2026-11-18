"""Fetch reproducible reference checkouts without modifying any existing checkout."""
import argparse
from pathlib import Path
import subprocess

SOURCES = {
    "event_flow": ("https://github.com/tudelft/event_flow.git", "e81f9630a145da8c7e57b2cfd38cf6c0b47ac14f"),
    "event_segmentation": ("https://github.com/Fulmen67/event_segmentation.git", "a6e28dce82720f41c0e62781e01d015a8f3b2a26"),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=[*SOURCES, "all"])
    p.add_argument("--root", default="external")
    args = p.parse_args()
    for name in SOURCES if args.source == "all" else [args.source]:
        url, sha = SOURCES[name]
        target = Path(args.root).resolve() / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", url, str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", sha], check=True)
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != sha:
            raise SystemExit(f"Existing {target} is at {actual}; expected {sha}. It has not been changed.")
        print(f"{name}: {sha} at {target}")

if __name__ == "__main__":
    main()
