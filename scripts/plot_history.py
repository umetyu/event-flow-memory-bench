"""Plot aggregated history curves, separated by scoring policy and sequence."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("csv"); p.add_argument("--metric", default="aee"); p.add_argument("--output", required=True)
args = p.parse_args()
with open(args.csv) as f:
    rows = [r for r in csv.DictReader(f) if r["metric"] == args.metric]
if not rows:
    raise SystemExit("No matching metric rows")
output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
for policy, sequence in sorted({(r["policy_id"], r["sequence"]) for r in rows}):
    fig, ax = plt.subplots(figsize=(7, 4))
    subset = [r for r in rows if r["policy_id"] == policy and r["sequence"] == sequence]
    for model, encoding in sorted({(r["model"], r["encoding"]) for r in subset}):
        values = sorted([r for r in subset if r["model"] == model and r["encoding"] == encoding], key=lambda x: int(x["history_events"]))
        ax.errorbar([int(r["history_events"]) for r in values], [float(r["mean"]) for r in values],
                    yerr=[float(r["std"] or 0) for r in values], marker="o", label=f"{model}/{encoding}")
    ax.set(xscale="log", xlabel="Input history (events)", ylabel=args.metric, title=Path(sequence).stem)
    ax.legend(fontsize=7); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output / f"{policy}-{Path(sequence).stem}-{args.metric}.png", dpi=180)
    plt.close(fig)
