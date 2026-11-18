"""Collect CMax results without treating missing/smoke jobs as completed evaluation."""
import argparse
import csv
import io
import json
from pathlib import Path

from event_flow_bench.config import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    reference = {(int(r["dt"]),r["sequence"]): r for r in csv.DictReader(open("configs/literature/cmax2022_table1.csv"))}
    results = {}
    for root in args.roots:
        for path in Path(root).glob("*/completed.json"):
            r = json.loads(path.read_text())
            if r["status"] != "completed" or r["limit"] != 0: raise ValueError("Not a full evaluation")
            key = (r["method"],r["profile"],r["sequence"],r["dt"])
            if key[0] not in ("patch2018","secrets2022") or key[1] not in ("matched1k","long") or (key[3],key[2]) not in reference:
                raise ValueError(f"Unexpected result: {key}")
            if key in results: raise ValueError(f"Duplicate result: {key}")
            results[key] = r
    if len({(r["config_sha256"],r["manifest_sha256"]) for r in results.values()}) > 1:
        raise ValueError("Do not aggregate different configurations or manifests")
    rows = []
    for method in ("patch2018","secrets2022"):
        for profile in ("matched1k","long"):
            for sequence in ("outdoor_day1","indoor_flying1","indoor_flying2","indoor_flying3"):
                for dt in (1,4):
                    r = results.get((method,profile,sequence,dt))
                    paper = reference[(dt,sequence)] if method == "secrets2022" else {}
                    row = {"method":method,"profile":profile,"sequence":sequence,"dt":dt,
                           "status":"completed" if r else "missing", "paper_aee":paper.get("aee"),
                           "paper_outlier_3px_pct":paper.get("outlier_3px_pct")}
                    row.update({k:r["summary"].get(k) if r else None for k in
                                ("aee","outlier_pct","outlier_3px_pct","windows","aee_invalid","optimization_seconds","estimated_pixel_fraction")})
                    rows.append(row)
    output = Path(args.output); output.mkdir(parents=True,exist_ok=True)
    write_json(output/"summary.json",{"completed_conditions":len(results),"expected_conditions":32,"results":rows})
    buf = io.StringIO(); writer = csv.DictWriter(buf,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/"comparison.csv").write_text(buf.getvalue())
    def fmt(value): return "—" if value is None else f"{float(value):.3f}"
    lines = ["# CMax MVSEC evaluation", "", f"Completed: {len(results)}/32 conditions.", "",
             "Paper values are references, not matched reproduction targets: our input is causal, cropped to 256×256, and reset for each window. See docs/CMAX.md.", "",
             "2018 has no corresponding paper MVSEC values. 2022 Burgers values: [Table 1](https://arxiv.org/html/2207.10022v2#S4.T1).", "",
             "| Method | Window | Sequence | dt | AEE | Outlier (3px AND 5% GT) % | 3px % | Paper AEE | Paper 3px % | sec/window | Status |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append("| "+" | ".join(str(r[k]) if k in ("method","profile","sequence","dt","status") else fmt(r[k]) for k in
                                     ("method","profile","sequence","dt","aee","outlier_pct","outlier_3px_pct","paper_aee","paper_outlier_3px_pct","optimization_seconds","status"))+" |")
    (output/"comparison.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"completed":len(results),"expected":32,"output":str(output)}))
    if len(results) != 32 and not args.allow_incomplete: raise SystemExit(1)


if __name__ == "__main__": main()
