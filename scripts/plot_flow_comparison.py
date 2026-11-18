"""Actual MVSEC predictions: author ConvGRU and a snapshotted voxel5 checkpoint.

Select the first GT interval starting after a fixed event index (no selection
by prediction quality). Align full 1k-event updates to the selected GT endpoint.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from event_flow_bench.config import DEFAULT, digest, write_json
from event_flow_bench.data import EventFile, batch, encode, load_manifest
from event_flow_bench.engine import load_checkpoint, sha256
from event_flow_bench.flow_visualization import aligned_chunks, event_rgb, flow_rgb, to_gt_displacement
from event_flow_bench.metrics import flow_metrics
from event_flow_bench.models import build_model, EVENT_FLOW_SHA
from event_flow_bench.reproduce import load_weights, read_training_config


def snapshot(source, destination):
    """Training saves by atomic rename. Verify a consistent copy before inference."""
    for _ in range(3):
        before = sha256(source)
        shutil.copyfile(source, destination)
        after = sha256(source)
        if before == after == sha256(destination): return before
    raise RuntimeError("Checkpoint changed during snapshot; retry after training update")


@torch.inference_mode()
def capture(models, configs, entry, dt, anchor):
    c = configs[1]  # produces both count and voxel5; count model ignores voxel tensor
    for model in models: model.reset()
    stride = c["window"]["stride_events"]
    with EventFile(entry["path"], c["data"]["resolution"]) as stream:
        for index, (t0,t1,gt,valid) in enumerate(stream.gt_intervals(f"gtflow_dt{dt}")):
            a, b = stream.index(t0), stream.index(t1)
            if a < anchor: continue
            # Partition the complete prefix without overlaps or gaps. Only the
            # first warmup chunk may be short; the scored input is exactly 1k.
            # This avoids interpreting a 1k-history prediction as displacement
            # over a shorter remainder at a GT boundary.
            outputs = inputs = None
            final_start = None
            for start, end in aligned_chunks(b,stride):
                inputs = batch([encode(stream.read(start,end),stream.resolution,5)],c["device"])
                outputs = [model(inputs)["flow"][-1] for model in models]
                final_start = start
            if a < c["window"]["warmup_events"] or outputs is None:
                raise ValueError("Selected interval is empty or precedes warmup")
            input_seconds = float(inputs["dt_input"])
            predictions = [to_gt_displacement(out, cc["eval"]["flow_scale"], input_seconds,t1-t0)
                           for out,cc in zip(outputs,configs)]
            events = stream.read(a,b)
            encoded_events = encode(events,stream.resolution,5)
            mask = encoded_events["event_mask"].bool().to(c["device"])
            gt, valid = gt.to(c["device"]), valid.to(c["device"])
            metrics = [flow_metrics(pred,gt,valid,mask,"gt") for pred in predictions]
            common_mask = (valid & mask & torch.isfinite(gt).all(1))[0].cpu().numpy()
            counts = encoded_events["event_cnt"].numpy()
            arrays = {"gt_flow":gt[0].cpu().numpy(), "convgru_flow":predictions[0][0].cpu().numpy(),
                      "voxel5_flow":predictions[1][0].cpu().numpy(), "gt_valid":valid[0].cpu().numpy(),
                      "event_mask":mask[0].cpu().numpy(), "display_mask":common_mask,
                      "event_counts":counts, "event_xy_tp":np.column_stack(events),
                      "final_input_xy_tp":np.column_stack(stream.read(max(0,b-1000),b))}
            cap = 5. if dt == 1 else 20.
            clipped = {key: float((np.linalg.norm(arrays[key],axis=0)[common_mask]>cap).mean()) if common_mask.any() else None
                       for key in ("gt_flow","convgru_flow","voxel5_flow")}
            record = {"sequence":entry["id"], "dt":dt, "gt_interval_index":index,
                      "timestamp_start":t0,"timestamp_end":t1,"event_time_origin":stream.origin,
                      "relative_start_seconds":t0-stream.origin,"relative_end_seconds":t1-stream.origin,
                      "start_event":a,"end_event":b,"history_start_event":max(0,b-1000),
                      "final_stride_start_event":final_start,"input_events":b-final_start,
                      "initial_warmup_chunk_events":b % stride or stride,
                      "input_seconds":input_seconds,"gt_seconds":t1-t0,
                      "convgru_metrics":metrics[0],"voxel5_metrics":metrics[1],"color_max_px":cap,
                      "saturated_fraction":clipped,
                      "array_sha256":{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}}
            for model in models: model.reset()
            return arrays, record
    raise ValueError("No GT interval after requested anchor")


def render(records, output, checkpoint_label, training_complete=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for dt in (1,4):
        selected = [r for r in records if r["dt"]==dt]
        scale = 5. if dt==1 else 20.
        for subset, stem in ((selected,f"flow-comparison-dt{dt}"),
                             ([r for r in selected if r["sequence"]=="indoor_flying2"], f"flow-detail-indoor_flying2-dt{dt}")):
            n = len(subset)
            width, panel, gap, top, bottom = 13.6, 3.0, .72, 1.05, 1.45
            height = top + n*panel + (n-1)*gap + bottom
            fig = plt.figure(figsize=(width,height))
            axes = np.empty((n,4),dtype=object)
            for row in range(n):
                y = height-top-panel-row*(panel+gap)
                for col in range(4):
                    axes[row,col] = fig.add_axes([(.35+3.3*col)/width,y/height,panel/width,panel/height])
            fig.suptitle(f"MVSEC optical flow | dt = {dt} | 256 x 256", fontsize=17, y=1-.13/height, weight="bold")
            stage="training complete" if training_complete else "INTERIM - training unfinished"
            fig.text(.5,1-.53/height,f"ConvGRU: author pretrained weights    |    Voxel5: {checkpoint_label} ({stage})",ha="center",fontsize=10,color="#4b5563")
            for row, record in enumerate(subset):
                with np.load(output/record["arrays_file"]) as data:
                    pictures = [event_rgb(data["event_counts"])] + [flow_rgb(data[k],scale,data["display_mask"]) for k in ("gt_flow","convgru_flow","voxel5_flow")]
                    titles = ["Events in GT interval","GT flow","Estimated: ConvGRU EV-FlowNet","Estimated: EV-FlowNet voxel5"]
                    for col,(ax,picture) in enumerate(zip(axes[row],pictures)):
                        ax.imshow(picture,interpolation="nearest"); ax.set_axis_off()
                        ax.set_title(titles[col],fontsize=10,pad=7)
                        if col==0:
                            text = f"{record['sequence']}\nt={record['relative_start_seconds']:.3f}--{record['relative_end_seconds']:.3f} s"
                        elif col==1:
                            text = f"Common valid event pixels: {record['convgru_metrics']['valid_pixels']:,}"
                        else:
                            metric=record["convgru_metrics" if col==2 else "voxel5_metrics"]
                            text=f"AEE = {metric['aee']:.3f} px (this interval)" if metric['aee'] is not None else "No valid pixels"
                        ax.text(.5,-.04,text,ha="center",va="top",transform=ax.transAxes,fontsize=9,color="#374151")
            ax = fig.add_axes([.5/width,.23/height,.7/width,.7/height])
            y,x=np.mgrid[-1:1:101j,-1:1:101j]
            wheel=flow_rgb(np.stack((x,y)),1.,x*x+y*y<=1)
            ax.imshow(wheel,extent=(-1,1,1,-1),interpolation="nearest"); ax.set_axis_off()
            ax.text(1.14,0,"right",va="center",fontsize=7); ax.text(0,1.2,"down",ha="center",fontsize=7)
            anchor=subset[0]["anchor_events"]
            fig.text(.15,.58/height,f"Same color scale for all flow panels: 0--{scale:g} px. Hue = direction; saturation = magnitude.\nWhite = zero flow; black = outside GT-valid AND event mask. Values above {scale:g} px saturate.\nSamples fixed by event index (first GT interval starting after event {anchor:,}); no selection by accuracy.",
                     fontsize=9,va="center",color="#374151")
            fig.savefig(output/f"{stem}.png",dpi=160,facecolor="white")
            fig.savefig(output/f"{stem}.svg",facecolor="white")
            plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest")
    p.add_argument("--author-run")
    p.add_argument("--voxel-checkpoint")
    p.add_argument("--output",required=True)
    p.add_argument("--anchor-events",type=int,default=500000)
    p.add_argument("--device",default="cuda")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--trust-author-pickle",action="store_true")
    p.add_argument("--render-only",action="store_true")
    args=p.parse_args()
    output=Path(args.output).resolve()
    if args.render_only:
        metadata=json.loads((output/"metadata.json").read_text())
        render(metadata["examples"],output,metadata["voxel_checkpoint"]["label"],metadata["voxel_checkpoint"]["training_complete"])
        return
    if not all((args.manifest,args.author_run,args.voxel_checkpoint)):
        p.error("manifest, author-run and voxel-checkpoint are required for inference")
    if args.anchor_events < 50000: p.error("Anchor must be after common warmup")
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    snapshots=output/"checkpoints"; snapshots.mkdir()
    voxel_hash=snapshot(args.voxel_checkpoint,snapshots/"voxel5.pt")
    state=torch.load(snapshots/"voxel5.pt",map_location="cpu",weights_only=True)
    c_voxel=deepcopy(state["config"])
    if c_voxel["model"]["name"]!="EVFlowNet" or c_voxel["model"]["encoding"]!="voxel" or c_voxel["model"]["num_bins"]!=5:
        raise ValueError("Expected feedforward EVFlowNet with five voxel bins")
    if c_voxel["window"]["history_events"]!=1000 or c_voxel["window"]["stride_events"]!=1000:
        raise ValueError("This comparison requires 1k history and stride")
    epoch=int(state["epoch"])
    voxel_record={"sha256":voxel_hash,"epoch_zero_based":epoch,"epochs_completed_at_checkpoint":epoch+1,
                  "training_complete":(Path(args.voxel_checkpoint).parent/"completed.json").exists(),
                  "planned_epochs":state["config"]["train"]["epochs"],"best_validation_rsat":state["best_rsat"],
                  "training_manifest_sha256":state["manifest_sha256"],"training_config":deepcopy(state["config"]),
                  "label":f"epoch {epoch+1}/{state['config']['train']['epochs']}, validation-best snapshot"}
    del state
    c_voxel["device"]=args.device
    c_voxel["data"].update(dataset="mvsec",manifest=args.manifest,resolution=[256,256])
    c_voxel["eval"]["mode"]="gtflow_dt1"
    voxel, loaded=load_checkpoint(snapshots/"voxel5.pt",c_voxel); del loaded
    training=read_training_config(run_dir=args.author_run)
    c_gru=deepcopy(c_voxel)
    c_gru["model"]={**deepcopy(training["model"]),"spiking_neuron":training.get("spiking_neuron")}
    if c_gru["model"]["name"]!="RecEVFlowNet" or c_gru["model"]["encoding"]!="cnt":
        raise ValueError("Expected author count-input ConvGRU RecEVFlowNet")
    author_checkpoint=Path(args.author_run)/"artifacts/model/data/model.pth"
    author_hash=sha256(author_checkpoint)
    gru=build_model(c_gru)
    gru.net.load_state_dict(load_weights(author_checkpoint,args.trust_author_pickle),strict=True)
    models=[gru.eval(),voxel.eval()]
    manifest=load_manifest(args.manifest)
    if manifest["dataset"]!="mvsec": raise ValueError("MVSEC manifest required")
    entries={e["id"]:e for e in manifest["test"]}
    records=[]; started=time.monotonic()
    for dt in (1,4):
        for name in ("outdoor_day1","indoor_flying1","indoor_flying2","indoor_flying3"):
            print(json.dumps({"phase":"inference","sequence":name,"dt":dt,"voxel_checkpoint":voxel_record["label"]}),flush=True)
            arrays,record=capture(models,[c_gru,c_voxel],entries[name],dt,args.anchor_events)
            filename=f"{name}-dt{dt}.npz"
            np.savez_compressed(output/filename,**arrays)
            record.update(arrays_file=filename,npz_sha256=sha256(output/filename),anchor_events=args.anchor_events)
            records.append(record)
            print(json.dumps({"phase":"captured","sequence":name,"dt":dt,"gru_aee":record['convgru_metrics']['aee'],"voxel_aee":record['voxel5_metrics']['aee']}),flush=True)
    metadata={"generated_utc":datetime.now(timezone.utc).isoformat(),"protocol":"gt-aligned-1k-qualitative-v1",
              "inference_settings":{"history_events":1000,"stride_events":1000,"sequence_reset":"at sequence start", "state_warmup":"entire prefix from event 0", "hot_pixel_filter":False,"flow_scale":128,
                                    "displacement_formula":"network_output * 128 * GT_duration / full_1k_input_duration",
                                    "chunk_alignment":"full nonoverlapping 1k chunks ending at selected GT endpoint; only first warmup chunk may be short",
                                    "temporal_assumption":"final-input average velocity held constant over GT interval"},
              "sampling":{"rule":"first GT interval whose start event index >= anchor", "anchor_events":args.anchor_events,"chosen_using_predictions":False},
              "author_checkpoint":{"sha256":author_hash,"model":"RecEVFlowNet","training_config":training},
              "voxel_checkpoint":voxel_record,"evaluation_configs":[c_gru,c_voxel],"manifest_sha256":digest(manifest),
              "upstream_revision":EVENT_FLOW_SHA,"script_sha256":sha256(__file__),"torch":str(torch.__version__),
              "gpu":torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
              "inference_seconds":time.monotonic()-started,"examples":records,
              "limitations":["Voxel checkpoint is from an unfinished training run, not final accuracy.",
                             "Author ConvGRU and voxel5 differ in training data split/conditions.",
                             "GT-aligned full 1k input inference differs from Table 1 reproduction and generic matched-v1 GT-boundary partial strides.",
                             "AEE labels are per selected interval, not full-sequence averages."]}
    write_json(output/"metadata.json",metadata)
    render(records,output,voxel_record["label"],voxel_record["training_complete"])
    print(json.dumps({"phase":"complete","examples":len(records),"output":str(output),"voxel_checkpoint":voxel_record["label"]}),flush=True)


if __name__=="__main__": main()
