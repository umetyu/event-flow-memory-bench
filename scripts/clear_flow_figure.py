"""Inspect scenes/GT before selecting readable, English-only flow comparisons."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from event_flow_bench.config import digest, write_json
from event_flow_bench.data import EventFile, batch, encode, load_manifest
from event_flow_bench.engine import sha256
from event_flow_bench.flow_visualization import aligned_chunks, event_rgb, flow_rgb, to_gt_displacement
from event_flow_bench.models import build_model
from event_flow_bench.metrics import flow_metrics
from event_flow_bench.reproduce import read_training_config, load_weights
from event_flow_bench.zhu2019 import ZhuFlowNet, displacement, PROTOCOL


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def datasets(group):
    records=[]
    group.visititems(lambda name,obj: records.append((float(obj.attrs["timestamp"]),name)) if hasattr(obj,"shape") else None)
    return sorted(records)


def preview(args):
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    if (root/"preview.json").exists(): raise FileExistsError(root/"preview.json")
    manifest=load_manifest(args.manifest); records=[]; plt=pyplot()
    for entry in manifest["test"]:
        selected=[]
        with EventFile(entry["path"],[256,256]) as stream:
            gt_records=datasets(stream.file["flow_dt4"])
            image_records=datasets(stream.file["images"])
            for fraction in (.15,.28,.41,.54,.67,.80):
                i=max(1,min(len(gt_records)-1,round(fraction*(len(gt_records)-1))))
                t1,name=gt_records[i]; ds=stream.file["flow_dt4/"+name]
                t0=float(ds.attrs.get("timestamp_start",gt_records[i-1][0]))
                gt=np.asarray(ds,dtype=np.float32)
                mask_key="flow_dt4_valid/"+name
                valid=np.asarray(stream.file[mask_key],dtype=bool) if mask_key in stream.file else np.any(gt!=0,axis=0)
                valid &= np.isfinite(gt).all(0)
                a,b=stream.index(t0),stream.index(t1)
                events=stream.read(a,b); encoded=encode(events,[256,256],9)
                image_time,image_name=min(image_records,key=lambda r:abs(r[0]-(t0+t1)/2))
                frame=np.asarray(stream.file["images/"+image_name])
                identifier=f"{entry['id']}-gt{i-1:04d}-dt4"
                filename=identifier+".npz"
                np.savez_compressed(root/filename,gt_flow=gt,gt_valid=valid,scene=frame,
                                    event_counts=encoded["event_cnt"].numpy(),event_mask=encoded["event_mask"][0].bool().numpy())
                magnitude=np.linalg.norm(gt,axis=0)[valid]
                row={"id":identifier,"sequence":entry["id"],"dt":4,"gt_interval_index":i-1,
                     "gt_dataset":name,"timestamp_start":t0,"timestamp_end":t1,
                     "relative_start_seconds":t0-stream.origin,"relative_end_seconds":t1-stream.origin,
                     "start_event":a,"end_event":b,"gt_seconds":t1-t0,"event_time_origin":stream.origin,
                     "frame_timestamp":image_time,"frame_dataset":image_name,"arrays_file":filename,
                     "gt_valid_fraction":float(valid.mean()),"gt_median_px":float(np.median(magnitude)),
                     "gt_p95_px":float(np.percentile(magnitude,95)),"interval_events":b-a,
                     "npz_sha256":sha256(root/filename)}
                selected.append(row); records.append(row)
        fig,axes=plt.subplots(6,3,figsize=(9,19))
        for row,record in enumerate(selected):
            with np.load(root/record["arrays_file"]) as data:
                pictures=[data["scene"],event_rgb(data["event_counts"]),flow_rgb(data["gt_flow"],20,data["gt_valid"])]
                for col,picture in enumerate(pictures):
                    ax=axes[row,col]
                    if col==0: ax.imshow(picture,cmap="gray",vmin=0,vmax=255)
                    else: ax.imshow(picture,interpolation="nearest")
                    ax.set_axis_off()
                    ax.set_title(("Scene","Events","GT flow: 0–20 px")[col],fontsize=9)
                axes[row,0].text(.5,-.04,f"{record['id']}\nt={record['relative_start_seconds']:.2f}–{record['relative_end_seconds']:.2f}s",transform=axes[row,0].transAxes,ha="center",va="top",fontsize=8)
                axes[row,2].text(.5,-.04,f"GT valid {record['gt_valid_fraction']:.0%} | median {record['gt_median_px']:.1f}px\np95 {record['gt_p95_px']:.1f}px | {record['interval_events']:,} events",transform=axes[row,2].transAxes,ha="center",va="top",fontsize=8)
        fig.suptitle(entry["id"]+" | candidates inspected before inference",fontsize=13)
        fig.subplots_adjust(top=.97,bottom=.04,hspace=.3,wspace=.08)
        fig.savefig(root/(entry["id"]+"-candidates.png"),dpi=130); plt.close(fig)
    write_json(root/"preview.json",{"manifest_sha256":digest(manifest),"fraction_candidates":[.15,.28,.41,.54,.67,.8],
                                    "predictions_used_for_selection":False,"candidates":records})
    print(json.dumps({"candidates":len(records),"output":str(root)}),flush=True)


@torch.inference_mode()
def infer(args):
    root=Path(args.output); selection=json.loads(Path(args.selection).read_text())
    candidate_root=Path(args.candidates); candidates=json.loads((candidate_root/"preview.json").read_text())
    manifest=load_manifest(args.manifest)
    if digest(manifest)!=candidates["manifest_sha256"]: raise ValueError("Manifest differs from scene selection")
    if root.exists() and any(root.iterdir()): raise FileExistsError(root)
    root.mkdir(parents=True)
    cstate=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    if cstate["config"]["protocol"]!=PROTOCOL: raise ValueError("Wrong voxel checkpoint")
    voxel=ZhuFlowNet(cstate["config"]).to(args.device)
    voxel.load_state_dict(cstate["state_dict"],strict=True); voxel.eval()
    voxel_record={"sha256":sha256(args.checkpoint),"checkpoint_epoch":int(cstate["epoch"])+1,
                  "kind":"UZH-trained Zhu 2019 paper-based reimplementation","best_validation_rsat":cstate["best_rsat"],
                  "training_config":deepcopy(cstate["config"])}
    del cstate
    training=read_training_config(run_dir=args.author_run)
    from event_flow_bench.config import DEFAULT
    config=deepcopy(DEFAULT); config["device"]=args.device; config["data"]["resolution"]=[256,256]
    config["model"]={**training["model"],"spiking_neuron":training.get("spiking_neuron")}
    if config["model"]["name"]!="RecEVFlowNet" or config["model"]["encoding"]!="cnt": raise ValueError("Count ConvGRU required")
    gru=build_model(config).eval()
    author=Path(args.author_run)/"artifacts/model/data/model.pth"
    gru.net.load_state_dict(load_weights(author,args.trust_author_pickle),strict=True)
    entries={r["id"]:r for r in manifest["test"]}; records=[]
    for identifier in selection["ids"]:
        old=next(r for r in candidates["candidates"] if r["id"]==identifier)
        if sha256(candidate_root/old["arrays_file"])!=old["npz_sha256"]: raise ValueError("Candidate data changed")
        with np.load(candidate_root/old["arrays_file"]) as data: arrays={k:data[k].copy() for k in data.files}
        with EventFile(entries[old["sequence"]]["path"],[256,256]) as stream:
            a,b=stream.index(old["timestamp_start"]),stream.index(old["timestamp_end"])
            assert (a,b)==(old["start_event"],old["end_event"])
            np.testing.assert_array_equal(arrays["gt_flow"],np.asarray(stream.file["flow_dt4/"+old["gt_dataset"]]))
            gru.reset()
            for start,end in aligned_chunks(b,1000):
                inputs=batch([encode(stream.read(start,end),[256,256],2)],args.device)
                flow=gru(inputs)["flow"][-1]
                if end%1000000<1000: print(json.dumps({"id":identifier,"warmup_end_event":end,"target":b}),flush=True)
            gru_seconds=float(inputs["dt_input"])
            arrays["convgru_raw"]=flow[0].cpu().numpy()
            arrays["convgru_flow"]=to_gt_displacement(flow,128,gru_seconds,old["gt_seconds"])[0].cpu().numpy()
            arrays["convgru_input_xy_tp"]=np.column_stack(stream.read(b-1000,b))
            n=30000 if old["sequence"].startswith("outdoor") else 15000
            events=stream.read(b-n,b); assert events[2][-1]+stream.origin<old["timestamp_end"]
            inputs=batch([encode(events,[256,256],9)],args.device)
            native=voxel(inputs["event_voxel"])[-1]; voxel_seconds=float(inputs["dt_input"])
            arrays["voxel9_raw"]=native[0].cpu().numpy()
            arrays["voxel9_flow"]=displacement(native,voxel_seconds,old["gt_seconds"])[0].cpu().numpy()
            arrays["voxel9_input_xy_tp"]=np.column_stack(events)
            arrays["display_mask"]=arrays["gt_valid"] & arrays["event_mask"]
        record=deepcopy(old)
        record.update(convgru_input_events=1000,convgru_input_seconds=gru_seconds,
                      voxel9_input_events=n,voxel9_input_seconds=voxel_seconds,
                      color_max_px=selection["color_max_px"])
        gt=torch.from_numpy(arrays["gt_flow"])[None]; valid=torch.from_numpy(arrays["gt_valid"])[None]
        record["metrics"]={}
        for key in ("convgru_flow","voxel9_flow"):
            assert arrays[key].shape==(2,256,256) and np.isfinite(arrays[key]).all()
            record["metrics"][key]=flow_metrics(torch.from_numpy(arrays[key])[None],gt,valid,torch.from_numpy(arrays["event_mask"])[None],"gt")
        np.savez_compressed(root/old["arrays_file"],**arrays)
        record.update(npz_sha256=sha256(root/old["arrays_file"]),
                      array_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()})
        records.append(record)
        print(json.dumps({"captured":identifier,"metrics":record["metrics"]}),flush=True)
    metadata={"generated_utc":datetime.now(timezone.utc).isoformat(),"protocol":"readable-three-column-flow-v1",
              "selection":selection,"candidate_metadata_sha256":sha256(candidate_root/"preview.json"),
              "manifest_sha256":digest(manifest),"author_checkpoint_sha256":sha256(author),"voxel9_checkpoint":voxel_record,
              "inference_script_sha256":sha256(__file__),"examples":records,
              "settings":{"convgru_input_events":1000,"convgru_state":"from event 0; full nonoverlapping 1k windows aligned to GT endpoint",
                          "voxel9_input_events":{"outdoor":30000,"indoor":15000},"future_events":False,
                          "displacement_units":"GT-interval pixels; ConvGRU uses 128*GTdt/inputdt; voxel9 uses 8*GTdt/inputdt"},
              "limitations":["Selected for scene/GT readability before model inference; these are illustrative intervals, not dataset averages.",
                             "Input histories and training conditions differ between the models.",
                             "Voxel9 is the same UZH reimplementation checkpoint as before; near-zero outputs are not altered."]}
    write_json(root/"metadata.json",metadata); shutil.copyfile(__file__,root/"inference-script.py")
    print(json.dumps({"complete":True,"output":str(root)}),flush=True)


def render(args):
    root=Path(args.output); metadata=json.loads((root/"metadata.json").read_text()); plt=pyplot()
    titles=("GT flow","ConvGRU EV-FlowNet","EV-FlowNet + voxel (9 bins)")
    for r in metadata["examples"]:
        for arrows in (False,True):
            fig=plt.figure(figsize=(13.6,6.6),facecolor="white")
            fig.text(.5,.965,"Optical flow comparison",ha="center",va="top",fontsize=22,weight="bold")
            fig.text(.5,.89,f"MVSEC {r['sequence']}  |  t = {r['relative_start_seconds']:.3f}–{r['relative_end_seconds']:.3f} s  |  dt = 4",ha="center",fontsize=11,color="#475569")
            with np.load(root/r["arrays_file"]) as data:
                for col,key in enumerate(("gt_flow","convgru_flow","voxel9_flow")):
                    ax=fig.add_axes([(.42+4.4*col)/13.6,1.4/6.6,4/13.6,4/6.6])
                    ax.imshow(flow_rgb(data[key],r["color_max_px"],data["gt_valid"]),interpolation="nearest")
                    ax.set_title(titles[col],fontsize=13,pad=11,weight="bold"); ax.set_axis_off()
                    if arrows:
                        yy,xx=np.mgrid[12:256:24,12:256:24]; good=data["gt_valid"][yy,xx]
                        u,v=data[key][:,yy,xx]
                        ax.quiver(xx[good],yy[good],u[good],v[good],angles="xy",scale_units="xy",scale=.5,color="#20252a",width=.003,headwidth=3.2)
                    text=("Ground truth", "Count input (no voxel)","9-bin voxel input")[col]
                    ax.text(.5,-.045,text,ha="center",va="top",transform=ax.transAxes,fontsize=11,color="#475569")
                wheel=fig.add_axes([.035,.025,.062,.13]); yy,xx=np.mgrid[-1:1:101j,-1:1:101j]
                wheel.imshow(flow_rgb(np.stack((xx,yy)),1,xx*xx+yy*yy<=1),extent=(-1,1,1,-1)); wheel.set_axis_off()
                wheel.text(1.15,0,"right",va="center",fontsize=8); wheel.text(0,1.2,"down",ha="center",fontsize=8)
                extra=" Arrows: 2x displacement." if arrows else ""
                note=f"Direction = hue; magnitude = saturation. Same 0–{r['color_max_px']:g} px scale in all panels; larger values saturate.{extra}\nWhite = zero flow. Black = invalid GT. Displayed on all GT-valid pixels.\nVoxel model: UZH-trained reimplementation; its current output is near zero. Input windows and training setups differ."
                fig.text(.16,.105,note,fontsize=9,va="center",color="#475569")
            stem=r["id"]+("-arrows" if arrows else "")
            fig.savefig(root/f"{stem}.png",dpi=180,facecolor="white"); fig.savefig(root/f"{stem}.svg",facecolor="white"); plt.close(fig)
        # Scene context is supplied separately, without adding a fourth flow column.
        with np.load(root/r["arrays_file"]) as data:
            fig,axes=plt.subplots(1,2,figsize=(7,4))
            axes[0].imshow(data["scene"],cmap="gray",vmin=0,vmax=255); axes[0].set_title("Scene")
            axes[1].imshow(event_rgb(data["event_counts"])); axes[1].set_title("Events during GT interval")
            for ax in axes: ax.set_axis_off()
            fig.suptitle(r["sequence"]+f" | t = {r['relative_start_seconds']:.3f} s")
            fig.tight_layout(); fig.savefig(root/(r["id"]+"-scene.png"),dpi=160); plt.close(fig)
    metadata["render"]={"script_sha256":sha256(__file__),"language":"English","pending_panels":False,"titles":list(titles),
                        "files":{p.name:sha256(p) for pattern in ("*.png","*.svg") for p in sorted(root.glob(pattern))}}
    write_json(root/"metadata.json",metadata)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("stage",choices=("preview","infer","render"))
    p.add_argument("--output",required=True); p.add_argument("--manifest"); p.add_argument("--candidates"); p.add_argument("--selection")
    p.add_argument("--checkpoint"); p.add_argument("--author-run"); p.add_argument("--trust-author-pickle",action="store_true")
    p.add_argument("--device",default="cuda"); p.add_argument("--threads",type=int,default=4)
    args=p.parse_args(); torch.set_num_threads(args.threads); torch.manual_seed(0)
    {"preview":preview,"infer":infer,"render":render}[args.stage](args)


if __name__=="__main__": main()
