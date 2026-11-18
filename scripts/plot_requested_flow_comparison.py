"""GT / ConvGRU count / [56]-type voxel9, with two explicit pending slots.

Reuse verified ConvGRU predictions from plot_flow_comparison.py. The user asked
to show status only for feedforward count and Shiba CMax, not interim flows.
"""
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
from event_flow_bench.flow_visualization import flow_rgb
from event_flow_bench.metrics import flow_metrics
from event_flow_bench.zhu2019 import ZhuFlowNet, PROTOCOL, displacement
from plot_flow_comparison import snapshot


def verify_arrays(arrays, record):
    """Verify saved identity, a common mask, causal time units and the scores."""
    expected = arrays["gt_valid"] & np.isfinite(arrays["gt_flow"]).all(0)
    np.testing.assert_array_equal(arrays["dense_mask"], expected)
    np.testing.assert_array_equal(arrays["display_mask"], expected & arrays["event_mask"])
    assert arrays["voxel9_input_xy_tp"].shape == (record["voxel9_input_events"],4)
    assert np.isclose(np.ptp(arrays["voxel9_input_xy_tp"][:,2]),record["voxel9_input_seconds"],rtol=0,atol=1e-10)
    assert arrays["voxel9_input_xy_tp"][-1,2] < record["relative_end_seconds"]
    assert "count_flow" not in arrays and "cmax_flow" not in arrays and "voxel5_flow" not in arrays
    for key in ("convgru_flow","voxel9_flow"):
        assert arrays[key].shape == arrays["gt_flow"].shape == (2,256,256)
        assert np.isfinite(arrays[key]).all()
        errors = np.linalg.norm(arrays[key]-arrays["gt_flow"],axis=0)
        for mask_name,score_name in (("display_mask","event_metrics"),("dense_mask","dense_metrics")):
            observed=float(errors[arrays[mask_name]].mean())
            assert abs(observed-record[score_name][key]["aee"]) < 1e-5


@torch.inference_mode()
def infer(args):
    root, reference = Path(args.output).resolve(), Path(args.reference).resolve()
    if root.exists() and any(root.iterdir()): raise FileExistsError(root)
    source = json.loads((reference/"metadata.json").read_text())
    if source["protocol"] != "gt-aligned-1k-qualitative-v1": raise ValueError("Unexpected ConvGRU source protocol")
    m = source["evaluation_configs"][0]["model"]
    if m["name"] != "RecEVFlowNet" or m["encoding"] != "cnt": raise ValueError("Count-input ConvGRU required")
    manifest = load_manifest(args.manifest)
    if manifest["dataset"] != "mvsec" or digest(manifest) != source["manifest_sha256"]:
        raise ValueError("Reference and inference must use the same MVSEC files")
    entries = {r["id"]:r for r in manifest["test"]}
    root.mkdir(parents=True,exist_ok=True); (root/"checkpoints").mkdir()
    fixed = root/"checkpoints/voxel9-zhu2019.pt"
    checkpoint_hash = snapshot(args.checkpoint,fixed)
    state = torch.load(fixed,map_location=args.device,weights_only=True)
    if state["config"]["protocol"] != PROTOCOL: raise ValueError("Expected the [56]-type reimplementation checkpoint")
    completion_path = Path(args.checkpoint).parent/"completed.json"
    completion = json.loads(completion_path.read_text())
    if completion["status"] != "completed": raise ValueError("The [56]-type training run must be complete")
    checkpoint = {"sha256":checkpoint_hash,"best_checkpoint_epoch":int(state["epoch"])+1,
                  "training_run_epochs":completion["epochs"],"training_complete":True,
                  "best_validation_rsat":state["best_rsat"],"training_config":deepcopy(state["config"]),
                  "manifest_sha256":state["manifest_sha256"],"kind":"paper-based reimplementation, not author checkpoint"}
    model = ZhuFlowNet(state["config"]).to(args.device)
    model.load_state_dict(state["state_dict"],strict=True); model.eval(); del state
    records = []
    for old in source["examples"]:
        p = reference/old["arrays_file"]
        if sha256(p) != old["npz_sha256"]: raise ValueError("Reference array hash mismatch")
        with np.load(p) as data:
            keep=("gt_flow","convgru_flow","gt_valid","event_mask","display_mask","event_counts","event_xy_tp","final_input_xy_tp")
            arrays={k:data[k].copy() for k in keep}
        arrays["dense_mask"] = arrays["gt_valid"] & np.isfinite(arrays["gt_flow"]).all(0)
        with EventFile(entries[old["sequence"]]["path"],[256,256]) as stream:
            selected = next(r for i,r in enumerate(stream.gt_intervals(f"gtflow_dt{old['dt']}")) if i==old["gt_interval_index"])
            t0,t1,gt,valid = selected
            if (t0,t1)!=(old["timestamp_start"],old["timestamp_end"]): raise ValueError("GT interval changed")
            np.testing.assert_array_equal(gt[0].numpy(),arrays["gt_flow"])
            np.testing.assert_array_equal(valid[0].numpy(),arrays["gt_valid"])
            a,b=stream.index(t0),stream.index(t1)
            assert (a,b)==(old["start_event"],old["end_event"])
            np.testing.assert_array_equal(np.column_stack(stream.read(a,b)),arrays["event_xy_tp"])
            n=30000 if old["sequence"].startswith("outdoor") else 15000
            if b<n: raise ValueError("Insufficient causal input history")
            events=stream.read(b-n,b)
            assert events[2][-1]+stream.origin < t1
            inputs=batch([encode(events,[256,256],9)],args.device)
            native=model(inputs["event_voxel"])[-1]
            pred=displacement(native,float(inputs["dt_input"]),t1-t0)
            arrays["voxel9_flow"]=pred[0].cpu().numpy()
            arrays["voxel9_native_pixels_per_bin"]=native[0].cpu().numpy()
            arrays["voxel9_input_xy_tp"]=np.column_stack(events)
        fields=("sequence","dt","gt_interval_index","timestamp_start","timestamp_end","event_time_origin",
                "relative_start_seconds","relative_end_seconds","start_event","end_event","anchor_events","gt_seconds")
        record={k:old[k] for k in fields}
        record.update(convgru_input_events=1000,convgru_input_seconds=old["input_seconds"],
                      voxel9_input_events=n,voxel9_input_start_event=b-n,voxel9_input_seconds=float(inputs["dt_input"]),
                      reference_npz_sha256=old["npz_sha256"],color_max_px=old["color_max_px"])
        for mask_name,score_name in (("display_mask","event_metrics"),("dense_mask","dense_metrics")):
            record[score_name]={k:flow_metrics(torch.from_numpy(arrays[k])[None],gt,valid,
                                             torch.from_numpy(arrays[mask_name])[None],"gt")
                                for k in ("convgru_flow","voxel9_flow")}
        verify_arrays(arrays,record)
        filename=old["arrays_file"]
        np.savez_compressed(root/filename,**arrays)
        record.update(arrays_file=filename,npz_sha256=sha256(root/filename),
                      array_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()})
        records.append(record)
        print(json.dumps({"sequence":record["sequence"],"dt":record["dt"],"zhu_aee":record["event_metrics"]["voxel9_flow"]["aee"]}),flush=True)
    metadata={"generated_utc":datetime.now(timezone.utc).isoformat(),"protocol":"requested-four-method-figure-v1",
              "convgru_source":{"metadata_sha256":sha256(reference/"metadata.json"),"checkpoint":source["author_checkpoint"],
                                "inference_settings":source["inference_settings"],"source_script_sha256":source["script_sha256"]},
              "voxel9_checkpoint":checkpoint,"manifest_sha256":digest(manifest),"sampling":source["sampling"],
              "script_sha256":sha256(__file__),"torch":str(torch.__version__),"examples":records,
              "pending_panels":{"evflownet_count":{"label":"学習中","flow_displayed":False},
                                "shiba_cmax":{"label":"最適化・評価中","training_required":False,"flow_displayed":False}},
              "limitations":["Different input event counts and training setups; not an isolated architecture comparison.",
                             "[56]-type model is a local reimplementation; its validation RSAT remained near 1, so reproduction success is unconfirmed.",
                             "Only two estimated flow fields are shown; pending panels contain no prediction arrays.",
                             "Dense figures use GT-valid pixels; supplementary event-masked figures use GT-valid AND event pixels.",
                             "AEE in metadata is per selected interval, not a full-sequence result."]}
    write_json(root/"metadata.json",metadata)
    shutil.copyfile(__file__,root/"inference-script.py")
    print(json.dumps({"complete":True,"output":str(root),"voxel9_checkpoint":checkpoint_hash,
                      "best_epoch":checkpoint["best_checkpoint_epoch"],"run_epochs":checkpoint["training_run_epochs"]}),flush=True)


def render(root,font_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle
    root=Path(root); metadata=json.loads((root/"metadata.json").read_text())
    font=FontProperties(fname=font_path) if font_path else FontProperties()
    jp=bool(font_path)
    titles=["GT flow", "ConvGRU EV-FlowNet", "EV-FlowNet [56]-type", "EV-FlowNet + count", "CMax (Shiba et al.)"]
    subtitles=(["正解", "count入力・配布重み", "9-bin voxel・UZH再実装", "no voxel", "Optimization"] if jp else
               ["Ground truth", "count input | pretrained", "9-bin voxel | UZH reimplementation", "no voxel", "Optimization"])
    for dt in (1,4):
        selected=[r for r in metadata["examples"] if r["dt"]==dt]
        for masked in (False,True):
            suffix="-eventmasked" if masked else ""
            for subset,stem in ((selected,f"comparison-dt{dt}{suffix}"),
                                ([r for r in selected if r["sequence"]=="indoor_flying2"],f"detail-indoor_flying2-dt{dt}{suffix}")):
                n=len(subset); width=16.5; panel=2.95; gap=.67; top=1.45; bottom=1.4
                height=top+n*panel+(n-1)*gap+bottom
                fig=plt.figure(figsize=(width,height),facecolor="white")
                fig.text(.5,1-.13/height,f"MVSEC optical flow comparison  |  dt = {dt}",ha="center",va="top",fontsize=20,weight="bold")
                subtitle=("同じGT区間を比較  /  count・CMaxは進捗表示のみ" if jp else "Same GT interval / Count and CMax: status only")
                fig.text(.5,1-.58/height,subtitle,ha="center",va="top",fontsize=12,fontproperties=font,color="#475569")
                for col,title in enumerate(titles):
                    x=(.4+col*3.25+panel/2)/width
                    fig.text(x,1-.92/height,title,ha="center",va="top",fontsize=11,weight="bold")
                    fig.text(x,1-1.14/height,subtitles[col],ha="center",va="top",fontsize=10,fontproperties=font,color="#475569")
                for row,record in enumerate(subset):
                    with np.load(root/record["arrays_file"]) as data:
                        mask=data["display_mask" if masked else "dense_mask"]
                        y=height-top-panel-row*(panel+gap)
                        for col in range(5):
                            ax=fig.add_axes([(.4+col*3.25)/width,y/height,panel/width,panel/height])
                            ax.set_axis_off()
                            if col<3:
                                key=("gt_flow","convgru_flow","voxel9_flow")[col]
                                ax.imshow(flow_rgb(data[key],record["color_max_px"],mask),interpolation="nearest")
                            else:
                                ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,facecolor="#f1f5f9",edgecolor="#cbd5e1",linewidth=1.5))
                                label=("学習中" if col==3 else "最適化・評価中") if jp else ("Training in progress" if col==3 else "Optimization / evaluation\nin progress")
                                ax.text(.5,.53,label,ha="center",va="center",transform=ax.transAxes,fontproperties=font,fontsize=19 if jp else 14,color="#334155")
                                note=("推定フローは未掲載" if col==3 else "学習を行わない手法") if jp else ("Flow not shown" if col==3 else "No neural-network training")
                                ax.text(.5,.38,note,ha="center",va="center",transform=ax.transAxes,fontproperties=font,fontsize=10,color="#64748b")
                            if col==0: label=f"{record['sequence']}\nt = {record['relative_start_seconds']:.3f}–{record['relative_end_seconds']:.3f} s"
                            elif col==1: label="Input: 1,000 events + recurrent state"
                            elif col==2: label=f"Input: {record['voxel9_input_events']:,} events | 9 bins"
                            else: label=""
                            ax.text(.5,-.045,label,ha="center",va="top",transform=ax.transAxes,fontsize=9,color="#475569")
                ax=fig.add_axes([.55/width,.25/height,.73/width,.73/height])
                yy,xx=np.mgrid[-1:1:101j,-1:1:101j]
                ax.imshow(flow_rgb(np.stack((xx,yy)),1,xx*xx+yy*yy<=1),extent=(-1,1,1,-1)); ax.set_axis_off()
                ax.text(1.1,0,"right",va="center",fontsize=8); ax.text(0,1.2,"down",ha="center",fontsize=8)
                cap=selected[0]["color_max_px"]
                if jp:
                    mask_text="GT有効かつイベントのある画素" if masked else "GTが有効な全画素（イベントのない画素も表示）"
                    note=f"方向＝色相、移動量＝彩度。全フローパネルで共通の0–{cap:g} pxスケール（上限以上は飽和）。\n表示範囲：{mask_text}。黒＝表示対象外、白＝移動量0。\n[56]型は100 epoch学習した再実装の検証ベスト。論文精度の再現成功は未確認。入力窓・学習条件は手法間で異なります。"
                else:
                    note=f"Same 0–{cap:g} px scale for all flow panels; larger magnitudes saturate. Hue = direction; saturation = magnitude.\nBlack = excluded pixels; white = zero flow. Display mask: {'GT-valid AND event pixels' if masked else 'all GT-valid pixels'}.\n[56]-type: validation-best from a 100-epoch reimplementation; paper accuracy not reproduced. Input windows and training setups differ."
                fig.text(.12,.59/height,note,fontsize=10,va="center",fontproperties=font,color="#475569")
                fig.savefig(root/f"{stem}.png",dpi=160,facecolor="white")
                fig.savefig(root/f"{stem}.svg",facecolor="white")
                plt.close(fig)
    metadata["render"]={"script_sha256":sha256(__file__),"font":Path(font_path).name if font_path else "default",
                        "files":{p.name:sha256(p) for pattern in ("*.png","*.svg") for p in sorted(root.glob(pattern))}}
    write_json(root/"metadata.json",metadata)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True); p.add_argument("--reference"); p.add_argument("--checkpoint"); p.add_argument("--manifest")
    p.add_argument("--device",default="cuda"); p.add_argument("--threads",type=int,default=4)
    p.add_argument("--render-only",action="store_true"); p.add_argument("--font")
    args=p.parse_args(); torch.set_num_threads(args.threads)
    if args.render_only: render(args.output,args.font)
    else:
        if not all((args.reference,args.checkpoint,args.manifest)): p.error("reference, checkpoint and manifest are required")
        torch.manual_seed(0); infer(args)


if __name__=="__main__": main()
