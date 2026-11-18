"""Metrics with explicit masks, invalid-window handling and threshold convention."""
import numpy as np
import torch
from .config import upstream_config
from ._vendor.event_flow.loss.flow import FWL, RSAT

def flow_metrics(pred, gt, valid, event_mask, outlier_reference="gt"):
    if pred.shape != gt.shape or pred.shape[1] != 2:
        raise ValueError("pred and GT must have identical (B,2,H,W) shapes")
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Nonfinite flow predictions")
    mask = valid.bool() & event_mask.bool() & torch.isfinite(gt).all(1)
    if mask.shape != gt[:, 0].shape:
        raise ValueError("Valid and event masks must be (B,H,W)")
    error = torch.linalg.vector_norm(pred - torch.nan_to_num(gt), dim=1)
    reference = gt if outlier_reference == "gt" else pred
    magnitude = torch.linalg.vector_norm(torch.nan_to_num(reference), dim=1)
    n = int(mask.sum())
    if n == 0:
        return {"aee": None, "outlier_pct": None, "valid_pixels": 0}
    outliers = ((error > 3.0) & (error > 0.05 * magnitude) & mask).sum()
    return {"aee": float(error[mask].mean()), "outlier_pct": float(outliers) * 100 / n, "valid_pixels": n}

class WarpMetrics:
    def __init__(self, c):
        self.metrics = [FWL(upstream_config(c), c["device"], c["eval"]["flow_scale"]),
                        RSAT(upstream_config(c), c["device"], c["eval"]["flow_scale"])]
        self.kept, self.events = 0, 0

    def add(self, flows, inputs):
        for metric in self.metrics:
            metric.event_flow_association(flows, inputs)
        # Track how many events remain in view after warping: FWL can reward collapse.
        self.events += inputs["event_list"].shape[1]

    def compute(self):
        result = {}
        for name, metric in zip(("fwl", "rsat"), self.metrics):
            value = metric().detach().cpu().numpy()
            result[name] = float(value.mean()) if np.isfinite(value).all() else None
        iwe = self.metrics[0].compute_window_iwe()
        result["warped_event_fraction"] = float(iwe.sum()) / max(1, self.events)
        return result

def mask_metrics(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch {pred.shape} != {gt.shape}")
    intersection = int((pred & gt).sum())
    union = int((pred | gt).sum())
    iou = intersection / union if union else 1.0
    false_pos = int((pred & ~gt).sum())
    return {"iou": iou, "detected": int(iou >= 0.5 and intersection >= false_pos)}

def summarize(rows):
    """Macro mean of window scores, plus explicit per-metric valid/invalid counts."""
    result = {"windows": len(rows)}
    for key in ("aee", "outlier_pct", "fwl", "rsat", "warped_event_fraction", "iou", "detected"):
        present = [r[key] for r in rows if key in r]
        if present:
            values = [x for x in present if x is not None and np.isfinite(x)]
            result[key] = float(np.mean(values)) if values else None
            result[key + "_n"] = len(values)
            result[key + "_invalid"] = len(present) - len(values)
    return result
