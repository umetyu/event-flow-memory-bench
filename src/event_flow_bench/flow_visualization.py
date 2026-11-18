"""Fixed-scale optical-flow colors and explicit displacement units for figures."""
import numpy as np


def aligned_chunks(end, size):
    """Cover [0,end) exactly, with a full-size last chunk ending at GT time.

    A shorter chunk, if necessary, belongs only to the start of warmup.
    """
    if size <= 0 or end < size:
        raise ValueError("A full input window is required")
    first = end % size or size
    yield 0, first
    for start in range(first, end, size):
        yield start, start + size


def flow_rgb(flow, max_displacement, mask=None):
    """Hue=atan2(v,u), saturation=clipped magnitude/scale, value=1.

    Positive u is right; positive v is down. Zero flow is white, missing data
    is black. The same max_displacement must be used for predictions and GT.
    """
    flow = np.asarray(flow)
    if flow.ndim != 3 or flow.shape[0] != 2 or max_displacement <= 0:
        raise ValueError("Expected (2,H,W) flow and positive color scale")
    if mask is None: mask = np.ones(flow.shape[1:], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != flow.shape[1:] or not np.isfinite(flow[:, mask]).all():
        raise ValueError("Invalid mask or nonfinite visible flow")
    u, v = np.nan_to_num(flow)
    hue = (np.arctan2(v, u) / (2*np.pi)) % 1
    saturation = np.clip(np.hypot(u, v)/max_displacement, 0, 1)
    rgb = np.clip(np.abs((hue[..., None]*6 + np.array([0., 4., 2.])) % 6 - 3)-1, 0, 1)
    rgb = 1 - saturation[..., None] + saturation[..., None]*rgb
    rgb[~mask] = 0
    return np.clip(rgb, 0, 1)


def to_gt_displacement(normalized_flow, flow_scale, input_seconds, gt_seconds):
    if input_seconds <= 0 or gt_seconds <= 0 or flow_scale <= 0:
        raise ValueError("Durations and flow_scale must be positive")
    return normalized_flow * flow_scale * gt_seconds / input_seconds


def event_rgb(counts):
    """Positive events orange, negative events cyan; equal activity is white."""
    positive, negative = np.asarray(counts) > 0
    rgb = np.zeros((*positive.shape, 3))
    rgb[positive] = [1., .42, .12]
    rgb[negative] = [.1, .7, 1.]
    rgb[positive & negative] = 1
    return rgb
