"""Event-only CMax solvers. All public outputs are (u, v) velocities in px/s.

2018: independent constant-translation patches, variance of unsigned IWEs.
2022: adapter to the unmodified official solver at a verified Git revision.
Neither solver receives ground-truth flow.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib
import io
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

SECRETS_SHA = "f3b899a001052d97ed68f0eec0478c4f6440c59c"


def splat(x, y, patch_ids, n_patches, size):
    """Bilinear unsigned event accumulation; out-of-canvas mass is discarded."""
    ix, iy = x.floor(), y.floor()
    image = x.new_zeros(n_patches * size * size)
    for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        xx, yy = ix + dx, iy + dy
        weight = (1 - (x - xx).abs()) * (1 - (y - yy).abs())
        valid = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
        index = patch_ids * size * size + yy.long().clamp(0, size-1) * size + xx.long().clamp(0, size-1)
        image.scatter_add_(0, index, weight * valid)
    return image.reshape(n_patches, 1, size, size)


class PatchCMax2018:
    def __init__(self, resolution, config, device="cpu"):
        self.resolution, self.config, self.device = tuple(resolution), deepcopy(config), device
        self.patch_size = int(config["patch_size"])
        if any(n % self.patch_size for n in resolution):
            raise ValueError("Resolution must be divisible by patch size")
        self.grid = (resolution[0] // self.patch_size, resolution[1] // self.patch_size)
        self.pad = int(np.ceil(config["max_displacement"])) + 4
        self.canvas = self.patch_size + 2 * self.pad
        sigma = float(config["blur_sigma"])
        x = torch.arange(-3, 4, dtype=torch.float32, device=device)
        k = torch.exp(-x.square() / (2 * sigma * sigma)); k /= k.sum()
        self.kernel = (k[:, None] * k[None, :])[None, None]

    def estimate(self, events, seed=0):
        x, y, t, _ = events
        h, w = self.resolution
        n = self.grid[0] * self.grid[1]
        zero = np.zeros((2, h, w), np.float32)
        if len(t) < 2 or t[-1] <= t[0]:
            return zero, {"status": "zero_duration", "estimated_pixel_fraction": 0.}
        duration = float(t[-1] - t[0])
        ids = (y.astype(np.int64) // self.patch_size) * self.grid[1] + x.astype(np.int64) // self.patch_size
        counts = np.bincount(ids, minlength=n)
        active = torch.as_tensor(counts >= self.config["min_events"], device=self.device)
        ids = torch.as_tensor(ids, device=self.device)
        xx = torch.as_tensor(x % self.patch_size + self.pad, dtype=torch.float32, device=self.device)
        yy = torch.as_tensor(y % self.patch_size + self.pad, dtype=torch.float32, device=self.device)
        tau = torch.as_tensor((t-t[0])/duration, dtype=torch.float32, device=self.device)
        def contrast(d):
            iwe = splat(xx - tau*d[ids, 0], yy - tau*d[ids, 1], ids, n, self.canvas)
            blurred = F.conv2d(iwe, self.kernel, padding=3)
            return blurred.flatten(1).var(1, unbiased=False)
        with torch.enable_grad():
            best = torch.zeros(n, 2, device=self.device)
            with torch.no_grad():
                initial = contrast(best)
                best_score = initial.clone()
                # Fixed search grid, chosen without MVSEC GT. Every patch keeps its own winner.
                for u in self.config["initial_displacements"]:
                    for v in self.config["initial_displacements"]:
                        candidate = torch.tensor([u, v], device=self.device).float().expand(n, 2)
                        score = contrast(candidate)
                        take = (score > best_score) & active
                        best[take] = candidate[take]; best_score[take] = score[take]
            d = best.detach().clone().requires_grad_()
            optimizer = torch.optim.Adam([d], lr=self.config["learning_rate"])
            for _ in range(self.config["iterations"]):
                optimizer.zero_grad(set_to_none=True)
                score = contrast(d)
                # A constant per-patch normalization changes no contrast maximizer.
                loss = -(score / initial.detach().clamp_min(1e-8) * active).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite patch CMax objective")
                with torch.no_grad():
                    take = (score > best_score) & active
                    best[take] = d[take]; best_score[take] = score[take]
                loss.backward(); optimizer.step()
                with torch.no_grad():
                    d.clamp_(-self.config["max_displacement"], self.config["max_displacement"])
            with torch.no_grad():
                score = contrast(d)
                take = (score > best_score) & active
                best[take] = d[take]; best_score[take] = score[take]
                best[~active] = 0
                velocity = best.T.reshape(1, 2, *self.grid) / duration
                dense = F.interpolate(velocity, size=(h, w), mode="nearest")[0].cpu().numpy()
        return dense, {"status": "ok", "active_patches": int(active.sum()), "patches": n,
                       "estimated_pixel_fraction": float(active.float().mean()),
                       "contrast_ratio": float((best_score/initial.clamp_min(1e-8))[active].mean()) if active.any() else None,
                       "bound_hit_patches": int((best.abs() >= self.config["max_displacement"]-.001).any(1).sum())}


def secrets_to_xy(dense):
    """Official code calls row/column x/y; our API uses column/row u/v."""
    return np.asarray(dense, dtype=np.float32)[[1, 0]].copy()


class SecretsCMax2022:
    def __init__(self, resolution, config, checkout):
        checkout = Path(checkout).resolve()
        revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if revision != SECRETS_SHA or dirty:
            raise ValueError("Official CMax checkout must be clean and pinned to " + SECRETS_SHA)
        # The official repository imports its package as `src`.
        if "src" in sys.modules and Path(sys.modules["src"].__file__).resolve() != checkout / "src/__init__.py":
            raise RuntimeError("Another package named src is already loaded; use an isolated process")
        sys.path.insert(0, str(checkout))
        upstream = importlib.import_module("src.solver.patch_contrast_pyramid")
        self.module = upstream
        c = deepcopy(config)
        c["solver"]["patch"].update(crop_height=resolution[0], crop_width=resolution[1])
        self.solver = upstream.PyramidalPatchContrastMaximization(tuple(resolution), {}, c["solver"], c["optimizer"], {}, None)
        # Remove only the profiling decorator (large per-window files/stdout); math is untouched.
        method = self.solver.run_scipy_over_scale
        self.solver.run_scipy_over_scale = types.MethodType(method.__wrapped__, self.solver)
        self.resolution = tuple(resolution)

    def estimate(self, events, seed=0):
        x, y, t, p = events
        if len(t) < 2 or t[-1] <= t[0]:
            return np.zeros((2, *self.resolution), np.float32), {"status": "zero_duration", "estimated_pixel_fraction": 0.}
        event_array = np.column_stack((y, x, t-t[0], p)).astype(np.float64)
        np.random.seed(seed); torch.manual_seed(seed)
        # No previous-window motion: both CMax methods have the same finite input history.
        self.solver.previous_frame_best_estimation = None
        sampler_cls = self.module.optuna.samplers.TPESampler
        rng = np.random.RandomState(seed)
        def seeded_sampler(*a, **kw):
            kw.setdefault("seed", int(rng.randint(0, 2**31-1)))
            return sampler_cls(*a, **kw)
        minimize = self.module.scipy_autograd.minimize
        convergence = []
        def recorded_minimize(*a, **kw):
            kw["options"] = {**kw.get("options", {}), "disp": False}
            result = minimize(*a, **kw)
            convergence.append({"success": bool(result.success), "status": int(result.status),
                                "iterations": int(result.nit), "objective": float(result.fun),
                                "message": str(result.message)})
            return result
        with torch.enable_grad(), patch.object(self.module.optuna.samplers, "TPESampler", seeded_sampler), \
                patch.object(self.module.scipy_autograd, "minimize", recorded_minimize), redirect_stdout(io.StringIO()):
            motion = self.solver.optimize(event_array)
            # Interpolate the optimized middle-time velocity without evolving it over 1 second.
            # The official interpolator also applies the solver's sign convention.
            dense = self.solver.interpolate_dense_flow_from_patch_numpy(motion[self.solver.current_scale])
        velocity = secrets_to_xy(dense)
        if not np.isfinite(velocity).all():
            raise FloatingPointError("Nonfinite official CMax prediction")
        return velocity, {"status": "ok", "estimated_pixel_fraction": 1., "scales": convergence,
                          "all_scales_converged": all(r["success"] for r in convergence)}
