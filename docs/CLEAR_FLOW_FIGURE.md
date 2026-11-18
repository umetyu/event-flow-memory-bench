# English three-column optical-flow figure

The current figure has exactly these columns:

1. **GT flow**
2. **ConvGRU EV-FlowNet** (count input, no voxel)
3. **EV-FlowNet + voxel (9 bins)**

The count-only and CMax progress panels have been removed. All labels and notes in the image are in English. The voxel model is the same UZH-trained paper-based reimplementation used previously; changing its display name does not change the checkpoint or establish successful reproduction of the paper.

## Scene selection

To replace the initial sequence segment with a more readable example, `scripts/clear_flow_figure.py preview` samples six evenly spaced positions in the available dt=4 GT records of each of the four MVSEC recordings. The 24 previews contain only the grayscale scene, events and GT; model predictions are not used to select the interval.

The selected interval is **indoor_flying3, 38.492991–38.620429 seconds** relative to the first event (GT interval index 301). A barrel and the surrounding background are visible in the scene, with a corresponding contour in GT flow. Valid GT covers 78.77% of the image. GT median displacement is 6.90 px and its 95th percentile is 9.44 px. A shared **0–10 px color scale** was fixed before neural inference so that these displacements are visible. Values above the scale saturate in every panel.

This is a qualitative example chosen for readability, not a random sample or a dataset-wide accuracy comparison. Its original scene and event image are saved separately as context.

## Inference and display

- ConvGRU uses the verified author checkpoint and 1,000-event count inputs. State is carried from event zero, with nonoverlapping windows aligned to the selected GT endpoint. Only the first warmup window may be shorter than 1,000 events.
- The voxel model uses the same validation-best checkpoint from the completed 100-epoch UZH run (checkpoint epoch 24), with 15,000 causal input events and 9 temporal bins.
- Outputs are converted to displacement over the same GT interval: `128 * GT_duration / input_duration` for ConvGRU, and `8 * GT_duration / input_duration` for the voxel model. The final-input average velocity is assumed to remain constant over the GT interval.
- Both estimated fields and GT use the same valid-GT mask and color scale. Black indicates invalid GT; white indicates zero displacement. No smoothing or independent panel normalization is added.
- The optional `-arrows` version overlays vectors on a regular grid with a common 2x display scale. The plain color version does not add arrows.
- Input history and training conditions differ between the models. The voxel checkpoint has produced near-zero flow; its original outputs are retained.

## Reproduce

```bash
PYTHONPATH=src python scripts/clear_flow_figure.py preview \
  --manifest runs/setup/mvsec.json --output runs/readable-scene-candidates

# Inspect the scene/GT contact sheets, then save selection.json before inference.
# For the delivered figure: ids=["indoor_flying3-gt0301-dt4"], color_max_px=10.
PYTHONPATH=src python scripts/clear_flow_figure.py infer \
  --manifest runs/setup/mvsec.json --candidates runs/readable-scene-candidates \
  --selection runs/readable-scene-candidates/selection.json \
  --author-run 'external/author-models/mlruns 2/42/EVFlowNet' \
  --checkpoint runs/zhu2019-uzh-seed0-3256469/best.pt \
  --trust-author-pickle --output runs/clear-flow-new --device cuda

PYTHONPATH=src python scripts/clear_flow_figure.py render --output runs/clear-flow-new
```

`scripts/slurm_izar_clear_flow.sh` runs the inference on Izar. The output includes PNG/SVG figures, the original scene preview, an NPZ with native and converted flows, and metadata containing selection rationale, hashes, exact input durations, and interval metrics. Data and model weights remain outside the Git repository.
