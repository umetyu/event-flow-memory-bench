# Sources and local changes

## Hagenaars et al., NeurIPS 2021

- [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/39d4b545fb02556829aab1db805021c3-Paper.pdf)
- [Code at e81f9630a145da8c7e57b2cfd38cf6c0b47ac14f](https://github.com/tudelft/event_flow/tree/e81f9630a145da8c7e57b2cfd38cf6c0b47ac14f)
- MIT, Copyright (c) 2021 TU Delft. Full license retained in the vendor directory.

models/, loss/, dataloader/, utils/iwe.py are vendored. The model, loss and metric mathematics are unchanged. Packaging changes:

1. models/submodules.py and models/spiking_submodules.py use relative spiking_util imports.
2. loss/flow.py removes a sys.path mutation and uses a relative utils.iwe import.
3. Empty enclosing package initializers are added.

The wrapper isolates mutable class-level neuron kwargs between instances. Matched data sampling, training control and metrics orchestration are new code. The matched AEE mask and threshold convention are documented alternatives to upstream. Reproduction uses upstream AEE.

configs/paper_reference.csv transcribes RecEVFlowNet (paper label EV-FlowNet) and FireNet from Table 1, page 8. These are reference targets, not results from this project.

## Zhu et al., CVPR 2019 (reference [56] in Hagenaars)

- [Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Unsupervised_Event-Based_Learning_of_Optical_Flow_Depth_and_Egomotion_CVPR_2019_paper.html)
- [Author preprint, sections 3 and 4](https://arxiv.org/html/1812.08156)

Selected for two comparisons: published results and a separate UZH-trained reimplementation. Only [56] is selected for additional retraining in the current phase. `configs/literature/zhu2019_table1.csv` contains the eight published MVSEC results, not project measurements. Native training uses MVSEC outdoor_day2 and 9-bin volumes; the original scoring convention is retained. These values remain distinct from future UZH-trained measurements. The author described release constraints in [Issue #12](https://github.com/daniilidis-group/EV-FlowNet/issues/12#issuecomment-644892449); a verified CVPR 2019 implementation/checkpoint has not been integrated. The paper-based reimplementation is now in `src/event_flow_bench/zhu2019.py`; it reuses the vendored MIT MultiResUNet components but implements a separate timestamp/smoothness loss and native pixels-per-bin conversion. Architectural and training choices that cannot be verified against author code are documented in [ZHU2019.md](docs/ZHU2019.md). The author's RSS 2018 EV-FlowNet repository and this project's Hagenaars 5-bin voxel control must not be relabeled as the CVPR 2019 method. See [BASELINES.md](docs/BASELINES.md).

## EV-LayerSegNet, Farah et al., CVPR Workshops 2025

- [Paper](https://arxiv.org/abs/2506.06596)
- [Code at a6e28dce82720f41c0e62781e01d015a8f3b2a26](https://github.com/Fulmen67/event_segmentation/tree/a6e28dce82720f41c0e62781e01d015a8f3b2a26)

No top-level license was found at the pinned revision. Code, pretrained models and data are not redistributed here. The optional adapter reads a user-side pinned checkout and temporarily scopes the original utils.LeakyDoReLU import. It exposes the original combined flow as a one-element flow list, retaining the original hard mask gating. Matched training is not claimed to reproduce the paper's training protocol.

## Contrast framework

[Gallego et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gallego_A_Unifying_Contrast_CVPR_2018_paper.pdf) supplies the optimization framework, not a neural baseline here. FWL uses event_flow's implementation attributed to Stoffregen and Scheerlinck et al., ECCV 2020; RSAT follows event_flow. Preserve all adaptation notices in the vendor files.
