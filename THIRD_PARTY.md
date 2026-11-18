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

## EV-LayerSegNet, Farah et al., CVPR Workshops 2025

- [Paper](https://arxiv.org/abs/2506.06596)
- [Code at a6e28dce82720f41c0e62781e01d015a8f3b2a26](https://github.com/Fulmen67/event_segmentation/tree/a6e28dce82720f41c0e62781e01d015a8f3b2a26)

No top-level license was found at the pinned revision. Code, pretrained models and data are not redistributed here. The optional adapter reads a user-side pinned checkout and temporarily scopes the original utils.LeakyDoReLU import. It exposes the original combined flow as a one-element flow list, retaining the original hard mask gating. Matched training is not claimed to reproduce the paper's training protocol.

## Contrast framework

[Gallego et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gallego_A_Unifying_Contrast_CVPR_2018_paper.pdf) supplies the optimization framework, not a neural baseline here. FWL uses event_flow's implementation attributed to Stoffregen and Scheerlinck et al., ECCV 2020; RSAT follows event_flow. Preserve all adaptation notices in the vendor files.
