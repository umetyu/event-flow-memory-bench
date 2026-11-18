"""Original upstream architectures, with a common flow-list interface."""
from copy import deepcopy
import importlib
import importlib.util
import hashlib
from pathlib import Path
import subprocess
import sys
import types
import torch
from ._vendor.event_flow.models import model as upstream

EVENT_FLOW_SHA = "e81f9630a145da8c7e57b2cfd38cf6c0b47ac14f"
LAYERSEG_SHA = "a6e28dce82720f41c0e62781e01d015a8f3b2a26"
MODEL_NAMES = ("EVFlowNet", "RecEVFlowNet", "RNNRecEVFlowNet", "LeakyRecEVFlowNet",
               "FireFlowNet", "FireNet", "RNNFireNet", "LeakyFireNet", "LeakyFireFlowNet",
               "LIFFireNet", "ALIFFireNet", "PLIFFireNet", "XLIFFireNet", "LIFFireFlowNet",
               "SpikingRecEVFlowNet", "ALIFRecEVFlowNet", "PLIFRecEVFlowNet", "XLIFRecEVFlowNet")

def load_layerseg(path):
    path = Path(path).resolve()
    if not (path / "models/model.py").is_file():
        raise FileNotFoundError("Run python scripts/bootstrap_upstream.py event_segmentation first")
    sha = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if sha != LAYERSEG_SHA:
        raise ValueError(f"EV-LayerSegNet checkout must be pinned to {LAYERSEG_SHA}")
    # Verify only imported Python sources against Git blob IDs. This avoids a
    # working-tree refresh over the large tracked run artifacts in upstream.
    tree = subprocess.check_output(["git", "-C", str(path), "ls-tree", "-r", "HEAD", "--", "models", "utils/LeakyDoReLU.py"], text=True, timeout=30)
    for line in tree.splitlines():
        info, relative = line.split("\t", 1)
        if not relative.endswith(".py"):
            continue
        expected = info.split()[2]
        content = (path / relative).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if actual != expected:
            raise ValueError(f"EV-LayerSegNet source differs from pinned revision: {relative}")
    name = "efbench_layerseg_models"
    if name in sys.modules and Path(sys.modules[name].__file__).parent != path / "models":
        raise ValueError("Use one EV-LayerSegNet checkout per process")
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path / "models/__init__.py", submodule_search_locations=[str(path / "models")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    # Upstream uses an absolute utils.LeakyDoReLU import. Scope that alias to the
    # import, then restore the host process; never edit the source checkout.
    aliases = {key: sys.modules.get(key) for key in ("utils", "utils.LeakyDoReLU")}
    try:
        utils = types.ModuleType("utils")
        utils.__path__ = [str(path / "utils")]
        sys.modules["utils"] = utils
        spec = importlib.util.spec_from_file_location("utils.LeakyDoReLU", path / "utils/LeakyDoReLU.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["utils.LeakyDoReLU"] = module
        spec.loader.exec_module(module)
        return importlib.import_module(f"{name}.model").EVFlowNet_Segmentation
    finally:
        for key, previous in aliases.items():
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous

class FlowModel(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        kwargs = deepcopy(c["model"])
        name = kwargs["name"]
        self.layerseg = name == "EVLayerSegNet"
        external = kwargs.pop("source", "external/event_segmentation")
        if self.layerseg:
            if kwargs["base_num_channels"] != 32 or any(n % 16 for n in c["data"]["resolution"]):
                raise ValueError("Upstream EV-LayerSegNet requires base channels=32 and H,W divisible by 16")
            cls = load_layerseg(external)
        else:
            if name not in MODEL_NAMES:
                raise ValueError(f"Unknown model {name}; choices: {MODEL_NAMES}")
            cls = getattr(upstream, name)
            if hasattr(cls, "kwargs"):
                cls = type(name, (cls,), {"kwargs": deepcopy(cls.kwargs)})
        self.net = cls(kwargs)
        h, w = c["data"]["resolution"]
        self.net.init_cropping(w, h)

    def forward(self, inputs):
        out = self.net(inputs["event_voxel"], inputs["event_cnt"])
        if isinstance(out["flow"], torch.Tensor):
            out["flow"] = [out["flow"]]
        return out

    def reset(self):
        self.net.reset_states()

    def detach(self):
        self.net.detach_states()

def build_model(c):
    return FlowModel(c).to(c["device"])
