"""
Exports the object detector (D-FINE-L, pretrained on Objects365) to models/dfine_l_obj365.onnx.
Run it with scripts\\export_detector.bat, which prepares a separate export environment (PyTorch CPU,
D-FINE source, weights); the app itself only needs onnxruntime-directml.

usage: python export_detector.py <D-FINE repo> <weights.pth> <out.onnx> [size: n|s|m|l|x]

Changes compared with D-FINE's own tools/deployment/export_onnx.py:
  - static batch size 1 (DirectML cannot run the dynamic-batch Reshape nodes)
  - Integral uses a 2-D MatMul ([N,33] x [33,1]); DirectML computes the original 1-D-weight
    MatMul (F.linear with a vector) incorrectly, which breaks every box
"""
import os
import sys

repo, weights, out = sys.argv[1:4]
size = sys.argv[4] if len(sys.argv) > 4 else "l"
sys.path.insert(0, repo)

import torch                                    # noqa: E402
import torch.nn as nn                           # noqa: E402
import torch.nn.functional as F                 # noqa: E402
from src.core import YAMLConfig                 # noqa: E402
from src.zoo.dfine import dfine_decoder         # noqa: E402


def _integral_forward(self, x, project):
    """Same maths as D-FINE's Integral.forward, DirectML-safe."""
    shape = x.shape
    x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
    x = torch.matmul(x, project.to(x.device).reshape(-1, 1)).reshape(-1, 4)
    return x.reshape(list(shape[:-1]) + [-1])


dfine_decoder.Integral.forward = _integral_forward

config = os.path.join(repo, "configs", "dfine", "objects365", "dfine_hgnetv2_%s_obj365.yml" % size)
cfg = YAMLConfig(config, resume=weights)
if "HGNetv2" in cfg.yaml_cfg:
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
ckpt = torch.load(weights, map_location="cpu")
cfg.model.load_state_dict(ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)


model = Model().eval()
data = torch.rand(1, 3, 640, 640)
sizes = torch.tensor([[640, 640]])
with torch.no_grad():
    model(data, sizes)
    torch.onnx.export(model, (data, sizes), out, input_names=["images", "orig_target_sizes"],
                      output_names=["labels", "boxes", "scores"], opset_version=17,
                      do_constant_folding=True, dynamo=False)

import onnx                                     # noqa: E402
import onnxslim                                 # noqa: E402

onnx.save(onnxslim.slim(onnx.load(out)), out)
print("Exported %s (%.0f MB)" % (out, os.path.getsize(out) / 1e6))
