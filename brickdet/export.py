"""
Export a trained checkpoint for an on-device runtime and check it against PyTorch.

  python -m brickdet.export <target> <config> <checkpoint> <out>
  python -m brickdet.export ncnn configs/nanodet-plus-m_416.yml \\
      work/runs/brick_416/model_best/nanodet_model_best.pth work/export/brick416

Writes the target's files next to <out> plus <out>.json (input/output description, see SPEC.md).
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch

from . import WORK, use_nanodet
from .targets import TARGETS


class Detector(torch.nn.Module):
    """Raw network without post-processing: [1, 3, H, W] -> [1, N, 1 + 4 * (reg_max + 1)]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        m = self.model
        return m.head._forward_onnx(m.fpn(m.backbone(x)))


def meta(cfg, target):
    w, h = cfg.data.val.input_size
    mean, std = cfg.data.val.pipeline.normalize
    head = cfg.model.arch.head
    return {
        'target': target.__name__.rsplit('.', 1)[-1],
        'io': target.IO,
        'input_size': [w, h],
        'channels': 'BGR',
        'resize': 'stretch',
        'mean': list(mean),
        'std': list(std),
        'strides': list(head.strides),
        'reg_max': head.reg_max,
        'classes': list(cfg.class_names),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', choices=TARGETS)
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('out')
    a = ap.parse_args()
    use_nanodet()
    from nanodet.model.arch import build_model
    from nanodet.util import Logger, cfg, load_config, load_model_weight

    load_config(cfg, a.config)
    model = build_model(cfg.model)
    load_model_weight(model, torch.load(a.checkpoint, map_location='cpu'), Logger(-1, str(WORK), use_tensorboard=False))
    net = Detector(model).eval()
    w, h = cfg.data.val.input_size
    x = torch.randn(1, 3, h, w)
    with torch.no_grad():
        ref = net(x).numpy()

    target = importlib.import_module(f'{__package__}.targets.{a.target}')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    files = target.export(net, x, a.out)
    json.dump(meta(cfg, target), open(a.out + '.json', 'w'), indent=1)
    diff = float(np.abs(target.load(a.out)(x.numpy()) - ref).max())
    print('output', ref.shape, 'max diff vs torch', round(diff, 4))
    for f in files + [a.out + '.json']:
        print('written', f, os.path.getsize(f) // 1024, 'KB')


if __name__ == '__main__':
    main()
