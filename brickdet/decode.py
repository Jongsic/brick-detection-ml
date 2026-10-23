"""
Reference pre/post-processing for an exported model (SPEC.md). Ports in apps should match this.
`meta` is the <out>.json written by brickdet.export.
"""
import math

import cv2
import numpy as np


def preprocess(bgr, meta):
    """BGR uint8 image -> float32 [1, 3, H, W]"""
    w, h = meta['input_size']
    x = (cv2.resize(bgr, (w, h)).astype(np.float32) - np.float32(meta['mean'])) / np.float32(meta['std'])
    return x.transpose(2, 0, 1)[None].copy()


def priors(meta):
    """[N, 3] (cx, cy, stride) in input pixels, in output row order."""
    w, h = meta['input_size']
    out = []
    for s in meta['strides']:
        ys, xs = np.mgrid[0:math.ceil(h / s), 0:math.ceil(w / s)]
        out.append(np.stack([xs.ravel() * s, ys.ravel() * s, np.full(xs.size, s)], 1))
    return np.concatenate(out).astype(np.float32)


def decode(y, meta, image_w, image_h, score=0.35):
    """Model output [1, N, C] -> [K, 5] (score, x1, y1, x2, y2) in image pixels."""
    y = y.reshape(-1, y.shape[-1])
    p = priors(meta)
    keep = y[:, 0] >= score
    y, p = y[keep], p[keep]
    bins = meta['reg_max'] + 1
    d = y[:, 1:].reshape(-1, 4, bins)
    d = np.exp(d - d.max(2, keepdims=True))
    d /= d.sum(2, keepdims=True)
    l, t, r, b = ((d * np.arange(bins)).sum(2) * p[:, 2:3]).T
    w, h = meta['input_size']
    sx, sy = image_w / w, image_h / h
    return np.stack([y[:, 0], (p[:, 0] - l) * sx, (p[:, 1] - t) * sy, (p[:, 0] + r) * sx, (p[:, 1] + b) * sy], 1)


def nms(boxes, iou=0.6):
    boxes = boxes[np.argsort(-boxes[:, 0], kind='stable')]
    keep = []
    for bx in boxes:
        if all(overlap(bx, k) < iou for k in keep):
            keep.append(bx)
    return np.array(keep).reshape(-1, 5)


def overlap(a, b):
    iw = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ih = max(0, min(a[4], b[4]) - max(a[2], b[2]))
    union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - iw * ih
    return iw * ih / union if union > 0 else 0


def detect(run, bgr, meta, score=0.35, iou=0.6):
    h, w = bgr.shape[:2]
    return nms(decode(run(preprocess(bgr, meta)), meta, w, h, score), iou)
