"""
Run an exported model on images and draw the boxes.

  python -m brickdet.check <model> <out_dir> image... [--score 0.35] [--iou 0.6]
  python -m brickdet.check work/export/brick416 work/check work/data/coco/images/val/photos_002838.jpg
"""
import argparse
import importlib
import json
import os

import cv2

from .decode import detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', help='export <out> path (reads <model>.json)')
    ap.add_argument('out_dir')
    ap.add_argument('images', nargs='+')
    ap.add_argument('--score', type=float, default=0.35)
    ap.add_argument('--iou', type=float, default=0.6)
    a = ap.parse_args()
    meta = json.load(open(a.model + '.json'))
    run = importlib.import_module(f'{__package__}.targets.{meta["target"]}').load(a.model)
    os.makedirs(a.out_dir, exist_ok=True)
    for p in a.images:
        img = cv2.imread(p)
        boxes = detect(run, img, meta, a.score, a.iou)
        for _, x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), max(2, img.shape[1] // 300))
        cv2.imwrite(os.path.join(a.out_dir, os.path.basename(p)), img)
        print(os.path.basename(p), len(boxes), 'boxes')


if __name__ == '__main__':
    main()
