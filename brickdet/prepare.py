"""
Pascal VOC (Boiński 2023) -> COCO with one category "brick", images downscaled.

  python -m brickdet.prepare [--src work/data/raw] [--out work/data/coco] [--size 640]

Groups (file name prefix):
  photos_   real photos, 1-32 bricks each           tagged1/photos
  renders_  renders on a conveyor belt              tagged1/renders
  single_   one labelled part per folder (part 2)   tagged2; other parts in the
            same photo may be unlabelled, so it is left out of train.json
Outputs: images/{train,val}/, annotations/train.json (photos+renders),
  train_all.json (+single), val.json, summary.json.
Val: 30% of multi-brick photos + 3% of the rest, fixed seed.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

from . import WORK

GROUPS = [
    ('photos', '02 - tagged1/photos/*/*.xml'),
    ('renders', '02 - tagged1/renders/*/*.xml'),
    ('single', '03 - tagged2/*/*/*.xml'),
]
TRAIN_GROUPS = ('photos', 'renders')
SOURCE = 'Boiński T.M. (2023) Photos and rendered images of LEGO bricks. Scientific Data 10:811, CC BY 4.0'
BOX = re.compile(r'<bndbox>\s*<xmin>([\d.]+)</xmin>\s*<ymin>([\d.]+)</ymin>\s*<xmax>([\d.]+)</xmax>\s*<ymax>([\d.]+)</ymax>', re.S)


def read_boxes(xml):
    return [tuple(float(v) for v in m) for m in BOX.findall(open(xml, encoding='utf-8', errors='replace').read())]


def convert(job):
    """Resize one image; returns (name, w, h, [x, y, w, h] boxes) or None."""
    xml, dst, size = job
    jpg = xml[:-4] + '.jpg'
    if not os.path.exists(jpg):
        return None
    boxes = read_boxes(xml)
    if not boxes:
        return None
    im = Image.open(jpg).convert('RGB')  # xml size == pixel size, no EXIF rotation
    w, h = im.size
    s = min(1.0, size / max(w, h))
    nw, nh = round(w * s), round(h * s)
    if s < 1:
        im = im.resize((nw, nh), Image.LANCZOS)
    im.save(dst, quality=90)
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, min(w, x1)) * s, max(0, min(w, x2)) * s))
        y1, y2 = sorted((max(0, min(h, y1)) * s, max(0, min(h, y2)) * s))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            out.append([round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)])
    return (os.path.basename(dst), nw, nh, out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(WORK / 'data' / 'raw'))
    ap.add_argument('--out', default=str(WORK / 'data' / 'coco'))
    ap.add_argument('--size', type=int, default=640, help='long side in px')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()

    items = [(g, p) for g, pattern in GROUPS for p in sorted(glob.glob(os.path.join(a.src, pattern)))]
    if not items:
        sys.exit(f'no data in {a.src} (see README)')
    rnd = random.Random(7)
    split = {}
    for group, p in items:
        multi = group == 'photos' and len(read_boxes(p)) >= 2
        split[p] = 'val' if rnd.random() < (0.30 if multi else 0.03) else 'train'

    for s in ('train', 'val'):
        os.makedirs(os.path.join(a.out, 'images', s), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'annotations'), exist_ok=True)
    jobs = [(p, os.path.join(a.out, 'images', split[p], f'{g}_{i:06d}.jpg'), a.size) for i, (g, p) in enumerate(items)]

    coco = {k: {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'brick'}]} for k in ('train', 'train_all', 'val')}
    summary = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for n, ((group, p), r) in enumerate(zip(items, ex.map(convert, jobs, chunksize=32)), start=1):
            if n % 2000 == 0:
                print(f'{n}/{len(jobs)}', flush=True)
            if r is None:
                continue
            name, w, h, boxes = r
            s = split[p]
            keys = ['val'] if s == 'val' else ['train_all'] + (['train'] if group in TRAIN_GROUPS else [])
            for k in keys:
                c = coco[k]
                iid = len(c['images']) + 1
                c['images'].append({'id': iid, 'file_name': name, 'width': w, 'height': h})
                for b in boxes:
                    c['annotations'].append({'id': len(c['annotations']) + 1, 'image_id': iid, 'category_id': 1,
                                             'bbox': b, 'area': round(b[2] * b[3], 1), 'iscrowd': 0})
            g = summary.setdefault(f'{s}/{group}', {'images': 0, 'boxes': 0})
            g['images'] += 1
            g['boxes'] += len(boxes)

    for k, c in coco.items():
        json.dump(c, open(os.path.join(a.out, 'annotations', f'{k}.json'), 'w'))
    summary['_'] = {'size': a.size, 'source': SOURCE}
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
