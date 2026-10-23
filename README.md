# brick-detection-ml

[한국어](README.ko.md)

Trains a small detector that finds where toy bricks are in a photo, and exports it for on-device runtimes.
One class (`brick`); identifying the part is left to something else (e.g. cropping each box and sending it to a recognizer).

- Model: [NanoDet-Plus](https://github.com/RangiLyu/nanodet)-m, 416 × 416 (2.3 MB as ncnn fp16)
- Data: Boiński (2023), real photos + renders, CC BY 4.0
- Export: ncnn (Android). Other runtimes can be added as targets (see [Extending](#extending)).
- Input/output contract for apps: [SPEC.md](SPEC.md)

## Results

416, 100 epochs, validation set (multi-brick real photos):

| epoch | mAP | AP50 | AP75 | AP small |
|---|---|---|---|---|
| 10 | 0.695 | 0.942 | 0.827 | 0.450 |
| 40 | 0.775 | 0.944 | 0.893 | 0.531 |
| **80 (best)** | **0.797** | 0.940 | **0.902** | **0.574** |
| 100 | 0.793 | 0.935 | 0.892 | 0.563 |

## Setup

Tested with Python 3.13.

```sh
PYTHON=python3.13 ./setup.sh     # .venv + work/nanodet (pinned commit, patches/nanodet.patch applied)
source .venv/bin/activate
```

## Data

Download from the Most Wiedzy repository and unzip into `work/data/raw/`:

- [Tagged images with LEGO bricks](https://doi.org/10.34808/anq4-rn44) → `work/data/raw/02 - tagged1/{photos,renders}/`
- [Tagged images with LEGO bricks part 2](https://doi.org/10.34808/7kk9-tn08) → `work/data/raw/03 - tagged2/` (optional; only used for `train_all.json` and part of val)

```sh
python -m brickdet.prepare       # → work/data/coco (640 px long side, COCO json)
```

## Train

```sh
python -m brickdet.train configs/nanodet-plus-m_416.yml            # --device auto|mps|cuda|cpu
python -m brickdet.bench configs/nanodet-plus-m_416.yml            # throughput only
```

Output: `work/runs/brick_416/` (`model_best/nanodet_model_best.pth`, `model_last.ckpt`, logs).
To resume, set `schedule.resume` in the config.

## Export

```sh
python -m brickdet.export ncnn configs/nanodet-plus-m_416.yml \
    work/runs/brick_416/model_best/nanodet_model_best.pth work/export/brick416
python -m brickdet.check work/export/brick416 work/check path/to/photo.jpg
```

`export` writes `brick416.param`, `brick416.bin`, `brick416.json` and prints the max difference from PyTorch (≈ 0.05 with fp16 weights).
`check` runs the exported file with the reference post-processing and draws the boxes.

## Layout

```
configs/            NanoDet configs (paths relative to the repo root)
patches/            changes to NanoDet (PyTorch 2, MPS)
brickdet/
  prepare.py        dataset → COCO
  train.py          NanoDet training with --device
  bench.py          training throughput
  export.py         checkpoint → target files + <out>.json, checked against PyTorch
  targets/          one module per runtime (ncnn.py)
  decode.py         reference pre/post-processing (SPEC.md)
  check.py          draw boxes with an exported model
work/               data, NanoDet checkout, runs, exports (git-ignored)
```

## Extending

- **Export target** (Core ML, TFLite, ONNX, …): add `brickdet/targets/<name>.py` with `IO`, `export(net, x, out)` and `load(out)`
  (see `brickdet/targets/__init__.py`), add its packages to `requirements.txt`, and a row to the table in SPEC.md.
  `python -m brickdet.export <name> …` and `python -m brickdet.check …` then work unchanged.
- **Training machine**: `--device cuda|cpu` works as is. For several GPUs, edit `GPU_IDS` in `brickdet/train.py`.
- **Input size / model**: copy a file in `configs/`. If strides, `reg_max`, mean/std or output layout change, update SPEC.md;
  `decode.py` reads them from `<out>.json`.
- **Data or classes**: `brickdet/prepare.py` (`GROUPS`, `TRAIN_GROUPS`, category), then `num_classes` / `class_names` in the configs.
  More than one class changes the output layout (SPEC.md, `decode.py`).

## License

Code: [Apache License 2.0](LICENSE). Uses NanoDet (Apache-2.0); see [NOTICE](NOTICE).

Training data: Boiński, T. M. (2023). *Photos and rendered images of LEGO bricks*. Scientific Data 10, 811.
[doi:10.1038/s41597-023-02682-2](https://doi.org/10.1038/s41597-023-02682-2), CC BY 4.0.
Models trained on it should carry this attribution when distributed.
