# Model I/O

What an app needs to run an exported model. The same contract holds for every export target;
only the file format and tensor names differ. `brickdet/decode.py` is the reference implementation.

The model finds where bricks are (one class, `brick`). It does not identify the part.

## Input

| | |
|---|---|
| shape | float32 `[1, 3, H, W]`, NCHW |
| size | `W × H` = 416 × 416 (or 320 × 320) |
| resize | stretch the whole image to `W × H` (no letterbox, aspect ratio not kept) |
| channels | BGR |
| normalize | `(x - mean) / std` per channel, mean `[103.53, 116.28, 123.675]`, std `[57.375, 57.12, 58.395]` |

## Output

float32 `[1, N, 1 + 4 × 8]`, one row per prior point.

- Priors: for each stride `s` in `8, 16, 32, 64` (in this order), a grid of
  `ceil(H/s)` rows × `ceil(W/s)` columns, row-major. The point at grid `(gx, gy)` is `(cx, cy) = (gx·s, gy·s)`.
  N = 3598 for 416, 2125 for 320.
- Column 0: score, sigmoid already applied.
- Columns 1–32: left, top, right, bottom distances, 8 logits each.
  `distance = s × Σₖ k · softmax(logits)ₖ` (k = 0..7).
- Box in input pixels: `(cx − left, cy − top, cx + right, cy + bottom)`.
  Scale x by `image_w / W` and y by `image_h / H` to get image pixels.

## Post-processing

Keep rows with score ≥ 0.35, sort by score, greedy NMS with IoU 0.6.

## Files

`brickdet.export` writes the target's model files and `<out>.json` with these values:

```json
{
 "target": "ncnn",
 "io": {"input": "in0", "output": "out0"},
 "input_size": [416, 416],
 "channels": "BGR",
 "resize": "stretch",
 "mean": [103.53, 116.28, 123.675],
 "std": [57.375, 57.12, 58.395],
 "strides": [8, 16, 32, 64],
 "reg_max": 7,
 "classes": ["brick"]
}
```

| target | files | input / output tensor |
|---|---|---|
| ncnn | `<out>.param`, `<out>.bin` (fp16 weights) | `in0` / `out0` |
