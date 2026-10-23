# brick-detection-ml

[English](README.md)

사진 속 블록 장난감 부품이 있는 자리를 찾는 작은 모델을 학습하고, 기기 안 런타임용으로 내보낸다.
한 종류(`brick`)만 찾는다. 무슨 부품인지는 다른 곳에서 알아낸다 (예: 상자마다 잘라서 인식기로 보내기).

- 모델: [NanoDet-Plus](https://github.com/RangiLyu/nanodet)-m, 416 × 416 (ncnn fp16 으로 2.3 MB)
- 데이터: Boiński (2023), 실사진 + 렌더, CC BY 4.0
- 내보내기: ncnn (안드로이드). 다른 런타임은 대상(target)으로 추가 ([확장](#확장))
- 앱이 따라야 할 입출력 약속: [SPEC.md](SPEC.md)

## 결과

416, 100바퀴, 검증 세트(여러 개 든 실사진):

| 바퀴 | mAP | AP50 | AP75 | 작은 부품 |
|---|---|---|---|---|
| 10 | 0.695 | 0.942 | 0.827 | 0.450 |
| 40 | 0.775 | 0.944 | 0.893 | 0.531 |
| **80 (최고)** | **0.797** | 0.940 | **0.902** | **0.574** |
| 100 | 0.793 | 0.935 | 0.892 | 0.563 |

## 준비

Python 3.13 에서 확인.

```sh
PYTHON=python3.13 ./setup.sh     # .venv + work/nanodet (고정 커밋, patches/nanodet.patch 적용)
source .venv/bin/activate
```

## 데이터

Most Wiedzy 저장소에서 받아 `work/data/raw/` 에 푼다:

- [Tagged images with LEGO bricks](https://doi.org/10.34808/anq4-rn44) → `work/data/raw/02 - tagged1/{photos,renders}/`
- [Tagged images with LEGO bricks part 2](https://doi.org/10.34808/7kk9-tn08) → `work/data/raw/03 - tagged2/` (선택. `train_all.json` 과 검증 일부에만 쓰임)

```sh
python -m brickdet.prepare       # → work/data/coco (긴 변 640 px, COCO json)
```

## 학습

```sh
python -m brickdet.train configs/nanodet-plus-m_416.yml            # --device auto|mps|cuda|cpu
python -m brickdet.bench configs/nanodet-plus-m_416.yml            # 속도만 재기
```

결과: `work/runs/brick_416/` (`model_best/nanodet_model_best.pth`, `model_last.ckpt`, 기록).
이어 하려면 설정의 `schedule.resume` 을 켠다.

## 내보내기

```sh
python -m brickdet.export ncnn configs/nanodet-plus-m_416.yml \
    work/runs/brick_416/model_best/nanodet_model_best.pth work/export/brick416
python -m brickdet.check work/export/brick416 work/check path/to/photo.jpg
```

`export` 는 `brick416.param`, `brick416.bin`, `brick416.json` 을 쓰고 PyTorch 와의 최대 차이를 보여 준다 (fp16 저장이라 0.05 정도).
`check` 는 내보낸 파일을 참고 후처리로 돌려 상자를 그린다.

## 구조

```
configs/            NanoDet 설정 (경로는 레포 루트 기준)
patches/            NanoDet 수정분 (PyTorch 2, MPS)
brickdet/
  prepare.py        데이터 → COCO
  train.py          NanoDet 학습 (--device)
  bench.py          학습 속도
  export.py         체크포인트 → 대상 파일 + <out>.json, PyTorch 와 비교
  targets/          런타임마다 모듈 하나 (ncnn.py)
  decode.py         참고 전처리·후처리 (SPEC.md)
  check.py          내보낸 모델로 상자 그리기
work/               데이터, NanoDet, 학습 결과, 내보낸 파일 (git 제외)
```

## 확장

- **내보내기 대상** (Core ML, TFLite, ONNX …): `brickdet/targets/<이름>.py` 에 `IO`, `export(net, x, out)`, `load(out)` 를 만들고
  (`brickdet/targets/__init__.py` 참고) `requirements.txt` 에 패키지를, SPEC.md 표에 한 줄을 더한다.
  `python -m brickdet.export <이름> …`, `python -m brickdet.check …` 는 그대로 쓴다.
- **학습 기기**: `--device cuda|cpu` 는 그대로 된다. GPU 여러 개는 `brickdet/train.py` 의 `GPU_IDS`.
- **입력 크기·모델**: `configs/` 파일을 복사한다. stride, `reg_max`, mean/std, 출력 모양이 바뀌면 SPEC.md 를 고친다
  (`decode.py` 는 `<out>.json` 에서 읽는다).
- **데이터·종류**: `brickdet/prepare.py` (`GROUPS`, `TRAIN_GROUPS`, 카테고리), 설정의 `num_classes`·`class_names`.
  종류가 둘 이상이면 출력 모양이 바뀐다 (SPEC.md, `decode.py`).

## 라이선스

코드: [Apache License 2.0](LICENSE). NanoDet (Apache-2.0) 사용, [NOTICE](NOTICE) 참고.

학습 데이터: Boiński, T. M. (2023). *Photos and rendered images of LEGO bricks*. Scientific Data 10, 811.
[doi:10.1038/s41597-023-02682-2](https://doi.org/10.1038/s41597-023-02682-2), CC BY 4.0.
이 데이터로 학습한 모델을 배포할 때는 이 출처를 함께 적는다.
