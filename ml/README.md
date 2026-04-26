# AgroLens ML

This package contains the training pipeline that produced the EfficientNet-B1 PlantVillage classifier used by the Next.js app.

## Setup

```bash
pnpm ml:setup
```

This creates `ml/.venv` with Python 3.12 and installs the training dependencies with `uv`.

## Production Training

```bash
pnpm ml:train
```

Equivalent command from inside `ml/`:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0 .venv/bin/python train_plant_disease.py --preset full --model-name efficientnet_b1 --epochs 15 --batch-size 64 --num-workers 4 --out-dir outputs_production_b1_15ep
```

This is the command used for the current bundled model. On the MacBook MPS run, it completed in about 193 minutes and reached 99.94% best validation accuracy on the PlantVillage validation split.

## Fast Smoke Training

```bash
pnpm ml:train:fast
```

This only verifies that the pipeline runs. It is intentionally not production quality.

## Sync Model Into The App

After a successful production run:

```bash
pnpm ml:sync
```

That copies the ONNX export and class labels into `../models`, where the Next.js API route loads them.

## Training Artifacts

The current production run visuals are kept in `ml/artifacts/`:

```text
ml/artifacts/sample_batch.png
ml/artifacts/training_curves.png
```
