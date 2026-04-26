# AgroLens

AgroLens is a plant disease detection MVP. It accepts a leaf image, runs a locally bundled EfficientNet-B1 ONNX model through a Next.js API route, and returns the top disease predictions with confidence scores.

## Monorepo

```text
.
├── src/            Next.js app and API routes
├── models/         ONNX model artifacts loaded by the API
└── ml/             PyTorch training and ONNX export pipeline
```

## Run Locally

```bash
pnpm install
pnpm dev
```

Open `http://localhost:3000`.

## Production Build

```bash
pnpm build
pnpm start
```

## Train The Model

```bash
pnpm ml:setup
pnpm ml:train
pnpm ml:sync
```

`pnpm ml:train` runs the production EfficientNet-B1 training command:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0 .venv/bin/python train_plant_disease.py --preset full --model-name efficientnet_b1 --epochs 15 --batch-size 64 --num-workers 4 --out-dir outputs_production_b1_15ep
```

The current bundled model was trained for 15 epochs and reached 99.94% best validation accuracy on the PlantVillage validation split.

## Model Files

The app expects these files under `models/`:

```text
models/class_names.json
models/plant_disease_model.onnx
models/plant_disease_model.onnx.data
```

Keep the `.onnx` and `.onnx.data` files together. The API route loads them from `process.cwd()/models`.

## API

`POST /api/predict`

Multipart form field:

```text
image: JPEG, PNG, or WebP, max 8 MB
```

Response:

```json
{
  "model": "efficientnet_b1_plant_disease",
  "confidenceBand": "high",
  "predictions": [
    {
      "label": "Potato___Early_blight",
      "crop": "Potato",
      "condition": "Early blight",
      "confidence": 0.876
    }
  ]
}
```

## Notes

Validation accuracy was high on PlantVillage-style images. Real field photos may be harder because of lighting, background, blur, and multiple leaves. Treat low-confidence predictions as uncertain and ask for a clearer image.
