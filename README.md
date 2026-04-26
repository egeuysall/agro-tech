# AgroLens

AgroLens is a Next.js plant disease detection MVP. It accepts a leaf image, runs a locally bundled EfficientNet-B1 ONNX model through a Next.js API route, and returns the top disease predictions with confidence scores.

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
