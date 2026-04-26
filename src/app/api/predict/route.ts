import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";
import * as ort from "onnxruntime-node";
import sharp from "sharp";

export const runtime = "nodejs";

const MODEL_DIR = path.join(process.cwd(), "models");
const MODEL_PATH = path.join(MODEL_DIR, "plant_disease_model.onnx");
const CLASS_NAMES_PATH = path.join(MODEL_DIR, "class_names.json");
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const TOP_K = 5;
const IMAGE_SIZE = 224;
const RESIZE_SIZE = 256;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

type Prediction = {
  label: string;
  crop: string;
  condition: string;
  confidence: number;
};

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let classNamesPromise: Promise<string[]> | null = null;

function getSession() {
  sessionPromise ??= ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["cpu"],
  });

  return sessionPromise;
}

async function getClassNames() {
  classNamesPromise ??= readFile(CLASS_NAMES_PATH, "utf8").then((contents) => {
    const mapping = JSON.parse(contents) as Record<string, string>;
    return Object.entries(mapping)
      .sort(([left], [right]) => Number(left) - Number(right))
      .map(([, label]) => label);
  });

  return classNamesPromise;
}

function parsePlantLabel(label: string) {
  const [rawCrop, rawCondition = "unknown"] = label.split("___");
  return {
    crop: rawCrop.replaceAll("_", " "),
    condition: rawCondition.replaceAll("_", " "),
  };
}

function softmax(logits: Float32Array) {
  let max = -Infinity;
  for (const value of logits) {
    if (value > max) max = value;
  }

  const probabilities = new Float32Array(logits.length);
  let sum = 0;
  for (let index = 0; index < logits.length; index += 1) {
    const value = Math.exp(logits[index] - max);
    probabilities[index] = value;
    sum += value;
  }

  for (let index = 0; index < probabilities.length; index += 1) {
    probabilities[index] = probabilities[index] / sum;
  }

  return probabilities;
}

async function preprocessImage(buffer: Buffer) {
  const pixels = await sharp(buffer, { failOn: "error" })
    .rotate()
    .resize(RESIZE_SIZE, RESIZE_SIZE, {
      fit: "cover",
      position: "centre",
    })
    .extract({
      left: Math.floor((RESIZE_SIZE - IMAGE_SIZE) / 2),
      top: Math.floor((RESIZE_SIZE - IMAGE_SIZE) / 2),
      width: IMAGE_SIZE,
      height: IMAGE_SIZE,
    })
    .removeAlpha()
    .raw()
    .toBuffer();

  const input = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE);
  const planeSize = IMAGE_SIZE * IMAGE_SIZE;

  for (let pixelIndex = 0; pixelIndex < planeSize; pixelIndex += 1) {
    const sourceOffset = pixelIndex * 3;
    input[pixelIndex] = pixels[sourceOffset] / 255;
    input[planeSize + pixelIndex] = pixels[sourceOffset + 1] / 255;
    input[planeSize * 2 + pixelIndex] = pixels[sourceOffset + 2] / 255;
  }

  for (let channel = 0; channel < 3; channel += 1) {
    const channelOffset = channel * planeSize;
    for (let pixelIndex = 0; pixelIndex < planeSize; pixelIndex += 1) {
      input[channelOffset + pixelIndex] =
        (input[channelOffset + pixelIndex] - MEAN[channel]) / STD[channel];
    }
  }

  return new ort.Tensor("float32", input, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function createPredictions(probabilities: Float32Array, classNames: string[]) {
  return Array.from(probabilities)
    .map((confidence, index) => ({
      confidence,
      index,
      label: classNames[index] ?? `class_${index}`,
    }))
    .sort((left, right) => right.confidence - left.confidence)
    .slice(0, TOP_K)
    .map(({ label, confidence }) => {
      const parsed = parsePlantLabel(label);
      return {
        label,
        crop: parsed.crop,
        condition: parsed.condition,
        confidence,
      } satisfies Prediction;
    });
}

export async function POST(request: Request) {
  const contentLength = Number(request.headers.get("content-length") ?? 0);
  if (contentLength > MAX_IMAGE_BYTES + 1024 * 1024) {
    return NextResponse.json(
      { error: "Request body is too large." },
      { status: 413 },
    );
  }

  const formData = await request.formData();
  const image = formData.get("image");

  if (!(image instanceof File)) {
    return NextResponse.json({ error: "Upload exactly one image file." }, { status: 400 });
  }

  if (!ALLOWED_TYPES.has(image.type)) {
    return NextResponse.json(
      { error: "Only JPEG, PNG, and WebP images are supported." },
      { status: 415 },
    );
  }

  if (image.size <= 0 || image.size > MAX_IMAGE_BYTES) {
    return NextResponse.json(
      { error: "Image must be smaller than 8 MB." },
      { status: 413 },
    );
  }

  try {
    const [session, classNames] = await Promise.all([getSession(), getClassNames()]);
    const tensor = await preprocessImage(Buffer.from(await image.arrayBuffer()));
    const outputs = await session.run({ image: tensor });
    const logits = outputs.logits?.data;

    if (!(logits instanceof Float32Array)) {
      return NextResponse.json({ error: "Model returned an invalid output." }, { status: 500 });
    }

    const predictions = createPredictions(softmax(logits), classNames);
    const top = predictions[0];
    const confidenceBand =
      top.confidence >= 0.85 ? "high" : top.confidence >= 0.5 ? "medium" : "low";

    return NextResponse.json({
      model: "efficientnet_b1_plant_disease",
      image: {
        name: image.name,
        type: image.type,
        size: image.size,
      },
      confidenceBand,
      predictions,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Prediction failed unexpectedly.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
