"use client";

import Image from "next/image";
import Link from "next/link";
import { DragEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Prediction = {
  label: string;
  crop: string;
  condition: string;
  confidence: number;
};

type PredictionResponse = {
  model: string;
  confidenceBand: "high" | "medium" | "low";
  predictions: Prediction[];
};

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
const BUTTON_BASE =
  "inline-flex h-10 items-center justify-center rounded-md px-4 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50";
const BUTTON_PRIMARY = `${BUTTON_BASE} bg-primary text-primary-foreground hover:bg-primary/90`;
const BUTTON_SECONDARY = `${BUTTON_BASE} border border-border bg-background hover:bg-secondary`;
const MIN_PREDICTION_MS = 1000;
const APP_SHELL = "mx-auto w-full max-w-[92rem] px-6";

function formatFileSize(bytes: number) {
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(mb >= 1 ? 1 : 2)} MB`;
}

function formatConfidence(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}

function createObjectUrl(file: File | null) {
  if (!file) return "";
  return URL.createObjectURL(file);
}

function validateImage(file: File) {
  if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
    return "Upload a JPEG, PNG, or WebP image.";
  }

  if (file.size <= 0 || file.size > MAX_IMAGE_BYTES) {
    return `Image must be smaller than ${formatFileSize(MAX_IMAGE_BYTES)}.`;
  }

  return "";
}

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const previewUrl = useMemo(() => createObjectUrl(file), [file]);
  const topPrediction = result?.predictions[0];

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const setSelectedFile = (nextFile: File | null) => {
    if (!nextFile) return;

    const validationError = validateImage(nextFile);
    if (validationError) {
      setError(validationError);
      return;
    }

    setFile(nextFile);
    setResult(null);
    setError(null);
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    setSelectedFile(event.dataTransfer.files[0] ?? null);
  };

  const submitImage = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!file || isPredicting) return;

    const formData = new FormData();
    formData.append("image", file);

    setIsPredicting(true);
    setResult(null);
    setError(null);

    try {
      const [response] = await Promise.all([
        fetch("/api/predict", {
          method: "POST",
          body: formData,
        }),
        wait(MIN_PREDICTION_MS),
      ]);

      const payload = (await response.json()) as
        | PredictionResponse
        | { error?: string };

      if (!response.ok) {
        throw new Error(
          "error" in payload && payload.error
            ? payload.error
            : `Prediction failed with status ${response.status}.`,
        );
      }

      if (!("predictions" in payload) || payload.predictions.length === 0) {
        throw new Error("The model returned no predictions.");
      }

      setResult(payload);
    } catch (caughtError) {
      setResult(null);
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Prediction failed.",
      );
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="border-b border-border bg-card">
        <div className={`${APP_SHELL} flex items-center justify-between gap-4 py-5`}>
          <div>
            <p className="text-2xl font-semibold">AgroLens</p>
            <p className="text-sm text-muted-foreground sm:text-base">
              Plant disease detection from leaf images
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden text-right text-xs text-muted-foreground sm:block">
              <p>EfficientNet-B1</p>
              <p>38 PlantVillage classes</p>
            </div>
            <Link className={BUTTON_SECONDARY} href="/settings">
              Settings
            </Link>
          </div>
        </div>
      </header>

      <section className={`${APP_SHELL} grid flex-1 items-center gap-6 py-8 lg:grid-cols-[minmax(0,1fr)_30rem]`}>
        <form onSubmit={submitImage} className="min-w-0">
          <div
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={(event) => {
              event.preventDefault();
              if (
                event.relatedTarget === null ||
                !event.currentTarget.contains(event.relatedTarget as Node)
              ) {
                setIsDragging(false);
              }
            }}
            onDrop={onDrop}
            className={`flex min-h-[50rem] flex-col rounded-md border border-dashed bg-card shadow-sm ${
              isDragging ? "border-primary bg-primary/5" : "border-border"
            }`}
          >
            <div className="flex items-center justify-between gap-3 border-b border-border px-5 py-4">
              <div className="min-w-0">
                <p className="text-base font-medium">Image upload</p>
                <p className="truncate text-xs text-muted-foreground">
                  {file ? `${file.name} · ${formatFileSize(file.size)}` : "JPEG, PNG, or WebP up to 8 MB"}
                </p>
              </div>
              <button
                type="button"
                className={`${BUTTON_SECONDARY} shrink-0`}
                onClick={() => inputRef.current?.click()}
              >
                Choose image
              </button>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED_IMAGE_TYPES.join(",")}
                className="hidden"
                onChange={(event) =>
                  setSelectedFile(event.target.files?.[0] ?? null)
                }
              />
            </div>

            <div className="flex flex-1 items-center justify-center p-5">
              {previewUrl ? (
                <div className="relative aspect-[4/3] w-full max-w-5xl overflow-hidden rounded-md border border-border bg-background">
                  <Image
                    src={previewUrl}
                    alt={file?.name ?? "Uploaded plant image"}
                    fill
                    unoptimized
                    sizes="(min-width: 1024px) 720px, 100vw"
                    className="object-contain"
                  />
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  className="flex min-h-[34rem] w-full max-w-5xl flex-col items-center justify-center rounded-md border border-border bg-background px-6 text-center transition-colors hover:bg-secondary/60"
                >
                  <span className="text-xl font-semibold">Drop a leaf image here</span>
                  <span className="mt-2 max-w-md text-sm text-muted-foreground">
                    The model works best on clear leaf photos with one primary subject.
                  </span>
                </button>
              )}
            </div>

            <div className="flex items-center justify-between gap-3 border-t border-border px-5 py-4">
              <p className="min-w-0 truncate text-sm text-muted-foreground">
                {error ??
                  (isPredicting
                    ? "Preparing diagnosis with the local model."
                    : "Model runs locally through the Next.js API route.")}
              </p>
              <button
                type="submit"
                className={`${BUTTON_PRIMARY} shrink-0`}
                disabled={!file || isPredicting}
              >
                {isPredicting ? "Analyzing" : "Analyze"}
              </button>
            </div>
          </div>
        </form>

        <aside className="min-h-[50rem] rounded-md border border-border bg-card shadow-sm">
          <div className="border-b border-border px-5 py-4">
            <p className="text-base font-medium">Diagnosis</p>
            <p className="text-xs text-muted-foreground">
              Top model predictions and confidence
            </p>
          </div>

          <div className="p-5">
            {isPredicting ? (
              <div className="flex min-h-[38rem] flex-col items-center justify-center rounded-md border border-border bg-background p-6 text-center">
                <div className="size-11 animate-spin rounded-full border-2 border-border border-t-primary" />
                <p className="mt-5 text-base font-semibold">Analyzing leaf image</p>
                <p className="mt-2 max-w-xs text-sm text-muted-foreground">
                  Running the trained disease model and ranking likely matches.
                </p>
              </div>
            ) : topPrediction ? (
              <div>
                <div className="rounded-md border border-border bg-background p-5">
                  <p className="text-xs uppercase text-muted-foreground">
                    Most likely
                  </p>
                  <p className="mt-2 text-2xl font-semibold">
                    {topPrediction.condition}
                  </p>
                  <p className="mt-1 text-base text-muted-foreground">
                    {topPrediction.crop}
                  </p>
                  <div className="mt-5 h-3 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full bg-primary"
                      style={{ width: formatConfidence(topPrediction.confidence) }}
                    />
                  </div>
                  <p className="mt-3 text-sm">
                    {formatConfidence(topPrediction.confidence)} confidence
                  </p>
                </div>

                <ul className="mt-5 space-y-2.5">
                  {result.predictions.map((prediction) => (
                    <li
                      key={prediction.label}
                      className="rounded-md border border-border bg-background p-4"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium">
                            {prediction.condition}
                          </p>
                          <p className="truncate text-xs text-muted-foreground">
                            {prediction.crop}
                          </p>
                        </div>
                        <p className="shrink-0 text-sm tabular-nums">
                          {formatConfidence(prediction.confidence)}
                        </p>
                      </div>
                    </li>
                  ))}
                </ul>

                {result.confidenceBand !== "high" ? (
                  <p className="mt-4 rounded-md border border-border bg-background p-3 text-xs text-muted-foreground">
                    Confidence is not high. Retake the photo with better light and a
                    single leaf filling most of the frame.
                  </p>
                ) : null}
              </div>
            ) : (
              <div className="flex min-h-[38rem] items-center rounded-md border border-border bg-background p-5 text-sm text-muted-foreground">
                Upload a plant leaf image to get the top disease predictions.
              </div>
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}
