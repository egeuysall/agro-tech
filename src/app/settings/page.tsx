import Link from "next/link";

const APP_SHELL = "mx-auto w-full max-w-[92rem] px-6";

export default function SettingsPage() {
  return (
    <main className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="border-b border-border bg-card">
        <div className={`${APP_SHELL} flex items-center justify-between gap-4 py-5`}>
          <div>
            <p className="text-2xl font-semibold">Model Settings</p>
            <p className="text-sm text-muted-foreground sm:text-base">
              Local inference configuration for AgroLens
            </p>
          </div>
          <Link
            href="/"
            className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            Back to app
          </Link>
        </div>
      </header>

      <section className={`${APP_SHELL} flex flex-1 items-center py-8`}>
        <div className="grid min-h-[50rem] w-full grid-rows-[auto_auto_1fr] gap-6 rounded-md border border-border bg-card p-8 shadow-sm">
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_30rem]">
            <div>
              <p className="text-xl font-semibold">Production Model</p>
              <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                AgroLens uses the locally trained EfficientNet-B1 plant disease
                model through the Next.js API route. External AI gateway
                configuration is no longer used.
              </p>
            </div>

            <div className="rounded-md border border-border bg-background p-8">
              <p className="text-sm font-medium">Runtime</p>
              <p className="mt-3 text-base text-muted-foreground">
                Local server-side inference with ONNX Runtime and Sharp image
                preprocessing.
              </p>
            </div>
          </div>

          <div className="grid gap-7">
            <dl className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
              <div className="min-h-44 rounded-md border border-border bg-background p-8">
                <dt className="text-sm text-muted-foreground">Route</dt>
                <dd className="mt-6 text-2xl font-semibold">/api/predict</dd>
              </div>
              <div className="min-h-44 rounded-md border border-border bg-background p-8">
                <dt className="text-sm text-muted-foreground">Model</dt>
                <dd className="mt-6 text-2xl font-semibold">EfficientNet-B1 ONNX</dd>
              </div>
              <div className="min-h-44 rounded-md border border-border bg-background p-8">
                <dt className="text-sm text-muted-foreground">Input</dt>
                <dd className="mt-6 text-2xl font-semibold">JPEG, PNG, WebP</dd>
              </div>
              <div className="min-h-44 rounded-md border border-border bg-background p-8">
                <dt className="text-sm text-muted-foreground">Classes</dt>
                <dd className="mt-6 text-2xl font-semibold">38 PlantVillage labels</dd>
              </div>
            </dl>

            <div className="grid gap-6 lg:grid-cols-3">
              <div className="min-h-48 rounded-md border border-border bg-background p-7">
                <p className="text-lg font-semibold">Upload Rules</p>
                <ul className="mt-5 space-y-3 text-sm text-muted-foreground">
                  <li>One image per prediction</li>
                  <li>Maximum upload size is 8 MB</li>
                  <li>Images are normalized to 224 x 224</li>
                </ul>
              </div>

              <div className="min-h-48 rounded-md border border-border bg-background p-7">
                <p className="text-lg font-semibold">Confidence Bands</p>
                <dl className="mt-5 space-y-3 text-sm">
                  <div className="flex items-center justify-between gap-4">
                    <dt className="text-muted-foreground">High</dt>
                    <dd className="font-medium">85%+</dd>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <dt className="text-muted-foreground">Medium</dt>
                    <dd className="font-medium">50-85%</dd>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <dt className="text-muted-foreground">Low</dt>
                    <dd className="font-medium">Below 50%</dd>
                  </div>
                </dl>
              </div>

              <div className="min-h-48 rounded-md border border-border bg-background p-7">
                <p className="text-lg font-semibold">Output</p>
                <ul className="mt-5 space-y-3 text-sm text-muted-foreground">
                  <li>Top five disease predictions</li>
                  <li>Crop and condition labels</li>
                  <li>Confidence score for each result</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
