import argparse
import copy
import json
import os
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dataset_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser()
    else:
        import kagglehub

        root = Path(kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset"))

    candidates = [
        root / "train",
        root
        / "New Plant Diseases Dataset(Augmented)"
        / "New Plant Diseases Dataset(Augmented)"
        / "train",
    ]
    for train_dir in candidates:
        valid_dir = train_dir.parent / "valid"
        if train_dir.exists() and valid_dir.exists():
            return train_dir.parent
    raise FileNotFoundError(f"Could not find train/valid under {root}")


def make_subset(ds: ImageFolder, max_samples: int | None) -> ImageFolder | Subset:
    if not max_samples or max_samples >= len(ds):
        return ds
    rng = np.random.default_rng(42)
    indices = rng.choice(len(ds), size=max_samples, replace=False)
    return Subset(ds, indices.tolist())


def make_balanced_subset(ds: ImageFolder, max_samples: int | None) -> ImageFolder | Subset:
    if not max_samples or max_samples >= len(ds):
        return ds
    rng = np.random.default_rng(42)
    targets = np.array(ds.targets)
    classes = np.unique(targets)
    per_class = max(1, max_samples // len(classes))
    selected: list[int] = []
    for class_idx in classes:
        class_indices = np.flatnonzero(targets == class_idx)
        take = min(per_class, len(class_indices))
        selected.extend(rng.choice(class_indices, size=take, replace=False).tolist())
    remaining = max_samples - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        candidates = np.array([i for i in range(len(ds)) if i not in selected_set])
        selected.extend(rng.choice(candidates, size=min(remaining, len(candidates)), replace=False).tolist())
    rng.shuffle(selected)
    return Subset(ds, selected[:max_samples])


def save_batch_preview(loader: DataLoader, class_names: list[str], out_path: Path, n: int = 8) -> None:
    imgs, labels = next(iter(loader))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    imgs = (imgs[:n].cpu() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(1, min(n, len(imgs)), figsize=(2 * min(n, len(imgs)), 2.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, img, lbl in zip(axes, imgs, labels[:n]):
        ax.imshow(img)
        ax.set_title(class_names[lbl.item()].split("___")[-1][:12], fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_model(num_classes: int, model_name: str, freeze_backbone: bool) -> nn.Module:
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not any(head_key in name for head_key in ("classifier", "fc", "head")):
                param.requires_grad = False
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    train: bool,
    desc: str,
) -> tuple[float, float]:
    model.train(train)
    running_loss = 0.0
    running_correct = 0
    total = 0
    use_amp = device.type == "cuda"

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        preds = logits.argmax(1)
        running_loss += loss.item() * imgs.size(0)
        running_correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, running_correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epochs: int,
    class_names: list[str],
    img_size: int,
    out_dir: Path,
) -> tuple[nn.Module, dict[str, list[float]]]:
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>9} {'Time':>7}", flush=True)
    print("-" * 75, flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True, desc=f"epoch {epoch}/{epochs} train"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, train=False, desc=f"epoch {epoch}/{epochs} valid"
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = " *" if val_acc > best_val_acc else ""
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.4f}  {val_loss:>9.4f}  {val_acc:>8.4f}  {lr_now:>8.2e}  {elapsed:>5.0f}s{marker}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_weights,
                    "val_acc": best_val_acc,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "img_size": img_size,
                },
                out_dir / "best_model.pth",
            )

    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)", flush=True)
    model.load_state_dict(best_weights)
    return model, history


def plot_history(history: dict[str, list[float]], out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot([a * 100 for a in history["train_acc"]], label="Train")
    ax2.plot([a * 100 for a in history["val_acc"]], label="Val")
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("EfficientNet-B3 Plant Disease Detection", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def per_class_accuracy(model: nn.Module, loader: DataLoader, class_names: list[str], device: torch.device) -> np.ndarray:
    model.eval()
    correct = torch.zeros(len(class_names))
    total = torch.zeros(len(class_names))
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs).argmax(1)
            for c in range(len(class_names)):
                mask = labels == c
                total[c] += mask.sum().item()
                correct[c] += (preds[mask] == c).sum().item()
    acc = (correct / total.clamp(min=1)).numpy()
    worst = np.argsort(acc)[:5]
    best = np.argsort(acc)[-5:][::-1]
    print("Top-5 best classes", flush=True)
    for i in best:
        print(f"  {class_names[i]:45s} {acc[i] * 100:6.2f}%", flush=True)
    print("Top-5 hardest classes", flush=True)
    for i in worst:
        print(f"  {class_names[i]:45s} {acc[i] * 100:6.2f}%", flush=True)
    return acc


def export_onnx(model: nn.Module, img_size: int, out_path: Path) -> None:
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model_cpu,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )

    import onnx

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    logits = sess.run(["logits"], {"image": dummy.numpy()})[0]
    print(f"ONNX smoke test passed. Dummy logits shape: {logits.shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--preset", choices=["full", "fast-5m"], default="full")
    parser.add_argument("--model-name", default="efficientnet_b3")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-per-class", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    args = parser.parse_args()

    if args.preset == "fast-5m":
        args.epochs = 1 if args.epochs == 3 else args.epochs
        args.batch_size = 96 if args.batch_size == 32 else args.batch_size
        args.num_workers = 4 if args.num_workers == 2 else args.num_workers
        args.max_train_samples = 3800 if args.max_train_samples is None else args.max_train_samples
        args.max_val_samples = 1140 if args.max_val_samples is None else args.max_val_samples
        args.freeze_backbone = True
        args.skip_onnx = True
        args.skip_per_class = True
        args.skip_package = True

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = resolve_dataset_root(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "valid"
    device = resolve_device()
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {data_root}", flush=True)

    train_tf = T.Compose(
        [
            T.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(args.img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds_full = ImageFolder(train_dir, transform=train_tf)
    val_ds_full = ImageFolder(val_dir, transform=val_tf)
    train_ds = make_balanced_subset(train_ds_full, args.max_train_samples)
    val_ds = make_balanced_subset(val_ds_full, args.max_val_samples)
    class_names = train_ds_full.classes
    idx_to_class = {v: k for k, v in train_ds_full.class_to_idx.items()}
    (out_dir / "class_names.json").write_text(json.dumps(idx_to_class, indent=2))
    print(f"Classes: {len(class_names)}", flush=True)
    print(f"Train images: {len(train_ds):,} | Val images: {len(val_ds):,}", flush=True)

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    save_batch_preview(train_loader, class_names, out_dir / "sample_batch.png")
    model = build_model(len(class_names), args.model_name, args.freeze_backbone).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_name}", flush=True)
    print(f"Trainable params: {trainable_params:,} / {total_params:,}", flush=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start = time.time()
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device,
        args.epochs,
        class_names,
        args.img_size,
        out_dir,
    )
    print(f"Total training time: {(time.time() - start) / 60:.1f} min", flush=True)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, out_dir / "training_curves.png")
    if not args.skip_per_class:
        acc = per_class_accuracy(model, val_loader, class_names, device)
        (out_dir / "per_class_accuracy.json").write_text(
            json.dumps({class_names[i]: float(acc[i]) for i in range(len(class_names))}, indent=2)
        )

    onnx_path = out_dir / "plant_disease_model.onnx"
    if not args.skip_onnx:
        export_onnx(model, args.img_size, onnx_path)

    zip_path = out_dir / "plant_disease_outputs.zip"
    if not args.skip_package:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in [
                "best_model.pth",
                "plant_disease_model.onnx",
                "class_names.json",
                "training_curves.png",
                "sample_batch.png",
                "history.json",
                "per_class_accuracy.json",
                "plant_disease_model.onnx.data",
            ]:
                path = out_dir / fname
                if path.exists():
                    zf.write(path, arcname=fname)
        print(f"Packaged: {zip_path}", flush=True)


if __name__ == "__main__":
    main()
