"""Extended validation-only training for the final mammography comparison.

The original ten-epoch runs in notebook 05 remain untouched. This experiment
uses the same data, architecture, preprocessing, optimizer, and focal loss.
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
RESULT_DIR = ROOT / "results" / "training_stability"
MODEL_DIR = ROOT / "models" / "training_stability"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 512
DEFAULT_BATCH_SIZE = 8
MIN_EFFECTIVE_BATCH_SIZE = 8
DEFAULT_WORKERS = 4
ENCODER_LR = 1e-5
HEAD_LR = 1e-4
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResizeWithPadding:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.copy()
        image.thumbnail((self.size, self.size), Image.Resampling.BILINEAR)
        width, height = image.size
        padding = (
            (self.size - width) // 2,
            (self.size - height) // 2,
            self.size - width - (self.size - width) // 2,
            self.size - height - (self.size - height) // 2,
        )
        return TF.pad(image, padding, fill=0)


TRANSFORM = transforms.Compose(
    [
        ResizeWithPadding(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class MammographyDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, multiview: bool):
        self.rows = rows.reset_index(drop=True)
        self.multiview = multiview

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        with Image.open(row["CC_path"]) as image:
            cc = TRANSFORM(image.convert("RGB"))
        label = torch.tensor(row["target"], dtype=torch.float32)
        if not self.multiview:
            return cc, label
        with Image.open(row["MLO_path"]) as image:
            mlo = TRANSFORM(image.convert("RGB"))
        return cc, mlo, label


class MultiViewResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.cc_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.mlo_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feature_dim = self.cc_encoder.fc.in_features
        self.cc_encoder.fc = nn.Identity()
        self.mlo_encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor) -> torch.Tensor:
        cc_features = self.cc_encoder(cc)
        mlo_features = self.mlo_encoder(mlo)
        return self.classifier(torch.cat([cc_features, mlo_features], dim=1))


class FocalLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce)
        alpha_t = FOCAL_ALPHA * targets + (1 - FOCAL_ALPHA) * (1 - targets)
        return (alpha_t * (1 - pt) ** FOCAL_GAMMA * bce).mean()


def create_model(multiview: bool):
    if multiview:
        model = MultiViewResNet50()
        encoders = [model.cc_encoder, model.mlo_encoder]
        head = model.classifier
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)
        encoders = [model]
        head = model.fc

    for parameter in model.parameters():
        parameter.requires_grad = False
    encoder_parameters = []
    for encoder in encoders:
        for parameter in encoder.layer4.parameters():
            parameter.requires_grad = True
        encoder_parameters.extend(encoder.layer4.parameters())
    for parameter in head.parameters():
        parameter.requires_grad = True
    return model.to(DEVICE), encoder_parameters, list(head.parameters())


def freeze_batchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def train_epoch(
    model, loader, criterion, optimizer, scaler, multiview: bool,
    accumulation_samples: int = MIN_EFFECTIVE_BATCH_SIZE,
) -> float:
    model.train()
    freeze_batchnorm(model)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    accumulated = 0

    for batch in loader:
        if multiview:
            cc, mlo, labels = batch
            mlo = mlo.to(DEVICE)
        else:
            cc, labels = batch
        cc = cc.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)
        batch_size = len(labels)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(cc, mlo) if multiview else model(cc)
            loss = criterion(logits, labels)
        scaler.scale(loss * batch_size).backward()
        running_loss += loss.item() * batch_size
        accumulated += batch_size
        if accumulated >= accumulation_samples:
            optimizer_step(model, optimizer, scaler, accumulated)
            accumulated = 0

    if accumulated:
        optimizer_step(model, optimizer, scaler, accumulated)
    return running_loss / len(loader.dataset)


def optimizer_step(model, optimizer, scaler, accumulated: int) -> None:
    scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(accumulated)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def evaluate(model, loader, criterion, multiview: bool):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    with torch.inference_mode():
        for batch in loader:
            if multiview:
                cc, mlo, labels = batch
                mlo = mlo.to(DEVICE)
            else:
                cc, labels = batch
            cc = cc.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                logits = model(cc, mlo) if multiview else model(cc)
                loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy().ravel())
            all_probs.extend(torch.sigmoid(logits.float()).cpu().numpy().ravel())
    return total_loss / len(loader.dataset), np.array(all_labels), np.array(all_probs)


def update_early_stopping(
    score: float, reference: float, stale_epochs: int, min_delta: float
) -> tuple[float, int]:
    """Count epochs without a meaningful improvement over the last reference."""
    if score > reference + min_delta:
        return score, 0
    return reference, stale_epochs + 1


def save_plot(history: pd.DataFrame, best_epoch: int, path: Path) -> None:
    changing_lr = any(history[column].nunique() > 1 for column in ("Encoder LR", "Head LR"))
    fig = Figure(figsize=(15 if changing_lr else 10, 4))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3 if changing_lr else 2)
    axes[0].plot(history["Epoch"], history["Train loss"], label="Train")
    axes[0].plot(history["Epoch"], history["Val loss"], label="Validation")
    axes[0].set_ylabel("Focal loss")
    axes[1].plot(history["Epoch"], history["AUPRC"], label="Validation AUPRC")
    axes[1].plot(history["Epoch"], history["AUROC"], label="Validation AUROC")
    axes[1].set_ylabel("Score")
    if changing_lr:
        axes[2].plot(history["Epoch"], history["Encoder LR"], label="Encoder")
        axes[2].plot(history["Epoch"], history["Head LR"], label="Classifier")
        axes[2].set_yscale("log")
        axes[2].set_ylabel("Learning rate")
    for axis in axes:
        axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.6, label="Best checkpoint")
        if history["Epoch"].max() > 10:
            axis.axvline(10, color="tab:red", linestyle=":", alpha=0.8, label="Original 10-epoch limit")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(SPLIT_DIR / "train_pairs.csv")
    val = pd.read_csv(SPLIT_DIR / "val_pairs.csv")
    test_ids = set(pd.read_csv(SPLIT_DIR / "test_pairs.csv", usecols=["study_id"])["study_id"])
    for name, rows in [("train", train), ("validation", val)]:
        assert {"study_id", "CC_path", "MLO_path", "target"} <= set(rows.columns)
        for view in ("CC", "MLO"):
            missing = rows.loc[~rows[f"{view}_path"].map(lambda p: Path(p).is_file())]
            if not missing.empty:
                raise FileNotFoundError(f"{name}: {len(missing)} missing {view} images")
    train_ids = set(train["study_id"])
    val_ids = set(val["study_id"])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Study IDs overlap between data splits")
    return train, val


def run_experiment(model_name: str, seed: int, train, val, args):
    run_name = f"{model_name}_seed{seed}_bs{args.batch_size}"
    if args.fixed_lr:
        run_name += "_fixedlr"
    history_path = RESULT_DIR / f"{run_name}_history.csv"
    partial_path = RESULT_DIR / f"{run_name}_history.partial.csv"
    plot_suffix = "_curves_2panel.png" if args.fixed_lr else "_curves.png"
    plot_path = RESULT_DIR / f"{run_name}{plot_suffix}"
    checkpoint_path = MODEL_DIR / f"{run_name}_best.pth"
    if history_path.exists() and checkpoint_path.exists() and not partial_path.exists():
        history = pd.read_csv(history_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if history["Epoch"].tolist() != list(range(1, len(history) + 1)):
            raise RuntimeError(f"Invalid epoch history: {history_path}")
        if checkpoint.get("training_config") != training_config(args):
            raise RuntimeError(f"Existing run uses different settings: {run_name}")
        if not plot_path.exists():
            save_plot(history, int(checkpoint["best_epoch"]), plot_path)
        print(f"Reusing {run_name}: best epoch {checkpoint['best_epoch']}")
        return summary_row(run_name, history, checkpoint)
    if history_path.exists() or checkpoint_path.exists() or partial_path.exists():
        raise RuntimeError(
            f"Incomplete run {run_name}. Archive its files in {RESULT_DIR} and "
            f"{MODEL_DIR} before restarting."
        )

    multiview = model_name == "multi_view"
    effective_batch_size = max(MIN_EFFECTIVE_BATCH_SIZE, args.batch_size)
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        MammographyDataset(train, multiview), batch_size=args.batch_size,
        shuffle=True, generator=generator, num_workers=args.workers,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        MammographyDataset(val, multiview), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model, encoder_params, head_params = create_model(multiview)
    optimizer = torch.optim.Adam(
        [{"params": encoder_params, "lr": ENCODER_LR},
         {"params": head_params, "lr": HEAD_LR}]
    )
    scheduler = None if args.fixed_lr else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience,
        threshold=args.min_delta, threshold_mode="abs", min_lr=1e-7,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    criterion = FocalLoss().to(DEVICE)
    history = []
    best_ap = -1.0
    best_epoch = 0
    stop_reference = -1.0
    stale_epochs = 0
    print(f"\nTraining {run_name} on {DEVICE}; max {args.max_epochs} epochs")
    try:
        for epoch in range(1, args.max_epochs + 1):
            start = time.perf_counter()
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, scaler, multiview,
                accumulation_samples=effective_batch_size,
            )
            val_loss, labels, probabilities = evaluate(
                model, val_loader, criterion, multiview
            )
            auprc = float(average_precision_score(labels, probabilities))
            auroc = float(roc_auc_score(labels, probabilities))
            row = {
                "Epoch": epoch, "Train loss": train_loss, "Val loss": val_loss,
                "AUROC": auroc, "AUPRC": auprc,
                "Encoder LR": optimizer.param_groups[0]["lr"],
                "Head LR": optimizer.param_groups[1]["lr"],
                "Time (s)": time.perf_counter() - start,
            }
            history.append(row)

            if auprc > best_ap:
                best_ap, best_epoch = auprc, epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "architecture": model_name,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "val_auprc": best_ap,
                        "image_size": IMAGE_SIZE,
                        "loss_name": "focal",
                        "focal_alpha": FOCAL_ALPHA,
                        "focal_gamma": FOCAL_GAMMA,
                        "encoder_lr": ENCODER_LR,
                        "head_lr": HEAD_LR,
                        "physical_batch_size": args.batch_size,
                        "gradient_accumulation_steps": effective_batch_size // args.batch_size,
                        "effective_batch_size": effective_batch_size,
                        "num_workers": args.workers,
                        "lr_schedule": "fixed" if args.fixed_lr else "plateau",
                        "optimizer": "Adam",
                        "augmentation": "None",
                        "pretraining": "ImageNet",
                        "finetuning": "layer4 + classifier",
                        "training_config": training_config(args),
                    }, checkpoint_path,
                )

            stop_reference, stale_epochs = update_early_stopping(
                auprc, stop_reference, stale_epochs, args.min_delta
            )
            if scheduler is not None:
                scheduler.step(auprc)
            pd.DataFrame(history).to_csv(partial_path, index=False)
            print(
                f"Epoch {epoch:02d}/{args.max_epochs} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | AUROC {auroc:.4f} | AUPRC {auprc:.4f} | "
                f"best {best_ap:.4f} at {best_epoch} | "
                f"no meaningful gain {stale_epochs}/{args.patience} | "
                f"head LR {optimizer.param_groups[1]['lr']:.2g}"
            )
            if epoch >= args.min_epochs and stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best checkpoint: epoch {best_epoch}")
                break
    finally:
        del model, optimizer, scheduler, scaler
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    partial_path.replace(history_path)
    history_df = pd.DataFrame(history)
    save_plot(history_df, best_epoch, plot_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result = summary_row(run_name, history_df, checkpoint)
    print(f"Saved {history_path}, {checkpoint_path}, {plot_path}")
    return result


def training_config(args) -> dict:
    config = {
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "lr_patience": args.lr_patience,
        "min_delta": args.min_delta,
        "batch_size": args.batch_size,
        "effective_batch_size": max(MIN_EFFECTIVE_BATCH_SIZE, args.batch_size),
        "workers": args.workers,
    }
    if args.fixed_lr:
        config["lr_schedule"] = "fixed"
    return config


def summary_row(run_name: str, history: pd.DataFrame, checkpoint: dict) -> dict:
    best_epoch = int(checkpoint["best_epoch"])
    best_row = history.loc[history["Epoch"] == best_epoch].iloc[0]
    return {
        "Run": run_name,
        "Model": checkpoint["architecture"],
        "Seed": int(checkpoint["seed"]),
        "Physical batch size": int(checkpoint["physical_batch_size"]),
        "Effective batch size": int(checkpoint["effective_batch_size"]),
        "LR schedule": checkpoint.get("lr_schedule", "plateau"),
        "Epochs trained": len(history),
        "Best epoch": best_epoch,
        "Best val AUROC": float(best_row["AUROC"]),
        "Best val AUPRC": float(best_row["AUPRC"]),
        "Min val loss": float(history["Val loss"].min()),
        "Final train loss": float(history["Train loss"].iloc[-1]),
        "Final val loss": float(history["Val loss"].iloc[-1]),
        "Stop reason": (
            "early_stopping" if len(history) < checkpoint["training_config"]["max_epochs"]
            else "max_epochs"
        ),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["single_view", "multi_view", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 10, 21])
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--fixed-lr", action="store_true", help="keep the original learning rates fixed")
    args = parser.parse_args(argv)
    if not (1 <= args.min_epochs <= args.max_epochs):
        parser.error("min-epochs must be between 1 and max-epochs")
    if args.patience <= args.lr_patience or args.min_delta < 0:
        parser.error("patience must exceed lr-patience; min-delta must be nonnegative")
    if args.batch_size not in (1, 2, 4, 8, 16, 32):
        parser.error("batch-size must be one of 1, 2, 4, 8, 16, or 32")
    if not 0 <= args.workers <= 8:
        parser.error("workers must be between 0 and 8")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train, val = load_splits()
    models_to_run = ["multi_view", "single_view"] if args.model == "both" else [args.model]
    for model_name in models_to_run:
        for seed in args.seeds:
            run_experiment(model_name, seed, train, val, args)
    results = []
    for history_path in sorted(RESULT_DIR.glob("*_history.csv")):
        run_name = history_path.name.removesuffix("_history.csv")
        checkpoint_path = MODEL_DIR / f"{run_name}_best.pth"
        if checkpoint_path.exists():
            history = pd.read_csv(history_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            results.append(summary_row(run_name, history, checkpoint))
    summary_path = RESULT_DIR / "validation_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\nValidation summary: {summary_path}")
    print(pd.DataFrame(results).to_string(index=False))
    print("Held-out test set was not evaluated.")


if __name__ == "__main__":
    main()
