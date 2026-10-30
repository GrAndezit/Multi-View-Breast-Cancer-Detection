"""Generate missing CC/MLO Grad-CAM figures for the selected test errors."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation"
CAM_DIR = RESULTS / "gradcam"
SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResizeWithPadding:
    def __call__(self, image):
        image = image.copy()
        image.thumbnail((SIZE, SIZE), Image.Resampling.BILINEAR)
        width, height = image.size
        padding = (
            (SIZE - width) // 2,
            (SIZE - height) // 2,
            SIZE - width - (SIZE - width) // 2,
            SIZE - height - (SIZE - height) // 2,
        )
        return TF.pad(image, padding, fill=0)


resize = ResizeWithPadding()
transform = transforms.Compose(
    [
        resize,
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class MultiViewResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.cc_encoder = models.resnet50(weights=None)
        self.mlo_encoder = models.resnet50(weights=None)
        features = self.cc_encoder.fc.in_features
        self.cc_encoder.fc = nn.Identity()
        self.mlo_encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(features * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, cc, mlo):
        return self.classifier(
            torch.cat([self.cc_encoder(cc), self.mlo_encoder(mlo)], dim=1)
        )


def load_model():
    checkpoint = torch.load(
        ROOT / "models" / "final" / "multi_view_seed42_best.pth",
        map_location="cpu",
        weights_only=True,
    )
    model = MultiViewResNet50()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(DEVICE).eval()


def make_cam(model, row):
    activations = {}
    gradients = {}
    handles = []

    for view, encoder in [("CC", model.cc_encoder), ("MLO", model.mlo_encoder)]:
        layer = encoder.layer4[-1].conv3

        def hook(module, inputs, output, view=view):
            activations[view] = output.detach()
            output.register_hook(lambda grad, view=view: gradients.__setitem__(view, grad.detach()))

        handles.append(layer.register_forward_hook(hook))

    images = {}
    tensors = {}
    for view in ("CC", "MLO"):
        with Image.open(row[f"{view}_path"]) as image:
            image = image.convert("RGB")
            images[view] = np.asarray(resize(image.convert("L")))
            tensors[view] = transform(image).unsqueeze(0).to(DEVICE)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensors["CC"], tensors["MLO"])
        probability = torch.sigmoid(logits[0, 0]).item()
        logits[0, 0].backward()
        cams = {}
        for view in ("CC", "MLO"):
            weights = gradients[view].mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations[view]).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()
            cams[view] = cam
        return images, cams, probability
    finally:
        for handle in handles:
            handle.remove()


def save_figure(row, images, cams, probability, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for column, view in enumerate(("CC", "MLO")):
        axes[0, column].imshow(images[view], cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(f"{view} Original")
        axes[1, column].imshow(images[view], cmap="gray", vmin=0, vmax=255)
        axes[1, column].imshow(cams[view], cmap="jet", alpha=0.40, vmin=0, vmax=1)
        axes[1, column].set_title(f"{view} Grad-CAM")
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(
        f"True label: {int(row['target'])} | Multi-view score: {probability:.3f} "
        f"| Outcome: {row['Multi outcome']}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(RESULTS / "qualitative_error_review.csv").drop_duplicates("Test row")
    model = load_model()
    for _, row in cases.iterrows():
        outcome = row["Multi outcome"]
        path = CAM_DIR / f"multiview_seed42_{outcome}_row{int(row['Test row'])}.png"
        if path.exists():
            continue
        images, cams, probability = make_cam(model, row)
        if abs(probability - row["Multi probability"]) > 0.005:
            raise ValueError(f"Checkpoint score differs from saved prediction for row {row['Test row']}")
        save_figure(row, images, cams, probability, path)
        print(f"Saved {path.relative_to(ROOT)} (score {probability:.4f})")


if __name__ == "__main__":
    main()
