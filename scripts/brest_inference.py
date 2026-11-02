"""Inference helpers for the published BREST-AI four-view model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import imageio.v3 as iio
import kornia.augmentation as KA
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from models import BREST_16bit

VIEW_ORDER = (("CC", "L"), ("CC", "R"), ("MLO", "L"), ("MLO", "R"))
MEAN, STD = 0.0856, 0.1687
IMAGE_SIZE = 1792
PUBLISHED_CHECKPOINT_SHA256 = "03f7e93ea03d0a4ff5fe5fb89c19d84268d6d6e3f9ec3f0520e834307d1c9493"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_brest_model(checkpoint: str | Path) -> BREST_16bit:
    """Load tensors only, on CPU, and require an exact architecture match."""
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".safetensors":
        state = load_file(str(checkpoint), device="cpu")
    elif checkpoint.suffix in {".pth", ".pt"}:
        # The published runtime predates later pickle-loader security fixes.
        # Only the hash-verified original is accepted in this legacy format.
        if sha256_file(checkpoint) != PUBLISHED_CHECKPOINT_SHA256:
            raise ValueError("Legacy checkpoint must match the verified published v1 hash; use safetensors otherwise")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    else:
        raise ValueError("Use the released .safetensors file or a trusted tensor-only .pth file")
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty model state dictionary")
    clean = OrderedDict()
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Checkpoint must contain model tensors only")
        name = key.removeprefix("module.")
        if name in clean:
            raise ValueError("Duplicate parameter name after removing the DataParallel prefix")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Checkpoint contains nonfinite values")
        clean[name] = value
    model = BREST_16bit()
    model.load_state_dict(clean, strict=True)
    return model.eval()


def preprocessing() -> torch.nn.Sequential:
    # Keep the original v1 Kornia resize and normalisation, including antialiasing.
    return torch.nn.Sequential(
        KA.Resize((IMAGE_SIZE, IMAGE_SIZE), resample="bilinear", align_corners=False, antialias=True),
        KA.Normalize(mean=torch.tensor([MEAN] * 3), std=torch.tensor([STD] * 3)),
    )


def load_png(path: Path) -> torch.Tensor:
    image = iio.imread(path)
    # Pillow can expose a 16-bit PNG as int32. An 8-bit image is incompatible.
    if image.ndim != 2 or image.dtype not in (np.dtype("uint16"), np.dtype("int32")):
        raise ValueError("BREST requires grayscale 16-bit PNGs from the documented preprocessing")
    if image.size == 0 or image.min() < 0 or image.max() > 65535:
        raise ValueError("PNG pixel values must lie between 0 and 65535")
    image = image.astype(np.float32) / 65535.0
    return torch.from_numpy(np.repeat(image[:, :, None], 3, axis=2)).permute(2, 0, 1).contiguous()


class EpisodeDataset(Dataset):
    """Validate every examination and order its four images explicitly."""

    def __init__(self, csv_file: str | Path, image_folder: str | Path, preserve_row_order: bool = False):
        frame = pd.read_csv(csv_file, dtype=str, keep_default_na=False)
        required = {"ClientID", "path"}
        if not preserve_row_order:
            required.update({"ViewPosition", "ImageLaterality"})
        if not required.issubset(frame.columns):
            raise ValueError("CSV requires ClientID, path, ViewPosition and ImageLaterality")
        if frame.empty or frame[list(required)].eq("").any().any():
            raise ValueError("Metadata is empty or has missing required values")
        if "EpisodeID" in frame and frame["EpisodeID"].eq("").any():
            raise ValueError("EpisodeID must be present for every row when that column is supplied")
        groups = ["ClientID", "EpisodeID"] if "EpisodeID" in frame else "ClientID"
        self.records = []
        self.root = Path(image_folder).resolve()
        self.transform = preprocessing()
        for _, rows in frame.groupby(groups, sort=False):
            if len(rows) != 4:
                raise ValueError("Every examination must contain exactly four images")
            if not preserve_row_order:
                pairs = list(zip(rows.ViewPosition, rows.ImageLaterality))
                if set(pairs) != set(VIEW_ORDER):
                    raise ValueError("Every examination must have exactly one L-CC, R-CC, L-MLO and R-MLO image")
                positions = {pair: i for i, pair in enumerate(pairs)}
                rows = rows.iloc[[positions[pair] for pair in VIEW_ORDER]]
            paths = []
            for value in rows.path:
                relative = Path(value if value.lower().endswith(".png") else value + ".png")
                path = (self.root / relative).resolve()
                if not path.is_relative_to(self.root):
                    raise ValueError("Image path leaves the supplied image root")
                if not path.is_file():
                    raise ValueError("An examination image is missing; no examinations were silently discarded")
                paths.append(path)
            record = {"ClientID": rows.ClientID.iloc[0], "paths": paths}
            if "EpisodeID" in rows:
                record["EpisodeID"] = rows.EpisodeID.iloc[0]
            if "EpisodeOutcome" in rows:
                outcomes = rows.EpisodeOutcome.unique()
                if len(outcomes) != 1 or outcomes[0] == "":
                    raise ValueError("EpisodeOutcome must be one consistent nonempty value per examination")
                # Preserve the published evaluation script's N-versus-other mapping.
                record["label"] = int(outcomes[0] != "N")
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.stack([
            self.transform(load_png(path).unsqueeze(0)).squeeze(0)
            for path in self.records[index]["paths"]
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Published BREST-AI v1 examination-level inference")
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--image_root_dir", required=True)
    parser.add_argument("--final_csv_path", required=True)
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--roc_plot_path", help="Optional ROC plot for a labelled, two-class cohort")
    parser.add_argument("--gpu_id", default="0", help="Visible GPU IDs; use --device cpu for CPU inference")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preserve_row_order", action="store_true",
                        help="Reproduce the original v1 loader using the supplied four-row order, without view-tag sorting")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch_size must be positive and num_workers nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(66)
    np.random.seed(66)
    # Match the preserved PyTorch 2.1 inference settings used for the paper.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = EpisodeDataset(args.metadata_csv, args.image_root_dir, preserve_row_order=args.preserve_row_order)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # Load before wrapping: saved module.* names are canonicalised exactly once.
    model = load_brest_model(args.model_checkpoint).to(device)
    if use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    scores = []
    with torch.inference_mode():
        for images in loader:
            scores.extend(torch.sigmoid(model(images.to(device))).flatten().cpu().tolist())
    output = pd.DataFrame([
        {key: record[key] for key in ("ClientID", "EpisodeID") if key in record}
        for record in dataset.records
    ])
    if "label" in dataset.records[0]:
        output["Episode Label"] = [record["label"] for record in dataset.records]
    output["Predicted Probability"] = scores
    destination = Path(args.final_csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    report = {"examinations": len(output), "device": str(device)}
    if "Episode Label" in output and output["Episode Label"].nunique() == 2:
        from sklearn.metrics import roc_auc_score, roc_curve
        report["auc"] = float(roc_auc_score(output["Episode Label"], scores))
        if args.roc_plot_path:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fpr, tpr, _ = roc_curve(output["Episode Label"], scores)
            fig, axis = plt.subplots()
            axis.plot(fpr, tpr, label=f"AUC = {report['auc']:.3f}")
            axis.plot([0, 1], [0, 1], "--", color="gray")
            axis.set(xlabel="False positive rate", ylabel="True positive rate")
            axis.legend()
            plot = Path(args.roc_plot_path)
            plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot, dpi=150, bbox_inches="tight")
            plt.close(fig)
    elif args.roc_plot_path:
        report["roc"] = "Not computed: a labelled cohort with both classes is required"
    print(json.dumps(report))


if __name__ == "__main__":
    main()
