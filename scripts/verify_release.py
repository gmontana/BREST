"""Check downloaded weight identity and inference on generated pixels only."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import torch

from brest_inference import EpisodeDataset, VIEW_ORDER, load_brest_model, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    fixture = json.loads((Path(__file__).resolve().parents[1] / "tests/generated_reference.json").read_text())
    if sha256_file(args.checkpoint) != fixture["weights_sha256"]:
        raise SystemExit("Weights differ from the published tensor-only release")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model = load_brest_model(args.checkpoint).to(args.device).eval()
    with tempfile.TemporaryDirectory(prefix="brest-generated-") as folder:
        root = Path(folder)
        rng = np.random.default_rng(fixture["seed"])
        rows = []
        for index, (view, side) in enumerate(VIEW_ORDER):
            name = f"generated_{index}.png"
            pixels = rng.integers(fixture["integer_low_inclusive"], fixture["integer_high_exclusive"],
                                  size=fixture["image_shape"], dtype=np.uint16)
            Image.fromarray(pixels).save(root / name)
            rows.append(dict(ClientID="generated", path=name, ViewPosition=view, ImageLaterality=side))
        metadata = root / "metadata.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)
        images = EpisodeDataset(metadata, root)[0].unsqueeze(0).to(args.device)
        with torch.inference_mode():
            logit = float(model(images).item())
        reference = fixture["expected_logit_by_device"][args.device]
        error = abs(logit - reference)
        passed = bool(np.isfinite(logit) and error <= fixture["logit_absolute_tolerance"])
        print(json.dumps({"weights_sha256_verified": True, "generated_input_logit": logit,
                          "reference_logit": reference, "absolute_error": error,
                          "absolute_tolerance": fixture["logit_absolute_tolerance"],
                          "device": args.device, "passed": passed}, indent=2))
        if not passed:
            raise SystemExit("Generated-input regression failed; check the documented runtime and preprocessing")


if __name__ == "__main__":
    main()
