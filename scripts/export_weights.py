"""Export only architecture-validated v1 tensors; never copy training metadata."""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from brest_inference import load_brest_model, sha256_file

SOURCE_SHA256 = "03f7e93ea03d0a4ff5fe5fb89c19d84268d6d6e3f9ec3f0520e834307d1c9493"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.source) != SOURCE_SHA256:
        raise SystemExit("Source is not the verified published v1 checkpoint")
    model = load_brest_model(args.source)
    state = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "brest-ai-v1.safetensors"
    if target.exists():
        raise SystemExit("Output already exists; use a new release directory")
    save_file(state, str(target), metadata={"format": "pt"})
    restored = load_file(str(target))
    assert state.keys() == restored.keys()
    assert all(torch.equal(value, restored[key]) for key, value in state.items())
    manifest = {
        "model": "BREST-AI", "version": "1.0", "repository": "dnamodel/BREST-AI",
        "revision": "main", "filename": target.name, "sha256": sha256_file(target),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "tensor_count": len(state), "tensor_elements": sum(t.numel() for t in state.values()),
        "paper_doi": "10.1038/s41523-026-01069-x",
    }
    (args.output_dir / "weights_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    config = {
        "architecture": "BREST_16bit", "version": "1.0", "input_dtype": "uint16",
        "prepared_image_size": [2048, 2048], "model_image_size": [1792, 1792],
        "view_order": ["L-CC", "R-CC", "L-MLO", "R-MLO"],
        "pixel_divisor": 65535.0, "mean": [0.0856] * 3, "std": [0.1687] * 3,
        "resize": {"implementation": "kornia.Resize", "mode": "bilinear", "align_corners": False, "antialias": True},
        "inference_precision": {"dtype": "float32", "matmul_allow_tf32": False, "cudnn_allow_tf32": True},
        "output": "sigmoid of examination-level risk logit; requires population calibration for absolute risk",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"tensors": len(state), "roundtrip_equal": True, "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
