"""Download the published v1 tensors and verify the release checksum."""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from brest_inference import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "weights_manifest.json").read_text())
    cached = Path(hf_hub_download(
        repo_id=manifest["repository"], filename=manifest["filename"],
        revision=manifest["revision"],
    ))
    if sha256_file(cached) != manifest["sha256"]:
        raise SystemExit("Downloaded weights do not match the published release checksum")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / manifest["filename"]
    if output.resolve() != cached.resolve():
        shutil.copyfile(cached, output)
    print(f"Verified BREST-AI v1 weights: {output}")


if __name__ == "__main__":
    main()
