#!/usr/bin/env python3
"""
DICOM -> 16-bit PNG (2048x2048) with breast mask.

-----------How to run-----------
python dicom_to_png16_2048_background-cleaned.py \
  --csv-file-path /path/to/meta.csv \
  --dicom-dir /path/to/dicom_root \
  --processed-images-dir /path/to/output_png \
  --breast-mask-dir /path/to/output_masks \
  --max-workers 64
-------------------------------

----Data processing oioline----
DICOM (.dcm)
   │
   ├─► dcmj2pnm (+on2)
   │      └─ write temporary 16-bit PNG
   │
   ├─► load temp PNG with PIL → uint16 numpy array
   │
   ├─► if ImageLaterality == 'R':
   │        horizontal flip (np.fliplr)
   │
   ├─► pad to square
   │        - pad right if height > width
   │        - pad bottom if width > height
   │
   ├─► resize to 2048 × 2048 (LANCZOS)
   │
   ├─► generate breast mask
   │        - Otsu threshold
   │        - largest connected component
   │        - binary closing (structuring element)
   │        - remove small holes
   │
   ├─► apply mask
   │        - inside: original uint16 values
   │        - outside: low-level noise (seeded) via bg_cleanwithnoise
   │
   └─► save:
            - cleaned 16-bit PNG
            - 16-bit mask PNG (65535 inside breast, 0 outside)

"""

#!/usr/bin/env python3
"""
DICOM -> 16-bit PNG (2048x2048) with breast mask.

Converted from notebook cells into a standalone Python 3.9 CLI script.
Logic is preserved; only minimal changes for CLI execution.
"""

import os
import io
import subprocess
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
import tempfile

from skimage.filters import threshold_otsu
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from typing import Optional


# -------------------- Utils --------------------

def save_uint16_png(array_u16: np.ndarray, out_path: str) -> None:
    """
    Save a uint16 array as a 16-bit greyscale PNG ('I;16').
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = Image.fromarray(array_u16, mode="I;16")
    img.save(out_path)


def make_structuring_element(size: int = 8) -> np.ndarray:
    """
    Create a square structuring element of given size for binary closing.
    """
    return np.ones((size, size), dtype=bool)


def bg_cleanwithnoise(img_u16: np.ndarray, breast_mask: np.ndarray, t: int, seed: int = 42) -> np.ndarray:
    """
    Keep breast pixels; fill background with uniform noise in [0, t * 0.01].
    """
    assert img_u16.dtype == np.uint16
    assert breast_mask.dtype == bool and breast_mask.shape == img_u16.shape

    rng = np.random.default_rng(seed)
    noise = rng.integers(0, int(t * 0.01), img_u16.shape, dtype=np.uint16)

    cleaned = np.where(breast_mask, img_u16, noise).astype(np.uint16)
    return cleaned


# -------------------- DICOM → uint16 array (via dcmj2pnm) --------------------

def dicom_to_uint16_array_via_dcmj2pnm(dicom_path: str) -> np.ndarray:
    """
    Convert a DICOM image to a uint16 numpy array using dcmj2pnm via a temporary PNG file.
    No persistent intermediates are kept.
    """
    # Temporary PNG file so dcmj2pnm uses PNG encoder and we can reload with PIL.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # +on2 keeps 16-bit depth where possible.
        subprocess.run(
            ['dcmj2pnm', '+on2', dicom_path, tmp_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        with Image.open(tmp_path) as im:
            if im.mode != "I;16":
                im = im.convert("I;16")
            arr = np.array(im, dtype=np.uint16)

        return arr
    finally:
        # Ensure cleanup even if reading fails.
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# -------------------- Worker --------------------

def process_one(path_rel: str,
                dicom_root: str,
                processed_images_dir: str,
                breast_mask_dir: str,
                image_laterality: Optional[str] = None,
                se_size: int = 16,
                hole_area: int = 40000) -> str:
    """
    Worker:
    - DICOM -> uint16 array (via dcmj2pnm)
    - Flip horizontally if ImageLaterality == 'R'
    - Pad to square (right or bottom)
    - Resize to 2048x2048 (LANCZOS)
    - Otsu threshold -> largest CC -> binary closing -> remove small holes
    - Apply mask to image using bg_cleanwithnoise
    - Save cleaned image (16-bit) and mask (16-bit)

    Returns a status string or raises on error.
    """
    dcm_path = os.path.join(dicom_root, path_rel + ".dcm")
    out_img_path = os.path.join(processed_images_dir, path_rel + ".png")
    out_mask_path = os.path.join(breast_mask_dir, path_rel + ".png")

    # Load from DICOM
    img_u16 = dicom_to_uint16_array_via_dcmj2pnm(dcm_path)

    # Flip horizontally if right breast
    if image_laterality == 'R':
        img_u16 = np.fliplr(img_u16)

    # Pad to square toward right/bottom
    height, width = img_u16.shape
    if width < height:
        pad_right = height - width
        new_img = np.zeros((height, height), dtype=np.uint16)
        new_img[:, :width] = img_u16
    elif width > height:
        pad_bottom = width - height
        new_img = np.zeros((width, width), dtype=np.uint16)
        new_img[:height, :] = img_u16
    else:
        new_img = img_u16

    # Resize to 2048 x 2048
    im_resized = Image.fromarray(new_img, mode="I;16").resize(
        (2048, 2048),
        resample=Image.LANCZOS
    )
    img_u16_resized = np.array(im_resized, dtype=np.uint16)

    # Mask creation
    t = threshold_otsu(img_u16_resized)
    binary = img_u16_resized > t * 0.01

    label_img, num_labels = ndimage.label(binary, structure=np.ones((3, 3), dtype=bool))

    if num_labels == 0:
        cleaned = np.zeros_like(img_u16_resized, dtype=np.uint16)
        mask_u16 = np.zeros_like(img_u16_resized, dtype=np.uint16)
        save_uint16_png(cleaned, out_img_path)
        save_uint16_png(mask_u16, out_mask_path)
        return f"EMPTY_FOREGROUND: {dcm_path}"

    sizes = ndimage.sum(binary, label_img, index=np.arange(1, num_labels + 1))
    largest_label = int(np.argmax(sizes) + 1)
    breast_mask = (label_img == largest_label)

    se = make_structuring_element(se_size)
    breast_mask = ndimage.binary_closing(breast_mask, structure=se)

    from skimage.morphology import remove_small_holes
    breast_mask = remove_small_holes(breast_mask, area_threshold=hole_area)

    # Apply mask with noise fill
    seed = (hash(path_rel) ^ 42) & 0x7FFFFFFF
    cleaned_u16 = bg_cleanwithnoise(img_u16_resized, breast_mask, t, seed=seed)

    # Save outputs (16-bit)
    save_uint16_png(cleaned_u16, out_img_path)
    mask_u16 = np.where(breast_mask, 65535, 0).astype(np.uint16)
    save_uint16_png(mask_u16, out_mask_path)

    return f"OK: {dcm_path}"


# -------------------- Main --------------------

def main(csv_file_path: str,
         dicom_dir: str,
         processed_images_dir: str,
         breast_mask_dir: str,
         max_workers: int = 128,
         se_size: int = 16,
         hole_area: int = 40000) -> None:
    """
    Control flow for batch processing.
    """
    df = pd.read_csv(csv_file_path)

    rel_paths = df['path'].astype(str).tolist()
    if 'ImageLaterality' in df.columns:
        laterality = (
            df['ImageLaterality']
            .astype(str)
            .str.upper()
            .replace({'NAN': None})
            .tolist()
        )
    else:
        laterality = [None] * len(rel_paths)

    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(breast_mask_dir, exist_ok=True)

    logs_ok, logs_warn, logs_err = [], [], []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                process_one,
                p,
                dicom_dir,
                processed_images_dir,
                breast_mask_dir,
                lat,
                se_size,
                hole_area
            ): p
            for p, lat in zip(rel_paths, laterality)
        }

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Processing DICOM → 2048 PNG"):
            rel = futures[fut]
            try:
                msg = fut.result()
                if msg.startswith("OK"):
                    logs_ok.append(msg)
                elif msg.startswith("EMPTY_FOREGROUND"):
                    logs_warn.append(msg)
                else:
                    logs_warn.append(msg)
            except Exception as e:
                logs_err.append(f"ERROR: {rel} -> {e}")

    print(f"Done. {len(logs_ok)} images processed successfully.")
    if logs_warn:
        print(f"{len(logs_warn)} warnings.")
        for m in logs_warn[:10]:
            print(m)
    if logs_err:
        print(f"{len(logs_err)} errors.")
        for m in logs_err[:20]:
            print(m)


# -------------------- CLI entrypoint --------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch convert DICOMs to cleaned 16-bit 2048x2048 PNGs with breast masks."
    )

    parser.add_argument(
        "--csv-file-path",
        type=str,
        default="/montana-storage02/fast2/Data/cruk-wei_j/SingleEpisode-Assessment_JWei/DataProcessing/6-year_Risk/External_Longitudinal_6Y_v4.csv",
        help="Path to CSV metadata file with a 'path' column and  'ImageLaterality'."
    )
    parser.add_argument(
        "--dicom-dir",
        type=str,
        default="/montana-storage04/slow/Users/OMI-DB/impr/DICOM_6Y/",
        help="Root directory containing DICOM files (relative paths from CSV plus .dcm)."
    )
    parser.add_argument(
        "--processed-images-dir",
        type=str,
        default="/montana-storage04/slow/Users/OMI-DB/impr/16bit_PNG_6Y_2048_test/watermark_rm/",
        help="Output directory for cleaned 16-bit PNG images."
    )
    parser.add_argument(
        "--breast-mask-dir",
        type=str,
        default="/montana-storage04/slow/Users/OMI-DB/impr/16bit_PNG_6Y_2048_test/breast_mask/",
        help="Output directory for corresponding 16-bit breast masks."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=128,
        help="Maximum number of parallel worker processes."
    )
    parser.add_argument(
        "--se-size",
        type=int,
        default=16,
        help="Structuring element size for binary closing."
    )
    parser.add_argument(
        "--hole-area",
        type=int,
        default=40000,
        help="Area threshold for removing small holes in the mask."
    )

    args = parser.parse_args()

    main(
        csv_file_path=args.csv_file_path,
        dicom_dir=args.dicom_dir,
        processed_images_dir=args.processed_images_dir,
        breast_mask_dir=args.breast_mask_dir,
        max_workers=args.max_workers,
        se_size=args.se_size,
        hole_area=args.hole_area,
    )
