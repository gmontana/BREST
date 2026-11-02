"""Generated pixels exercise conversion logic without any clinical data."""

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

source = Path(__file__).resolve().parents[1] / "dataProcessing/dicom_to_png16_2048_background-cleaned.py"
spec = importlib.util.spec_from_file_location("preprocessing_script", source)
preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing)


def test_generated_breast_conversion_preserves_foreground_and_flips_right(tmp_path, monkeypatch):
    pixels = np.zeros((2048, 2048), dtype=np.uint16)
    pixels[256:1792, 128:896] = 16000
    monkeypatch.setattr(preprocessing, "dicom_to_uint16_array_via_dcmj2pnm", lambda _: pixels.copy())
    for side in ("L", "R"):
        preprocessing.process_one(f"generated_{side}", str(tmp_path), str(tmp_path / "png"),
                                  str(tmp_path / "mask"), side)
        actual = np.asarray(Image.open(tmp_path / "png" / f"generated_{side}.png"))
        mask = np.asarray(Image.open(tmp_path / "mask" / f"generated_{side}.png")) > 0
        expected = pixels if side == "L" else np.fliplr(pixels)
        assert actual.shape == (2048, 2048) and actual.dtype == np.uint16
        np.testing.assert_array_equal(actual[mask], expected[mask])
        assert set(np.unique(mask)) == {False, True}


def test_background_randomness_is_seeded_and_foreground_unchanged():
    image = np.full((16, 16), 10000, dtype=np.uint16)
    mask = np.zeros_like(image, dtype=bool)
    mask[:, :8] = True
    first = preprocessing.bg_cleanwithnoise(image, mask, 10000, seed=42)
    second = preprocessing.bg_cleanwithnoise(image, mask, 10000, seed=42)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[mask], image[mask])
    assert first[~mask].max() < 100


def test_zero_threshold_has_zero_background_without_changing_tissue():
    image = np.full((16, 16), 10000, dtype=np.uint16)
    mask = np.zeros_like(image, dtype=bool)
    mask[:, :8] = True
    actual = preprocessing.bg_cleanwithnoise(image, mask, 0, seed=42)
    np.testing.assert_array_equal(actual[mask], image[mask])
    assert not actual[~mask].any()
