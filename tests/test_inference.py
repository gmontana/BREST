"""Release tests use synthetic metadata and generated pixels only."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import brest_inference as inference


def metadata(tmp_path, labelled=True):
    rows = []
    for index, (view, side) in enumerate(inference.VIEW_ORDER):
        path = tmp_path / f"image_{index}.png"
        path.touch()
        row = dict(ClientID="synthetic_001", EpisodeID="example", path=path.stem,
                   ViewPosition=view, ImageLaterality=side)
        if labelled:
            row["EpisodeOutcome"] = "N"
        rows.append(row)
    return pd.DataFrame(rows)


def write(frame, tmp_path):
    target = tmp_path / "metadata.csv"
    frame.to_csv(target, index=False)
    return target


def test_orders_views_and_preserves_identifiers(tmp_path):
    frame = metadata(tmp_path).iloc[[3, 0, 2, 1]]
    frame["ClientID"] = "0001"
    dataset = inference.EpisodeDataset(write(frame, tmp_path), tmp_path)
    assert len(dataset) == 1
    assert dataset.records[0]["ClientID"] == "0001"
    assert [p.stem for p in dataset.records[0]["paths"]] == [f"image_{i}" for i in range(4)]


def test_unlabelled_inference_and_multiple_examinations(tmp_path):
    frame = metadata(tmp_path, labelled=False)
    other = frame.copy()
    other["EpisodeID"] = "second"
    dataset = inference.EpisodeDataset(write(pd.concat([frame, other]), tmp_path), tmp_path)
    assert len(dataset) == 2
    assert "label" not in dataset.records[0]


def test_original_row_order_requires_explicit_compatibility_mode(tmp_path):
    frame = metadata(tmp_path).iloc[[3, 0, 2, 1]].drop(columns=["ViewPosition", "ImageLaterality"])
    path = write(frame, tmp_path)
    with pytest.raises(ValueError):
        inference.EpisodeDataset(path, tmp_path)
    dataset = inference.EpisodeDataset(path, tmp_path, preserve_row_order=True)
    assert [p.stem for p in dataset.records[0]["paths"]] == ["image_3", "image_0", "image_2", "image_1"]


@pytest.mark.parametrize("problem", ["duplicate_view", "missing_image", "incomplete", "inconsistent_label", "outside_root"])
def test_invalid_examinations_fail_without_dropping_rows(tmp_path, problem):
    frame = metadata(tmp_path)
    if problem == "duplicate_view":
        frame.loc[0, ["ViewPosition", "ImageLaterality"]] = ["MLO", "R"]
    elif problem == "missing_image":
        (tmp_path / "image_0.png").unlink()
    elif problem == "incomplete":
        frame = frame.iloc[:3]
    elif problem == "inconsistent_label":
        frame.loc[0, "EpisodeOutcome"] = "M"
    elif problem == "outside_root":
        frame.loc[0, "path"] = "../outside"
    with pytest.raises(ValueError):
        inference.EpisodeDataset(write(frame, tmp_path), tmp_path)


def test_pixels_preserve_16bit_scale(tmp_path):
    pixels = np.array([[0, 65535], [1024, 32768]], dtype=np.uint16)
    path = tmp_path / "pixels.png"
    Image.fromarray(pixels).save(path)
    actual = inference.load_png(path)
    assert actual.shape == (3, 2, 2)
    np.testing.assert_array_equal(actual[0].numpy(), pixels.astype(np.float32) / 65535.0)
    assert torch.equal(actual[0], actual[1])


def test_8bit_input_is_rejected(tmp_path):
    path = tmp_path / "eight_bit.png"
    Image.fromarray(np.ones((8, 8), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="16-bit"):
        inference.load_png(path)


def test_loader_strips_prefix_and_is_strict(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "BREST_16bit", lambda: torch.nn.Linear(2, 1))
    checkpoint = tmp_path / "model.safetensors"
    save_file({"module.weight": torch.ones(1, 2), "module.bias": torch.zeros(1)}, str(checkpoint))
    model = inference.load_brest_model(checkpoint)
    assert not model.training
    assert torch.equal(model.weight, torch.ones(1, 2))
    save_file({"module.weight": torch.ones(1, 3), "module.bias": torch.zeros(1)}, str(checkpoint))
    with pytest.raises(RuntimeError):
        inference.load_brest_model(checkpoint)


def test_loader_rejects_extra_metadata_and_nonfinite_tensors(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "BREST_16bit", lambda: torch.nn.Linear(2, 1))
    checkpoint = tmp_path / "model.pth"
    torch.save({"weight": torch.ones(1, 2), "patient_records": ["synthetic"]}, checkpoint)
    with pytest.raises(ValueError, match="verified published"):
        inference.load_brest_model(checkpoint)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.full((1, 2), float("nan")), "bias": torch.zeros(1)}, str(checkpoint))
    with pytest.raises(ValueError, match="nonfinite"):
        inference.load_brest_model(checkpoint)
