---
license: other
license_name: warwick-brest-research-license
license_link: LICENSE
library_name: pytorch
pipeline_tag: image-classification
tags:
  - mammography
  - breast-cancer-risk
  - medical-imaging
  - brest-ai
---

# BREST-AI: three-year breast cancer risk from screening mammograms

This release provides the **published BREST-AI v1, 16-bit, four-view model** developed at the University of Warwick with NHS and academic collaborators. It estimates an examination-level three-year risk score from a screening mammogram. The model uses a ResNeXt-50 encoder and attentional feature fusion.

**Paper:** Wei J, Zhou X, Santeramo R, et al. *BREST-AI: Three-year breast cancer risk prediction from screening mammograms*. npj Breast Cancer (2026). [doi:10.1038/s41523-026-01069-x](https://doi.org/10.1038/s41523-026-01069-x).

**Code and preprocessing:** [github.com/gmontana/BREST](https://github.com/gmontana/BREST).

## Download and run

```bash
git clone https://github.com/gmontana/BREST.git
cd BREST
python -m pip install -r requirements-published-cuda.lock
python scripts/download_weights.py --output-dir models
python scripts/patientLevel-inference.py \
    --metadata_csv /data/examinations.csv \
    --image_root_dir /data/processed_png \
    --model_checkpoint models/brest-ai-v1.safetensors \
    --final_csv_path /data/predictions.csv
```

Use Python 3.11 on Linux. The locked environment pins PyTorch 2.1.0 / CUDA 11.8, Kornia 0.7.1 and their dependencies. Use this same installation for CPU inference with `--device cpu`.

The repository supplies the model definition and checks the downloaded weights against a fixed SHA-256 checksum. No remote Python code needs to be executed from this model repository. GPU inference is selected when available; add `--device cpu` for CPU inference.

## Inputs and output

Use the repository's documented DICOM preprocessing to produce breast-aligned 2048 × 2048 grayscale 16-bit PNGs. Inference divides pixels by 65535, repeats the grayscale channel three times, resizes to 1792 × 1792 using the original Kornia bilinear antialiased resize, and normalises with mean 0.0856 and standard deviation 0.1687 per channel. Right-breast flipping belongs to DICOM preprocessing; inference does not flip the prepared images again.

Each examination requires one image for each of **left CC, right CC, left MLO and right MLO**, in that network order. The CSV supplies `ClientID`, `path`, `ViewPosition` and `ImageLaterality`; `EpisodeID` distinguishes repeat examinations. The loader validates and orders these views. Outcome labels are optional for prediction. For an original v1 evaluation CSV whose rows are already ordered, `--preserve_row_order` reproduces the original loader without sorting historical view tags.

The output is the sigmoid of one examination-level risk logit. Population calibration is required before interpreting it as an absolute three-year probability. Its value is not the probability that an MRI will detect a cancer immediately.

## Evaluation and scope

The published retrospective UK Hologic case–control evaluation included 7,596 women (1,899 cases); BREST-AI AUC was 0.751 (95% CI 0.738–0.764), compared with 0.700 (0.687–0.714) for MIT's Mirai. The paper also reports Siemens and US external evaluations. Acquisition differences can affect performance; evaluate the model and calibration in the intended population.

## Release contents and privacy

The weights are stored as architecture-validated model tensors in `safetensors` format. The release includes the input configuration, checksums, source-checkpoint hash and the software licence. Patient images, patient-level metadata, predictions, training logs, optimiser state and institutional file paths are excluded. Examples in the code repository are synthetic metadata.

These file-level controls do not constitute a differential-privacy guarantee for the learned parameters. The model was trained using controlled-access research data; those data are obtained separately through their custodians.

## Licence and citation

This software includes BREST code © 2025, The University of Warwick.

The [University of Warwick BREST software licence](LICENSE) covers the code and neural-network weights. It permits academic research and non-commercial, non-clinical evaluation. Retain the complete licence and copyright notice on redistribution. Commercial licensing enquiries: ventures@warwick.ac.uk. Research enquiries: g.montana@warwick.ac.uk.

```bibtex
@article{wei2026brest,
  title = {BREST-AI: Three-year breast cancer risk prediction from screening mammograms},
  author = {Wei, Jiefei and Zhou, Xinyu and Santeramo, Ruggiero and Al-Attar, Miaad and Muthyala, Sreenivas and Gilbert, Fiona and Brentnall, Adam R. and Montana, Giovanni},
  journal = {npj Breast Cancer},
  year = {2026},
  doi = {10.1038/s41523-026-01069-x}
}
```
