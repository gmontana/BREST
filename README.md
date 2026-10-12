<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-light.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-dark.png">
    <img src="assets/logo-light.png" alt="BREST" width="420" />
  </picture>
</p>

# BREST: Breast Risk Evaluation from Screening Test

BREST is an artificial-intelligence model that estimates three-year breast cancer risk from the four standard images of a screening mammogram. It was developed at the University of Warwick with NHS and academic collaborators, using the OPTIMAM Mammography Image Database. The research aims to support risk-adapted breast screening and earlier cancer detection.

We share BREST software and neural networks with the research community to support independent evaluation and further research. This repository provides the preprocessing code, network architecture and inference code; trained weights and an inference Docker image are available on request. Access is governed by the [University of Warwick BREST software licence](LICENSE), which permits academic research and non-commercial, non-clinical evaluation.

## Publication

**Wei J, Zhou X, Santeramo R, Al-Attar M, Muthyala S, Gilbert F, Brentnall AR, Montana G. BREST-AI: Three-year breast cancer risk prediction from screening mammograms. *npj Breast Cancer*, to appear.**

Please cite this study when using BREST. This repository implements the model described in that publication.

## Modelling approach

BREST uses a ResNeXt-50 image encoder and Attentional Feature Fusion (AFF) to combine the craniocaudal (CC) and mediolateral oblique (MLO) views from both breasts. Training progresses from annotated lesion patches to full-image cancer detection and risk prediction, followed by fusion of all four views into an examination-level risk score. This curriculum builds on our finding that stronger cancer-detection models can improve risk prediction ([Santeramo et al., 2024](https://doi.org/10.1186/s13058-024-01775-z)).

<p align="center">
  <img src="Images/Model-Overview.png" alt="BREST model architecture and training curriculum" width="700" />
</p>

## Results reported in the forthcoming paper

The study evaluated three-year risk discrimination in independent retrospective case–control cohorts, comparing BREST with MIT's Mirai model on the same women. These results use the final 16-bit BREST configuration.

| External validation cohort | Women (cancer cases) | BREST AUC (95% CI) | Mirai AUC (95% CI) |
|---|---:|---:|---:|
| UK, Hologic | 7,596 (1,899) | 0.751 (0.738–0.764) | 0.700 (0.687–0.714) |
| UK, Siemens | 1,600 (400) | 0.719 (0.689–0.748) | 0.685 (0.656–0.715) |
| US, Hologic | 388 (97) | 0.670 (0.609–0.731) | 0.633 (0.570–0.696) |

AUC measures discrimination; CI denotes confidence interval. The smaller US evaluation is exploratory. Use of scores as absolute risks requires calibration to the target screening population.

## Using BREST

### Environment and trained weights

Contact [Giovanni Montana](mailto:g.montana@warwick.ac.uk) to request trained weights and the inference Docker image. The repository's [environment record](requirements.txt) lists Python 3.9, PyTorch 2.4 and CUDA 12.4 alongside the analysis dependencies. It contains both Conda and Python package specifications; use it as an environment reference. DICOM conversion also requires DCMTK's `dcmj2pnm` executable.

### Prepare the images

Run commands from the repository root. The preprocessing script converts DICOM mammograms to breast-aligned, 2048 × 2048, 16-bit PNGs and produces breast masks. Its metadata CSV requires `path` (relative to the DICOM root, without the `.dcm` extension) and `ImageLaterality` (`L` or `R`).

```bash
python dataProcessing/dicom_to_png16_2048_background-cleaned.py \
    --csv-file-path /data/image_metadata.csv \
    --dicom-dir /data/dicom \
    --processed-images-dir /data/processed_png \
    --breast-mask-dir /data/breast_masks \
    --max-workers 16
```

### Run inference and evaluate a labelled cohort

The evaluation script requires one complete four-view examination per `ClientID`, with rows ordered **left CC, right CC, left MLO, right MLO**. Its CSV requires `ClientID`, `path` (relative to the PNG root, without `.png`) and `EpisodeOutcome`. For a cancer-versus-control risk evaluation, `N` identifies the cancer-free controls; the script maps every other outcome value to a positive label. Prepare the cohort and outcome labels accordingly.

[Example metadata](data/test_metadata_example.csv) and a [view-ordering utility](data/sort_csv.py) are provided. The utility additionally requires `EpisodeID`, `ViewPosition` and `ImageLaterality`. The inference loader resizes the prepared images to 1792 × 1792 pixels.

```bash
python scripts/patientLevel-inference.py \
    --metadata_csv /data/risk_evaluation_metadata.csv \
    --image_root_dir /data/processed_png \
    --final_csv_path /data/output/results.csv \
    --roc_plot_path /data/output/roc_curve.png \
    --model_checkpoint /data/checkpoints/episode-Level-3yrisk.pth \
    --gpu_id 0
```

The script saves scores and outcome labels, reports the cohort AUC and produces a receiver operating characteristic curve.

## Repository contents

| Location | Contents |
|---|---|
| [`dataProcessing/`](dataProcessing/) | DICOM conversion and breast-background processing |
| [`scripts/`](scripts/) | Neural network definitions and examination-level inference |
| [`data/`](data/) | Example metadata and view ordering |
| [`Images/`](Images/) | Model diagrams and research illustrations |
| [`requirements.txt`](requirements.txt) | Inference environment reference |
| [`LICENSE`](LICENSE) | Terms covering the software and neural network weights |

## Team, funding and data

The research is led by Professor Giovanni Montana at the University of Warwick, with collaborators at Queen Mary University of London, the University of Cambridge, University Hospitals of Leicester and University Hospitals Coventry and Warwickshire. The publication above lists the study authors.

Development was supported by **Cancer Research UK (C49757/A28689)** and the **MRC Gap Fund (UKRI086)**. We acknowledge the OPTIMAM project team, Royal Surrey NHS Foundation Trust and Cancer Research UK for the UK imaging resource. US external validation used the EMory BrEast imaging Dataset (EMBED).

Access to OPTIMAM images and clinical records is managed separately through the [OPTIMAM data-access process](https://medphys.royalsurrey.nhs.uk/omidb/apply-for-access/).

Research and model-access enquiries: [g.montana@warwick.ac.uk](mailto:g.montana@warwick.ac.uk). Commercial licensing enquiries: [Warwick Ventures](mailto:ventures@warwick.ac.uk).
