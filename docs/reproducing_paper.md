# Reproducing the published BREST-AI evaluation

This document describes the v1 model and study reported by Wei et al., *BREST-AI: Three-year breast cancer risk prediction from screening mammograms*, **npj Breast Cancer** (2026), [doi:10.1038/s41523-026-01069-x](https://doi.org/10.1038/s41523-026-01069-x).

## Data and prediction target

The model predicts a breast cancer diagnosis within three years of a negative index screening mammogram. The positive group includes subsequent screen-detected and interval cancers. Each input examination contains four mammographic images. The clinical evaluation uses the original study's patient-level partitions and case–control membership.

Development used OPTIMAM screening data from three UK services (Sites 1–3 in the paper): **65,419 women**, comprising **58,591 training women** and **6,828 internal-validation women**. Training included 9,020 malignant, 1,633 benign and 47,938 cancer-free women. Internal validation included 1,317 malignant, 189 benign and 5,322 cancer-free women. The curriculum progresses from lesion patches to full-image detection/risk learning and four-view examination-level fusion.

For the patch pretraining phase, the paper reports **235,911 normal patches and 82,348 cancer patches**. The training data contained **9,805 annotated lesion bounding boxes in 8,964 images from 4,653 malignant patients**. These patch, annotation and image counts describe supervision within the development cohort; they are not additional patients. The later phases used full mammograms and then four-view examinations, with shared patient partitions throughout.

The primary external evaluations were:

| Original study cohort | Women | Cases | Controls | BREST AUC (95% CI) | Mirai AUC (95% CI) |
|---|---:|---:|---:|---|---|
| UK Hologic, OPTIMAM Sites 4–5 | 7,596 | 1,899 | 5,697 | 0.751 (0.738–0.764) | 0.700 (0.687–0.714) |
| UK Siemens, OPTIMAM Site 6 | 1,600 | 400 | 1,200 | 0.719 (0.689–0.748) | 0.685 (0.656–0.715) |
| US Hologic, two EMBED cohorts | 388 | 97 | 291 | 0.670 (0.609–0.731) | 0.633 (0.570–0.696) |

The UK study included women aged 46–73, with controls matched by age, scanner type and site at a 1:3 ratio. Controls were identified using the recorded subsequent screening follow-up. Benign cases were excluded from external validation. For EMBED, age restrictions were relaxed and controls were matched by age, race, breast density and cohort, also at 1:3. The paper gives the complete selection criteria and subgroup definitions. Its external evaluations are retrospective case–control studies.

## Obtain authorised access

OPTIMAM images and clinical records are supplied by the data custodians through the [OMI-DB application process](https://medphys.royalsurrey.nhs.uk/omidb/apply-for-access/). EMBED access is described by the [EMBED project](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/); the paper used an eligible matched subset of the original study data.

Patient images, cohort lists, clinical metadata and individual predictions are kept with the approved research data. They are not distributed in this repository or on the model Hub. To reproduce the exact study, obtain access to its original cohort membership and input examinations under the applicable approval. Recreating a different matched sample with the same size produces a new evaluation.

## Prepare the inputs

Use the DICOM conversion described in the repository README to produce breast-aligned **2048 × 2048 grayscale 16-bit PNGs**. Choose `--profile paper-hologic` for the Hologic cohorts, using 16-bit rendering and Lanczos resizing. The original Siemens preparation is available as `--profile paper-siemens`: it renders an 8-bit image, expands its range to uint16 by multiplying by 257, and applies the original NumPy bilinear resize. Both profiles save 16-bit PNGs. The right breast is flipped during conversion. Inference repeats the grayscale channel, divides pixel values by 65535, resizes to **1792 × 1792** with Kornia bilinear antialiasing, and applies mean **0.0856** and standard deviation **0.1687** to each channel.

Background cleaning uses the original fixed random seed of 42. Keep the conversion settings and dependency versions fixed; prepared images include both breast pixels and the seeded background fill.

The network's view order is **left CC, right CC, left MLO, right MLO**. Use the original ordered evaluation metadata for a paper reproduction. The explicit `--preserve_row_order` option retains the original v1 loader's four-row order, including historical view-tag anomalies; it does not infer a new order or remove those examinations. For new inputs, the default loader validates and orders the four standard views.

The CSV needs `ClientID` and `path` for original-order inference, with `EpisodeID` if repeat examinations are present. Default view-validated inference also needs `ViewPosition` and `ImageLaterality`. To evaluate risk, supply `EpisodeOutcome=N` for controls and a non-`N` code for women diagnosed within the three-year horizon. These are outcome labels, not evidence that cancer was visible at the index screen.

## Run the released model and evaluation

Use Python 3.11 on Linux with `requirements-published-cuda.lock`, which pins the published PyTorch 2.1.0 / CUDA 11.8 and Kornia 0.7.1 runtime and its dependencies. The weights use the tensor-only safetensors format. The same installation supports CPU inference with `--device cpu`.

```bash
python -m pip install -r requirements-published-cuda.lock
python scripts/download_weights.py --output-dir models

python scripts/patientLevel-inference.py \
    --metadata_csv /data/original_study/uk_hologic_images.csv \
    --image_root_dir /data/original_study/png16 \
    --model_checkpoint models/brest-ai-v1.safetensors \
    --final_csv_path /data/results/uk_hologic_predictions.csv \
    --preserve_row_order --batch_size 4 --gpu_id 0

python scripts/evaluate_paper.py \
    --cohort uk_hologic \
    --predictions /data/results/uk_hologic_predictions.csv \
    --output /data/results/uk_hologic_metrics.json
```

Use `uk_siemens` and `us_hologic` for the other two fixed cohorts. Evaluation reports the AUC and a normal DeLong 95% confidence interval and checks the published sample counts. Identical counts alone do not verify cohort identity; retain the original study membership and image-selection records.
