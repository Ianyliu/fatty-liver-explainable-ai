<h1 align="center">Explainable Fatty Liver Classification from Multi-Image Ultrasound Studies</h1>

<p align="center"><strong>多張超音波影像之可解釋脂肪肝分類</strong></p>

<p align="center">
Research code for explaining subject-level fatty liver predictions made from multiple abdominal ultrasound images using graph-based image aggregation, perturbation sampling, marginal association analysis, and regularized surrogate models.
</p>

<!-- <p align="center">
  <em>By Ian Liu 劉以恆 [^1] [^2] [^3]</em><br>
  <em>Project mentor/advisor: Tso-Jung Yen 顏佐榕, PhD</em>
</p> -->

###### By: Ian Liu 劉以恆[^1] [^2] [^3]

###### Project Mentor/Advisor: Tso-Jung Yen 顏佐榕, PhD[^3]

---

## Overview

This repository contains an explainability pipeline for a multi-image ultrasound classifier. In the original classification setting, each subject may have many ultrasound image crops. A pretrained image encoder first converts each image into features, and a graph neural network aggregates the image-level information into a subject-level disease prediction.

The goal of this repository is not only to classify fatty liver disease, but also to estimate **which images within a subject's ultrasound study most influenced the model prediction**.

The workflow treats each ultrasound image as a candidate explanatory unit. It repeatedly samples image subsets, runs the trained model on each sampled subset, records the model prediction, and then fits interpretable post-hoc explanation models to estimate image-level influence.

![Workflow](https://github.com/user-attachments/assets/7583f3b0-f4e1-4042-b206-ca105c4656b2)
<p align="center"><em>Figure 1. Overall explanation workflow for multi-image ultrasound classification.</em></p>

---

## What Makes This Project Different

Most standard image explanation tools, such as LIME-style perturbation, are designed for a **single image**. This project extends the same local-perturbation intuition to a **multi-image graph classification setting**.

Key ideas:

- **Multi-image explanation unit**: instead of perturbing superpixels inside one image, the pipeline perturbs which ultrasound images are included in a subject-level graph.
- **Subject-level prediction tracing**: sampled image subsets are passed through the trained image-encoder + graph-classifier pipeline to observe how predictions change.
- **Marginal and conditional influence estimates**:
  - Pearson/Spearman/Matthews-style marginal relationships estimate how each image is associated with the model's prediction across sampled subsets.
  - Ridge and ElasticNet surrogate classifiers estimate conditional image importance after accounting for other sampled images.
- **Uncertainty-aware summaries**: coefficients and correlations are reported with standard errors, confidence intervals, and significance indicators.
- **Class-balanced sampling**: sampling routines attempt to avoid explanation sets dominated by only one prediction class.

---

## Repository Structure

```text
.
├── README.md
├── xai_env.yml
├── generate_samples_and_marginal_relations.py
├── sampling_marginal_relation_pipeline.py
├── sampler.py
├── marginal_relation.py
├── classifiers.py
├── ridge_run.py
├── elastic_net_run.py
├── organize_output.ipynb
└── deprecated_and_archived/
```

Main files:

| File | Purpose |
| --- | --- |
| `generate_samples_and_marginal_relations.py` | Example driver script for running the explanation pipeline on a selected test split. |
| `sampling_marginal_relation_pipeline.py` | Original LIME-style subject-level perturbation pipeline. Samples image subsets, runs predictions, computes marginal relations, and saves plots/tables. |
| `sampler.py` | Alternative sampler class with clustering-aware image subset sampling and graph-model prediction utilities. |
| `marginal_relation.py` | Standalone marginal-correlation analysis from saved prediction result matrices. |
| `classifiers.py` | Ridge, ElasticNet, and GLMNET-style surrogate classifiers with bootstrap/statistical summaries. |
| `ridge_run.py` | Fits Ridge surrogate models on saved perturbation results. |
| `elastic_net_run.py` | Fits ElasticNet surrogate models on saved perturbation results. |
| `organize_output.ipynb` | Notebook for reorganizing explanation outputs into positive, negative, neutral, CSV, and plot folders. |
| `xai_env.yml` | Conda environment file used for the original experiments. |

---

## Method Summary

For each subject:

1. Load all ultrasound image IDs belonging to the subject.
2. Encode image crops using a pretrained image encoder.
3. Build a graph representation from the encoded images.
4. Run the trained graph classifier to obtain a subject-level prediction.
5. Generate many perturbed image subsets.
6. Re-run the classifier on each subset.
7. Store a binary design matrix indicating which images were included in each sampled subset.
8. Estimate image-level influence using:
   - marginal correlation between image inclusion and model prediction;
   - Ridge surrogate classification coefficients;
   - ElasticNet surrogate classification coefficients.
9. Save explanation tables and bar plots with image thumbnails overlaid on the bars.

---

## Example Results

<p align="center">
<img width="806" alt="Correlation coefficient explanation plot" src="https://github.com/user-attachments/assets/9b7c1cdf-78e4-4bdb-a180-9ec11ac07492">
<br>
<em>Figure 2a. Correlation coefficients for each image, interpreted as marginal image influence.</em>
</p>

<p align="center">
<img width="793" alt="ElasticNet coefficient explanation plot" src="https://github.com/user-attachments/assets/c39ebec2-2211-4c89-be99-015ad734dfaa">
<br>
<em>Figure 2b. ElasticNet coefficients for each image, interpreted as conditional image influence. Faded bars indicate statistical insignificance.</em>
</p>

<p align="center">
<img width="782" alt="Ridge coefficient explanation plot" src="https://github.com/user-attachments/assets/da72f591-0afe-40a2-96b2-78e8bcc3edbc">
<br>
<em>Figure 2c. Ridge coefficients for each image, interpreted as conditional image influence. Faded bars indicate statistical insignificance.</em>
</p>

---

## Setup

Create the original conda environment:

```bash
conda env create -f xai_env.yml
conda activate xai
```

The original experiments also depend on a local package/module named `usflc_xai`, which provides the dataset loaders and model definitions used by the pipeline.

You will also need local access to:

- cropped ultrasound images;
- metadata CSV files;
- train/test split CSV files;
- trained model checkpoints such as `best_results.ckpt`;
- optional clustering result JSON files if using `sampler.py`.

---

## Configuration

Several scripts were written for the original experiment directory layout and include absolute or project-specific paths. Before running the pipeline, update these paths for your own machine.

At minimum, create a `.env` file with:

```bash
CROP_IMAGE_DIR_PATH=/path/to/cropped_ultrasound_images
```

Then check or update the following script-level paths:

- metadata CSV path, usually `meta_data/TWB_ABD_expand_modified_gasex_21072022.csv`;
- test split path, usually `fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/datasetXX/test_datasetXX.csv`;
- model checkpoint path, usually `model_tl_twbabdXX/best_results.ckpt`;
- output directory for explanation results;
- clustering result directory if using clustering-aware sampling.

---

## Running the Pipeline

### 1. Generate perturbation samples and marginal relations

Edit the test split ID and CUDA device in `generate_samples_and_marginal_relations.py`, then run:

```bash
python generate_samples_and_marginal_relations.py
```

This creates per-subject folders containing sampled prediction matrices, marginal correlation summaries, heatmaps, and bar plots.

### 2. Run Ridge surrogate explanations

After perturbation prediction results have been generated, update `result_dir` and `all_subj_save_dir` in `ridge_run.py`, then run:

```bash
python ridge_run.py
```

### 3. Run ElasticNet surrogate explanations

Similarly, update `result_dir` and `all_subj_save_dir` in `elastic_net_run.py`, then run:

```bash
python elastic_net_run.py
```

### 4. Organize outputs

Use `organize_output.ipynb` to collect image-level explanation results into folders such as:

```text
organized_output_folder/
├── correlations/
│   ├── positive/
│   ├── negative/
│   └── neutral/
├── csv/
│   ├── 0-elastic_net_coefficients.csv
│   ├── 1-ridge_coefficients.csv
│   └── 2-correlations.csv
├── ridge/
│   ├── positive/
│   ├── negative/
│   └── neutral/
├── elastic/
│   ├── positive/
│   ├── negative/
│   └── neutral/
└── plots/
    ├── 0-vbar.png
    └── 1-hbar.png
```

Interpretation:

- `positive/`: images that support or push the prediction toward the positive fatty-liver class;
- `negative/`: images that support or push the prediction away from the positive class;
- `neutral/`: images whose estimated influence is not statistically significant;
- `csv/`: tabular explanation summaries;
- `plots/`: bar plots and visual summaries.

---

## Outputs

Typical per-subject outputs include:

| Output | Description |
| --- | --- |
| `design_matrix.csv` | Binary matrix where each row is a sampled image subset and each column is an image. |
| `pred_results.csv` | Design matrix plus model prediction labels for each sampled subset. |
| `corr_results.csv` or `correlations.csv` | Marginal image-prediction association statistics. |
| `ridge_coefficients.csv` | Ridge surrogate coefficients and uncertainty summaries. |
| `elastic_net_coefficients.csv` | ElasticNet surrogate coefficients and uncertainty summaries. |
| `hbar.png`, `vbar.png` | Horizontal and vertical image-level explanation plots. |
| `sampling_corr.csv` | Correlation structure among perturbation samples. |
| `encoded_img_corr.csv` | Correlation among encoded image features when generated. |

---

## Notes and Limitations

- This is research code from an experimental explainability workflow, not a polished clinical software package.
- The scripts assume access to private/local ultrasound image data and trained checkpoints, which are not included in this repository.
- Several paths are currently hard-coded and should be parameterized before reuse.
- The explanation scores describe the behavior of the trained model under perturbation sampling. They should not be interpreted as causal clinical biomarkers without additional validation.
- Statistical significance here refers to the surrogate explanation analysis, not to clinical diagnostic certainty.

---

## Citation / Attribution

If you use or adapt this repository, please cite or acknowledge the project as:

Ian Liu, Tso-Jung Yen. (2025). Ianyliu/fatty-liver-explainable-ai: First Version (0.0.0). Zenodo. https://doi.org/10.5281/zenodo.14601896

---


[^1]: Department of Data Science, Fei Tian College Middletown, Middletown NY
[^2]: Department of Biostatistics, Brown University, Providence RI
[^3]: Institute of Statistical Science, Academia Sinica, Taiwan
