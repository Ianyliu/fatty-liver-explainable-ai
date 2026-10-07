# Prompt for the agent on the original server

Copy the prompt below into the agent on the original server. All exports should stay in private storage and be transferred directly, never through GitHub.

---

We are migrating `fatty-liver-explainable-ai` to another cluster. Please perform a **read-only reproducibility investigation** and prepare a private transfer bundle. Do not run inference, sampling, model training, bootstrap fits, clustering, or change the scientific code. Do not overwrite existing outputs.

1. Start with `/home/liuusa_tw/twbabd_image_xai_20062024/` (adjust if moved). Search its `custom_lime_results/` and other result directories for these files:
   - `<run>/<MI_ID>/pred_results.csv`: one binary inclusion column per image ID, followed by `yhat` and `y`.
   - `<run>/<MI_ID>/design_matrix.csv`: the same inclusion columns without labels.
   - `<run>/<MI_ID>/corr_results.csv`, `correlations.csv`, `sampling_corr.csv`, and `encoded_img_corr.csv`.
   - `<run>/<MI_ID>/ridge_coefficients.csv`, `elastic_net_coefficients.csv`, `hbar.png`, `vbar.png`, and heatmap PNGs.
   Search filenames first (for example `rg --files --hidden --no-ignore custom_lime_results`) rather than printing patient records. A missing design matrix can be derived later from pred_results by dropping `yhat,y`; preserve the original if it exists.

2. The checked-in code points to these candidate runs; verify their actual existence and relationships rather than assuming they belong together:
   - Sampling/predictions: `custom_lime_results/07-12-2024-03-57-58/`.
   - Ridge: `custom_lime_results/ridge-08-06-2024-01-24-40/`.
   - Elastic net: `custom_lime_results/elastic-net-old-dataset-08-01-2024-06-59-39/`.
   - Correlation: `custom_lime_results/correlation-old-dataset-08-01-2024-03-37-02/`.
   - Other references: `TEMP-elastic-net-08-02-2024-00-32-01`, patient-specific `*-RIDGE-RESULTS`, and organized-output directories.
   - Legacy only: `custom_lime_results/clustering-07-29-2024-06-05-39/<MI_ID>/clustering_result_dict.json`. Identify these if present, but do not regenerate clustering or make it a dependency of the current adaptive sampling method.

3. Select an initial reproduction patient from test split **09**, with `liver_fatty > 0` and at least 20 images in `IMG_ID_LIST`. Require:
   - Saved `pred_results.csv`, both binary prediction classes, and preferably at least 10 unique rows in each class after the original runner's `drop_duplicates()`.
   - Every referenced cropped image exists, including the full metadata image list for that patient.
   - At least one corresponding Ridge reference CSV, preferably correlation and Elastic-Net CSVs/plots as well.
   Among qualifying patients, prefer the smallest image count, then stable MI_ID order; do not select for unusually strong or attractive results. Keep the chosen MI_ID and detailed candidate table inside the private bundle. If no patient qualifies, explain which requirement fails. The destination upload currently has 52,851 JPEGs but **no fully covered test-09 patient**; test09 references 16,545 images, of which 14,957 are missing. The destination can provide its private `outputs/reproducibility/missing_test_images.csv` for exact transfer matching.

4. Prepare a bundle containing the selected patient's entire original result folder, matching reference CSVs/plots from the separate stages, all of their `<MI_ID>_<IMG_ID>.jpg` cropped images, and a manifest mapping original paths to bundle-relative paths with SHA-256 and byte sizes. Preserve the original patient directory name and image IDs; do not rename columns or substitute images. Copy files into the bundle; retain originals. If practical, separately bundle missing images listed by the destination report for a later full-cohort transfer.

5. Record provenance in a private text/JSON report:
   - Git commit/branch/status of the actual code used, and copies/hashes of uncommitted scientific files, especially `sampling_marginal_relation_pipeline.py`, `classifiers.py`, `marginal_relation.py`, `ridge_run.py`, `elastic_net_run.py`, `organize_output.py`, and all `usflc_xai/*.py` files.
   - Python version, active environment name/path, `python -m pip freeze`, `conda list --explicit` and `conda env export` if applicable, Torch/TorchVision/PyG/NumPy/SciPy/scikit-learn/glmnet versions. Use the original environment without modifying it. Include OS, compiler/BLAS versions, CUDA/driver/GPU information and available Slurm launch scripts/logs.
   - Dataset-09 train/valid/test lists, metadata filename/hash, the graph checkpoint `model_tl_twbabd09/best_results.ckpt` hash, and exact pretrained image encoder weights. Check the original Torch cache, normally `~/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth`, and include the actual DenseNet121 ImageNet weights used. Do not assume the graph checkpoint contains these encoder weights.
   - Original command/config, sample count and subset-size bounds, class-balancing parameters, graph correlation threshold, image transforms, seeds/RNG state (if recorded), CV grids/folds, bootstrap counts, significance rule, and stage-to-stage provenance. Mark unknown values as unknown.
   - Inspect whether the actual source used for the saved run has the correct image-ID-to-column mapping. The current GitHub code has `self.img_to_indx = dict(enumerate(self.img_list))`, which appears reversed. Compare, report, and preserve the used code; do not fix it during this investigation.

6. Report missing files and ambiguity explicitly. Do not fabricate seeds, claim runs match merely because timestamps are similar, execute `.ckpt/.pt/.pth` pickle payloads, export `.env` or credentials, or place patient data in a public repository. Return only aggregate findings and the private bundle's location in chat.

---

## Destination layout after transfer

- Original predictions/design matrices: `outputs/original/07-12-2024-03-57-58/<MI_ID>/` (or set `XAI_PREDICTION_ROOT` to the verified run).
- Reference stage outputs: `outputs/original/<original-stage-run>/<MI_ID>/`.
- Images: `data/cropped_images/<MI_ID>_<IMG_ID>.jpg`.
- Model checkpoint: `checkpoints/model_tl_twbabd09/best_results.ckpt`.
- DenseNet weights: `checkpoints/torch/hub/checkpoints/densenet121-a639ec97.pth`, with `TORCH_HOME` set to the absolute `checkpoints/torch` directory.
- Provenance/environment manifests: `outputs/original/provenance/`.

All these destinations are ignored by Git. Keep a replay in a new `outputs/replay/` directory so reference results remain intact.
