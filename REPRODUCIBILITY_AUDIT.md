# Reproducibility audit

**Complete-patient study update:** the prospective n=1 study has now completed all three seeds on Discovery (array `9557485`), with 1,000 training rows per sampling strategy, 200 shared evaluation draws per seed and 6,741 total model calls. All tasks exited `0:0`; independent CSV checks, artifact hashes, four inspected figures, the environment and 23 tests pass. Random-trained Ridge had lower shared-novel probability MAE in all seeds, but improved on its constant baseline in only one; adaptive missed its 50/50 target. Both strategies' ranked deletion curves had lower AUC than seeded random controls. See [the complete-patient execution record](docs/COMPLETE_PATIENT_STUDY.md). This new methods study does not establish cohort generalization or change historical fitters, formulas or saved artifacts. The earlier audit scope below is retained as a historical record.

**October 7 execution update:** the locked environment and one-patient CPU/GPU smoke tests pass. Sixteen checked historical masks (eight July, eight November) match their saved GNN labels. Discovery jobs `9557444` (random) and `9557446` (repaired adaptive) completed successfully, submitted directly from Polaris. A first job exposed a Polaris-local Python path; the identical interpreter is now on shared storage. The adaptive mapping/NumPy label/count/subset-size/pool-semantics defects described below have subsequently been repaired and tested; realized class balance remains a target, not a guarantee. Earlier no-execution/no-correction boundaries below describe prior audit/upload stages. Historical correlation formulas, fitters, bootstrap semantics and saved outputs remain unchanged. See [the next-run guide](docs/NEXT_RUNS.md) for exact behavior changes, evidence, the 152-image pilot upload plan and remaining gates.

Audit date: 2026-10-07 (updated after the second artifact upload). Scope: migration and prerequisite verification only. **No training, sampling, inference, bootstrap fit, clustering, or full experiment was run. Scientific calculations and method defects were not repaired.** Filesystem defaults were made configurable; saved notebook outputs were removed from the published tree, with originals retained privately.

## 1. Required files and expected locations

Paths below are relative to the repository unless indicated otherwise. Patient identifiers, records, file-level manifests, and missing-image lists are intentionally kept out of this document and Git.

| Requirement | Expected destination | Status and use |
| --- | --- | --- |
| Cropped ultrasound images | `data/cropped_images/<MI_ID>_<IMG_ID>.jpg` | Present but incomplete: 52,868 JPEGs after adding 17 bundle images (the original 52,851 JPEG upload totaled 1,430,353,098 bytes). Required for model inference and explanation thumbnails. |
| Subject metadata | `data/meta_data/TWB_ABD_expand_modified_gasex_21072022.csv` | Present, 9,787,510 bytes, 22,062 rows. Required columns: `MI_ID`, `liver_fatty`, `IMG_ID_LIST` (Python literal list). Contains clinical data; never commit. |
| Binary classification splits | `data/fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/datasetXX/{train,valid,test}_datasetXX.csv` | Present: all 30 CSVs for splits 01–10. Split09 has 7,210 train, 802 validation, 890 test rows. These are inputs, not recreated random splits. |
| Trained graph model | `checkpoints/model_tl_twbabd09/best_results.ckpt` | Present, 18,976,357 bytes. ZIP/pickle opcode inspection finds `model_state_dict` and `optimizer_state_dict`, with no DenseNet denseblock keys. No checkpoint payload was executed. Model architecture compatibility remains unverified. |
| Graph checkpoints for other splits | `checkpoints/model_tl_twbabdXX/best_results.ckpt` | Missing for 01–08 and 10; only needed if those splits are requested. |
| External project package | `usflc_xai/{__init__,datasets,models,training,utils}.py` | All five uploaded source files present locally. No package manifest/version/commit accompanied the upload. Kept ignored rather than publishing uploaded source without provenance. A fresh clone must receive this source separately. |
| Image encoder weights | Torch cache `hub/checkpoints/densenet121-a639ec97.pth` under `TORCH_HOME` (normally `~/.cache/torch`) | **Present and hash verified** at `checkpoints/torch/hub/checkpoints/densenet121-a639ec97.pth` (32,342,954 bytes). The full SHA-256 matches the original-server report and filename hash prefix. Local `.env` sets absolute `TORCH_HOME`. `models.image_encoder_model` requests DenseNet121 `IMAGENET1K_V1`; model weights were not loaded/executed. |
| Alternate encoder weights | ResNet50 `IMAGENET1K_V2`; timm `vit_large_patch16_224_in21k` | Not uploaded; only required for those nondefault encoders. Original architecture/checkpoint matching must be established first. |
| Saved sampled predictions | `outputs/original/07-12-2024-03-57-58/<MI_ID>/pred_results.csv` | **Present for the selected patient.** A separate complete test09 run is also present at `outputs/original/11-28-2024-03-27-14/` (890 patient tables). Historical July path is explicitly used by marginal/Ridge/Elastic-Net scripts. Columns: image IDs with binary inclusion indicators, `yhat`, `y`. Essential to replay downstream explanations without resampling. |
| Saved design matrices | Same patient folder, `design_matrix.csv` | **Present for the selected July patient and all 890 November patients.** Inclusion columns/rows match their prediction tables in the saved-run audit. |
| Sampling correlation artifacts | Same folder: `corr_results.csv`, `sampling_corr.csv`, `encoded_img_corr.csv` | **Present for the selected July patient and all November patients.** The current sampling module produces these plus `sampling-corr-heatmap.png`, `image-encoded-corr-heatmap.png`, `hbar.png`, `vbar.png`. |
| Original Ridge explanation | `outputs/original/ridge-08-06-2024-01-24-40/<MI_ID>/{ridge_coefficients.csv,hbar.png,vbar.png}` | **Present for the selected patient.** This historical run is referenced in the organizer. Patient identity/artifact presence alone does not prove fit-stage provenance. |
| Original Elastic-Net explanation | `outputs/original/elastic-net-old-dataset-08-01-2024-06-59-39/<MI_ID>/{elastic_net_coefficients.csv,hbar.png,vbar.png}` | **Present for the selected patient.** Verify fit-stage provenance independently. |
| Original reprocessed correlations | `outputs/original/correlation-old-dataset-08-01-2024-03-37-02/<MI_ID>/{correlations.csv,hbar.png,vbar.png}` | **Present for the selected patient.** Its saved coefficients and SEs match the existing marginal formula applied to the July deduplicated predictions to floating-point precision. This calculation differs from sampling-module correlations; do not interchange reference files. |
| Organized explanation outputs | `outputs/original/<organized-run>/<MI_ID>/{csv,plots,correlations,ridge,elastic}/...` | **Present for the selected patient** in `outputs/original/organized-output-11-29-2024-05-23-42/`. Optional reference; organizer copies patient images. An organized-output timestamp does not establish its input sampling run. |
| Original runtime/run provenance | `outputs/original/provenance/` | **Current original-server environment captures, source snapshot, Git state, hashes, and 12 logs are present** under `outputs/original/bundles/<bundle>/`. Exact historical runtime/source-used provenance, original RNG states and complete run commands remain unknown. Both base and xai environment captures are included; they must not be conflated. A subsequent user-supplied report confirms current imported versions, while historical runtime provenance remains unknown (see follow-up section). |
| Legacy cluster assignments | `outputs/original/clustering-07-29-2024-06-05-39/<MI_ID>/clustering_result_dict.json` | **Present for all 135 legacy cohort patients**, plus the run-level summary and archived figures (3,778 files total). Required **only** by `clustering_sampler.Sampler`. Keys include `mi_id`, `img_id_list`, `agglomerative.best_cluster_labels`. Do not regenerate or introduce these into the current adaptive pipeline. |
| Other legacy clustering outputs | Historical `clustering-07-19-2024-01-46-55/<MI_ID>/clustering_results.csv` and figures | Missing; only used by archived clustering notebooks. |
| Legacy embedding/cluster summaries | `outputs/original/<clustering-run>/<MI_ID>/...` and run-level `all_subj_best_results.csv` | July29 cluster JSONs/summary/figures are present. Other per-patient embedding/precomputed CSVs are not established by that inventory; archived clustering can optionally read caller-supplied `pre_computed_results_dirs`. Not required for the current adaptive method. |
| Training resume artifacts/logs | Caller-configured `backup_file_name`, `epoch_<N>.ckpt`, result/confusion-matrix logs | Missing, apart from the best graph checkpoint. Resume expects scheduler state and last epoch in addition to model/optimizer state. Not required for explanation replay; do not start training to reconstruct them. |
| Cached encoded graphs/features | `fattyliver_<encoder>_<dataset_name>_dataset/pretrained_<encoder>_<MI_ID>.pt` | Missing. Used by uploaded `AUSDataset_train/valid/test`, `AUSDataset`, and `dataset_container`; not required by `single_data_loader` for explanations, which encodes JPEGs directly. Producer/preprocessing script is not supplied. |
| Legacy single-image examples | `/home/liuusa_tw/data/TWBABD_US_images_5_instances_13062024/<MI_ID>/*.jpg` | Missing; archived `LIME_test.ipynb` references these. Not a prerequisite for default multi-image replay. |
| Local configuration | Ignored `.env`; tracked `.env.example` | Created. `CROP_IMAGE_DIR_PATH` must retain a trailing slash for uploaded `datasets.single_data_loader` string concatenation. |

Uploaded directories were renamed into their destinations without rewriting their contents. Original Python bytecode remains in `temp/original_usflc_xai_pycache/`; macOS AppleDouble sidecars remain in `temp/__MACOSX/` and are not image data. Original notebooks with saved outputs remain in `temp/original_notebooks/`. These are all ignored.

## 2. What is present versus missing: verified input coverage

The local audit uses CSV parsing, literal image lists, and filename membership, without opening images or loading models:

| Check | Result |
| --- | ---: |
| Metadata rows / duplicate MI_IDs | 22,062 / 0 |
| Test09 rows / duplicate MI_IDs | 890 / 0 |
| Test09 IDs without metadata | 0 |
| Malformed test09 metadata image lists | 0 |
| Test09 image references | 16,545 |
| Available test09 image references | 1,605 |
| Missing test09 image references | **14,940** |
| Test09 patients with every metadata image available | **1 / 890** |
| Positive test09 patients with at least 20 metadata images | 135 |
| Such patients with complete image coverage | **1 / 135** |

**One patient now passes the input-only replay preflight.** Its 20 images match the bundle SHA-256 hashes and decode with Pillow; three were already present and 17 were added without overwriting existing files. Directory existence and total JPEG count do not establish full dataset completeness. Other image content, full metadata/train/validation coverage, checkpoint architecture compatibility and historical-run provenance remain unverified.

Rerun `.venv/bin/python scripts/audit_reproducibility.py --strict` after new uploads. The script exits nonzero for incomplete/ambiguous inputs. Its detailed reports are private:

- `outputs/reproducibility/missing_test_images.csv`: exact patient/image IDs and expected filenames.
- `outputs/reproducibility/candidate_patients.csv`: complete positive patients meeting the existing >=20-image rule; now contains one candidate.
- `outputs/reproducibility/artifact_manifest.json`: SHA-256/byte sizes for uploaded CSVs, checkpoint, and external Python source; does not hash every JPEG.
- `outputs/reproducibility/source_inventory.json`: imports and old absolute literals in Python, plus notebook cell references.
- `outputs/reproducibility/summary.json`: aggregate checks above.

The audit's default coverage is **test09**, not every subject in metadata or all ten splits. Use explicit `--split`, `--metadata`, and `--images` arguments for other input inventories.

## 3. Pipeline entry points and dependencies

The local branch initially ended at `0d9e100`. Before migration edits, it was fast-forwarded to the four newer upstream default-branch commits through **`3a5a718`**. GitHub's default branch is `master`; no `main` branch exists. Upstream restores `sampling_marginal_relation_pipeline.py` and `organize_output.py`, refactors `RidgeRun`, and renames `sampler.py` to `clustering_sampler.py`. The sampling-module/organizer absence observed before syncing is therefore **resolved by upstream**, not a remaining upload request.

| Entry point | Inputs and dependencies | Outputs / important execution behavior |
| --- | --- | --- |
| `generate_samples_and_marginal_relations.py` | Imports `LIME_all_subj_pipeline` from `sampling_marginal_relation_pipeline`, metadata/test split, cropped JPEGs, DenseNet weights, graph checkpoint, Torch/PyG/timm | Immediately constructs the pipeline and runs **all** test subjects on import/execution. Hard-coded `test_data_id='09'`, `cuda_device_no=1`; default `n_samples=1000`, min subset size 3, target 0.5. Do not run during this audit. |
| `sampling_marginal_relation_pipeline.LIME_subj_pipeline` | Subject images/label plus a prediction callback | Two-stage random/adaptive sampling, correlations, inclusion tables and figures. Constructor itself calls the callback on all images. Current runtime/scientific defects are listed below. |
| `marginal_relation.py` | Existing `pred_results.csv`, metadata/test09 split, JPEGs, NumPy/pandas/SciPy/plotting | Standalone `corr_pipeline()` writes separate `correlations.csv` and plots. Metadata is read at import; import is not a neutral dependency test. |
| `ridge_run.py` | `RidgeRun`, existing predictions, metadata/test09 split, JPEGs; `classifiers.py` eagerly imports Torch/PyG/timm/usflc_xai/glmnet | `run()` deduplicates rows and feature columns, restricts to positive subjects with >=20 metadata images, fits RidgeClassifierCV and bootstraps, writes coefficient CSV/plots. Main guard prevents fitting merely on import. |
| `elastic_net_run.py` | Existing predictions and same cohort/plotting requirements; `classifiers.ElasticNetClassifierWithStats` | Top-level code reads metadata and runs the cohort even on import. Active estimator is sklearn elastic-net **logistic** regression, not the GLMNET class. |
| `organize_output.py` | Metadata/test split, three stage output directories, original JPEGs | `OutputOrganizer.run()` copies CSVs, plots, and ranked/neutral images. Main guard; it does not perform inference. Missing/unselected subject directories can produce dictionary-key errors. |
| `classifiers.py` | NumPy, SciPy indirectly, pandas, sklearn, joblib, plotting/Pillow, dotenv, Torch/TorchVision, `usflc_xai`, `glmnet.LogitNet` | Contains GLMNET Elastic-Net, sklearn Elastic-Net, and Ridge implementations. Its eager imports mean even Ridge currently requires external graph/model packages and glmnet; do not silently remove them under an environment-only migration. |
| `clustering_sampler.py` | Legacy cluster JSONs plus model/data/image inputs | Obsolete cluster-stratified sampling; not called by the current primary entry point. Importing alone does not sample, but constructing `Sampler` loads models and cluster files. |
| `deprecated_and_archived/image_clustering.py` | Legacy prediction/correlation outputs, images/model source, clustering/embedding dependencies | `CustomClustering` and its pipeline generate cluster dictionaries/CSVs/plots. Not part of the current method. |
| `ridge_classification.ipynb`, `elastic_net_classification_test.ipynb`, `organize_output.ipynb` | Historical patient/run paths, stage files and external packages | Exploratory/historical code, multiple estimator variants, no stable command-line interface. Use a kernel from `.venv` and repository-root working directory only after auditing cells. |
| `deprecated_and_archived/*.ipynb`, `LIME_custom_pipeline.py`, `lime_custom_explainer/LIME_pipeline.py`, `custom_regressor.py` | Legacy LIME/clustering/single-image workflows; additional imports listed below | Retained as historical source. `cluster_based_sampling.ipynb` still imports `sampler`, which upstream renamed; archived imports also assume particular working directories. |
| New `scripts/check_environment.py` | Installed environment plus uploaded external source | Import-only smoke check and CUDA availability report. Does not instantiate scientific classes, load weights, infer or fit. Reports broken auxiliary `usflc_xai.utils` separately. |
| New `scripts/reproduce_patient.py` | One original prediction table, matching metadata/split/images, new output location | Validation only by default. `--execute` calls the existing `RidgeRun` with only the selected patient's original table exposed. It preserves original deduplication, CV/alpha grid and 10,000 bootstrap settings. It does not resample or regenerate GNN predictions. |

README's original statement that the generation entry imports `marginal_relation.py` is inaccurate: it imports `sampling_marginal_relation_pipeline.py`. Ridge is linear ridge classification; calling both conditional stages logistic regression is inaccurate.

## 4. Environment requirements and missing dependencies

The host's unqualified `python` is 3.6.8 and unsuitable: source uses dataclasses, TypedDict and evaluated `list[...]` annotations. A uv-managed Python **3.9.25** environment is created at `.venv`; uv **0.12.23** is available locally at ignored `.uv/bin/uv`. `pyproject.toml`, `.python-version`, and `uv.lock` are tracked. This is a compatible migration environment, **not a proven reconstruction of the original run**.

| External import / distribution | Migration pin | Reason / original export evidence |
| --- | --- | --- |
| `numpy` | 1.22.4 | Intentional compatibility adjustment: original export lists NumPy 1.20.0 and Matplotlib 3.8.0, which requires >=1.21. NumPy 1.22.4 also satisfies SciPy 1.7.3's <1.23 bound. A subsequent report identifies current original-server imports as NumPy 1.20.0 and Matplotlib 3.5.1; those differ from this migration combination. Historical-run versions are not proven. |
| `scipy` | 1.7.3 | Original conda export; correlations and numerical solvers. |
| `pandas` | 1.4.2 | Original conda export; preserves historical indexing behavior. |
| `sklearn` / `scikit-learn` | 1.0.2 | Original conda export; classifiers, CV, metrics. Do not install deprecated `sklearn==0.0` shim. |
| `skimage` / `scikit-image` | 0.19.2 | Original conda export; segmentation imports. |
| `matplotlib` | 3.8.0 | Original pip export; plots. Use `MPLBACKEND=Agg` on Slurm. |
| `seaborn` | 0.13.2 | Original pip export; heatmaps. |
| `PIL` / `pillow` | 9.0.1 | Original conda export; image loading. |
| `joblib` | 1.1.0 | Original conda export; bootstrap parallelism. |
| `tqdm` | 4.64.0 | Original conda export; progress. |
| `dotenv` / `python-dotenv` | 1.0.1 | Original pip export; local paths/config. |
| `torch` | 2.1.0 | Original pip export; checkpoint/model/graph computations. Linux wheel uses CUDA 12.1 runtime dependencies. |
| `torchvision` | 0.16.0 | Original pip export paired with Torch 2.1.0; transforms and DenseNet121 weights. |
| `torch_geometric` / `torch-geometric` | 2.3.1 | Original pip export; graph data and GAT layers. Optional compiled `torch_scatter`, `torch_sparse`, `torch_cluster`, `pyg_lib` were not directly imported in inspected project source or pinned in the old export. Verify any needed operator on an allocated GPU before a later inference run; do not install arbitrary incompatible wheels. |
| `timm` | 0.6.7 | Original pip export; imported eagerly even for DenseNet, alternate ViT encoder. |
| `lime` | 0.2.0.1 | Original pip export; eager legacy imports and archived LIME workflows. |
| `glmnet.LogitNet` / `glmnet` | 2.2.1 | Original conda export; compiled legacy dependency imported by all classifier variants. Requires GCC/gfortran and NumPy distutils. Build isolation is disabled specifically for glmnet, with pinned NumPy/setuptools/wheel preinstalled by setup script. |
| `setuptools`, `wheel` | 59.8.0, 0.45.1 | Migration build tooling for glmnet; intentionally different from exported setuptools 71.1.0. No estimator source patch. |
| `usflc_xai` | Unversioned uploaded source | Local, not a public PyPI dependency. All five Python files must be transferred separately for a fresh checkout; hashes are private in the artifact manifest. |
| `ipykernel`, `nbformat` | 6.29.5, 5.10.4 | Optional `notebooks` extra. Not required for the script pipeline; not part of default environment installation. Notebook 5.10.4 format tooling matches the old export. |

Standard-library imports (ast, collections, dataclasses, datetime, gc, glob, inspect, itertools, json, math, multiprocessing, os, random, shutil, sys, time, typing) need no separate package. All inspected Python/notebook imports are recorded in the private source inventory.

Additional **historical/obsolete** dependencies, intentionally excluded from the active environment:

- `umap` → `umap-learn==0.5.6`, `adjustText==1.2.0`, `communities==3.0.0`, `graph_based_clustering` → `graph-based-clustering==0.1.0`: imported by archived clustering code; old export has these pins.
- `openTSNE` and `tsnecuda`: eagerly imported by archived clustering code but **not present in `xai_env.yml`**. `tsnecuda` requires a separately compatible GPU/CUDA build. Missing from the new environment; not required for adaptive explanations.
- `hdbscan==0.8.37`, `pynndescent==0.5.13`: in the old export, associated with embedding/clustering; no direct HDBSCAN import in inspected project code. UMAP plotting may additionally need its plotting extras (datashader/bokeh/holoviews, etc.); not an active requirement.
- `sklearnex` → `scikit-learn-intelex`: imported and patches sklearn in archived `lime_custom_explainer/LIME_pipeline.py`; the old export has the conda distribution. This can change computation; do not enable it in the active environment.
- `regressors==0.0.3` and `torchsummary==1.5.1`: imported by historical notebooks, not the current script workflow; missing from the default migration environment. `imageio` was exported at 2.9.0 and is present transitively at 2.37.2 via scikit-image; its historical notebook behavior is unverified.
- `usflc_xai.utils` contains `import training.forward_backward_prop as forward_backward_prop`. No top-level `training` package is supplied. The uploaded `usflc_xai/training.py` exists, but the obsolete import cannot resolve as written. Auxiliary utility import remains broken; default explanation path imports only datasets/models.
- Old notebooks refer to local `LIME_custom_pipeline`, `lime_custom_explainer`, `image_clustering`, and `sampler` names; these are repository modules moved into `deprecated_and_archived/` or renamed, not PyPI packages to install.

`xai_env.yml` includes hundreds of unrelated packages, old conda build strings, mixed pip/conda duplicates, and a machine-specific `/home/tjyen/anaconda3/envs/xai` prefix. Do not mechanically convert/install the entire export. Its Python 3.9.12 and NumPy pins do not prove the environment actually used for the original outputs. CUDA/driver, BLAS, compiler, transitive versions and original random states are still needed for numerical reproducibility.

Setup: `bash scripts/setup_env.sh` requires network access, uv, GCC and gfortran. The script places uv cache/managed Python under `UV_CACHE_DIR` (defaults to `$SCRATCH/fatty-liver-uv-cache` or `/tmp/fatty-liver-uv-cache`) and installs locked packages into `.venv`. This session's cache/Python live under `/scratch/f008hzc/uv-cache` and `/scratch/f008hzc/uv-python`. Compiled wheels and the environment can consume several GB; none are committed. Jobs use the installed environment and do not install packages on compute nodes.

Slurm templates are provided for import/input preflight and an explicitly requested single-patient Ridge replay. Account/partition/QoS are site-specific `sbatch` flags, not assumed or embedded. CPU/memory/time values are initial resource requests, not measured run requirements. No Slurm job has been submitted. GPU inference/device compatibility and compute-node execution are not validated by an import check on the login host. The original generator uses CUDA device 1, which would be invalid inside a typical one-visible-GPU allocation; this remains documented rather than launching the full pipeline.

## 5. Hard-coded paths and configuration

`project_paths.py` resolves relative settings from the repository root, reads `.env`, and normalizes `CROP_IMAGE_DIR_PATH` before the uploaded package captures it. Active path changes only redirect filesystem inputs/outputs; fitting/sampling/model settings remain as upstream.

| Original location/value | Consumers | Config / migration destination |
| --- | --- | --- |
| `/home/liuusa_tw/data/cropped_images/` and `CROP_IMAGE_DIR_PATH` | Sampling, legacy sampler, stage plotters, uploaded loader | `CROP_IMAGE_DIR_PATH`, default `data/cropped_images/` |
| `meta_data/TWB_ABD_expand_modified_gasex_21072022.csv` and absolute project-prefixed form | All current stages; legacy sampler | `XAI_METADATA_PATH`, default `data/meta_data/...csv` |
| `fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/datasetXX/test_datasetXX.csv` | All current stages; legacy sampler | `XAI_TEST_SPLIT_PATH`; default under `data/` |
| `model_tl_twbabdXX/best_results.ckpt` and absolute project-prefixed form | Sampling; legacy sampler | `XAI_CHECKPOINT_PATH`; default under `checkpoints/` |
| `/home/liuusa_tw/twbabd_image_xai_20062024/custom_lime_results` | New sampled/stage output roots | `XAI_OUTPUT_ROOT`, default `outputs/` |
| Same root, `07-12-2024-03-57-58/` | Ridge/Elastic-Net/marginal input | `XAI_PREDICTION_ROOT`, default `outputs/original/07-12-2024-03-57-58` |
| Same root, `clustering-07-29-2024-06-05-39/` | Legacy sampler only | `XAI_CLUSTERING_ROOT`, default under `outputs/original/` |
| Same root, `ridge-08-06-2024-01-24-40` | Original Ridge override/comment; organizer reference | `XAI_RIDGE_OUTPUT`; Ridge otherwise creates a timestamped destination |
| Same root, `elastic-net-old-dataset-08-01-2024-06-59-39` | Elastic runner and organizer | `XAI_ELASTIC_OUTPUT`; historical default retained under `outputs/original/` |
| Same root, `correlation-old-dataset-08-01-2024-03-37-02` | Organizer | `XAI_CORRELATION_OUTPUT`; marginal stage otherwise creates a timestamped destination |
| Same root, `organized_output` | Organizer | `XAI_ORGANIZED_OUTPUT`, default `outputs/organized_output` |

Legacy/notebook literals are intentionally retained and must become explicit config/CLI inputs before those workflows are reused. Their exhaustive locations are in `source_inventory.json`:

- `deprecated_and_archived/image_clustering.py`: metadata/split/checkpoint paths, July prediction/correlation result root, image root, timestamped clustering destination.
- `deprecated_and_archived/lime_custom_explainer/LIME_pipeline.py`: image root; relative metadata/split/checkpoint names; output locations and working-directory CSV writes.
- `deprecated_and_archived/LIME_custom_pipeline.py`: relative metadata/split/checkpoint paths; old custom result root and image env usage.
- `ridge_classification.ipynb`: original prediction path for a fixed example patient; patient-specific Ridge destination; `temp_FOLDCER`; `corr_results.csv` lookup; metadata/split/checkpoint paths; raw string concatenation of image root.
- `elastic_net_classification_test.ipynb`: fixed example patient images; historical prediction and timestamped Elastic-Net result roots; relative metadata/split/checkpoint paths.
- `organize_output.ipynb`: old prediction and Ridge roots, `TEMP-elastic-net-08-02-2024-00-32-01`, patient-specific `*-RIDGE-RESULTS`, August stage directories, `test_organized_output`, timestamped organized destinations.
- `deprecated_and_archived/LIME_test.ipynb`: legacy raw image tree `TWBABD_US_images_5_instances_13062024`, fixed single-image paths, `lime_test_results/`, relative metadata/splits/checkpoint references.
- `deprecated_and_archived/LIME_custom_test.ipynb`: metadata/splits, custom result root, fixed patient IDs; one checkpoint literal is misspelled `best_r\`esults.ckpt`.
- `deprecated_and_archived/LIME_pipeline_example.ipynb`: old `lime_test_results/` and cropped image root.
- `deprecated_and_archived/image_clustering_custom.ipynb`: July19 clustering CSV root, custom result root, `temp_test` clustering destinations.
- `usflc_xai/datasets.py`: train/valid/test CSV construction and relative encoded feature `.pt` templates in `dataset_container`; concatenated JPEG path needs a trailing slash.
- `usflc_xai/training.py`: caller-relative epoch/best checkpoint paths and result-log output construction.
- `xai_env.yml`: obsolete original conda installation prefix.

Other fixed run choices to expose later, with scientific review: test split 09, example patient IDs, CUDA device, sample counts/bounds/target proportion, correlation threshold 0.95, model architecture/dimensions/layers, bootstrap counts, CV grids/folds and significance thresholds. **They were not changed in this setup.** Relative filesystem defaults are resolved from root in active stages, while historical notebooks still require deliberate working-directory handling.

## 6. Current code versus methodology and remaining defects

These findings are based on source inspection. They are blockers/risks to validate against the original used source, not permission to redesign the method.

1. **Obsolete cluster sampling:** upstream explicitly names the old implementation `clustering_sampler.py`, but archived `cluster_based_sampling.ipynb` still imports `sampler`. The old sampler selects subjects from cluster JSONs rather than the current adaptive workflow. Missing cluster JSONs and `openTSNE`/`tsnecuda` must not be used to justify generating clusters for the current method. It also performs a suspicious file-existence comprehension using `v` before its local comprehension binding, and loads clustering results twice.
2. **Image-to-column mapping defect in current sampling:** `LIME_subj_pipeline.__post_init__` sets both maps using `dict(enumerate(self.img_list))`. `img_to_indx` is later indexed by image ID strings in matrix generation and plotting, so this appears to cause `KeyError`. The archived implementation constructs the reverse map. Preserve/obtain the original run's source to determine which version generated its results; no mapping fix was made here.
3. **Scalar-label type mismatch:** the all-subject pipeline passes the pandas/graph-derived `y` into a constructor that requires built-in `int`. The uploaded loader passes through a metadata scalar; binary relabeling only assigns an integer when `y > 0`. NumPy integer zero labels can therefore fail the type check, while positive labels are typically converted to built-in 1. Verify before all-cohort use.
4. **Adaptive count bug:** when negative quota is met first, the positive top-up requests `target_negative_count - current_negative_count` instead of the remaining positive count. It can request zero or the wrong count, yielding fewer than `n_samples`.
5. **Class balance is a target, not a guarantee:** pools distinguish predictions that agree/disagree with ground truth. For negative-ground-truth subjects those pool meanings differ from positive/negative disease class. Stage two uses 0.85/0.15 image-pool proportions and does not guarantee the resulting graph labels meet target proportions. The one-pool fallback and clipping can change final counts/subset sizes; duplicates are not reliably excluded despite unique-sample counting language.
6. **Correlations differ across paths:** sampling-module Pearson standard error uses the number of sample rows, while `marginal_relation.corr_pipeline` uses the number of retained image columns (`num_img-2`). Bounds use approximately one SE; the `conf_level` parameter is not used to apply a confidence multiplier. These are materially different uncertainty definitions. The sampling path also sets a constant-column `corr_CI` to scalar zero then later tries to unpack it as an interval; it can mark a constant column with `p=0`. Do not silently correct these while comparing old results.
7. **Different cohorts:** generation iterates all test rows. Downstream marginal/Ridge/Elastic-Net stages select ground-truth-positive subjects with >=20 metadata images and skip one-class prediction tables. This distinction must match the methodology/cohort definition and original run logs.
8. **Duplicate rows/columns affect interpretation:** downstream stages drop duplicate prediction rows and identical image inclusion columns. That changes weighting and may remove separate image features. Bootstrap/reference comparisons must preserve the original processing; do not synthesize matrices or silently relabel dropped features.
9. **Bootstrap setting mismatch:** `RidgeRun.n_bootstrap_iterations` advertises 50,000 but `run()` hard-codes `n_bootstrap=10000`. Elastic-Net runner uses 10,000. Class defaults are 250. Replay preserves the actual runner's 10,000. The Ridge bootstrap uses its own default fold count rather than clearly forwarding every fit-stage setting.
10. **CV edge cases:** Ridge and sklearn Elastic-Net reduce `n_splits` by one when below 10, which can produce invalid folds for a small minority class. One-class checks alone do not establish valid CV/bootstrap samples. Ridge accesses `RidgeClassifierCV.best_score_`; inspection of installed sklearn 1.0.2 confirms this attribute is assigned for explicit CV. Numerical fitting itself was not tested.
11. **Estimator terminology:** RidgeClassifierCV is not logistic regression. sklearn Elastic-Net uses `LogisticRegressionCV(..., penalty='elasticnet', solver='saga')`; the separate GLMNET implementation exists but is not the active Elastic-Net runner. Figures use SE-based significance while bootstrap-percentile significance is also reported; the intended rule must be confirmed. Typoed historical `Boostrap...` column names are preserved because the organizer consumes original schemas.
12. **Unrecorded stochastic state:** sampling uses global NumPy randomness; legacy sampler also uses Python `random`; bootstrap draws use global NumPy state and joblib workers. Estimator `random_state=42/0` does not seed all those draws. Original sample tables can reproduce fitted inputs, but exact bootstrap intervals cannot be promised without original RNG states/backend/versions. The migration intentionally does not invent a seed or alter reproducibility semantics.
13. **Device/weights provenance:** original generator defaults to CUDA index 1; `torch.load` lacks `map_location`, which can fail loading GPU-saved checkpoints on CPU. Image encoder weights may download implicitly when instantiating the model. DenseNet grayscale/resize/normalize, correlation adjacency >0.95, graph encoder `SETNET_GAT`, 1024 dimensions/one layer/two classes must match the original checkpoint/run.
14. **Import/run side effects and resume behavior:** generator and Elastic-Net runner execute at top level; marginal reads clinical CSVs on import. Output directories or existing subject folders are sometimes used as completion markers without verifying successful files, allowing incomplete results to be skipped. Plotting uses `plt.show()` and one marginal function assigns `plt.xlabel` a string; Agg avoids display requirements but does not fix code. Legacy organizer copies JPEG bytes to some `.png` filenames rather than re-encoding. New replay requires a fresh output destination and checks that the existing runner actually produced its coefficient CSV.
15. **External-source provenance:** uploaded `usflc_xai` is not versioned/packaged and its auxiliary utility has a broken `training` import. Feature cache producer scripts, original model training configuration, and original preprocessing history are not supplied. Keep these separate from the narrowly scoped default explanation replay.

## 7. Minimal path to reproduce one patient's existing explanation

**Recommended first target:** a deterministic, input-complete positive test09 patient with >=20 images, both classes in saved predictions (prefer >=10 unique rows per class after deduplication), and a matching original Ridge coefficient CSV. Among qualifying cases choose the smallest image count, then stable MI_ID order. This minimizes work without selecting based on explanation strength. The new original-server bundle supplies one patient selected by this rule, and destination input validation passes. Its identity is recorded only in private reports/bundle paths, not this public audit. It has 1,000 saved July prediction rows (304/696 classes); after the original row deduplication there are 589 rows (54/535 classes).

1. Use [the original-server handoff prompt](docs/ORIGINAL_SERVER_HANDOFF.md) to identify the original prediction run and chosen patient. Obtain that patient's prediction/design tables, **all** metadata-listed JPEGs, corresponding reference CSV/plots, exact source used, original environment/version dump, and run settings/seeds if recorded. Match hashes and preserve original names. For a downstream Ridge replay, the graph checkpoint, DenseNet weights and cluster results are not computational inputs, although current eager imports still require the graph software stack.
2. Put images in `data/cropped_images/`, prediction folder under the verified `outputs/original/<run>/<MI_ID>/`, references under their original stage directories, and provenance under `outputs/original/provenance/`. Configure `.env`. Never overwrite references with replay outputs. A missing design matrix can be derived from verified `pred_results.csv` by removing labels, after preserving the supplied table and documenting the derivation.
3. Run `bash scripts/setup_env.sh`, then `.venv/bin/python scripts/check_environment.py`. Resolve active import failures first. Compare the migration environment against the actual original runtime, especially NumPy, sklearn API, BLAS/CUDA and compiled glmnet. Current import success alone does not resolve source/methodology blockers. The new xai capture reports sklearn 1.5.1/joblib 1.4.2, versus migration sklearn 1.0.2/joblib 1.1.0. Preserve the locked migration baseline pending a deliberate original-runtime reconciliation; do not silently upgrade the fitter environment.
4. Inspect the private missing-image/candidate reports. The all-cohort strict audit may still fail after uploading only one complete patient; the single-patient validator checks that target independently. Validate only:

   ```bash
   .venv/bin/python scripts/reproduce_patient.py \
     --patient <MI_ID> \
     --predictions outputs/original/<verified-run>/<MI_ID>/pred_results.csv \
     --output outputs/replay/<new-run>/ridge
   ```

   Replace bracketed placeholders; do not execute the example literally. No inference or fit happens without `--execute`.
5. For a controlled comparison in the documented migration environment, explicitly run the single-patient existing Ridge stage by adding `--execute`, preferably with `sbatch slurm/reproduce_patient.sbatch` from the repository root. Record the environment differences and treat this as a comparison rather than proven exact historical reproduction; no method rewrite is a prerequisite for this saved-input attempt. The wrapper exposes only this patient's saved predictions to `RidgeRun`; it does not regenerate samples or change the fitter. Current upstream defects may still make this fail; document rather than substitute another estimator.
6. Compare image/column identities, duplicate handling, selected alpha, coefficient estimates, interval/significance columns, and plots to original reference outputs. Use tolerances justified by the original versions/settings; do not claim exact bootstrap reproduction when original RNG state is missing. Report deviations and unknowns. Only then expand to the existing Elastic-Net/correlation stages or consider an end-to-end GNN regeneration.

For end-to-end regeneration, additionally restore exact DenseNet weights, verify graph checkpoint loading on an allocated GPU, confirm device index and original source implementation, and resolve the mapping/label/sampling/uncertainty discrepancies through a subsequent reviewed change. **Do not run the full generator or obsolete clustering pipeline as a substitute for recovering saved samples.**

## Validation and publication boundary

Only audit/setup checks, saved-table/reference arithmetic consistency checks, image decoding/hash validation, Python compilation, shell syntax validation, dependency resolution/installation, and import smoke checks are in scope. Input audit correctly reports missing images. No Slurm job or experiment was launched. Setup/environment and publication results are recorded below once validated.

`.gitignore` protects `data/`, `checkpoints/`, `outputs/`, `logs/`, `temp/`, `.venv`, `.uv/`, `.env` variants (except `.env.example`), model binaries, private key extensions and local agent/credential directories. Uploaded `usflc_xai/` remains ignored. Notebook source is retained while execution outputs/counts and execution metadata are cleared; private originals are preserved. Historical GitHub commits already contain old notebook outputs; this migration clears the branch's current tree and adds no new patient artifacts, without rewriting shared history.

Validated migration results: all 17 direct third-party imports and the active local datasets/models/classifier/sampling/Ridge/organizer module imports pass; `uv pip check` reports all 61 installed distributions compatible. Torch reports CUDA build 12.1, with no usable GPU on this login host. The known auxiliary `usflc_xai.utils` import fails with `No module named training` and remains unchanged. The test09 strict audit exits 1 for the documented missing images. Python compilation, shell syntax, `git diff --check`, notebook cell-source preservation, and synthetic replay validation/no-fit/no-overwrite checks pass. Installed versions are saved privately in `outputs/reproducibility/migration.freeze.txt`. No numerical fitting or GPU execution was used for validation.

## Second-upload verification and remaining boundary

The additional uploads have been organized without overwriting existing inputs or copying original-server source over the active scientific code:

- `outputs/original/11-28-2024-03-27-14/`: 8,010 files / 648,127,093 bytes across 890 test-patient folders. Kept separate from July predictions and August reference fits.
- `outputs/original/clustering-07-29-2024-06-05-39/`: 3,778 legacy files / 1,101,675,564 bytes, including 135 cluster JSONs and figures. Archived only; no obsolete clustering packages were installed and no clustering was run.
- `outputs/original/bundles/<private-bundle>/`: preserved 144-file / 109,206,268-byte bundle with 20 images, reference stages, source snapshot, models, logs, environment captures and original manifest. Relevant images, stage outputs and DenseNet weights were also placed at their configured canonical locations. All destinations are ignored by Git.

Bundle manifest verification passes for **142 of 143 listed records**, including all scientific payloads. The sole mismatch is `provenance/file_sizes.tsv`: manifest declares zero bytes, whereas the supplied listing is 9,804 bytes; its hash also differs. Original manifest/listing are preserved, with the mismatch recorded in `outputs/reproducibility/new_uploads/bundle_manifest_verification.json`. This administrative-file discrepancy is not treated as evidence of a failed image/model/prediction transfer. The already-uploaded metadata, graph checkpoint, three split09 CSVs and all five external `usflc_xai` sources match the newly supplied source-server copies/hashes.

The selected July patient's `pred_results.csv` and `design_matrix.csv` agree, including column order. All 20 JPEGs hash-match and decode; exact DenseNet SHA-256 matches the source report. Recalculation solely to inspect existing artifacts confirms the July sampling correlation coefficients to max absolute difference <1e-14 and its SEs <2e-16; August standalone correlations from deduplicated July rows match within <4e-15 and their SEs <2e-16. This supports the correlation-stage association and confirms the differing SE denominators documented above; it does **not** prove Ridge/Elastic-Net fit-stage provenance or regenerate an explanation.

The selected patient's November prediction table is different from its July table. Do not use November predictions to reproduce August coefficients merely because patient IDs match. `scripts/audit_saved_run.py` verifies binary tables, metadata/split membership, design-matrix equality, class support, conflicting labels for identical masks and local image coverage without inference/fitting. Run it separately for each sampling run; detailed reports stay in ignored storage. The November audit passes for all 890 prediction/design table pairs (878,013 prediction rows): 806 patients have both prediction classes, 84 have one class, and no identical inclusion mask has conflicting predictions within a patient. All tables have binary values and match metadata/test membership. This is intermediate-table integrity, not full image/model reproducibility.

New provenance narrows—but does not eliminate—the runtime uncertainty. The original server was on `3a5a718`, with an uncommitted Elastic-Net runner refactor that also excludes two specific subjects. Its classifiers and other four top-level scientific source snapshots match upstream; the active migration only differs in filesystem defaults. The uncommitted source is preserved rather than applied. The xai environment capture reports scikit-learn **1.5.1** and joblib **1.4.2**, while the migration environment has **1.0.2** and **1.1.0**. Base-environment packages differ again. NumPy metadata remains contradictory (exported pip 1.20.0, conda explicit 1.21.5). At that stage actual imported NumPy/BLAS versions were not established; the subsequent follow-up report below addresses current imports, while the environment used to create the historical outputs remains unproven. Original seeds/RNG states remain unknown.

**No refit, resampling, model loading/inference or Slurm submission was performed after this upload.** The single-patient validator passes; full test09 strict coverage still fails on the remaining 14,940 missing references. A narrowly scoped saved-July Ridge comparison can proceed in the documented migration environment, with new outputs kept separate from references and without claiming historical runtime equivalence. An independently configured environment matching reported current original-server imports would be a separate comparison, not a proven reconstruction of the historical environment.

## Original-server follow-up report: current runtime established

A user-supplied report labeled **2026-10-08** records read-only queries using the original xai environment's Python. Its original text and a SHA-256 receipt are preserved under ignored `outputs/original/provenance/followup_reports/`. Only the report has been received here; the new follow-up bundle and its raw runtime capture/corrected manifests have not been transferred or independently verified on this server.

| Component | Reported current original-server import | Verified destination installation |
| --- | --- | --- |
| Python | 3.9.12 | 3.9.25 |
| NumPy | **1.20.0** | **1.22.4** |
| SciPy | 1.7.3 | 1.7.3 |
| scikit-learn | **1.5.1** | **1.0.2** |
| joblib | **1.4.2** | **1.1.0** |
| pandas | 1.4.2 | 1.4.2 |
| matplotlib | **3.5.1** | **3.8.0** |
| Torch | 2.1.0+cu121 | Distribution 2.1.0; CUDA build 12.1 previously verified |

The report resolves the present imported NumPy/matplotlib ambiguity: NumPy 1.20.0 and matplotlib 3.5.1 are imported on the original server, despite contradictory package/export records. It reports NumPy's OpenBLAS libraries under `/usr/local/lib`; the raw `numpy.show_config()` file remains in the untransferred follow-up bundle. Pip/conda/export metadata must not replace imported-version evidence. The destination lockfile and existing local environment were left unchanged by this report ingestion.

The original server currently reports two RTX 3080 GPUs, driver 580.178.04, driver-reported CUDA 13.0, and nvcc 11.5. These describe different parts of its software/hardware stack; the Torch runtime is cu121. They do not establish the hardware/runtime used for the historical outputs or require installing CUDA 13.0 here for the saved-prediction Ridge comparison.

Still unknown after the follow-up investigation:

- The exact source/environment used to produce the historical July sampling tables and August fits. Current sampling source retains the reversed image-ID mapping; archived source differs.
- A launch command/config proving July predictions were the inputs to the supplied Ridge/Elastic-Net fits. Our correlation arithmetic checks support the correlation-stage association only.
- Original random seeds/RNG states, full invocation settings, and historically used bootstrap/CV settings. Source defaults and preserved logs are evidence, not proof of the invocation.

The report says separate corrected payload/file-size manifests were created, excluding self-referential generated listings. Those corrected files are not yet available here, so the original local manifest discrepancy remains recorded. No scientific payload corruption was found in the prior verification.

The original-server agent could not prepare missing-image transfers because it did not receive the destination's refreshed list. For full-cohort work, privately transfer **`outputs/reproducibility/missing_test_images.csv`** to that server, then obtain the exact corresponding JPEGs. The latest local audit still reports **14,940 missing test09 image references**; the selected patient's inputs remain complete.

**Decision:** this server is ready for development and an explicitly invoked, controlled single-patient saved-input comparison. Additional original-server investigation is not a prerequisite for that comparison. Numerical differences must be measured and attributed cautiously; exact historical bootstrap reproduction cannot be claimed from the evidence available. No inference, fitting, environment changes or scientific source changes were performed while ingesting this report. Existing local edits outside the audit were preserved.
