"""Filesystem configuration only; no model, sampling, or statistical settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def configured_path(name, default):
    path = Path(os.getenv(name, str(default))).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


def metadata_path():
    return configured_path("XAI_METADATA_PATH", "data/meta_data/TWB_ABD_expand_modified_gasex_21072022.csv")


def split_path(test_data_id="09"):
    return configured_path("XAI_TEST_SPLIT_PATH", f"data/fattyliver_2_class_certained_0_123_4_20_40_dataset_lists/dataset{test_data_id}/test_dataset{test_data_id}.csv")


def checkpoint_path(test_data_id="09"):
    return configured_path("XAI_CHECKPOINT_PATH", f"checkpoints/model_tl_twbabd{test_data_id}/best_results.ckpt")


def image_dir():
    # Uploaded datasets.single_data_loader concatenates strings, so keep the slash.
    return configured_path("CROP_IMAGE_DIR_PATH", "data/cropped_images") + os.sep


def output_root():
    return configured_path("XAI_OUTPUT_ROOT", "outputs")


def prediction_root():
    return configured_path("XAI_PREDICTION_ROOT", "outputs/original/07-12-2024-03-57-58")

# Normalize before usflc_xai captures this variable at import time.
os.environ["CROP_IMAGE_DIR_PATH"] = image_dir()
# Normalize the encoder cache too, including when called outside the repo root.
os.environ["TORCH_HOME"] = configured_path("TORCH_HOME", "checkpoints/torch")
