#!/usr/bin/env python3
"""Import smoke check; no checkpoint loading, downloads, inference, or fitting."""
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
modules = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "sklearn": "scikit-learn",
    "skimage": "scikit-image", "matplotlib": "matplotlib", "seaborn": "seaborn",
    "PIL": "pillow", "joblib": "joblib", "tqdm": "tqdm", "dotenv": "python-dotenv",
    "torch": "torch", "torchvision": "torchvision", "torch_geometric": "torch-geometric",
    "timm": "timm", "lime": "lime", "glmnet": "glmnet",
    "usflc_xai.datasets": None, "usflc_xai.models": None, "classifiers": None,
    "sampling_marginal_relation_pipeline": None, "clustering_sampler": None,
    "ridge_run": None, "organize_output": None,
}
failures = []
for module, distribution in modules.items():
    try:
        importlib.import_module(module)
        print(module, importlib.metadata.version(distribution) if distribution else "local: import OK")
    except Exception as exc:
        failures.append({"module": module, "error": str(exc)})
        print(module, "FAILED:", exc)
if "torch" in sys.modules:
    import torch
    print("torch CUDA build:", torch.version.cuda, "CUDA usable here:", torch.cuda.is_available())
# This separate uploaded utility has a known obsolete absolute import.
try:
    importlib.import_module("usflc_xai.utils")
except Exception as exc:
    print("Known auxiliary blocker: usflc_xai.utils:", exc)
print(json.dumps({"active_import_failures": failures}))
raise SystemExit(bool(failures))
