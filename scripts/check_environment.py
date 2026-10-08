#!/usr/bin/env python3
"""Import checks; --numerics also fits tiny synthetic Ridge/glmnet models. No patient inference."""
import importlib
import importlib.metadata
import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--numerics", action="store_true", help="Also fit tiny synthetic Ridge and compiled glmnet models")
args = parser.parse_args()
print("Python executable:", sys.executable, "base prefix:", sys.base_prefix)
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
if args.numerics and not failures:
    try:
        import numpy as np
        from glmnet import LogitNet
        from sklearn.linear_model import Ridge
        rng = np.random.RandomState(42)
        x = rng.normal(size=(80, 5))
        y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
        ridge = Ridge(alpha=1.0).fit(x, y)
        assert np.isfinite(ridge.predict(x)).all()
        glmnet = LogitNet(alpha=0.5, n_lambda=20, n_splits=3, n_jobs=1, random_state=42)
        glmnet.fit(x, y)
        probabilities = glmnet.predict_proba(x)
        assert probabilities.shape == (80, 2) and np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1)
        print("Synthetic Ridge fit and compiled glmnet fit/predict: OK")
    except Exception as exc:
        failures.append({"module": "numerical checks", "error": str(exc)})
        print("Numerical check FAILED:", exc)
print(json.dumps({"active_import_failures": failures}))
raise SystemExit(bool(failures))
