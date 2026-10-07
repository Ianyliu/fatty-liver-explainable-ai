#!/usr/bin/env python3
"""Reject private artifact paths, oversized files, saved outputs, and common secrets."""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1024 * 1024
PRIVATE_ROOTS = {"data", "checkpoints", "outputs", "logs", "temp", "usflc_xai", ".venv", ".uv", ".aws", ".agents", ".codex", "sources", "meta_data", "custom_lime_results", "lime_test_results"}
PRIVATE_SUFFIXES = {".ckpt", ".pt", ".pth", ".pkl", ".pickle", ".npy", ".npz", ".csv", ".parquet", ".h5", ".hdf5", ".sif", ".simg", ".pem", ".key", ".p12", ".pfx"}
PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "AWS access key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "Slack token": re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}"),
    "OpenAI token": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{40,}"),
    "credential in URL": re.compile(rb"https?://[^\s/]+:[^\s/]+@"),
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def inspect_blob(name, data, failures):
    path = Path(name)
    if any(part in PRIVATE_ROOTS for part in path.parts) or path.suffix.lower() in PRIVATE_SUFFIXES:
        failures.append(f"Private artifact path: {name}")
    if path.name.startswith('.env') and path.name != '.env.example':
        failures.append(f"Environment file: {name}")
    if len(data) > MAX_BYTES:
        failures.append(f"File exceeds 1 MiB: {name} ({len(data)} bytes)")
    for label, pattern in PATTERNS.items():
        if pattern.search(data):
            failures.append(f"Possible {label}: {name} (value suppressed)")
    if path.suffix == '.ipynb':
        notebook = json.loads(data)
        if any(cell.get('outputs') or cell.get('execution_count') is not None for cell in notebook['cells']):
            failures.append(f"Saved notebook execution data: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='origin/master', help='Already-public base to exclude from outgoing history')
    args = parser.parse_args()
    failures = []
    files = [name for name in git('ls-files', '-z').decode().split('\0') if name]
    sizes = []
    for name in files:
        data = git('show', ':' + name)
        inspect_blob(name, data, failures)
        sizes.append((len(data), name))
    # Inspect every newly reachable blob, not just the final tree. This prevents
    # committing private files and subsequently deleting them before push.
    introduced = 0
    for line in git('rev-list', '--objects', args.base + '..HEAD').decode().splitlines():
        parts = line.split(' ', 1)
        sha = parts[0]
        if git('cat-file', '-t', sha).strip() != b'blob':
            continue
        introduced += 1
        inspect_blob(parts[1] if len(parts) > 1 else sha, git('cat-file', 'blob', sha), failures)
    print(f"Inspected {len(files)} indexed files and {introduced} outgoing history blobs.")
    print('Largest indexed files:', [(name, size) for size, name in sorted(sizes, reverse=True)[:5]])
    for failure in sorted(set(failures)):
        print('FAIL:', failure)
    if failures:
        raise SystemExit(1)
    print('Publication checks passed. Pattern scans supplement manual diff review; they are not a guarantee against every secret format.')


if __name__ == '__main__':
    main()
