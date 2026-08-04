#!/usr/bin/env python3
"""准备 Kaggle notebook - 注入参数并更新 kernel metadata"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", default="act")
    parser.add_argument("--steps", default="20000")
    parser.add_argument("--hf-token", required=True)
    args = parser.parse_args()

    notebook_path = Path("kaggle-notebook/train_act.ipynb")
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}")
        return

    import nbformat
    nb = nbformat.read(str(notebook_path), as_version=4)

    params_cell = nbformat.v4.new_code_cell(
        source=f"""
# === Pipeline Parameters (auto-injected) ===
DATASET_REPO = "{args.dataset}"
MODEL_REPO = "{args.model}"
POLICY_TYPE = "{args.policy}"
TRAINING_STEPS = {args.steps}
# HF_TOKEN from Kaggle Secret (not embedded in source)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set in environment")
HF_ENDPOINT = "https://hf-mirror.com"
# === End Parameters ===
""".strip()
    )
    nb.cells.insert(0, params_cell)

    nbformat.write(nb, str(notebook_path))

    metadata_path = Path("kaggle-notebook/kernel-metadata.json")
    meta = {
        "id": "xieyucheng/so101-train-act",
        "title": "so101-train-act",
        "code_file": "train_act.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Notebook prepared with dataset={args.dataset}, model={args.model}, policy={args.policy}, steps={args.steps}")


if __name__ == "__main__":
    main()
