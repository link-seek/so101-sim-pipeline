#!/usr/bin/env python3
"""Push Kaggle notebook via API with Bearer token (KGAT_ format support)"""

import argparse
import json
import requests
import sys
from pathlib import Path


def push_kaggle_notebook(notebook_path, metadata, kaggle_token):
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
        if 'source' in cell and isinstance(cell['source'], list):
            cell['source'] = ''.join(cell['source'])

    script_body = json.dumps(nb)

    slug = metadata.get('id', '')
    if '/' in slug:
        kernel_slug = slug
    else:
        kernel_slug = slug

    payload = {
        "slug": kernel_slug,
        "newTitle": metadata.get('title', kernel_slug.split('/')[-1]),
        "text": script_body,
        "language": metadata.get('language', 'python'),
        "kernelType": metadata.get('kernel_type', 'notebook'),
        "isPrivate": metadata.get('is_private', True),
        "enableGpu": metadata.get('enable_gpu', True),
        "enableTpu": metadata.get('enable_tpu', False),
        "enableInternet": metadata.get('enable_internet', True),
        "datasetDataSources": metadata.get('dataset_sources', []),
        "competitionDataSources": metadata.get('competition_sources', []),
        "kernelDataSources": metadata.get('kernel_sources', []),
        "modelDataSources": metadata.get('model_sources', []),
        "categoryIds": metadata.get('keywords', []),
    }

    headers = {
        "Authorization": f"Bearer {kaggle_token}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        "https://www.kaggle.com/api/v1/kernels/push",
        headers=headers,
        json=payload,
        timeout=60,
    )

    if resp.status_code == 200:
        data = resp.json()
        print(f"Pushed: {data.get('url', 'unknown')}")
        print(f"Version: {data.get('versionNumber', '?')}")
        return True
    else:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        return False


def check_kaggle_status(slug, kaggle_token):
    headers = {"Authorization": f"Bearer {kaggle_token}"}
    resp = requests.get(
        f"https://www.kaggle.com/api/v1/kernels/status?user_name={slug.split('/')[0]}&slug={slug.split('/')[1]}",
        headers=headers,
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()
    return None


def main():
    parser = argparse.ArgumentParser(description="Push notebook to Kaggle")
    parser.add_argument("--notebook", default="kaggle-notebook/train_act.ipynb")
    parser.add_argument("--token", required=True)
    parser.add_argument("--slug", default="xieyucheng123/so101-train-act")
    parser.add_argument("--title", default="so101-train-act")
    args = parser.parse_args()

    metadata = {
        "id": args.slug,
        "title": args.title,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
    }

    ok = push_kaggle_notebook(args.notebook, metadata, args.token)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
