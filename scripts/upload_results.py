#!/usr/bin/env python3
"""上传结果到 HuggingFace 和华为云 OBS"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("/data/pipeline/results")


def upload_hf(model_path, hf_token, hf_repo):
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_repo, exist_ok=True)
    api.upload_folder(folder_path=model_path, repo_id=hf_repo, repo_type="model")
    print(f"Model uploaded to HF: {hf_repo}")

    report = RESULTS_DIR / "eval_report.json"
    if report.exists():
        api.upload_file(path_or_fileobj=str(report), path_in_repo="eval_report.json", repo_id=hf_repo, repo_type="model")

    video = RESULTS_DIR / "eval_video.mp4"
    if video.exists():
        api.upload_file(path_or_fileobj=str(video), path_in_repo="eval_video.mp4", repo_id=hf_repo, repo_type="model")


def upload_obs(obs_ak, obs_sk, obs_bucket, obs_endpoint):
    from obs import ObsClient
    obs = ObsClient(access_key_id=obs_ak, secret_access_key=obs_sk, server=obs_endpoint)

    for f in RESULTS_DIR.glob("*"):
        if f.is_file():
            key = f"so101-pipeline/{f.name}"
            obs.putFile(obs_bucket, key, str(f))
            print(f"Uploaded to OBS: {obs_bucket}/{key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/pipeline/training/checkpoints/last/pretrained_model")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--hf-repo", default=os.environ.get("HF_MODEL_REPO", "xieyucheng123/so101-act-sim"))
    parser.add_argument("--obs-ak", default=os.environ.get("OBS_AK"))
    parser.add_argument("--obs-sk", default=os.environ.get("OBS_SK"))
    parser.add_argument("--obs-bucket", default=os.environ.get("OBS_BUCKET", "robotwin-assets"))
    parser.add_argument("--obs-endpoint", default=os.environ.get("OBS_ENDPOINT", "obs.cn-north-4.myhuaweicloud.com"))
    parser.add_argument("--skip-obs", action="store_true")
    args = parser.parse_args()

    try:
        upload_hf(args.model_path, args.hf_token, args.hf_repo)
    except Exception as e:
        print(f"Warning: HF upload failed ({e})")

    if not args.skip_obs and args.obs_ak:
        try:
            upload_obs(args.obs_ak, args.obs_sk, args.obs_bucket, args.obs_endpoint)
        except Exception as e:
            print(f"Warning: OBS upload failed ({e})")

    print("\n=== Upload Complete ===")


if __name__ == "__main__":
    main()
