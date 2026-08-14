#!/usr/bin/env python3
"""Pre-download lerobot dataset without requiring version tag.

Downloads dataset files via huggingface_hub and places them in the
lerobot cache directory structure.
"""

import os
import shutil
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

def main():
    from huggingface_hub import snapshot_download

    repo_id = sys.argv[1] if len(sys.argv) > 1 else "ataghof/so101nexus-cube500-binary"
    dest_root = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Downloading {repo_id} via huggingface_hub...")

    snapshot_path = snapshot_download(
        repo_id,
        repo_type="dataset",
        endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    )
    print(f"Snapshot downloaded to: {snapshot_path}")

    # Determine lerobot cache path
    if dest_root:
        lerobot_path = os.path.join(dest_root, repo_id)
    else:
        cache_home = os.path.expanduser("~/.cache/huggingface/lerobot")
        lerobot_path = os.path.join(cache_home, repo_id)

    os.makedirs(os.path.dirname(lerobot_path), exist_ok=True)

    # Remove old symlink/dir if exists
    if os.path.islink(lerobot_path) or os.path.exists(lerobot_path):
        if os.path.islink(lerobot_path):
            os.unlink(lerobot_path)
        elif os.path.isdir(lerobot_path):
            shutil.rmtree(lerobot_path)

    # Create symlink
    os.symlink(snapshot_path, lerobot_path)
    print(f"Symlinked: {lerobot_path} -> {snapshot_path}")

    # Verify
    info_path = os.path.join(lerobot_path, "meta", "info.json")
    if os.path.exists(info_path):
        print(f"Verified: {info_path} exists")
    else:
        print(f"ERROR: {info_path} not found!")
        sys.exit(1)

    # Print dataset info
    import json
    with open(info_path) as f:
        info = json.load(f)
    print(f"Dataset: {repo_id}")
    print(f"  episodes: {info.get('total_episodes')}")
    print(f"  frames: {info.get('total_frames')}")
    print(f"  fps: {info.get('fps')}")
    print(f"  codebase_version: {info.get('codebase_version')}")

if __name__ == "__main__":
    main()
