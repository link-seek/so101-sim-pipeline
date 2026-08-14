#!/usr/bin/env python3
"""Wrapper that patches lerobot get_safe_version to skip version tag check,
then calls lerobot-train."""
import sys

import lerobot.datasets.utils as _utils
_orig = _utils.get_safe_version
def _patched(repo_id, revision):
    try:
        return _orig(repo_id, revision)
    except RuntimeError:
        return "main"
_utils.get_safe_version = _patched

# Also patch the already-imported reference in dataset_metadata
import lerobot.datasets.dataset_metadata as _meta
if hasattr(_meta, 'get_safe_version'):
    _meta.get_safe_version = _patched

from lerobot.scripts.lerobot_train import main
main()
