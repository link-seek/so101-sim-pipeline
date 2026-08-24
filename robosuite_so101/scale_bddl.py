"""Scale LIBERO BDDL task files for SO101 workspace.

SO101 max reach: ~0.47m vs Panda ~0.86m
Scale factor: 0.55
"""
import os
import re
import shutil
from pathlib import Path

SCALE_FACTOR = 0.55

def scale_ranges(content, factor=SCALE_FACTOR):
    """Scale all region ranges in BDDL content."""
    def scale_match(m):
        nums = [float(x) for x in m.groups()]
        scaled = [n * factor for n in nums]
        return f"({scaled[0]:.6f} {scaled[1]:.6f} {scaled[2]:.6f} {scaled[3]:.6f})"

    pattern = r'\((-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)'
    return re.sub(pattern, scale_match, content)


def scale_bddl_file(src_path, dst_path, factor=SCALE_FACTOR):
    """Scale a single BDDL file."""
    with open(src_path, 'r') as f:
        content = f.read()

    scaled = scale_ranges(content, factor)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w') as f:
        f.write(scaled)


def scale_suite(src_dir, dst_dir, factor=SCALE_FACTOR):
    """Scale all BDDL files in a suite directory."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists():
        print(f"SKIP: {src_dir} does not exist")
        return 0

    count = 0
    for bddl_file in src_path.glob("*.bddl"):
        dst_file = dst_path / bddl_file.name
        scale_bddl_file(str(bddl_file), str(dst_file), factor)
        count += 1

    print(f"Scaled {count} files from {src_dir} -> {dst_dir}")
    return count


if __name__ == "__main__":
    import libero

    libero_root = Path(libero.__file__).parent
    bddl_root = libero_root / "bddl_files"

    suites = [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_pro_swap",
        "libero_pro_object",
        "libero_pro_lan",
        "libero_pro_task",
        "libero_pro_env",
    ]

    output_root = Path("bddl_files_so101")
    total = 0
    for suite in suites:
        src = bddl_root / suite
        dst = output_root / suite
        total += scale_suite(str(src), str(dst))

    print(f"\nTotal: {total} BDDL files scaled for SO101 (factor={SCALE_FACTOR})")
