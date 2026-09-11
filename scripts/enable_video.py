"""Enable per-episode mp4 recording in a vla-eval benchmark YAML.

Usage:
    python3 enable_video.py <bench_yaml> <true|false> [fps]

Idempotent: skips files that already contain a ``record_video`` key.
Exits 0 in all skip paths so ``set -e`` pipeline steps keep going.
"""
import sys

BLOCK_TMPL = "    recording:\n      record_video: true\n      video_fps: {fps}\n"
ANCHOR = "    episodes_per_task:"


def main() -> int:
    path, flag = sys.argv[1], sys.argv[2].strip().lower()
    fps = sys.argv[3] if len(sys.argv) > 3 else "20"
    if flag != "true":
        print("video off (RENDER_VIDEO=%s), skip %s" % (sys.argv[2], path))
        return 0
    text = open(path).read()
    if "record_video" in text:
        print("recording exists, skip %s" % path)
        return 0
    if ANCHOR not in text:
        print("anchor %r not found in %s, skip" % (ANCHOR, path))
        return 0
    open(path, "w").write(
        text.replace(ANCHOR, BLOCK_TMPL.format(fps=fps) + ANCHOR, 1)
    )
    print("recording injected into %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
