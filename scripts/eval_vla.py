#!/usr/bin/env python3
"""VLA 评估编排脚本 - 使用 vla-evaluation-harness 运行 LIBERO + LIBERO-PRO"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("MUJOCO_GL", "egl")

CONFIGS_DIR = Path("/workspace/configs")
RESULTS_DIR = Path("/data/eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LIBERO_BENCHMARKS = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
]

LIBERO_PRO_BENCHMARKS = [
    "libero_pro_swap",
    "libero_pro_object",
    "libero_pro_lan",
    "libero_pro_task",
    "libero_pro_env",
]


def start_model_server(model_config, checkpoint=None):
    cfg_path = CONFIGS_DIR / "model_servers" / model_config
    cmd = ["vla-eval", "serve", "--config", str(cfg_path)]
    if checkpoint:
        cmd.extend(["--arg", f"checkpoint={checkpoint}"])

    print(f"Starting model server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for _ in range(300):
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            print("Model server is healthy")
            return proc
        except Exception:
            time.sleep(2)

    print("Model server failed to start within 600s")
    stdout_data = proc.stdout.read() if proc.stdout else ""
    print(f"Model server stdout:\n{stdout_data}")
    proc.kill()
    sys.exit(1)


def run_benchmark(benchmark_name, num_shards=1):
    cfg_path = CONFIGS_DIR / "benchmarks" / f"{benchmark_name}.yaml"
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}, skipping")
        return None

    cmd = ["vla-eval", "run", "--config", str(cfg_path)]
    if num_shards > 1:
        cmd.extend(["--num-shards", str(num_shards)])

    print(f"\n{'='*60}")
    print(f"Running benchmark: {benchmark_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
    if result.returncode != 0:
        print(f"Benchmark {benchmark_name} failed with code {result.returncode}")

    return result.returncode == 0


def merge_results(benchmark_name):
    cfg_path = CONFIGS_DIR / "benchmarks" / f"{benchmark_name}.yaml"
    output_path = RESULTS_DIR / benchmark_name / "merged.json"
    cmd = ["vla-eval", "merge", "--config", str(cfg_path), "-o", str(output_path)]
    print(f"Merging results for {benchmark_name}...")
    subprocess.run(cmd, capture_output=True, text=True)


def generate_summary(all_benchmarks):
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
    }

    for name in all_benchmarks:
        result_file = RESULTS_DIR / name / "merged.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            success_rate = data.get("success_rate", 0.0)
            summary["benchmarks"][name] = {
                "success_rate": success_rate,
                "num_episodes": data.get("num_episodes", 0),
                "num_tasks": data.get("num_tasks", 0),
            }
        else:
            summary["benchmarks"][name] = {"success_rate": None, "error": "results not found"}

    summary_path = RESULTS_DIR / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="VLA evaluation with vla-eval harness")
    parser.add_argument("--model-config", default="smolvla_so101.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint (e.g. lerobot/smolvla_base)")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Specific benchmarks to run (default: all)")
    parser.add_argument("--skip-libero", action="store_true")
    parser.add_argument("--skip-libero-pro", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    benchmarks = []
    if not args.skip_libero:
        benchmarks.extend(LIBERO_BENCHMARKS)
    if not args.skip_libero_pro:
        benchmarks.extend(LIBERO_PRO_BENCHMARKS)
    if args.benchmarks:
        benchmarks = args.benchmarks

    server_proc = start_model_server(args.model_config, args.checkpoint)

    results = {}
    try:
        for name in benchmarks:
            results[name] = run_benchmark(name, args.num_shards)
            merge_results(name)
    finally:
        print("Shutting down model server...")
        server_proc.send_signal(signal.SIGTERM)
        server_proc.wait()

    summary = generate_summary(benchmarks)

    all_passed = all(v is True for v in results.values())
    print(f"\nAll benchmarks passed: {all_passed}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
