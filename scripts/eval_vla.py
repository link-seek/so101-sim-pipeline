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

    print(f"Starting model server: {' '.join(cmd)}", flush=True)
    server_log = open(RESULTS_DIR / "server.log", "ab")
    proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, text=True)

    for _ in range(300):
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            print("Model server is healthy")
            return proc
        except Exception:
            time.sleep(2)

    print("Model server failed to start within 600s")
    server_log.flush()
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
    # vla-eval 0.4.0 actually writes LIBEROBenchmark_<bench>_aggregate.json
    # next to the results dir; locate it for honest scoring.
    cands = sorted((RESULTS_DIR / benchmark_name).glob("*_aggregate.json"))
    if cands:
        print(f"Aggregate found: {cands[0].name}")
        return cands[0]
    print(f"WARNING: no aggregate json under {RESULTS_DIR / benchmark_name}")
    return None


def score_aggregate(agg_path):
    """Score a LIBEROBenchmark_*_aggregate.json. Returns (ok, entry)."""
    try:
        with open(agg_path) as f:
            data = json.load(f)
    except Exception as e:
        return False, {"success_rate": None, "error": f"aggregate unreadable: {e}"}
    total, succ, errs = 0, 0, 0
    per_task = {}
    for t in data.get("tasks", []):
        eps = t.get("episodes", [])
        ts, te = 0, 0
        for ep in eps:
            total += 1
            ts += 1
            if ep.get("metrics", {}).get("success"):
                succ += 1
            elif ep.get("failure_reason"):
                errs += 1
                te += 1
        per_task[t.get("task", "?")] = {"episodes": ts, "errors": te}
    if total == 0:
        return False, {"success_rate": None, "error": "aggregate has 0 episodes"}
    entry = {"success_rate": succ / total, "num_episodes": total,
             "num_success": succ, "num_errors": errs, "per_task": per_task}
    # ok only when every episode actually ran (no harness exceptions)
    return errs == 0, entry


def generate_summary(all_benchmarks, aggregates):
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
    }
    all_ok = True
    for name in all_benchmarks:
        agg = aggregates.get(name)
        if agg is None:
            summary["benchmarks"][name] = {"success_rate": None, "error": "results not found"}
            all_ok = False
        else:
            ok, entry = score_aggregate(agg)
            summary["benchmarks"][name] = entry
            all_ok = all_ok and ok

    summary_path = RESULTS_DIR / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")
    return all_ok


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

    aggregates = {}
    try:
        for name in benchmarks:
            run_benchmark(name, args.num_shards)
            aggregates[name] = merge_results(name)
    finally:
        print("Shutting down model server...")
        server_proc.send_signal(signal.SIGTERM)
        server_proc.wait()

    # NOTE: vla-eval exits 0 even with 100 errored episodes; the aggregate
    # is the only honest gate.
    all_ok = generate_summary(benchmarks, aggregates)
    print(f"\nAll benchmarks clean: {all_ok}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
