"""Run LIBERO evaluation with SO101 robot.

This script is meant to run inside the so101-eval Docker container
which has robosuite, libero, and robosuite_so101 pre-installed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/robosuite_so101")


def scale_bddl_files(benchmarks, output_dir):
    """Scale BDDL files for SO101 workspace."""
    from scale_bddl import scale_suite
    import libero

    bddl_root = Path(libero.__file__).parent / "bddl_files"
    output_root = Path(output_dir) / "bddl_files_so101"
    for suite_name in benchmarks:
        scale_suite(str(bddl_root / suite_name), str(output_root / suite_name))
    return output_root


def load_policy(checkpoint_path):
    """Load SmolVLA policy from checkpoint."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    preprocess, postprocess = make_pre_post_processors(
        policy.config, checkpoint_path
    )
    return policy, preprocess, postprocess


def run_libero_suite(suite_name, policy, preprocess, postprocess,
                     episodes_per_task, bddl_dir, output_dir):
    """Run evaluation on a single LIBERO suite."""
    import libero
    from libero import get_suite

    suite = get_suite(suite_name)
    results = []

    for task_idx, task in enumerate(suite):
        task_name = task.name if hasattr(task, "name") else f"task_{task_idx}"
        print(f"  Task {task_idx}: {task_name}")

        for ep_idx in range(episodes_per_task):
            try:
                env = task.get_env()
                obs = env.reset()
                done = False
                success = False
                total_reward = 0.0
                steps = 0

                while not done and steps < 300:
                    action = policy.select_action(obs)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    success = success or info.get("success", False)
                    steps += 1

                results.append({
                    "suite": suite_name,
                    "task": task_name,
                    "task_idx": task_idx,
                    "episode": ep_idx,
                    "success": success,
                    "reward": total_reward,
                    "steps": steps,
                })
                print(f"    ep {ep_idx}: success={success}, reward={total_reward:.3f}, steps={steps}")
            except Exception as e:
                print(f"    ep {ep_idx}: ERROR {e}")
                results.append({
                    "suite": suite_name,
                    "task": str(task_idx),
                    "task_idx": task_idx,
                    "episode": ep_idx,
                    "success": False,
                    "reward": 0.0,
                    "steps": 0,
                    "error": str(e),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description="Run LIBERO evaluation with SO101")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmark names")
    parser.add_argument("--episodes_per_task", type=int, default=50)
    parser.add_argument("--output_dir", default="/data/eval/libero_results")
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== SO101 LIBERO Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Episodes per task: {args.episodes_per_task}")

    print("\n--- Scaling BDDL files for SO101 ---")
    bddl_dir = scale_bddl_files(benchmarks, args.output_dir)

    print("\n--- Loading policy ---")
    policy, preprocess, postprocess = load_policy(args.checkpoint)
    print(f"Policy loaded: {type(policy).__name__}")

    all_results = []
    start_time = time.time()

    for suite_name in benchmarks:
        print(f"\n--- Running {suite_name} ---")
        results = run_libero_suite(
            suite_name, policy, preprocess, postprocess,
            args.episodes_per_task, bddl_dir, args.output_dir,
        )
        all_results.extend(results)

        successes = [r["success"] for r in results]
        rate = sum(successes) / len(successes) if successes else 0
        print(f"  {suite_name}: {sum(successes)}/{len(successes)} = {rate:.1%}")

    elapsed = time.time() - start_time

    summary = {
        "checkpoint": args.checkpoint,
        "benchmarks": benchmarks,
        "episodes_per_task": args.episodes_per_task,
        "total_episodes": len(all_results),
        "total_successes": sum(r["success"] for r in all_results),
        "overall_success_rate": sum(r["success"] for r in all_results) / len(all_results) if all_results else 0,
        "elapsed_s": elapsed,
        "per_suite": {},
        "per_episode": all_results,
    }

    for suite_name in benchmarks:
        suite_results = [r for r in all_results if r["suite"] == suite_name]
        if suite_results:
            summary["per_suite"][suite_name] = {
                "success_rate": sum(r["success"] for r in suite_results) / len(suite_results),
                "num_episodes": len(suite_results),
                "num_successes": sum(r["success"] for r in suite_results),
            }

    output_file = Path(args.output_dir) / "libero_eval_summary.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Evaluation Complete ===")
    print(f"Overall: {summary['total_successes']}/{summary['total_episodes']} = {summary['overall_success_rate']:.1%}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
