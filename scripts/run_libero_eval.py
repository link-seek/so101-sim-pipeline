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

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np

sys.path.insert(0, "/workspace/robosuite_so101")


def setup_libero_config():
    """Pre-create LIBERO config to avoid interactive prompt."""
    import libero
    libero_root = Path(libero.__file__).parent / "libero"
    config_dir = Path.home() / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        config_content = f"""benchmark_root: {libero_root}
init_states: {libero_root / "init_files"}
datasets: {libero_root.parent / "datasets"}
bddl_files: {libero_root / "bddl_files"}
"""
        config_file.write_text(config_content)
        print(f"Created LIBERO config at {config_file}")


def register_so101():
    """Register SO101 robot and gripper in robosuite."""
    import robosuite as suite
    from so101_robot import MountedSO101
    from so101_gripper import SO101Gripper

    print(f"robosuite version: {suite.__version__}")
    print(f"ALL_ROBOTS: {suite.ALL_ROBOTS}")
    print(f"ALL_GRIPPERS: {suite.ALL_GRIPPERS}")

    import robosuite.robots as robots_pkg
    if hasattr(robots_pkg, "ROBOT_CLASS_MAPPING"):
        panda_class = robots_pkg.ROBOT_CLASS_MAPPING.get("Panda")
        robots_pkg.ROBOT_CLASS_MAPPING["MountedSO101"] = panda_class
        print(f"Added MountedSO101 -> {panda_class} to ROBOT_CLASS_MAPPING")

    import robosuite.models.robots as robots_mod
    import robosuite.models.grippers as grippers_mod
    robots_mod.MountedSO101 = MountedSO101
    grippers_mod.SO101Gripper = SO101Gripper

    import inspect
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    from robosuite.utils.mjcf_utils import find_elements, string_to_array
    from collections import OrderedDict

    _orig_mm_init = ManipulatorModel.__init__
    def _patched_mm_init(self, fname, idn=0):
        super(ManipulatorModel, self).__init__(fname, idn=idn)
        self.grippers = OrderedDict()
        if self.arm_type == "single":
            eef = self.eef_name["right"] if isinstance(self.eef_name, dict) else self.eef_name
            hand_element = find_elements(root=self.root, tags="body", attribs={"name": eef}, return_first=True)
            self.hand_rotation_offset = string_to_array(hand_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]
        else:
            self.hand_rotation_offset = {}
            for arm in ("right", "left"):
                hand_element = find_elements(root=self.root, tags="body", attribs={"name": self.eef_name[arm]}, return_first=True)
                self.hand_rotation_offset[arm] = string_to_array(hand_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]
        self.cameras = self.get_element_names(self.worldbody, "camera")
        self._base_actuators = []
        self._torso_actuators = []
        self._head_actuators = []
        self._legs_actuators = []
        self._arms_actuators = []
        self._base_joints = []
        self._torso_joints = []
        self._head_joints = []
        self._legs_joints = []
        self._arms_joints = []
    ManipulatorModel.__init__ = _patched_mm_init

    print("SO101 registration complete")


def load_policy(checkpoint_path, device="cuda"):
    """Load SmolVLA policy from checkpoint."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"Loading SmolVLA policy from {checkpoint_path}...")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        checkpoint_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"Policy loaded: {type(policy).__name__}")
    return policy, preprocess, postprocess


def libero_obs_to_policy_obs(obs, task_description, device="cuda"):
    """Convert LIBERO observation to SmolVLA policy input format."""
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    state = obs.get("robot0_joint_pos", np.zeros(6, dtype=np.float32))
    state = np.asarray(state, dtype=np.float32)
    if len(state) > 6:
        state = state[:6]
    elif len(state) < 6:
        state = np.pad(state, (0, 6 - len(state)))

    frame = {
        "observation.state": state,
    }

    if "robot0_eye_in_hand_image" in obs:
        frame["observation.images.camera1"] = np.asarray(
            obs["robot0_eye_in_hand_image"], dtype=np.uint8
        )
    if "agentview_image" in obs:
        frame["observation.images.camera2"] = np.asarray(
            obs["agentview_image"], dtype=np.uint8
        )
    if "robot0_eye_in_hand_image" in obs and "agentview_image" in obs:
        frame["observation.images.camera3"] = np.asarray(
            obs["agentview_image"], dtype=np.uint8
        )

    frame = prepare_observation_for_inference(
        frame, device, task=task_description, robot_type=""
    )
    return frame


def predict_action(policy, preprocess, postprocess, obs, task_description, device="cuda"):
    """Run policy inference and return action numpy array."""
    import torch

    frame = libero_obs_to_policy_obs(obs, task_description, device)
    frame = preprocess(frame)

    with torch.inference_mode():
        action = policy.select_action(frame)

    action = postprocess(action)
    if isinstance(action, dict):
        action = action["action"]
    if isinstance(action, torch.Tensor):
        action = action.squeeze().cpu().numpy()

    return np.asarray(action, dtype=np.float32)


def run_libero_suite(suite_name, policy, preprocess, postprocess,
                     episodes_per_task, output_dir, device="cuda"):
    """Run evaluation on a single LIBERO suite."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        print(f"  ERROR: suite '{suite_name}' not in {list(benchmark_dict.keys())}")
        return []

    task_suite = benchmark_dict[suite_name]()
    num_tasks = task_suite.n_tasks
    bddl_root = get_libero_path("bddl_files")

    max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }.get(suite_name, 300)

    results = []

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_desc = task.language
        bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        print(f"  Task {task_id}: {task_name} ({task_desc})")

        try:
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_file,
                robots=["SO101"],
                controller="JOINT_POSITION",
                camera_heights=128,
                camera_widths=128,
            )
            env.seed(0)
            init_states = task_suite.get_task_init_states(task_id)
        except Exception as e:
            import traceback
            print(f"    ERROR creating env: {e}")
            traceback.print_exc()
            for ep_idx in range(episodes_per_task):
                results.append({
                    "suite": suite_name, "task": task_name, "task_idx": task_id,
                    "episode": ep_idx, "success": False, "reward": 0.0,
                    "steps": 0, "error": str(e),
                })
            continue

        for ep_idx in range(min(episodes_per_task, len(init_states))):
            try:
                env.reset()
                obs = env.set_init_state(init_states[ep_idx])

                done = False
                success = False
                total_reward = 0.0
                steps = 0
                num_steps_wait = 10

                while not done and steps < max_steps + num_steps_wait:
                    if steps < num_steps_wait:
                        action = np.zeros(6, dtype=np.float32)
                        obs, reward, done, info = env.step(action)
                        steps += 1
                        continue

                    action = predict_action(
                        policy, preprocess, postprocess, obs, task_desc, device
                    )
                    if len(action) < 6:
                        action = np.pad(action, (0, 6 - len(action)))
                    elif len(action) > 6:
                        action = action[:6]

                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    success = success or env.check_success()
                    steps += 1

                results.append({
                    "suite": suite_name, "task": task_name, "task_idx": task_id,
                    "episode": ep_idx, "success": bool(success),
                    "reward": float(total_reward), "steps": steps,
                })
                print(f"    ep {ep_idx}: success={success}, reward={total_reward:.3f}, steps={steps}")
            except Exception as e:
                print(f"    ep {ep_idx}: ERROR {e}")
                results.append({
                    "suite": suite_name, "task": task_name, "task_idx": task_id,
                    "episode": ep_idx, "success": False, "reward": 0.0,
                    "steps": 0, "error": str(e),
                })

        try:
            env.close()
        except Exception:
            pass

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

    print("\n--- Registering SO101 ---")
    register_so101()

    print("\n--- Setting up LIBERO config ---")
    setup_libero_config()

    print("\n--- Loading policy ---")
    policy, preprocess, postprocess = load_policy(args.checkpoint)

    all_results = []
    start_time = time.time()

    for suite_name in benchmarks:
        print(f"\n--- Running {suite_name} ---")
        results = run_libero_suite(
            suite_name, policy, preprocess, postprocess,
            args.episodes_per_task, args.output_dir,
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
