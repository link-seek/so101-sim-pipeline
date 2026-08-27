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
    """Register SO101 robot and gripper in robosuite 1.5."""
    import robosuite as suite
    from so101_robot import MountedSO101
    from so101_gripper import SO101Gripper

    print(f"robosuite version: {suite.__version__}")
    print(f"ALL_ROBOTS: {suite.ALL_ROBOTS}")
    print(f"ALL_GRIPPERS: {suite.ALL_GRIPPERS}")

    from robosuite.robots.fixed_base_robot import FixedBaseRobot
    import robosuite.robots as robots_pkg
    robots_pkg.ROBOT_CLASS_MAPPING["SO101"] = FixedBaseRobot
    print(f"Added SO101 -> FixedBaseRobot to ROBOT_CLASS_MAPPING")

    import robosuite.models.robots as robots_mod
    import robosuite.models.grippers as grippers_mod
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    robots_mod.MountedSO101 = MountedSO101
    REGISTERED_ROBOTS["SO101"] = MountedSO101
    grippers_mod.SO101Gripper = SO101Gripper
    grippers_mod.GRIPPER_MAPPING["SO101Gripper"] = SO101Gripper
    print(f"Registered SO101 in REGISTERED_ROBOTS and GRIPPER_MAPPING")

    print("SO101 registration complete")

    import mujoco
    from robosuite.utils.binding_utils import MjModel

    _orig_get_joint_qpos_addr = MjModel.get_joint_qpos_addr
    _orig_get_joint_qvel_addr = MjModel.get_joint_qvel_addr

    def _patched_get_joint_qpos_addr(self, name):
        joint_id = self.joint_name2id(name)
        joint_type = int(self.jnt_type[joint_id])
        joint_addr = self.jnt_qposadr[joint_id]
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            ndim = 7
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            ndim = 4
        else:
            assert joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
            ndim = 1
        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    def _patched_get_joint_qvel_addr(self, name):
        joint_id = self.joint_name2id(name)
        joint_type = int(self.jnt_type[joint_id])
        joint_addr = self.jnt_dofadr[joint_id]
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            ndim = 6
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            ndim = 3
        else:
            assert joint_type in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
            ndim = 1
        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    MjModel.get_joint_qpos_addr = _patched_get_joint_qpos_addr
    MjModel.get_joint_qvel_addr = _patched_get_joint_qvel_addr
    print("Patched get_joint_qpos_addr and get_joint_qvel_addr for MuJoCo Enum compatibility")

    import torch
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    print("Patched torch.load for weights_only=False compatibility")


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


# SO101 calibration (from dyordan1/so101-mujoco calib/so101_robot.json, ENCODER_RES=4096)
# offsets are degrees added to the LeRobot .pos value to reach the model joint frame
SO101_CALIB_OFFSETS = np.array(
    [5.88998, -1.45055, -7.25275, -0.96704, -0.04396], dtype=np.float64
)
SO101_JOINT_LO = np.array(
    [-1.9198621772, -1.7453292520, -1.7453292520, -1.6580627970, -2.7925268970],
    dtype=np.float64,
)
SO101_JOINT_HI = np.array(
    [1.9198621772, 1.7453292520, 1.5707963268, 1.6580627818, 2.7925267094],
    dtype=np.float64,
)
SO101_GRIP_LO = -0.1745329252
SO101_GRIP_HI = 1.7453292520

SCENE_SCALE_FACTOR = 0.55  # SO101 reach (~0.47m) vs Panda reach (~0.86m)


def rescale_scene_to_reach(env, factor=SCENE_SCALE_FACTOR):
    """Shrink the xy position of all task-relevant objects (movable + fixtures)
    toward the table center so they fall within SO101's smaller reach envelope.
    Relative geometry (in contact / on top of / left of) is preserved because
    every object is scaled by the same factor about the origin."""
    import mujoco

    sim = env.env.sim
    model, data = sim.model, sim.data
    touched = []
    domain_env = getattr(env, "env", env)

    body_dicts = []
    for attr in ("objects_dict", "fixtures_dict"):
        d = getattr(domain_env, attr, None) or {}
        body_dicts.extend(d.values())

    seen = set()
    for obj in body_dicts:
        root_body = getattr(obj, "root_body", None)
        if root_body is None or root_body in seen:
            continue
        seen.add(root_body)
        if any(k in root_body.lower() for k in ("table", "desk", "floor", "wall")):
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
        if bid < 0:
            continue
        model.body_pos[bid][0] *= factor
        model.body_pos[bid][1] *= factor
        touched.append(root_body)

    # Also move free-joint bodies whose current qpos placement differs from body_pos
    for jadr in range(model.njnt):
        b = model.jnt_bodyid[jadr]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if name is None or name in seen:
            continue
        if any(k in name.lower() for k in ("table", "desk", "floor", "wall")):
            continue
        if model.jnt_type[jadr] == int(mujoco.mjtJoint.mjJNT_FREE):
            qadr = model.jnt_qposadr[jadr]
            data.qpos[qadr] *= factor
            data.qpos[qadr + 1] *= factor

    sim.forward()
    print(f"    [scene-rescale x{factor}] moved {len(touched)} bodies: {touched[:6]}...")


def _policy_state_from_obs(obs):
    """Convert LIBERO radian joint state to training-unit .pos state
    (5 arm joints in degrees relative to calibration midpoint + gripper 0-100)."""
    joint_rad = np.asarray(obs.get("robot0_joint_pos", np.zeros(5)), dtype=np.float64).reshape(-1)[:5]
    grip_q = float(np.asarray(obs.get("robot0_gripper_qpos", [0.0]), dtype=np.float64).reshape(-1)[0])
    arm_pos = np.degrees(joint_rad) - SO101_CALIB_OFFSETS
    grip_pos = (grip_q - SO101_GRIP_LO) / (SO101_GRIP_HI - SO101_GRIP_LO) * 100.0
    return np.concatenate([arm_pos, [grip_pos]]).astype(np.float32)


def _env_action_from_policy(action):
    """Convert policy action in .pos units (degrees + gripper 0-100) to robosuite env action:
    absolute joint targets in radians for JOINT_POSITION(absolute) + GRIP input for gripper."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size < 6:
        a = np.pad(a, (0, 6 - a.size))
    else:
        a = a[:6]
    arm_rad = np.radians(a[:5] + SO101_CALIB_OFFSETS)
    arm_rad = np.clip(arm_rad, SO101_JOINT_LO, SO101_JOINT_HI)
    grip_cmd = np.clip(a[5] / 50.0 - 1.0, -1.0, 1.0)
    return np.concatenate([arm_rad, [grip_cmd]]).astype(np.float32)


def libero_obs_to_policy_obs(obs, task_description, device="cuda"):
    """Convert LIBERO observation to SmolVLA policy input format."""
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    state = _policy_state_from_obs(obs)

    frame = {
        "observation.state": state,
    }

    if "robot0_eye_in_hand_image" in obs:
        frame["observation.images.camera1"] = np.asarray(
            obs["agentview_image"] if "agentview_image" in obs else obs["robot0_eye_in_hand_image"], dtype=np.uint8
        )
        frame["observation.images.camera2"] = np.asarray(
            obs["robot0_eye_in_hand_image"], dtype=np.uint8
        )
        if "birdview_image" in obs:
            frame["observation.images.camera3"] = np.asarray(obs["birdview_image"], dtype=np.uint8)
        elif "agentview_image" in obs:
            frame["observation.images.camera3"] = np.asarray(obs["agentview_image"], dtype=np.uint8)

    frame = prepare_observation_for_inference(
        frame, device, task=task_description, robot_type="so_follower"
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
            import robosuite as _suite
            _ctrl_path = os.path.join(
                os.path.dirname(_suite.__file__),
                "controllers", "configs", "robots", "default_so101.json"
            )
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_file,
                robots=["SO101"],
                controller=_ctrl_path,
                camera_names=["agentview", "birdview", "robot0_eye_in_hand"],
                camera_heights=128,
                camera_widths=128,
            )
            try:
                env.seed(0)
            except (TypeError, AttributeError):
                pass
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
                policy.reset()
                env.reset()
                obs = env.set_init_state(init_states[ep_idx])
                rescale_scene_to_reach(env)

                done = False
                success = False
                total_reward = 0.0
                steps = 0
                num_steps_wait = 10

                while not done and steps < max_steps + num_steps_wait:
                    if steps < num_steps_wait:
                        hold = _policy_state_from_obs(obs)
                        action = _env_action_from_policy(hold)
                        obs, reward, done, info = env.step(action)
                        steps += 1
                        continue

                    raw_action = predict_action(
                        policy, preprocess, postprocess, obs, task_desc, device
                    )
                    action = _env_action_from_policy(raw_action)
                    if steps % 50 == 0:
                        jp = np.asarray(obs.get("robot0_joint_pos", np.zeros(5)), dtype=np.float64).reshape(-1)[:5]
                        print(
                            f"      step {steps}: cur_arm_deg={np.round(np.degrees(jp), 1)}, "
                            f"tgt_pos={np.round(np.asarray(raw_action).reshape(-1)[:6], 1)}"
                        )

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
