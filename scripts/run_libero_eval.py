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

    # LIBERO X-embodiment bug: Libero_Floor_Manipulation never defines
    # robot_base_xpos_offset (referenced by bddl_base_domain._load_model),
    # so floor-arena tasks crash for every robot. Patch it here.
    # Note: register_problem has no return, so the module attribute is None;
    # fetch the real class from TASK_MAPPING.
    try:
        from libero.libero.envs.bddl_base_domain import TASK_MAPPING

        floor_cls = TASK_MAPPING.get("libero_floor_manipulation")
        if floor_cls is not None and not hasattr(floor_cls, "robot_base_xpos_offset"):
            floor_cls.robot_base_xpos_offset = {
                "bins": (-0.5, -0.1, 0),
                "empty": (-0.6, 0, 0),
                # SO101 sweet-spot working ring is ~0.2-0.25m from the base:
                # object regions (scaled 0.55) sit within +/-0.15m of origin,
                # so park the base diagonally to keep them in the ring
                "table": (-0.13, 0.05, 0),
            }
            print("Patched Libero_Floor_Manipulation.robot_base_xpos_offset")

        # tabletop problem classes place the base at Panda's working distance
        # (x=-0.66); SO101's short reach needs it much closer to the table center
        for name in (
            "libero_tabletop_manipulation",
            "libero_study_tabletop_manipulation",
            "libero_kitchen_tabletop_manipulation",
            "libero_living_room_tabletop_manipulation",
            "libero_coffee_table_manipulation",
        ):
            prob_cls = TASK_MAPPING.get(name)
            if prob_cls is None or not hasattr(prob_cls, "robot_base_xpos_offset"):
                continue
            if not getattr(prob_cls, "_so101_base_patched", False):
                orig = prob_cls.robot_base_xpos_offset

                def _so101_table_offset(table_length):
                    return (-0.30, 0, 0.90)

                prob_cls.robot_base_xpos_offset = {
                    **{k: v for k, v in orig.items() if k != "table"},
                    "table": _so101_table_offset,
                }
                prob_cls._so101_base_patched = True
        print("Patched tabletop robot_base_xpos_offset for SO101")
    except ImportError:
        pass

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
        try:
            bid = model.body_name2id(root_body)
        except Exception:
            continue
        if bid < 0:
            continue
        model.body_pos[bid][0] *= factor
        model.body_pos[bid][1] *= factor
        touched.append(root_body)

    # Also move free-joint bodies whose current qpos placement differs from body_pos
    for jadr in range(model.njnt):
        b = int(model.jnt_bodyid[jadr])
        try:
            name = model.body_id2name(b)
        except Exception:
            continue
        if name is None or name in seen:
            continue
        if any(k in name.lower() for k in ("table", "desk", "floor", "wall")):
            continue
        if int(model.jnt_type[jadr]) == int(mujoco.mjtJoint.mjJNT_FREE):
            qadr = int(model.jnt_qposadr[jadr])
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
    absolute joint targets for JOINT_POSITION(absolute) + GRIP input for gripper."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size < 6:
        a = np.pad(a, (0, 6 - a.size))
    else:
        a = a[:6]
    arm_rad = np.radians(a[:5] + SO101_CALIB_OFFSETS)
    arm_rad = np.clip(arm_rad, SO101_JOINT_LO, SO101_JOINT_HI)
    grip_cmd = np.clip(a[5] / 50.0 - 1.0, -1.0, 1.0)
    return np.concatenate([arm_rad, [grip_cmd]]).astype(np.float32)


class SnapGraspController:
    """Kinematic snap-grasp (the standard robosuite/MetaWorld approach).

    MuJoCo friction grasping with mesh contacts is unreliable (slip, single
    contact point, regularized friction has no stick state). All working
    SO101 sim pipelines (dyordan1 weld-latch, robosuite tasks, sim-engine)
    attach the object kinematically when the gripper closes around it.

    Usage per step:
        ctrl.maybe_attach(tip_pos, gripper_closed)
        ctrl.update(tip_pos)      # while attached: obj tracks tip + offset
        ctrl.detach()             # gripper opened: release with zero velocity
    """

    ATTACH_DIST = 0.06  # generous: tip must be within 6cm of object center

    def __init__(self, sim, obj_body_name):
        import mujoco

        self.mujoco = mujoco
        self.m = sim.model._model if hasattr(sim.model, "_model") else sim.model
        self.d = sim.data._data if hasattr(sim.data, "_data") else sim.data
        self.attached = False
        self._offset = None
        self._obj_quat = None

        # object body + its free joint
        self.obj_bid = self._find_body(obj_body_name)
        if self.obj_bid < 0:
            raise RuntimeError(f"snap-grasp: object body {obj_body_name} not found")
        jid = int(self.m.body_jntadr[self.obj_bid])
        if jid < 0 or int(self.m.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError(f"snap-grasp: {obj_body_name} has no free joint")
        self.obj_qadr = int(self.m.jnt_qposadr[jid])
        self.obj_vadr = int(self.m.jnt_dofadr[jid])

        # gripper tip body (robosuite renames: right_moving_jaw ->
        # gripper0_right_right_hand in the composed scene)
        self.tip_bid = self._find_body(
            "right_moving_jaw",
            extra_candidates=("gripper0_right_right_moving_jaw", "gripper0_right_right_hand"),
            fuzzy=("moving_jaw", "right_hand"),
        )
        if self.tip_bid < 0:
            raise RuntimeError("snap-grasp: gripper tip body not found")
        # TCP: if we resolved a hand body (origin at the wrist, not the claw
        # tip), the tip sits 7.5cm further along the hand's +z_local axis
        tip_name = self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_BODY, self.tip_bid)
        self.tcp = 0.075 if (tip_name and "hand" in tip_name.lower()) else 0.0

    def _find_body(self, name, extra_candidates=(), fuzzy=()):
        cands = [name, f"robot0_{name}", *extra_candidates]
        for c in cands:
            bid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_BODY, c)
            if bid >= 0:
                return bid
        if fuzzy:
            for b in range(self.m.nbody):
                bn = self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_BODY, b)
                if bn and any(k in bn.lower() for k in fuzzy):
                    return b
        return -1

    def tip_position(self):
        pos = np.array(self.d.xpos[self.tip_bid])
        zax = np.array(self.d.xmat[self.tip_bid]).reshape(3, 3)[:, 2]
        return pos + self.tcp * zax

    def object_position(self):
        return np.array(self.d.xpos[self.obj_bid])

    def maybe_attach(self, gripper_closed):
        if self.attached or not gripper_closed:
            return False
        tip = self.tip_position()
        obj = self.object_position()
        if np.linalg.norm(tip - obj) > self.ATTACH_DIST:
            return False
        # capture constant offset and orientation at the attach moment
        self._offset = obj - tip
        self._obj_quat = np.array(self.d.qpos[self.obj_qadr + 3 : self.obj_qadr + 7]).copy()
        self.attached = True
        return True

    def update(self):
        if not self.attached:
            return
        tip = self.tip_position()
        target = tip + self._offset
        self.d.qpos[self.obj_qadr : self.obj_qadr + 3] = target
        self.d.qpos[self.obj_qadr + 3 : self.obj_qadr + 7] = self._obj_quat
        self.d.qvel[self.obj_vadr : self.obj_vadr + 6] = 0.0
        self.mujoco.mj_forward(self.m, self.d)

    def detach(self):
        if self.attached:
            self.d.qvel[self.obj_vadr : self.obj_vadr + 6] = 0.0  # no flying off
        self.attached = False
        self._offset = None


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
                     episodes_per_task, output_dir, device="cuda", force_prompt=None):
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
        if force_prompt:
            task_desc = force_prompt
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

                # snap-grasp: object attaches when the policy closes the gripper
                # around it (must match the data-collection env mechanics)
                domain = getattr(env, "env", env)
                snap_obj = None
                parsed = getattr(domain, "parsed_problem", None)
                if parsed is not None and parsed.get("obj_of_interest"):
                    snap_obj = parsed["obj_of_interest"][0]
                if snap_obj and snap_obj in getattr(domain, "objects_dict", {}):
                    ctrl = SnapGraspController(
                        env.env.sim, domain.objects_dict[snap_obj].root_body
                    )
                else:
                    ctrl = None

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
                    if ctrl is not None:
                        ctrl.update()
                        grip_pos = float(np.asarray(raw_action).reshape(-1)[5])
                        if ctrl.attached:
                            if grip_pos > 55:
                                ctrl.detach()
                        elif grip_pos < 45:
                            if ctrl.maybe_attach(gripper_closed=True):
                                print(f"      [snap-grasp] ATTACHED at step {steps}")
                        success = success or env.check_success()
                    else:
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
    parser.add_argument("--force_prompt", default="", help="Override task language instruction")
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== SO101 LIBERO Evaluation ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Episodes per task: {args.episodes_per_task}")

    print("\n--- Setting up LIBERO config ---")
    setup_libero_config()

    print("\n--- Registering SO101 ---")
    register_so101()

    print("\n--- Loading policy ---")
    policy, preprocess, postprocess = load_policy(args.checkpoint)

    all_results = []
    start_time = time.time()

    for suite_name in benchmarks:
        print(f"\n--- Running {suite_name} ---")
        results = run_libero_suite(
            suite_name, policy, preprocess, postprocess,
            args.episodes_per_task, args.output_dir,
            force_prompt=(args.force_prompt or None),
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
