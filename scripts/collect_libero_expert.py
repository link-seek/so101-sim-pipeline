#!/usr/bin/env python3
"""Scripted-expert data collection on LIBERO with SO101.

Plan phase: numeric IK through reach->grasp->lift->place waypoints builds a
per-step action sequence in .pos units (degrees + gripper 0-100).
Execute phase: replays actions through the env (absolute joint control),
records camera frames + states + actions, keeps only episodes where
check_success() passes, stores as LeRobot v3 dataset compatible with the
dobri420/pick-cube-so101-sim feature layout used by SmolVLA finetuning.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from run_libero_eval import (  # noqa: E402
    register_so101,
    setup_libero_config,
    rescale_scene_to_reach,
    _env_action_from_policy,
    SO101_CALIB_OFFSETS,
    SO101_JOINT_LO,
    SO101_JOINT_HI,
)

CAMERA_MAP = {"camera1": "birdview", "camera2": "robot0_eye_in_hand", "camera3": "agentview"}
RENDER_HW = (480, 640)
FPS = 20
MAX_GRASP_RETRY = 2


def mj_raw(sim):
    m = sim.model._model if hasattr(sim.model, "_model") else sim.model
    d = sim.data._data if hasattr(sim.data, "_data") else sim.data
    return m, d


def body_id(sim, name):
    import mujoco

    m = mj_raw(sim)
    for candidate in (name, f"robot0_{name}"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if bid >= 0:
            return bid
    raise RuntimeError(f"body {name} not found")


def body_pos(sim, name):
    bid = body_id(sim, name)
    return np.array(sim.data.xpos[bid]).copy()


class ArmIK:
    """Damped-least-squares IK for the 5 arm joints: 3 position + 2 orientation
    (end-effector z-axis aligned straight down), matching MimicGen-style
    scripted grasp policies for low-DoF arms."""

    def __init__(self, sim):
        import mujoco

        self.mujoco = mujoco
        self.m, self.d = mj_raw(sim)
        self.eef_bid = body_id(sim, "right_hand")
        self.arm_qadr, self.arm_dofadr = [], []
        for j in range(self.m.njnt):
            name = self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.startswith("robot0_joint") and int(self.m.jnt_type[j]) == 3:
                self.arm_qadr.append(int(self.m.jnt_qposadr[j]))
                self.arm_dofadr.append(int(self.m.jnt_dofadr[j]))
        if len(self.arm_qadr) != 5:
            raise RuntimeError(f"expected 5 arm joints, got {len(self.arm_qadr)}")
        self.DOWN = np.array([0.0, 0.0, -1.0])

    def _fk(self, q):
        for a, v in zip(self.arm_qadr, q):
            self.d.qpos[a] = v
        self.mujoco.mj_forward(self.m, self.d)
        pos = np.array(self.d.xpos[self.eef_bid])
        z_axis = np.array(self.d.xmat[self.eef_bid]).reshape(3, 3)[:, 2]
        return pos, z_axis

    def solve(self, target_xyz, q_init, iters=200, damping=0.08, tol=0.004, ori_w=0.6):
        q = np.array(q_init, dtype=np.float64)
        for _ in range(iters):
            pos, z_axis = self._fk(q)
            e_pos = target_xyz - pos
            w = np.cross(z_axis, self.DOWN)  # rotation that brings z -> down
            err = np.concatenate([e_pos, ori_w * w[:2]])
            if np.linalg.norm(e_pos) < tol and np.linalg.norm(w) < 0.05:
                break
            jac_pos = np.zeros((3, self.m.nv))
            jac_rot = np.zeros((3, self.m.nv))
            self.mujoco.mj_jacBody(self.m, self.d, jac_pos, jac_rot, self.eef_bid)
            J = np.vstack([jac_pos[:, self.arm_dofadr], jac_rot[:2, self.arm_dofadr] * ori_w])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(5), err)
            q = np.clip(q + dq, SO101_JOINT_LO, SO101_JOINT_HI)
        return q


def plan_actions(domain, sim, ik, q_start_rad):
    """Return list of per-step action vectors in .pos units (5 arm deg + gripper 0-100)."""
    obj_name, basket_name = None, None
    for name in domain.objects_dict:
        if "basket" in name.lower():
            basket_name = name
        elif obj_name is None:
            obj_name = name
    if obj_name is None or basket_name is None:
        raise RuntimeError(f"could not resolve objects: {list(domain.objects_dict)}")
    p_obj = body_pos(sim, domain.objects_dict[obj_name].root_body)
    p_basket = body_pos(sim, domain.objects_dict[basket_name].root_body)
    print(f"    plan: obj={obj_name}@{np.round(p_obj, 3)} basket@{np.round(p_basket, 3)}")

    waypoints = [
        ("move", p_obj + np.array([0.0, 0.0, 0.14]), 1.0),
        ("move", p_obj + np.array([0.0, 0.0, 0.035]), 1.0),
        ("grip", p_obj + np.array([0.0, 0.0, 0.035]), 0.0),
        ("move", p_obj + np.array([0.0, 0.0, 0.22]), 0.0),
        ("move", p_basket + np.array([0.0, 0.0, 0.24]), 0.0),
        ("move", p_basket + np.array([0.0, 0.0, 0.12]), 0.0),
        ("release", p_basket + np.array([0.0, 0.0, 0.12]), 1.0),
        ("move", p_basket + np.array([0.0, 0.0, 0.30]), 1.0),
    ]

    HOLD, GRIP_HOLD = 10, 20
    q = np.array(q_start_rad, dtype=np.float64)
    actions = []
    for phase, tgt, grip_open in waypoints:
        q_sol = ik.solve(tgt, q)
        tgt_deg = np.degrees(q_sol) - SO101_CALIB_OFFSETS
        cur_deg = np.degrees(q) - SO101_CALIB_OFFSETS
        dist = float(np.max(np.abs(tgt_deg - cur_deg)))
        n = max(4, int(dist / 1.5))
        for k in range(1, n + 1):
            t = k / n
            a5 = (1 - t) * cur_deg + t * tgt_deg
            actions.append(np.concatenate([a5, [grip_open * 100.0]]))
        hold = GRIP_HOLD if phase in ("grip", "release") else HOLD
        for _ in range(hold):
            actions.append(np.concatenate([tgt_deg, [grip_open * 100.0]]))
        q = q_sol
    return actions


def run_episode(env, ik, max_steps=450):
    obs = env.reset()
    rescale_scene_to_reach(env)
    sim = env.env.sim
    domain = env.env

    q_start = np.array([sim.data.qpos[a] for a in ik.arm_qadr], dtype=np.float64)

    # Planning: IK solvers write sim state, so snapshot & restore afterwards
    qpos_snapshot = np.array(sim.data.qpos).copy()
    qvel_snapshot = np.array(sim.data.qvel).copy()
    actions = plan_actions(domain, sim, ik, q_start)
    sim.data.qpos[:] = qpos_snapshot
    sim.data.qvel[:] = qvel_snapshot
    import mujoco

    m_raw, d_raw = mj_raw(sim)
    mujoco.mj_forward(m_raw, d_raw)
    actions = actions[:max_steps]
    if not actions:
        return False, 0, 0.0, [], []

    obs_list, act_list = [], []
    success_streak = 0
    total_reward = 0.0
    for act6 in actions:
        obs_list.append({
            "camera1": np.asarray(obs["birdview_image"]).copy(),
            "camera2": np.asarray(obs["robot0_eye_in_hand_image"]).copy(),
            "camera3": np.asarray(obs["agentview_image"]).copy(),
            "state": np.concatenate([
                np.degrees(np.asarray(obs["robot0_joint_pos"], dtype=np.float64).reshape(-1)[:5])
                - SO101_CALIB_OFFSETS,
                [(float(np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)[0]) + 0.1745329252)
                 / (1.7453292520 + 0.1745329252) * 100.0],
            ]).astype(np.float32),
        })
        act_list.append(act6.astype(np.float32))
        obs, reward, done, info = env.step(_env_action_from_policy(act6))
        total_reward += float(reward)
        # LIBERO official practice: success latches only after 10 consecutive checks
        success_streak = success_streak + 1 if domain.check_success() else 0
        if done or success_streak >= 10:
            break

    success = success_streak >= 10
    return success, len(act_list), total_reward, obs_list, act_list


def save_dataset(episodes, out_dir, task_name):
    """Save episodes as a LeRobot v3 dataset (one dataset per suite run)."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.camera1": {"dtype": "video", "shape": [*RENDER_HW, 3], "names": None},
        "observation.images.camera2": {"dtype": "video", "shape": [*RENDER_HW, 3], "names": None},
        "observation.images.camera3": {"dtype": "video", "shape": [*RENDER_HW, 3], "names": None},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id="libero_so101_expert",
        fps=FPS,
        features=features,
        root=out_dir,
        robot_type="so_follower",
        use_videos=True,
    )
    for ep_idx, (frames, acts, lang) in enumerate(episodes):
        for t in range(len(acts)):
            ds_frame = {
                "observation.images.camera1": frames[t]["camera1"],
                "observation.images.camera2": frames[t]["camera2"],
                "observation.images.camera3": frames[t]["camera3"],
                "observation.state": frames[t]["state"],
                "action": acts[t],
                "task": lang,
            }
            ds.add_frame(ds_frame)
        ds.save_episode()
    print(f"    dataset saved: {out_dir} ({len(episodes)} episodes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--episodes_per_task", type=int, default=5)
    parser.add_argument("--tasks", default="", help="comma-separated task indices; default all")
    parser.add_argument("--output_dir", default="/data/datasets/libero_so101_expert")
    args = parser.parse_args()

    setup_libero_config()
    register_so101()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite as _suite

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    bddl_root = get_libero_path("bddl_files")

    task_ids = (
        [int(i) for i in args.tasks.split(",") if i != ""]
        if args.tasks else list(range(task_suite.n_tasks))
    )
    ctrl_path = os.path.join(
        os.path.dirname(_suite.__file__),
        "controllers", "configs", "robots", "default_so101.json",
    )

    all_episodes = []
    out_dir = Path(args.output_dir)
    out_root_suite = out_dir / args.suite
    out_root_suite.mkdir(parents=True, exist_ok=True)

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        print(f"\n=== Task {task_id}: {task.name} ===")

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            robots=["SO101"],
            controller=ctrl_path,
            camera_names=list(CAMERA_MAP.values()),
            camera_heights=RENDER_HW[0],
            camera_widths=RENDER_HW[1],
            control_freq=FPS,
            horizon=450,
        )
        try:
            ik = ArmIK(env.env.sim)
            episodes, tried = [], 0
            while len(episodes) < args.episodes_per_task and tried < args.episodes_per_task * 4:
                tried += 1
                try:
                    success, steps, reward, obs_list, act_list = run_episode(env, ik)
                except Exception as e:
                    import traceback
                    print(f"    trial {tried}: PLAN/EXEC ERROR {e}")
                    traceback.print_exc()
                    continue
                print(f"    trial {tried}: success={success} steps={steps} reward={reward:.3f}")
                if success:
                    episodes.append((obs_list, act_list, task.language))
            print(f"    saved {len(episodes)}/{args.episodes_per_task} episodes (tried {tried})")
            all_episodes.extend(episodes)
        finally:
            env.close()

    if all_episodes:
        save_dataset(all_episodes, out_root_suite, args.suite)
    print(f"\nTotal episodes collected: {len(all_episodes)}")


if __name__ == "__main__":
    main()
