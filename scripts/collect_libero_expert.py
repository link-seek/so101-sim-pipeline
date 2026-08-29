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


def body_id(sim, name, verbose=False):
    import mujoco

    m = mj_raw(sim)[0] if isinstance(mj_raw(sim), tuple) else mj_raw(sim)
    for candidate in (name, f"robot0_{name}"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if bid >= 0:
            return bid
    if verbose:
        names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)
        ]
        print(f"    [debug] bodies containing hand/grip/wrist: "
              f"{[n for n in names if n and any(k in n.lower() for k in ('hand', 'grip', 'wrist', 'eef'))]}")
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
        # position is tracked at the moving-jaw body (the actual grasp point);
        # orientation keeps the hand frame's z-axis pointing straight down
        try:
            self.pos_bid = body_id(sim, "right_moving_jaw")
        except RuntimeError:
            self.pos_bid = body_id(sim, "right_hand")
        self.rot_bid = body_id(sim, "right_hand")
        self.eef_bid = self.pos_bid
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
        pos = np.array(self.d.xpos[self.pos_bid])
        z_axis = np.array(self.d.xmat[self.rot_bid]).reshape(3, 3)[:, 2]
        return pos, z_axis

    def _jls(self, target_xyz, q0, iters, damping, tol, ori_w, ori_w_ramp=True):
        """Single-start damped least squares. Position first, orientation ramps in."""
        q = np.array(q0, dtype=np.float64)
        m, d = self.m, self.d
        for i in range(iters):
            pos, z_axis = self._fk(q)
            e_pos = target_xyz - pos
            w = np.cross(z_axis, self.DOWN)
            ow = ori_w if not ori_w_ramp else ori_w * min(1.0, i / max(1, iters // 2))
            err = np.concatenate([e_pos, ow * w[:2]])
            if np.linalg.norm(e_pos) < tol and np.linalg.norm(w) < 0.05:
                break
            jac_pos = np.zeros((3, m.nv))
            jac_rot = np.zeros((3, m.nv))
            self.mujoco.mj_jacBody(m, d, jac_pos, None, self.pos_bid)
            self.mujoco.mj_jacBody(m, d, None, jac_rot, self.rot_bid)
            J = np.vstack([jac_pos[:, self.arm_dofadr], jac_rot[:2, self.arm_dofadr] * ow])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(5), err)
            q = np.clip(q + dq, SO101_JOINT_LO, SO101_JOINT_HI)
        return q

    def solve(self, target_xyz, q_init, tol=0.006):
        """Multi-start JLS: tries the seed plus perturbed/fallback postures,
        returns the solution with the lowest weighted error."""
        rng = np.random.default_rng(0)
        seeds = [np.array(q_init, dtype=np.float64)]
        # standard grasp-ish postures (shoulder pitched forward, elbow bent)
        seeds.append(np.radians([0.0, -50.0, 60.0, -10.0, 0.0]))
        seeds.append(np.radians([0.0, -70.0, 80.0, 10.0, 0.0]))
        seeds.append(np.radians([0.0, -30.0, 40.0, -10.0, 0.0]))
        for _ in range(6):
            seeds.append(
                np.clip(
                    np.radians(rng.uniform(-60, 60, size=5))
                    + np.array([0.0, -50.0, 50.0, 0.0, 0.0]),
                    SO101_JOINT_LO,
                    SO101_JOINT_HI,
                )
            )

        best_q, best_cost = None, np.inf
        for q0 in seeds:
            # phase 1: position only
            q = self._jls(target_xyz, q0, iters=500, damping=0.05, tol=tol, ori_w=0.0)
            # phase 2: orientation refinement
            q = self._jls(target_xyz, q, iters=300, damping=0.05, tol=tol, ori_w=0.6)
            pos, z_axis = self._fk(q)
            e = np.linalg.norm(target_xyz - pos) + 0.3 * np.linalg.norm(
                np.cross(z_axis, self.DOWN)
            )
            if e < best_cost:
                best_q, best_cost = q, e
            if e < tol:
                break
        return best_q


def plan_actions(domain, sim, ik, q_start_rad):
    """Return list of per-step action vectors in .pos units (5 arm deg + gripper 0-100)."""
    # resolve the object to grasp and the goal container from BDDL obj_of_interest
    obj_name, basket_name = None, None
    interest = None
    parsed = getattr(domain, "parsed_problem", None)
    if parsed is not None:
        interest = parsed.get("obj_of_interest")
    if interest and len(interest) >= 2 and interest[0] in domain.objects_dict:
        obj_name, basket_name = interest[0], interest[1]
    else:
        for name in domain.objects_dict:
            if any(k in name.lower() for k in ("basket", "plate", "box")):
                basket_name = name
            elif obj_name is None:
                obj_name = name
    if obj_name is None or basket_name is None:
        raise RuntimeError(f"could not resolve objects: {list(domain.objects_dict)}")
    p_obj = body_pos(sim, domain.objects_dict[obj_name].root_body)
    p_basket = body_pos(sim, domain.objects_dict[basket_name].root_body)
    print(f"    plan: obj={obj_name}@{np.round(p_obj, 3)} goal={basket_name}@{np.round(p_basket, 3)}")

    # IK controls the moving-jaw body directly, so targets are in jaw-center
    # coordinates relative to the object/basket centers
    waypoints = [
        ("move", p_obj + np.array([0.0, 0.0, 0.10]), 1.0, 1.5),
        ("approach", p_obj + np.array([0.0, 0.0, 0.045]), 1.0, 0.8),
        ("grip", p_obj + np.array([0.0, 0.0, 0.035]), 0.0, 1.5),
        ("lift", p_obj + np.array([0.0, 0.0, 0.20]), 0.0, 1.5),
        ("move", p_basket + np.array([0.0, 0.0, 0.22]), 0.0, 1.5),
        ("move", p_basket + np.array([0.0, 0.0, 0.10]), 0.0, 1.5),
        ("release", p_basket + np.array([0.0, 0.0, 0.10]), 1.0, 1.5),
        ("move", p_basket + np.array([0.0, 0.0, 0.28]), 1.0, 1.5),
    ]

    HOLD, GRIP_HOLD = 10, 20
    q = np.array(q_start_rad, dtype=np.float64)
    actions = []
    checkpoints = {}
    for phase, tgt, grip_open, max_step_deg in waypoints:
        q_sol = ik.solve(tgt, q)
        _, z_axis = ik._fk(q_sol)
        pos_reached = np.array(ik.d.xpos[ik.pos_bid])
        print(
            f"    [ik:{phase}] err={np.linalg.norm(tgt - pos_reached):.4f}m "
            f"reached={np.round(pos_reached, 3)} z_axis={np.round(z_axis, 2)}"
        )
        tgt_deg = np.degrees(q_sol) - SO101_CALIB_OFFSETS
        cur_deg = np.degrees(q) - SO101_CALIB_OFFSETS
        dist = float(np.max(np.abs(tgt_deg - cur_deg)))
        n = max(4, int(dist / max_step_deg))
        for k in range(1, n + 1):
            t = k / n
            a5 = (1 - t) * cur_deg + t * tgt_deg
            actions.append(np.concatenate([a5, [grip_open * 100.0]]))
        hold = GRIP_HOLD if phase in ("grip", "release") else HOLD
        for _ in range(hold):
            actions.append(np.concatenate([tgt_deg, [grip_open * 100.0]]))
        checkpoints[len(actions)] = phase
        q = q_sol
    return actions, checkpoints


def run_episode(env, ik, max_steps=900):
    obs = env.reset()
    rescale_scene_to_reach(env)
    sim = env.env.sim
    domain = env.env

    q_start = np.array([sim.data.qpos[a] for a in ik.arm_qadr], dtype=np.float64)

    # Planning: IK solvers write sim state, so snapshot & restore afterwards
    qpos_snapshot = np.array(sim.data.qpos).copy()
    qvel_snapshot = np.array(sim.data.qvel).copy()
    actions, checkpoints = plan_actions(domain, sim, ik, q_start)
    sim.data.qpos[:] = qpos_snapshot
    sim.data.qvel[:] = qvel_snapshot
    import mujoco

    m_raw, d_raw = mj_raw(sim)
    mujoco.mj_forward(m_raw, d_raw)
    actions = actions[:max_steps]
    if not actions:
        return False, 0, 0.0, [], []

    # track the manipulated object body for diagnostics
    obj_name = next(
        (n for n in domain.objects_dict if "basket" not in n.lower()), None
    )
    obj_bid = body_id(sim, domain.objects_dict[obj_name].root_body) if obj_name else -1
    obj_z0 = float(np.array(sim.data.xpos[obj_bid])[2]) if obj_bid >= 0 else 0.0

    obs_list, act_list = [], []
    success_streak = 0
    total_reward = 0.0
    for step_idx, act6 in enumerate(actions):
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
        success_streak = success_streak + 1 if domain._check_success() else 0
        if (step_idx + 1) in checkpoints and obj_bid >= 0:
            label = checkpoints[step_idx + 1]
            p_obj = np.array(sim.data.xpos[obj_bid])
            print(f"    [{label}@{step_idx + 1}] obj_pos={np.round(p_obj, 3)}")
            # grasp verification: after lifting, the object must come up with us
            if label == "lift" and p_obj[2] < obj_z0 + 0.03:
                print("    [grasp-failed] object not lifted; aborting trial")
                return False, len(act_list), total_reward, obs_list, act_list
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
            horizon=950,
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
