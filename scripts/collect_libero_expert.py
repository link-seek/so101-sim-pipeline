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


def body_id(sim, name, contains_fallback=("jaw", "grip", "hand")):
    import mujoco

    m, _ = mj_raw(sim)
    cands = [name, f"robot0_{name}", f"{name}_main", f"robot0_{name}_main"]
    for candidate in cands:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if bid >= 0:
            return bid
    # last resort: fuzzy match but SKIP robot-prefixed bodies
    for b in range(m.nbody):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if bn and not bn.startswith("robot0") and any(k in bn.lower() for k in contains_fallback):
            print(f"    [body-fallback] {name} -> {bn}")
            return b
    raise RuntimeError(f"body {name} not found")


def body_pos(sim, name):
    bid = body_id(sim, name)
    return np.array(sim.data.xpos[bid]).copy()


class ArmIK:
    """SO101 arm IK: c-space sampling (claw-down filter) + JLS position
    refinement. Controls the moving-jaw tip directly.

    NOTE: SO101's claw extends along hand's +z_local, and a valid grasp
    requires wrist-roll (j5) far from zero -- uniform sampling with a
    claw-down filter is essential; plain JLS from q=0 points the claw
    upward and just sweeps objects away.
    """

    TIP = 0.075  # claw tip distance from right_hand origin along +z_local

    def __init__(self, sim):
        import mujoco

        self.mujoco = mujoco
        self.m, self.d = mj_raw(sim)
        self.pos_bid = body_id(sim, "right_moving_jaw")
        self.rot_bid = body_id(sim, "right_hand")
        self.eef_bid = self.pos_bid
        self.arm_qadr, self.arm_dofadr, self.arm_jids = [], [], []
        for j in range(self.m.njnt):
            name = self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.startswith("robot0_joint") and int(self.m.jnt_type[j]) == 3:
                self.arm_qadr.append(int(self.m.jnt_qposadr[j]))
                self.arm_dofadr.append(int(self.m.jnt_dofadr[j]))
                self.arm_jids.append(j)
        if len(self.arm_qadr) != 5:
            raise RuntimeError(f"expected 5 arm joints, got {len(self.arm_qadr)}")
        self.LO = self.m.jnt_range[self.arm_jids, 0].copy()
        self.HI = self.m.jnt_range[self.arm_jids, 1].copy()
        rng = np.random.default_rng(0)
        self.samples = rng.uniform(self.LO, self.HI, size=(40000, 5))
        self.rng = rng

    def _set_q(self, q):
        for a, v in zip(self.arm_qadr, q):
            self.d.qpos[a] = v
        self.mujoco.mj_forward(self.m, self.d)

    def _fk(self, q):
        self._set_q(q)
        pos = np.array(self.d.xpos[self.pos_bid])
        zax = np.array(self.d.xmat[self.rot_bid]).reshape(3, 3)[:, 2]
        tip = pos + self.TIP * zax
        return pos, zax, tip

    def solve(self, target, q_hint=None):
        """Return joint angles (rad) putting the claw tip at target, claw down."""
        rng = self.rng
        n_glob, n_near = 20000, 25000
        glob_idx = rng.choice(len(self.samples), size=n_glob, replace=False)
        cands = []
        for i in glob_idx:
            q = self.samples[i].copy()
            self._set_q(q)
            zax = np.array(self.d.xmat[self.rot_bid]).reshape(3, 3)[:, 2]
            if zax[2] > -0.85:
                continue
            p = np.array(self.d.xpos[self.pos_bid])
            e = np.linalg.norm(p + self.TIP * zax - target)
            cands.append((e, q))
        if q_hint is not None:
            near = self.rng.uniform(
                np.maximum(q_hint - 0.9, self.LO),
                np.minimum(q_hint + 0.9, self.HI),
                size=(n_near, 5),
            )
            for q in near:
                self._set_q(q)
                zax = np.array(self.d.xmat[self.rot_bid]).reshape(3, 3)[:, 2]
                if zax[2] > -0.5:
                    continue
                p = np.array(self.d.xpos[self.pos_bid])
                e = np.linalg.norm(p + self.TIP * zax - target)
                cands.append((e, q))
        cands.sort(key=lambda c: c[0])

        best, bc = None, np.inf
        for _, q0 in cands[:5]:
            q = np.array(q0, dtype=np.float64)
            for _ in range(200):
                self._set_q(q)
                pos = np.array(self.d.xpos[self.pos_bid])
                zax = np.array(self.d.xmat[self.rot_bid]).reshape(3, 3)[:, 2]
                err = target - (pos + self.TIP * zax)
                if np.linalg.norm(err) < 0.004:
                    break
                jp = np.zeros((3, self.m.nv))
                self.mujoco.mj_jacBody(self.m, self.d, jp, None, self.pos_bid)
                J = jp[:, self.arm_dofadr]
                dq = J.T @ np.linalg.solve(J @ J.T + 0.1 * np.eye(3), err)
                q = np.clip(q + dq, self.LO, self.HI)
            pos, zax, tip = self._fk(q)
            e = np.linalg.norm(tip - target)
            if zax[2] > -0.4:
                e += 1.0
            if e < bc:
                best, bc = q.copy(), e
        return best


def aim_base_at(sim, target_xy):
    """Rotate the free-standing base so SO101's front (-y local) faces target."""
    import mujoco

    m, d = mj_raw(sim)
    base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_bid < 0:
        return
    base_pos = np.array(d.xpos[base_bid])[:2]
    dir_xy = np.asarray(target_xy) - base_pos
    # front = -y_local; after R(z,θ) front becomes (sinθ, -cosθ)
    theta = np.arctan2(dir_xy[0], -dir_xy[1])
    half = theta / 2.0
    m.body_quat[base_bid] = [np.cos(half), 0.0, 0.0, np.sin(half)]
    mujoco.mj_forward(m, d)
    print(f"    base aimed at {np.round(target_xy, 3)} (yaw {np.degrees(theta):.0f} deg)")


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

    # SO101's home orientation faces -y (unlike Panda's +x); rotate the base
    # so its front (-y local) points at the grasp target
    aim_base_at(sim, p_obj[:2])

    # IK controls the moving-jaw body directly, so targets are in jaw-center
    # coordinates relative to the object/basket centers
    waypoints = [
        ("move", p_obj + np.array([0.0, 0.0, 0.10]), 1.0, 1.5),
        ("approach", p_obj + np.array([0.0, 0.0, 0.035]), 1.0, 0.8),
        ("grip", p_obj + np.array([0.0, 0.0, 0.035]), 0.0, 1.5),
        ("lift", p_obj + np.array([0.0, 0.0, 0.20]), 0.0, 0.8),
        ("move", p_basket + np.array([0.0, 0.0, 0.22]), 0.0, 1.5),
        ("move", p_basket + np.array([0.0, 0.0, 0.10]), 0.0, 1.5),
        ("release", p_basket + np.array([0.0, 0.0, 0.10]), 1.0, 1.5),
        ("move", p_basket + np.array([0.0, 0.0, 0.28]), 1.0, 1.5),
    ]

    HOLD, GRIP_HOLD = 10, 30
    q = np.array(q_start_rad, dtype=np.float64)
    actions = []
    checkpoints = {}
    for phase, tgt, grip_open, max_step_deg in waypoints:
        q_sol = ik.solve(tgt, q_hint=q)
        if q_sol is None:
            print(f"    [ik:{phase}] NO SOLUTION; aborting plan")
            return None, None
        pos, zax, tip = ik._fk(q_sol)
        print(
            f"    [ik:{phase}] tip_err={np.linalg.norm(tgt - tip):.4f}m "
            f"tip={np.round(tip, 3)} z_axis={np.round(zax, 2)}"
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


def _has_body(sim, name):
    import mujoco

    m = mj_raw(sim)[0]
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) >= 0


def run_episode(env, ik, max_steps=900):
    obs = env.reset()
    rescale_scene_to_reach(env)
    sim = env.env.sim
    domain = env.env

    q_start = np.array([sim.data.qpos[a] for a in ik.arm_qadr], dtype=np.float64)
    base_bid = body_id(sim, "base_link") if _has_body(sim, "base_link") else (
        body_id(sim, "right_hand")
    )
    print(f"    base/right_hand pos={np.round(np.array(sim.data.xpos[base_bid]), 3)}")

    # Planning: IK solvers write sim state, so snapshot & restore afterwards
    qpos_snapshot = np.array(sim.data.qpos).copy()
    qvel_snapshot = np.array(sim.data.qvel).copy()
    actions, checkpoints = plan_actions(domain, sim, ik, q_start)
    if actions is None:
        return False, 0, 0.0, [], []
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
    obj_bid = -1
    if obj_name:
        import mujoco as _mj
        m_raw, _ = mj_raw(sim)
        rb = domain.objects_dict[obj_name].root_body
        for cand in (rb, f"{rb}_main", rb.replace("_main", "")):
            bid = _mj.mj_name2id(m_raw, _mj.mjtObj.mjOBJ_BODY, cand)
            if bid >= 0:
                obj_bid = bid
                break
        if obj_bid < 0:
            names = [_mj.mj_id2name(m_raw, _mj.mjtObj.mjOBJ_BODY, b) for b in range(m_raw.nbody)]
            print(f"    [debug] no body for {rb}; candidates: "
                  f"{[n for n in names if n and obj_name.split('_')[0] in n.lower()]}")
    obj_z0 = float(np.array(sim.data.xpos[obj_bid])[2]) if obj_bid >= 0 else 0.0

    # kinematic snap-grasp: attach object to claw when gripper closes on it
    if obj_name:
        from run_libero_eval import SnapGraspController

        ctrl = SnapGraspController(sim, domain.objects_dict[obj_name].root_body)
    else:
        ctrl = None

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
        if ctrl is not None:
            ctrl.update()
        # LIBERO official practice: success latches only after 10 consecutive checks
        success_streak = success_streak + 1 if domain._check_success() else 0
        if (step_idx + 1) in checkpoints and obj_bid >= 0:
            label = checkpoints[step_idx + 1]
            p_obj = np.array(sim.data.xpos[obj_bid])
            print(f"    [{label}@{step_idx + 1}] obj_pos={np.round(p_obj, 3)}")
            if label == "grip" and ctrl is not None:
                if ctrl.maybe_attach(gripper_closed=True):
                    print("    [snap-grasp] ATTACHED")
                else:
                    print("    [snap-grasp] attach condition not met (tip too far)")
            if label == "release" and ctrl is not None:
                ctrl.detach()
                print("    [snap-grasp] DETACHED")
            # grasp verification: after lifting, the object must come up with us
            if label == "lift" and obj_bid >= 0 and ctrl is not None and not ctrl.attached:
                print("    [grasp-failed] object not attached; aborting trial")
                return False, len(act_list), total_reward, obs_list, act_list
        if done or success_streak >= 10:
            break

    # done=True with reward>=1 means the env registered success on the final
    # step (before our 10-step latch could fill); also re-check final state
    if not success:
        if done and total_reward >= 1.0:
            success = True
        else:
            try:
                success = bool(domain._check_success())
            except Exception:
                pass
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
