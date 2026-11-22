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
    find_grip_site_id,
    register_so101,
    setup_libero_config,
    place_base_at,
    _env_action_from_policy,
    SO101_CALIB_OFFSETS,
    SO101_JOINT_LO,
    SO101_JOINT_HI,
)

CAMERA_MAP = {"camera1": "birdview", "camera2": "robot0_eye_in_hand", "camera3": "agentview"}
RENDER_HW = (480, 640)
CART_STEP = 0.01  # metres between task-space IK samples along a path
# Nobody scales LIBERO's scene: its layout is sized for a Panda and scaling it
# pushed objects into the robot. We move the robot's base instead (see --base_xy).
BASE_XY = None
SNAP_GRASP = False  # physical contact only: no teleporting objects
# Community SO101 sims (masato-ka/gym-soarm) drive the gripper joint directly and
# treat qpos < 0 as "closed"; the lower range limit (-0.1745) is over-closed and
# pushes objects away. Close to this angle instead, converted through the
# actuator range (never a hand-tuned action value).
GRIP_CLOSED_Q = 0.0
# stream progress: docker logs is block-buffered otherwise, so a multi-hour run
# looks dead for tens of minutes
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
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

    # grasp point = model grip_site (see so101_gripper.xml); no coordinate constants

    def __init__(self, sim):
        import mujoco

        self.mujoco = mujoco
        self.m, self.d = mj_raw(sim)
        # composed name is gripper0_right_right_moving_jaw; do NOT let the fuzzy
        # fallback match "...right_hand" (that silently aimed the hand instead)
        self.pos_bid = body_id(sim, "right_moving_jaw",
                               contains_fallback=("moving_jaw", "jaw"))
        self.site_bid, self.site_name = find_grip_site_id(self.mujoco, self.m)
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

    def _robot_geoms(self):
        """Geom ids that belong to the robot itself (links + gripper)."""
        if getattr(self, "_rg", None) is not None:
            return self._rg
        ids = set()
        for g in range(self.m.ngeom):
            bn = self.mujoco.mj_id2name(
                self.m, self.mujoco.mjtObj.mjOBJ_BODY, int(self.m.geom_bodyid[g])
            ) or ""
            if bn.startswith("robot0") or bn.startswith("gripper0"):
                ids.add(g)
        self._rg = ids
        return ids

    def self_contacts(self, q):
        """Number of penetrating self-contacts at posture q (0 = feasible)."""
        self._set_q(q)
        rg = self._robot_geoms()
        n = 0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist < -0.001 and c.geom1 in rg and c.geom2 in rg:
                n += 1
        return n

    def _fk(self, q):
        self._set_q(q)
        pos = np.array(self.d.xpos[self.pos_bid])
        zax = np.array(self.d.xmat[self.rot_bid]).reshape(3, 3)[:, 2]
        # same fingertip definition as snap-grasp (hinge-origin aware)
        tip = np.array(self.d.site_xpos[self.site_bid])
        return pos, zax, tip

    def solve_local(self, target, q_hint, iters=80, tol=0.004):
        """Refine from a hint only: keeps the arm on one branch, ~1000x cheaper
        than the global search. Used for Cartesian path sampling."""
        q = np.array(q_hint, dtype=np.float64)
        for _ in range(iters):
            self._set_q(q)
            tip = np.array(self.d.site_xpos[self.site_bid])
            err = target - tip
            if np.linalg.norm(err) < tol:
                break
            jp = np.zeros((3, self.m.nv))
            self.mujoco.mj_jacBody(self.m, self.d, jp, None, self.pos_bid)
            J = jp[:, self.arm_dofadr]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05 * np.eye(3), err)
            q = np.clip(q + dq, self.LO, self.HI)
        self._set_q(q)
        tip = np.array(self.d.site_xpos[self.site_bid])
        zax = np.array(self.d.xmat[self.rot_bid]).reshape(3, 3)[:, 2]
        if np.linalg.norm(target - tip) > 0.02 or zax[2] > -0.35:
            return None
        return q

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
            e = np.linalg.norm(np.array(self.d.site_xpos[self.site_bid]) - target)
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
                e = np.linalg.norm(np.array(self.d.site_xpos[self.site_bid]) - target)
                cands.append((e, q))
        cands.sort(key=lambda c: c[0])

        scored = []
        for _, q0 in cands[:8]:
            q = np.array(q0, dtype=np.float64)
            for _ in range(200):
                self._set_q(q)
                pos = np.array(self.d.xpos[self.pos_bid])
                zax = np.array(self.d.xmat[self.rot_bid]).reshape(3, 3)[:, 2]
                err = target - (np.array(self.d.site_xpos[self.site_bid]))
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
            scored.append((e, q.copy()))
        if not scored:
            return None
        best_e = min(e for e, _ in scored)
        # joint-space continuity matters more than the last millimetre: among
        # near-optimal solutions take the one closest to the current posture.
        # Otherwise the arm flips elbow branches and drags itself through the
        # table on the way to the basket (observed: tracking breakdown at the
        # first post-lift waypoint).
        near = [(e, q) for e, q in scored if e <= best_e + 0.015]
        # IK has no notion of collision: solutions that fold the arm into itself
        # are physically unreachable (the servos just stall), so drop them.
        feasible = [(e, q) for e, q in near if self.self_contacts(q) == 0]
        pool = feasible if feasible else near
        if not feasible:
            print(f"    [ik] no collision-free solution (best {best_e*1000:.0f}mm, "
                  f"{len(near)} candidates)")
        if q_hint is not None:
            return min(pool, key=lambda t: float(np.linalg.norm(t[1] - q_hint)))[1]
        return min(pool, key=lambda t: t[0])[1]


def aim_base_at(sim, target_xy):
    """Rotate the free-standing base so SO101's front (-y local) faces target."""
    import mujoco

    m, d = mj_raw(sim)
    # composed body is robot0_base: looking up "base" alone silently no-ops,
    # which left the arm facing away from the basket (unreachable transport)
    base_bid = body_id(sim, "base")
    if base_bid < 0:
        print("    [aim] base body not found; arm not aimed")
        return
    base_pos = np.array(d.xpos[base_bid])[:2]
    dir_xy = np.asarray(target_xy) - base_pos
    # front = -y_local; after R(z,θ) front becomes (sinθ, -cosθ)
    theta = np.arctan2(dir_xy[0], -dir_xy[1])
    half = theta / 2.0
    m.body_quat[base_bid] = [np.cos(half), 0.0, 0.0, np.sin(half)]
    mujoco.mj_forward(m, d)
    print(f"    base aimed at {np.round(target_xy, 3)} (yaw {np.degrees(theta):.0f} deg)")


def _grip_action(open_flag, sim):
    """Action-space value (0-100) for the jaw: open = range max, closed = GRIP_CLOSED_Q."""
    import mujoco

    m, _ = mj_raw(sim)
    gj = [j for j in range(m.njnt) if "gripper" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")][0]
    act = [a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == gj][0]
    lo, hi = m.actuator_ctrlrange[act]
    q = hi if open_flag > 0.5 else GRIP_CLOSED_Q
    return 100.0 * (float(q) - float(lo)) / (float(hi) - float(lo))


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

    # Official scene, untouched: move OUR mount and face the bisector of the two
    # targets so both stay in front and within reach (a fixed-yaw base cannot
    # cover an object/basket pair that straddles it).
    base_now = np.array(sim.data.xpos[body_id(sim, "base")])[:2]
    target_base = np.array(BASE_XY) if BASE_XY is not None else base_now
    if BASE_XY is not None:
        place_base_at(sim, target_base)
    v1 = p_obj[:2] - target_base
    v2 = p_basket[:2] - target_base
    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
        u = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
        if np.linalg.norm(u) > 1e-6:
            u = u / np.linalg.norm(u)
            aim_base_at(sim, target_base + 0.2 * u)
    _bb = body_id(sim, "base")
    print(f"    [plan-base] xpos={np.round(np.array(sim.data.xpos[_bb]),3)} "
          f"quat={np.round(np.array(sim.data.xquat[_bb]),3)}")

    # IK controls the moving-jaw body directly, so targets are in jaw-center
    # coordinates relative to the object/basket centers
    waypoints = [
        ("move", p_obj + np.array([0.0, 0.0, 0.10]), 1.0, 1.5),
        ("approach", p_obj + np.array([0.0, 0.0, 0.050]), 1.0, 0.8),
        # close at the object CENTRE: staying 3.5cm above it closed on air and
        # only worked through the kinematic snap (objects then slipped)
        ("grip", p_obj + np.array([0.0, 0.0, 0.035]), 0.0, 1.5),
        ("lift", p_obj + np.array([0.0, 0.0, 0.20]), 0.0, 0.8),
        ("move", p_basket + np.array([0.0, 0.0, 0.22]), 0.0, 1.5),
        ("move", p_basket + np.array([0.0, 0.0, 0.10]), 0.0, 1.5),
        ("release", p_basket + np.array([0.0, 0.0, 0.10]), 1.0, 1.5),
        ("move", p_basket + np.array([0.0, 0.0, 0.28]), 1.0, 1.5),
    ]

    HOLD, GRIP_HOLD = 10, 70  # the GRIP servo ramps: ~60 steps to fully close
    a5_open = _grip_action(1.0, sim)
    a5_closed = _grip_action(0.0, sim)
    print(f"    [grip] open={a5_open:.1f} closed={a5_closed:.1f} closed_q={GRIP_CLOSED_Q}")
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
        # joint-space blend is fine now that the servos actually track: the
        # earlier table-drag came from 20deg of tracking error, not from the
        # interpolation itself. Cartesian sampling blew the action count up.
        tgt_deg = np.degrees(q_sol) - SO101_CALIB_OFFSETS
        cur_deg = np.degrees(q) - SO101_CALIB_OFFSETS
        dist = float(np.max(np.abs(tgt_deg - cur_deg)))
        n = max(4, int(dist / max_step_deg))
        for k in range(1, n + 1):
            t = k / n
            a5 = (1 - t) * cur_deg + t * tgt_deg
            actions.append(np.concatenate([a5, [a5_open if grip_open > 0.5 else a5_closed]]))
        q_prev = q_sol
        tgt_deg = np.degrees(q_sol) - SO101_CALIB_OFFSETS
        hold = GRIP_HOLD if phase in ("grip", "release") else HOLD
        for _ in range(hold):
            actions.append(np.concatenate([tgt_deg, [a5_open if grip_open > 0.5 else a5_closed]]))
        checkpoints[len(actions)] = phase
        q = q_sol
    return actions, checkpoints


def _has_body(sim, name):
    import mujoco

    m = mj_raw(sim)[0]
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) >= 0


def run_episode(env, ik=None, max_steps=900):
    obs = env.reset()
    # env.reset() swaps the live MjModel/MjData; an ArmIK built before the reset
    # keeps the stale buffers, so its FK disagrees with the simulator (the plan
    # lands 15cm away from where the same joints actually go).
    ik = ArmIK(env.env.sim)
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
    if obj_name and SNAP_GRASP:
        from run_libero_eval import SnapGraspController

        ctrl = SnapGraspController(sim, domain.objects_dict[obj_name].root_body)
    else:
        ctrl = None

    _bb = body_id(sim, "base")
    print(f"    [exec-base] xpos={np.round(np.array(sim.data.xpos[_bb]),3)} "
          f"quat={np.round(np.array(sim.data.xquat[_bb]),3)}")
    obs_list, act_list = [], []
    success = False
    success_streak = 0
    max_streak = 0
    total_reward = 0.0
    for step_idx, act6 in enumerate(actions):
        obs_list.append({
            **({"camera1": np.asarray(obs["birdview_image"]).copy()}
               if "birdview_image" in obs else {}),
            **({"camera2": np.asarray(obs["robot0_eye_in_hand_image"]).copy()}
               if "robot0_eye_in_hand_image" in obs else {}),
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
        max_streak = max(max_streak, success_streak)
        if (step_idx + 1) in checkpoints and obj_bid >= 0:
            label = checkpoints[step_idx + 1]
            p_obj = np.array(sim.data.xpos[obj_bid])
            try:
                import mujoco as _mj
                _m, _d = mj_raw(sim)
                _sid, _ = __import__("run_libero_eval").find_grip_site_id(_mj, _m)
                tip_now = np.array(_d.site_xpos[_sid])
            except Exception:
                tip_now = None
            q_now = np.degrees(np.array([sim.data.qpos[a] for a in ik.arm_qadr]))
            q_cmd = np.array(act6[:5]) + SO101_CALIB_OFFSETS
            print(f"    [q@{label}] cmd={np.round(q_cmd,1)} now={np.round(q_now,1)}")
            _gj = [j for j in range(m_raw.njnt) if "gripper" in (mujoco.mj_id2name(m_raw, mujoco.mjtObj.mjOBJ_JOINT, j) or "")][0]
            _jaw = float(sim.data.qpos[sim.model.jnt_qposadr[_gj]]) if hasattr(sim.model, "jnt_qposadr") else float(d_raw.qpos[m_raw.jnt_qposadr[_gj]])
            print(f"    [{label}@{step_idx + 1}] obj_pos={np.round(p_obj, 3)} jaw={_jaw:+.2f} "
                  f"tip={np.round(tip_now, 3) if tip_now is not None else None}")
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
    if obj_bid >= 0:
        p_obj = np.array(sim.data.xpos[obj_bid])
        print(f"    [diag] max_success_streak={max_streak}/10 obj_final={np.round(p_obj,3)}")
    return success, len(act_list), total_reward, obs_list, act_list


def _import_lerobot_dataset():
    """Import LeRobotDataset, auto-installing the lerobot[dataset] extra on ImportError."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        import subprocess
        import sys

        print("[save] missing lerobot dataset deps; pip installing lerobot[dataset] ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "lerobot[dataset]"],
            check=True,
        )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset


def save_dataset(episodes, out_dir, task_name):
    """Save episodes as a LeRobot v3 dataset (one dataset per suite run)."""
    import shutil
    from pathlib import Path

    LeRobotDataset = _import_lerobot_dataset()

    if Path(out_dir).exists():
        print(f"[save] removing stale dataset dir {out_dir}")
        shutil.rmtree(out_dir)

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
    parser.add_argument("--base_xy", nargs=2, type=float, default=None,
                        help="place robot0_base at this world xy (scene untouched)")
    parser.add_argument("--snap", action="store_true",
                        help="kinematic attach (cheating: teleports the object). Off by default.")
    parser.add_argument("--smoke", action="store_true",
                        help="L1 gate: 1 trial, 1 camera, no dataset write")
    args = parser.parse_args()
    global BASE_XY, SNAP_GRASP
    BASE_XY = tuple(args.base_xy) if args.base_xy else None
    SNAP_GRASP = bool(args.snap)
    if args.smoke:
        # smoke = gate mode: run the requested number of trials, write nothing
        print(f"[smoke] {args.episodes_per_task} trial(s), 1 camera, no dataset write")

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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_root_suite = out_dir / args.suite  # created by LeRobotDataset.create

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        print(f"\n=== Task {task_id}: {task.name} ===")

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            robots=["SO101"],
            controller=ctrl_path,
            camera_names=["agentview"] if args.smoke else list(CAMERA_MAP.values()),
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

    if args.smoke:
        print(f"[smoke] trials={len(all_episodes)} collected; dataset NOT written")
        return
    if all_episodes:
        save_dataset(all_episodes, out_root_suite, args.suite)
    print(f"\nTotal episodes collected: {len(all_episodes)}")


if __name__ == "__main__":
    main()
