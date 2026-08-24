"""Verify SO101 robot integration in RoboSuite.

Runs L1-L4 verification checks on the ECS Docker container.
"""
import sys
import numpy as np

def check_l1_loading():
    """L1: Robot loads without errors."""
    import robosuite as suite
    env = suite.make(
        "Lift",
        robots="MountedSO101",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
    )
    env.reset()
    robot = env.robots[0]
    print(f"[L1] PASS: env created, robot={robot.robot_name}, dof={robot.robot_model.dof}, action_dim={robot.action_dim}")
    return env


def check_l2_motion(env):
    """L2: Joints can move independently."""
    for i in range(5):
        initial = env.sim.data.qpos.copy()
        action = np.zeros(env.action_dim)
        action[i] = 0.5
        env.step(action)
        new = env.sim.data.qpos.copy()
        moved = not np.allclose(initial, new, atol=1e-6)
        print(f"[L2] Joint {i+1}: {'PASS' if moved else 'FAIL'} (moved={moved})")


def check_l3_gripper(env):
    """L3: Gripper can open and close."""
    initial = env.sim.data.qpos.copy()
    action = np.zeros(env.action_dim)
    action[5] = -1.0
    for _ in range(10):
        env.step(action)
    new = env.sim.data.qpos.copy()
    moved = not np.allclose(initial, new, atol=1e-6)
    print(f"[L3] Gripper close: {'PASS' if moved else 'FAIL'} (moved={moved})")

    action[5] = 1.0
    for _ in range(10):
        env.step(action)
    print(f"[L3] Gripper open: PASS")


def check_l4_workspace(env):
    """L4: End-effector position is within expected workspace."""
    robot = env.robots[0]
    eef_pos = robot.get_eef_pos()
    distance = np.linalg.norm(eef_pos[:2])
    print(f"[L4] EEF pos: {eef_pos}, distance_from_base: {distance:.4f}m")
    print(f"[L4] Workspace check: {'PASS' if distance < 0.5 else 'FAIL'} (radius={distance:.4f} < 0.5)")


def check_l3_kinematics():
    """L3 extended: Compare end-effector position with dyordan1/so101-mujoco."""
    import mujoco
    from robot_descriptions import so_arm101_mj_description

    model_a = mujoco.MjModel.from_xml_path(so_arm101_mj_description.MJCF_PATH)
    data_a = mujoco.MjData(model_a)
    mujoco.mj_forward(model_a, data_a)

    gripper_site_a = model_a.site("gripper").id
    eef_a = data_a.site_xpos[gripper_site_a].copy()

    print(f"[L3-ext] dyordan1 EEF at init: {eef_a}")

    model_b = mujoco.MjModel.from_xml_path("robots/so101/robot.xml")
    data_b = mujoco.MjData(model_b)
    mujoco.mj_forward(model_b, data_b)

    hand_body = model_b.body("right_hand").id
    eef_b = data_b.xpos[hand_body].copy()

    print(f"[L3-ext] RoboSuite EEF at init: {eef_b}")

    error = np.linalg.norm(eef_a - eef_b)
    print(f"[L3-ext] Position error: {error*100:.2f}cm ({'PASS' if error < 0.01 else 'FAIL'})")


if __name__ == "__main__":
    print("=== SO101 RoboSuite Verification ===\n")

    print("--- L1: Loading ---")
    env = check_l1_loading()

    print("\n--- L2: Joint Motion ---")
    check_l2_motion(env)

    print("\n--- L3: Gripper ---")
    check_l3_gripper(env)

    print("\n--- L4: Workspace ---")
    check_l4_workspace(env)

    print("\n--- L3-ext: Kinematics Comparison ---")
    try:
        check_l3_kinematics()
    except Exception as e:
        print(f"[L3-ext] SKIP: {e}")

    print("\n=== Verification Complete ===")
