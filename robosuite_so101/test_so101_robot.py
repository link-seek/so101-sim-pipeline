"""Unit tests for SO101 RoboSuite integration.

Tests:
    L1: Robot and gripper can be loaded without errors
    L2: Joints can move independently
    L3: Gripper can open/close
    L4: End-effector position matches expected workspace
"""

import numpy as np
import pytest

import robosuite as suite


@pytest.fixture
def env():
    """Create a Lift environment with SO101 robot."""
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
    return env


class TestL1Loading:
    """L1: Robot loads without errors."""

    def test_env_creation(self, env):
        assert env is not None

    def test_robot_loaded(self, env):
        assert len(env.robots) == 1
        robot = env.robots[0]
        assert robot.robot_name == "MountedSO101"

    def test_joint_count(self, env):
        robot = env.robots[0]
        assert robot.robot_model.dof == 5

    def test_action_dim(self, env):
        robot = env.robots[0]
        assert robot.action_dim == 6  # 5 arm + 1 gripper


class TestL2JointMotion:
    """L2: Joints can move independently."""

    def test_joints_move(self, env):
        robot = env.robots[0]
        initial_qpos = env.sim.data.qpos.copy()

        action = np.zeros(env.action_dim)
        action[0] = 0.5
        env.step(action)

        new_qpos = env.sim.data.qpos.copy()
        assert not np.allclose(initial_qpos, new_qpos)

    def test_all_joints_move(self, env):
        for i in range(5):
            initial_qpos = env.sim.data.qpos.copy()
            action = np.zeros(env.action_dim)
            action[i] = 0.5
            env.step(action)
            new_qpos = env.sim.data.qpos.copy()
            assert not np.allclose(initial_qpos, new_qpos), f"Joint {i} did not move"


class TestL3Gripper:
    """L3: Gripper can open and close."""

    def test_gripper_closes(self, env):
        initial_qpos = env.sim.data.qpos.copy()
        action = np.zeros(env.action_dim)
        action[5] = -1.0
        for _ in range(10):
            env.step(action)
        new_qpos = env.sim.data.qpos.copy()
        assert not np.allclose(initial_qpos, new_qpos)

    def test_gripper_opens(self, env):
        action = np.zeros(env.action_dim)
        action[5] = 1.0
        for _ in range(10):
            env.step(action)
        assert env.sim.data.qpos is not None


class TestL4Workspace:
    """L4: End-effector reaches expected workspace."""

    def test_eef_position(self, env):
        robot = env.robots[0]
        eef_pos = robot.get_eef_pos()
        assert eef_pos is not None
        assert eef_pos.shape == (3,)

    def test_workspace_radius(self, env):
        robot = env.robots[0]
        eef_pos = robot.get_eef_pos()
        distance_from_base = np.linalg.norm(eef_pos[:2])
        assert distance_from_base < 0.5, "EEF outside expected workspace radius"
