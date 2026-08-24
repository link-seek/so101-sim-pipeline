import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel

from robosuite.utils.mjcf_utils import xml_path_completion


class SO101Gripper(GripperModel):
    """SO101 parallel jaw gripper.

    Single-DoF parallel jaw gripper with STS3215 servo.
    Based on the moving_jaw mechanism from SO-ARM101.

    Args:
        idn (int or str): Gripper instance identifier
    """

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion("grippers/so101_gripper.xml"),
            idn=idn,
        )

    def format_action(self, action):
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([1.0]) * self.speed * np.sign(action),
            -1.0,
            1.0,
        )
        return self.current_action

    @property
    def init_qpos(self):
        return np.array([0.0])

    @property
    def speed(self):
        return 0.2

    @property
    def _important_geoms(self):
        return {
            "right_fingerpad": ["right_fingerpad_collision", "right_fingerpad_visual"],
            "left_fingerpad": ["right_fingerpad_visual", "right_fingerpad_collision"],
        }
