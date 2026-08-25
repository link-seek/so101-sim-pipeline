import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

from robosuite.utils.mjcf_utils import xml_path_completion


class MountedSO101(ManipulatorModel):
    """SO-ARM101: 5-DoF low-cost robotic arm by TheRobotCompany.

    Single-arm robot with 5 revolute joints (STS3215 servos) + 1-DoF gripper.
    Designed for tabletop pick-and-place tasks.

    Args:
        idn (int or str): Robot instance identifier
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion("robots/so101/robot.xml"),
            idn=idn,
        )

        self.set_joint_attribute(
            attrib="damping",
            values=np.array((0.60, 0.60, 0.60, 0.60, 0.60)),
        )

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "SO101Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_so101"}

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.3

    @property
    def arm_type(self):
        return "single"

    @property
    def _eef_name(self):
        return {"right": "right_hand"}
