"""Controller configuration for SO101 robot.

JOINT_POSITION controller for 5-DoF arm + 1-DoF gripper.
"""
import numpy as np

controller_config = {
    "type": "JOINT_POSITION",
    "input_max": 1.0,
    "input_min": -1.0,
    "output_max": 0.5,
    "output_min": -0.5,
    "kp": 50.0,
    "damping": 1.0,
    "velocity_limits": [-1.0, 1.0],
    "ramp_ratio": 0.2,
    "interpolation": "linear",
}
