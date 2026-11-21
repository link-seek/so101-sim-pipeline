"""Drive SO101 with the official MuJoCo position servos.

The official SO-ARM100 model (TheRobotStudio/SO-ARM100, Simulation/SO101/
so101_new_calib.xml) actuates every joint with

    <position kp="998.22" kv="2.731" forcerange="-3.35 3.35" ctrlrange="..."/>

i.e. a real servo loop evaluated at the physics rate. robosuite controllers,
however, assume <motor> torque actuators and write a TORQUE into data.ctrl;
with a position actuator that slot is a joint TARGET, so the stock controller
would command absurd values.

This subclass keeps JOINT_POSITION semantics but, for position-actuated models,
writes the goal posture instead of a computed torque and lets MuJoCo's servo do
the work. Models with <motor> actuators are untouched.
"""
import numpy as np

from robosuite.controllers.parts.generic.joint_pos import JointPositionController


class PositionServoJointController(JointPositionController):
    """JOINT_POSITION that is correct for both motor and position actuators."""

    def _position_actuated(self):
        """True if THIS arm's joints are driven by MuJoCo position servos."""
        model = self.sim.model
        bias = getattr(model, "actuator_biastype", None)
        trn = getattr(model, "actuator_trnid", None)
        if bias is None or trn is None:
            raw = getattr(model, "_model", model)
            bias = getattr(raw, "actuator_biastype", None)
            trn = getattr(raw, "actuator_trnid", None)
        if bias is None or trn is None:
            return False
        try:
            import mujoco

            affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
        except Exception:
            affine = 1
        bias = np.asarray(bias).reshape(-1)
        joint_of_act = np.asarray(trn).reshape(-1, 2)[:, 0]
        mine = np.isin(joint_of_act, np.asarray(self.joint_indexes["joints"]))
        if not bool(mine.any()):
            return False
        return bool(np.all(bias[mine] == affine))

    def run_controller(self):
        if not self._position_actuated():
            return super().run_controller()
        if self.goal_qpos is None:
            self.set_goal(np.zeros(self.control_dim))
        self.update()
        desired = np.array(self.goal_qpos, dtype=np.float64)
        if self.interpolator is not None and self.interpolator.order == 1:
            desired = np.array(self.interpolator.get_interpolated_goal(), dtype=np.float64)
        # robosuite writes run_controller()'s return value into data.ctrl
        self.torques = desired
        self.new_update = True
        return desired


def install():
    """Route JOINT_POSITION through the servo-aware implementation."""
    from robosuite.controllers.parts import generic

    generic.JointPositionController = PositionServoJointController
