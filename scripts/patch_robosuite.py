"""Fetch-time patch for robosuite FixedBaseRobot (runs on the GPU host).

Adds a `controller` compat property exposing ee_pos / ee_ori_mat /
use_delta (robosuite<=1.4 API) over 1.5.x composite/part controllers.
Fails loud if upstream anchors drift.
"""

COMPAT_CLS = '''
class _ControllerCompat1514:
    """robosuite<=1.4 controller API over a 1.5.x FixedBaseRobot (for vla-eval).

    ee_pos / ee_ori_mat report the *sensed* EE pose from sim sites, which is
    what the 1.4 OSC controller exposed. use_delta is accepted and propagated
    to part controllers when they support it.
    """

    def __init__(self, robot):
        self._robot = robot
        self._use_delta = True

    def _arm(self):
        arms = list(getattr(self._robot, "arms", ["right"]) or ["right"])
        return arms[0]

    def _site_id(self):
        return self._robot.eef_site_id[self._arm()]

    @property
    def ee_pos(self):
        try:
            return np.asarray(
                self._robot.sim.data.site_xpos[self._site_id()], dtype=np.float64
            ).copy()
        except Exception:
            buf = self._robot.recent_ee_pose[self._arm()][-1]
            return np.asarray(buf[:3], dtype=np.float64).copy()

    @property
    def ee_ori_mat(self):
        try:
            return np.asarray(
                self._robot.sim.data.site_xmat[self._site_id()].reshape(3, 3),
                dtype=np.float64,
            ).copy()
        except Exception:
            buf = self._robot.recent_ee_pose[self._arm()][-1]
            return np.asarray(T.quat2mat(buf[3:7]), dtype=np.float64).copy()

    @property
    def use_delta(self):
        return self._use_delta

    @use_delta.setter
    def use_delta(self, value):
        self._use_delta = value
        pcs = getattr(self._robot, "part_controllers", None) or {}
        try:
            items = list(pcs.values())
        except AttributeError:
            items = []
        for pc in items:
            if hasattr(pc, "use_delta"):
                try:
                    pc.use_delta = value
                except Exception:
                    pass


'''

PROP_DEF = '''    @property
    def controller(self):
        # _rs15_controller_compat: vla-eval (<=0.5) expects robosuite<=1.4
        # robot.controller with ee_pos / ee_ori_mat / use_delta. 1.5.x
        # replaced it with composite/part controllers; adapt here so the
        # harness file stays untouched.
        if getattr(self, "_compat_controller", None) is None:
            self._compat_controller = _ControllerCompat1514(self)
        return self._compat_controller

'''

path = "/data/scripts/fixed_base_robot.py"
src = open(path).read()

anchor_cls = "class FixedBaseRobot(Robot):"
assert src.count(anchor_cls) == 1, "FixedBaseRobot class anchor drift"
src = src.replace(anchor_cls, COMPAT_CLS + anchor_cls, 1)

anchor_meth = "    def _load_controller(self):"
assert src.count(anchor_meth) == 1, "_load_controller anchor drift"
src = src.replace(anchor_meth, PROP_DEF + anchor_meth, 1)

assert "_rs15_controller_compat" in src
open(path, "w").write(src)
print("fixed_base_robot patched OK", flush=True)
