"""LIBERO robot-switch shim (Ch7: config-level robot extension).

Loaded automatically as `sitecustomize` when its directory is on PYTHONPATH.
Reads the LIBERO_ROBOT env var (e.g. "Sawyer") and injects it as the default
`robots` argument for LIBERO env creation, so the vla-eval LIBERO benchmark
(which hardcodes Panda) runs on another robosuite robot with zero image change.

Container wiring (all in the benchmark YAML, no code change):
  docker:
    volumes: ["/data/scripts/robot_shim:/tmp/robot_shim:ro"]
    env: ["PYTHONPATH=/tmp/robot_shim", "LIBERO_ROBOT=Sawyer"]
"""

import os

try:
    _ROBOT = os.environ.get("LIBERO_ROBOT", "").strip()
    if _ROBOT:
        from libero.libero.envs import env_wrapper as _ew

        _orig_init = _ew.ControlEnv.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs.setdefault("robots", [_ROBOT])
            _orig_init(self, *args, **kwargs)

        _ew.ControlEnv.__init__ = _patched_init
        print(f"[robot-shim] LIBERO default robot overridden to: {_ROBOT}")
except Exception as _e:
    print(f"[robot-shim] WARNING: robot override inactive ({_e})")
