"""Robot-switch shim (loaded via sitecustomize on PYTHONPATH).

CRITICAL: never write to stdout - subprocesses (e.g. glfw's version check)
eval() captured stdout, so a single stray print breaks every episode.
All diagnostics go to stderr.

- Seed compat (robosuite>=1.5 made ControlEnv.seed a non-callable attr):
  applied unconditionally, required by Franka and Sawyer alike.
- Robot override (LIBERO_ROBOT=Sawyer, ...): patch LIBERO's
  OffScreenRenderEnv default robot list without rebuilding the image.
"""
import os
import sys

def _log(msg):
    sys.stderr.write(f"[eval-shim] {msg}\n")
    sys.stderr.flush()

try:
    from libero.libero.envs import env_wrapper

    _orig_seed = getattr(env_wrapper.ControlEnv, "seed", None)
    if not callable(_orig_seed):
        def _seed_compat(self, seed=None):
            base = getattr(super(env_wrapper.ControlEnv, self), "seed", None)
            if callable(base):
                return base(seed)
            return None
        env_wrapper.ControlEnv.seed = _seed_compat
        _log("seed compat installed")
    else:
        _log("native seed callable, no patch needed")

    _target = os.environ.get("LIBERO_ROBOT", "").strip()
    if _target:
        import inspect
        _orig_init = env_wrapper.ControlEnv.__init__
        _sig = inspect.signature(_orig_init)

        def _patched_init(self, *args, **kwargs):
            if "robots" in _sig.parameters and "robots" not in kwargs:
                kwargs["robots"] = [_target]
                _log(f"robots -> [{_target}]")
            _orig_init(self, *args, **kwargs)

        env_wrapper.ControlEnv.__init__ = _patched_init
        _log(f"robot override active for {_target}")
    else:
        _log("passthrough (LIBERO_ROBOT unset)")
except Exception as e:
    _log(f"WARNING inactive ({e})")
