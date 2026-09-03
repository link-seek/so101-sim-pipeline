"""Robot-switch shim (loaded via sitecustomize on PYTHONPATH).

- Default (LIBERO_ROBOT unset, e.g. Franka/Panda baseline): pure no-op,
  zero imports, zero behavior change vs. vanilla image.
- Override (LIBERO_ROBOT=Sawyer, ...): patch LIBERO's OffScreenRenderEnv
  default robot list + robosuite seed compat, without rebuilding the image.
"""
import os

_TARGET = os.environ.get("LIBERO_ROBOT", "").strip()
if not _TARGET:
    print("[eval-shim] passthrough (LIBERO_ROBOT unset)", flush=True)
else:
    try:
        from libero.libero.envs import env_wrapper
        import inspect

        _orig_init = env_wrapper.ControlEnv.__init__
        _sig = inspect.signature(_orig_init)
        _target = _TARGET

        def _patched_init(self, *args, **kwargs):
            if "robots" in _sig.parameters and "robots" not in kwargs:
                kwargs["robots"] = [_target]
                print(f"[eval-shim] robots -> [{_target}]", flush=True)
            _orig_init(self, *args, **kwargs)

        env_wrapper.ControlEnv.__init__ = _patched_init

        def _seed_compat(self, seed=None):
            base = getattr(super(env_wrapper.ControlEnv, self), "seed", None)
            if callable(base):
                return base(seed)
            return None

        _orig_seed = getattr(env_wrapper.ControlEnv, "seed", None)
        if not callable(_orig_seed):
            env_wrapper.ControlEnv.seed = _seed_compat
            print("[eval-shim] seed compat installed", flush=True)
        print(f"[eval-shim] active for robot={_target}", flush=True)
    except Exception as e:
        print(f"[eval-shim] WARNING: override inactive ({e})", flush=True)
