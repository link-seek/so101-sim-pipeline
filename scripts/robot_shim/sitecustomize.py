"""Eval compat shim (Ch6/Ch7: zero-image-change fixes).

Loaded automatically as `sitecustomize` when its directory is on PYTHONPATH
(e.g. `-e PYTHONPATH=/data/scripts/robot_shim` on the outer `docker run`;
vla-eval runs benchmarks in-process inside that container).

1. Robot switch (Ch7): reads LIBERO_ROBOT (e.g. "Sawyer") and injects it as
   the default `robots` arg for LIBERO env creation. The vla-eval LIBERO
   benchmark hardcodes Panda with no config-level switch, so this restores
   "change config, change robot" with zero image change.
   Enabled only when LIBERO_ROBOT is set.

2. Seed compat (Ch6): LIBERO calls `env.seed(s)`, but robosuite>=1.5 stores
   `seed` as a plain attribute (no `seed()` method), raising
   `TypeError: 'NoneType' object is not callable`. This patch makes
   `ControlEnv.seed` version-tolerant: call it when callable (<=1.4),
   otherwise refresh `env.rng` (1.5+).
   Always applied; inert when LIBERO is absent.
"""

import os

try:
    from libero.libero.envs import env_wrapper as _ew

    # --- 2. seed compat: patch ControlEnv.seed ---
    _orig_seed = _ew.ControlEnv.seed

    def _compat_seed(self, seed=None):
        fn = getattr(self.env, "seed", None)
        if callable(fn):
            return fn(seed)
        # robosuite>=1.5: `seed` is a plain attribute, not a method.
        try:
            self.env.seed = seed
        except Exception:
            pass
        try:
            import numpy as np

            self.env.rng = np.random.default_rng(seed)
        except Exception:
            pass
        return [seed] if seed is not None else []

    _ew.ControlEnv.seed = _compat_seed

    # --- 1. robot switch: patch ControlEnv.__init__ defaults ---
    _ROBOT = os.environ.get("LIBERO_ROBOT", "").strip()
    if _ROBOT:
        _orig_init = _ew.ControlEnv.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs.setdefault("robots", [_ROBOT])
            _orig_init(self, *args, **kwargs)

        _ew.ControlEnv.__init__ = _patched_init
        print(f"[eval-shim] LIBERO default robot overridden to: {_ROBOT}")
except Exception as _e:
    print(f"[eval-shim] WARNING: shim inactive ({_e})")
