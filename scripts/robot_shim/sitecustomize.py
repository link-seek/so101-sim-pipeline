"""Robot-switch shim (loaded via sitecustomize on PYTHONPATH).

Design constraints (learned the hard way):
- NEVER print to stdout: subprocesses (e.g. glfw's version check) eval()
  captured stdout, one stray print breaks every episode. Diagnostics -> stderr.
- NEVER import heavy libs (libero/robosuite/mujoco) at startup: every python
  process pays it, pipe buffers flood, spawn storms multiply warnings.
  Instead wrap __import__ and patch libero lazily on first use.

Patches (applied once, in whichever process imports libero first):
- seed compat (robosuite>=1.5 made ControlEnv.seed non-callable): always.
- robot override (LIBERO_ROBOT=Sawyer, ...): only when the env var is set.
"""
import builtins
import os
import sys

_TARGET_MOD = "libero.libero.envs.env_wrapper"
_PATCHED = "_eval_shim_patched"
_PATCHING = False


def _log(msg):
    try:
        sys.stderr.write(f"[eval-shim] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _apply(mod):
    if getattr(mod, _PATCHED, False):
        return
    # Mark first: _apply imports robosuite, which re-enters the __import__
    # wrapper below. Without the early mark + _PATCHING guard this recurses
    # until RecursionError and the patch is skipped.
    setattr(mod, _PATCHED, True)
    try:
        cls = mod.ControlEnv
    except AttributeError:
        return
    # Level 1 (LIBERO wrapper): ControlEnv.seed is normally a bound method,
    # keep this as a safety net only.
    if not callable(getattr(cls, "seed", None)):
        def _seed_compat(self, seed=None):
            base = getattr(super(cls, self), "seed", None)
            if callable(base):
                return base(seed)
            return None
        cls.seed = _seed_compat
        _log("seed compat installed (ControlEnv level)")
    # Level 2 (robosuite>=1.5): MujocoEnv.seed is a None placeholder
    # (era-appropriate robosuite<1.5 had a real seed() method). Patch the
    # base class so LIBERO's self.env.seed(seed) works again.
    try:
        from robosuite.environments import base as _rb_base
        _mj = _rb_base.MujocoEnv
        if not callable(getattr(_mj, "seed", None)):
            def _rs_seed(self, seed=None):
                try:
                    self._seed = seed
                    if seed is not None:
                        import numpy as _np
                        import random as _rd
                        _np.random.seed(seed % (2 ** 32))
                        _rd.seed(seed)
                except Exception:
                    pass
            _mj.seed = _rs_seed
            _log("seed compat installed (MujocoEnv level)")
        else:
            _log("native MujocoEnv.seed callable, no patch needed")
    except Exception as e:
        _log(f"MujocoEnv patch skipped ({e})")
    target = os.environ.get("LIBERO_ROBOT", "").strip()
    if target:
        import inspect
        orig_init = cls.__init__
        sig = inspect.signature(orig_init)

        def _patched_init(self, *args, **kwargs):
            if "robots" in sig.parameters and "robots" not in kwargs:
                kwargs["robots"] = [target]
                _log(f"robots -> [{target}]")
            orig_init(self, *args, **kwargs)

        cls.__init__ = _patched_init
        _log(f"robot override active for {target}")


_real_import = builtins.__import__


def _shim_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _real_import(name, globals, locals, fromlist, level)
    global _PATCHING
    if _PATCHING:
        return mod
    try:
        cand = sys.modules.get(_TARGET_MOD)
        if cand is not None and not getattr(cand, _PATCHED, False):
            _PATCHING = True
            try:
                _apply(cand)
            finally:
                _PATCHING = False
    except Exception as e:
        _log(f"patch skipped ({e})")
    return mod


builtins.__import__ = _shim_import
