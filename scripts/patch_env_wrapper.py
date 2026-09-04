"""Fetch-time patch for LIBERO env_wrapper.py (runs on the GPU host).

Replaces ControlEnv.seed (robosuite<=1.4 API: self.env.seed(seed) as a
method call) with the robosuite>=1.5 equivalent (seed is a ctor attr +
numpy default_rng). Optionally rewrites the robots=["Panda"] default for
robot-switch runs. Fails loud if upstream anchors drift.
"""
import re
import sys

path = "/data/scripts/env_wrapper.py"
robot = sys.argv[1] if len(sys.argv) > 1 else None

src = open(path).read()
anchor = re.compile(r"def seed\(self, seed\):\s*\n\s*self\.env\.seed\(seed\)")
repl = (
    "def seed(self, seed):\n"
    "        # _rs15_seed_compat: robosuite>=1.5 made seed a ctor attr,\n"
    "        # not a method. Mirror MujocoEnv.__init__ seeding instead.\n"
    "        self.env.seed = seed\n"
    "        try:\n"
    "            import numpy as _np\n"
    "            self.env.rng = _np.random.default_rng(seed)\n"
    "        except Exception:\n"
    "            pass"
)
src, n = anchor.subn(repl, src)
assert n == 1, f"seed anchor matched {n}x"
if robot:
    src, m = re.subn(r'robots=\["Panda"\]', 'robots=["%s"]' % robot, src)
    assert m >= 1, "robots default not found"
    print("robots default -> %s" % robot, flush=True)
assert "_rs15_seed_compat" in src
open(path, "w").write(src)
print("env_wrapper patched OK", flush=True)
