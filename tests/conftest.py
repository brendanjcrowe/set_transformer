"""Shared pytest configuration.

Since change 5.2 of the harness centralisation every bare training / pretraining / evaluation
run writes under ONE root (`set_transformer.rl.run_records.output_root`: the parent repo's
`runs/`, overridable with the RL_BMDP_RUNS environment variable). A test that drives an entry
point without an explicit `--output_root` would otherwise leave dry-run folders in the real
root, so the variable is pointed at a per-session temporary directory here, before any test
module imports the package. Subprocess-launched scripts inherit it through `os.environ`.
"""

import os
import tempfile

_RUNS_GUARD = tempfile.mkdtemp(prefix="rl_bmdp_test_runs_")
os.environ["RL_BMDP_RUNS"] = _RUNS_GUARD
