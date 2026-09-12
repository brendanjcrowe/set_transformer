"""Load a sibling module of this directory BY PATH, never by flat name.

Gap 12, and it is not hypothetical. experiments/odd_even/ and
experiments/ant_tag/ both contain `variants.py`, `2_collect_pf_dataset.py` and
`4_train_rl_cgf.py`. A plain `import variants` resolves off sys.path and picks
whichever of the two directories comes first -- and worse, off sys.modules,
which is a process-wide cache. If ANYTHING has already imported the Ant-Tag
`variants` (a shared launcher, a diagnostics script, or a test module that
inserts experiments/ant_tag/ at import time, which tests/test_st_pretrained_
load.py does), then every later `import variants` in this directory silently
returns the Ant-Tag registry. The symptom is
`ValueError: Unknown variant 'oe50'. Available: ['base', 'cdens', ...]`, which
names the wrong file only if you read the traceback carefully. It was measured
happening in the real test suite, not imagined.

A script's own directory is normally sys.path[0], so a sibling import usually
resolves correctly -- which is exactly what makes this dangerous: it works
until something else loads first, and then it is wrong rather than absent.

So sibling modules are loaded here with an explicit path and a
directory-qualified module key, which no other directory can collide with.
The digit-leading pipeline modules (`4_train_rl_cgf`) still go through
importlib.import_module in the arms that need them: those are only ever
imported by a script whose own directory is sys.path[0], and they are covered
by the same key convention when a test loads them.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: Package name the siblings are loaded UNDER, so their module keys are
#: unambiguous across the two experiment directories.
#:
#: (2026-09-12: the wrappers and the factory now live in the package,
#: `set_transformer.rl.domains.odd_even`, which a child imports by its real
#: name, so the paragraph below no longer bites for them; the mechanism stays
#: for the modules still loaded here, e.g. `4_train_rl_cgf`.)
#: It must be a name a FRESH PROCESS can import, not just a synthetic
#: sys.modules key. SubprocVecEnv cloudpickles the env-factory closures, and
#: cloudpickle serializes a class defined in an importable module BY
#: REFERENCE -- so the child re-imports
#: `_odd_even_experiments.odd_even_belief_env` to unpickle
#: StepIndexObservationWrapper. With a bare spec_from_file_location key that
#: import fails in the child with `PicklingError: ... import of module ...
#: failed`, and only at n_envs > 1, which is the default. So a real package
#: module with a __path__ is installed in sys.modules below, and the child
#: (which inherits sys.path through the forkserver) resolves the submodule
#: through it.
_PACKAGE = "_odd_even_experiments"


def _package() -> types.ModuleType:
    """The synthetic package the siblings live under, with a real __path__."""
    existing = sys.modules.get(_PACKAGE)
    if existing is not None:
        return existing
    package = types.ModuleType(_PACKAGE)
    package.__doc__ = (
        "Synthetic package for experiments/odd_even's sibling modules. "
        "Exists so their module keys cannot collide with the same-named "
        "files in experiments/ant_tag/, while staying importable in a "
        "SubprocVecEnv child process.")
    # __path__ is what makes `import _odd_even_experiments.<name>` work: the
    # standard machinery searches these directories for the submodule.
    package.__path__ = [str(_HERE)]
    sys.modules[_PACKAGE] = package
    return package


def load(name: str):
    """Import experiments/odd_even/<name>.py under the synthetic package."""
    _package()
    key = f"{_PACKAGE}.{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    path = _HERE / f"{name}.py"
    if not path.exists():
        raise ModuleNotFoundError(f"{path} does not exist")
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so a circular sibling import sees the partial
    # module rather than re-executing the file.
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    setattr(_package(), name, module)
    return module
