"""Generic pretraining objectives: the ones any domain can use because they need nothing a
problem must compute for them (``rl/domains/base.py::Objective``). Problem-specific
objectives are declared in the problem's own domain module (``Domain.pretraining``), never
here. ``rl/pretrain.py`` offers ``--objective`` as the union of the two.

Adding a generic objective: one module here ending in an ``OBJECTIVE = Objective(...)``
record, and one entry in :data:`GENERIC` below.
"""

from __future__ import annotations

from set_transformer.rl.domains.base import Objective
from set_transformer.rl.pretrain_objectives import reconstruction

#: ``objective name -> Objective``, available on every domain.
GENERIC: dict[str, Objective] = {
    reconstruction.OBJECTIVE.name: reconstruction.OBJECTIVE,
}

#: The objective ``rl/pretrain.py`` uses when neither the flag nor the domain names one.
DEFAULT_OBJECTIVE = reconstruction.OBJECTIVE.name

__all__ = ["DEFAULT_OBJECTIVE", "GENERIC", "Objective"]
