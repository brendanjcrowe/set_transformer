"""Per-domain pieces of the RL harness: variant registries, env wrappers and factories,
curriculum declarations, particle-filter glue, and one ``Domain`` record per problem.

One module per domain (``ant_tag``, ``car_flag``, ``hunt``, ``odd_even``). These modules import no plotting backend
and load without MuJoCo (``import pdomains`` only registers env ids); the ``Domain`` record
at the bottom of each (``ANT_TAG``, ``CAR_FLAG``, ``HUNT``, ``ODD_EVEN``) is what ``rl/train.py`` reads. Look a
record up by name with :func:`get`; the module is imported on first use, so asking for
``odd_even`` never imports the Ant-Tag particle filters.

Adding a problem: ``rl/domains/<name>.py`` ending in ``<NAME> = Domain(...)`` (see
``base.py``), plus one entry in :data:`DOMAIN_NAMES` below.
"""

from __future__ import annotations

import importlib

from set_transformer.rl.domains.base import Domain

#: ``--domain`` choices: domain name -> (module, record attribute).
DOMAIN_NAMES: dict[str, tuple[str, str]] = {
    "ant_tag": ("set_transformer.rl.domains.ant_tag", "ANT_TAG"),
    "car_flag": ("set_transformer.rl.domains.car_flag", "CAR_FLAG"),
    "hunt": ("set_transformer.rl.domains.hunt", "HUNT"),
    "odd_even": ("set_transformer.rl.domains.odd_even", "ODD_EVEN"),
}


def get(name: str | Domain) -> Domain:
    """The :class:`Domain` record for ``name`` (a record is returned as is)."""
    if isinstance(name, Domain):
        return name
    try:
        module_name, attribute = DOMAIN_NAMES[name]
    except KeyError:
        raise ValueError(f"Unknown domain {name!r}. Available: {sorted(DOMAIN_NAMES)}") from None
    return getattr(importlib.import_module(module_name), attribute)


def domain_of_variant(variant: str, env_id: str | None = None) -> Domain:
    """The Domain whose registry has ``variant`` (and, when given, registers it under
    ``env_id``). For scripts that start from a dataset rather than ``--domain``: the
    collector records the variant key and env id in the dataset's metadata."""
    matches = []
    for name in DOMAIN_NAMES:
        domain = get(name)
        if variant in domain.variants and (
                env_id is None or domain.resolve(variant).env_id == env_id):
            matches.append(domain)
    if len(matches) != 1:
        raise ValueError(
            f"variant {variant!r}" + (f" with env id {env_id!r}" if env_id else "")
            + f" belongs to {len(matches)} domains (" + ", ".join(d.name for d in matches)
            + f"); the choices are {sorted(DOMAIN_NAMES)}")
    return matches[0]


__all__ = ["DOMAIN_NAMES", "Domain", "domain_of_variant", "get"]
