"""Re-export of :mod:`set_transformer.rl.pretrained_encoder`.

The reload-after-PPO helpers are domain-independent and moved into the
package on 2026-09-10 so the Ant-Tag CGF arm can use them without importing
an Odd-Even script (Gap 12: flat module names must not cross experiment
directories). This shim keeps ``_sibling.load("pretrained_encoder")`` and
every existing import working.
"""

from set_transformer.rl.pretrained_encoder import (  # noqa: F401
    _cgf_reference_state,
    max_abs_delta,
    policy_extractors,
    reload_pretrained,
    reload_pretrained_cgf,
    verify_matches_checkpoint,
)

__all__ = ["max_abs_delta", "verify_matches_checkpoint", "reload_pretrained",
           "reload_pretrained_cgf", "policy_extractors"]
