"""Per-domain pieces of the RL harness: env wrappers, curriculum, particle-filter glue.

One module per domain (``ant_tag``, later ``odd_even``). These modules import no env
package and set no plotting backend, so they load without MuJoCo; registering the envs
(``import pdomains``) is the caller's job.
"""
