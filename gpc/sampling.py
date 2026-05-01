from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from hydrax.alg_base import Trajectory
from hydrax.algs.predictive_sampling import PSParams, PredictiveSampling
from mujoco import mjx

from gpc.policy import Policy


class BootstrappedPredictiveSampling(PredictiveSampling):
    """Predictive sampling augmented with generative-policy knot proposals."""

    def __init__(
        self,
        policy: Policy,
        observation_fn: Callable[[mjx.Data], jax.Array],
        num_policy_samples: int,
        warm_start_level: float = 0.0,
        inference_timestep: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize the controller.

        Args:
            policy: The generative policy to sample from.
            observation_fn: Produces an observation vector from ``mjx.Data``.
            num_policy_samples: Number of policy samples per iteration.
            warm_start_level: Flow-matching warm start in ``[0, 1]``.
            inference_timestep: Flow-matching integration step.
            **kwargs: Passed to :class:`PredictiveSampling`.
        """
        self.observation_fn = observation_fn
        self.policy = policy.replace(dt=inference_timestep)
        self.policy.model.eval()
        self.warm_start_level = jnp.clip(warm_start_level, 0.0, 1.0)
        self.num_policy_samples = num_policy_samples

        super().__init__(**kwargs)

    def optimize(self, state: mjx.Data, params: PSParams) -> Tuple[PSParams, Trajectory]:
        """Sample PS knots, add policy knots, roll out, and update."""
        tk = params.tk
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)

        def _optimize_scan_body(
            scan_params: PSParams, _: Any
        ) -> Tuple[PSParams, Trajectory]:
            knots_ps, scan_params = PredictiveSampling.sample_knots(
                self, scan_params
            )
            knots_ps = jnp.clip(
                knots_ps, self.task.u_min, self.task.u_max
            )

            rng, policy_rng, dr_rng = jax.random.split(scan_params.rng, 3)
            state_fwd = mjx.forward(self.task.model, state)
            y = self.observation_fn(state_fwd)
            policy_rngs = jax.random.split(policy_rng, self.num_policy_samples)
            policy_knots = jax.vmap(
                self.policy.apply, in_axes=(None, None, 0, None)
            )(
                scan_params.mean,
                y,
                policy_rngs,
                self.warm_start_level,
            )
            policy_knots = jnp.clip(
                policy_knots, self.task.u_min, self.task.u_max
            )
            knots = jnp.concatenate([knots_ps, policy_knots], axis=0)

            rollouts = self.rollout_with_randomizations(
                state, new_tk, knots, dr_rng
            )
            scan_params = scan_params.replace(rng=rng)
            scan_params = self.update_params(scan_params, rollouts)
            return scan_params, rollouts

        params, rollouts = jax.lax.scan(
            f=_optimize_scan_body,
            init=params,
            xs=jnp.arange(self.iterations),
        )

        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)
        return params, rollouts_final
