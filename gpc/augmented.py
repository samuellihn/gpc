from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from hydrax.alg_base import (
    SamplingBasedController,
    SamplingParams,
    Trajectory,
)


@dataclass
class PACParams(SamplingParams):
    """Parameters for the policy-augmented controller.

    Attributes:
        tk: Knot times for the control spline.
        mean: Mean spline knots μ.
        rng: PRNG key.
        policy_samples: Knot sequences from the generative policy.
    """

    policy_samples: jax.Array


class PolicyAugmentedController(SamplingBasedController):
    """SPC variant where rollout samples include a learned policy."""

    def __init__(
        self,
        base_ctrl: SamplingBasedController,
        num_policy_samples: int,
        mpc_seed: int = 0,
    ) -> None:
        """Initialize the policy-augmented controller.

        Args:
            base_ctrl: The base controller to augment.
            num_policy_samples: The number of samples to draw from the policy.
            mpc_seed: Random seed for domain-randomized dynamics. Must match
                ``base_ctrl`` if ``num_randomizations > 1``.
        """
        self.base_ctrl = base_ctrl
        self.num_policy_samples = num_policy_samples
        super().__init__(
            base_ctrl.task,
            num_randomizations=base_ctrl.num_randomizations,
            risk_strategy=base_ctrl.risk_strategy,
            seed=mpc_seed,
            plan_horizon=base_ctrl.plan_horizon,
            spline_type=base_ctrl.spline_type,
            num_knots=base_ctrl.num_knots,
            iterations=base_ctrl.iterations,
        )

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> PACParams:
        """Initialize the controller parameters."""
        base = self.base_ctrl.init_params(initial_knots, seed)
        policy_samples = jnp.zeros(
            (
                self.num_policy_samples,
                self.num_knots,
                self.task.model.nu,
            )
        )
        return PACParams(
            tk=base.tk,
            mean=base.mean,
            rng=base.rng,
            policy_samples=policy_samples,
        )

    def sample_knots(self, params: PACParams) -> Tuple[jax.Array, PACParams]:
        """Sample spline knots from the base controller and the policy."""
        base_sp = SamplingParams(
            tk=params.tk, mean=params.mean, rng=params.rng
        )
        knots_base, base_out = self.base_ctrl.sample_knots(base_sp)
        knots = jnp.concatenate([knots_base, params.policy_samples], axis=0)
        return knots, params.replace(rng=base_out.rng)

    def update_params(
        self, params: PACParams, rollouts: Trajectory
    ) -> PACParams:
        """Update parameters using the base controller's rule."""
        base_sp = SamplingParams(tk=params.tk, mean=params.mean, rng=params.rng)
        base_out = self.base_ctrl.update_params(base_sp, rollouts)
        return params.replace(mean=base_out.mean, rng=base_out.rng)

    def get_action(self, params: PACParams, t: jax.Array) -> jax.Array:
        """Get the control action at time ``t`` from the spline mean."""
        base_sp = SamplingParams(tk=params.tk, mean=params.mean, rng=params.rng)
        return self.base_ctrl.get_action(base_sp, t)

    def get_action_sequence(self, params: PACParams) -> jax.Array:
        """Dense open-loop controls from the current mean spline."""
        tk = params.tk
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        return self.interp_func(tq, tk, params.mean[None, ...])[0]
