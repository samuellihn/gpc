import jax
import jax.numpy as jnp
from hydrax.algs import PredictiveSampling
from hydrax.tasks.particle import Particle
from mujoco import mjx

from gpc.augmented import PolicyAugmentedController


def test_augmented() -> None:
    """Test the prediction-augmented controller."""
    task = Particle()
    num_knots = 4
    plan_horizon = 1.0
    ps = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=0.1,
        plan_horizon=plan_horizon,
        num_knots=num_knots,
        spline_type="cubic",
    )
    opt = PolicyAugmentedController(ps, num_policy_samples=32)
    jit_opt = jax.jit(opt.optimize)

    state = task.make_data()
    state = state.replace(
        mocap_pos=state.mocap_pos.at[0, 0:2].set(jnp.array([0.01, 0.01]))
    )
    params = opt.init_params()
    params = params.replace(
        policy_samples=jnp.ones((32, num_knots, task.model.nu))
    )

    for _ in range(10):
        params, rollouts = jit_opt(state, params)

    total_costs = jnp.sum(rollouts.costs, axis=1)
    best_idx = jnp.argmin(total_costs)
    best_ctrl = rollouts.controls[best_idx]

    assert jnp.all(best_ctrl != 0.0)
    assert jnp.all(params.policy_samples == 1.0)

    U = opt.get_action_sequence(params)
    assert U.shape == (opt.ctrl_steps, task.model.nu)


if __name__ == "__main__":
    test_augmented()
