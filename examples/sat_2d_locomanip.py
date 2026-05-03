"""Train / test GPC on fixed 2D satellite pick-and-place (imports planarsim)."""

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from flax import nnx
from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive as run_sampling
from mujoco import mjx

from gpc.architectures import DenoisingMLP
from gpc.envs import Locomanip2DEnv, build_fixed_locomanip_task
from gpc.policy import Policy
from gpc.sampling import BootstrappedPredictiveSampling
from gpc.training import train

from planarsim.sat_2d import config as sat2d_config
from planarsim.sat_2d.interactive_2d_locomanip import make_sim_step_callback
from planarsim.sat_2d.locomanip_task import PHASE_APPROACH
from planarsim.sat_2d.satellite_plant import create_satellite_plant


def _make_env_and_ctrl():
    task = build_fixed_locomanip_task()
    plan_horizon = 0.6
    num_knots = sat2d_config.MPPI_NUM_KNOTS
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=sat2d_config.MPPI_NOISE_LEVEL,
        temperature=sat2d_config.MPPI_TEMPERATURE,
        plan_horizon=plan_horizon,
        spline_type=sat2d_config.MPPI_SPLINE_TYPE,
        num_knots=num_knots,
    )
    env = Locomanip2DEnv(task, episode_length=400)
    return env, ctrl, plan_horizon, num_knots


def _test_locomanip_interactive(env: Locomanip2DEnv, policy: Policy, inference_timestep: float) -> None:
    """Interactive policy roll-out with MuJoCo phase/box callback (same as ``interactive_2d_locomanip``)."""
    rng = jax.random.key(0)
    task = env.task
    policy = policy.replace(dt=inference_timestep)
    policy.model.eval()
    jit_policy = jax.jit(partial(policy.apply, warm_start_level=1.0))

    start_q = sat2d_config.START_Q.copy()
    goal_q = sat2d_config.GOAL_Q.copy()
    mj_model, mj_data = create_satellite_plant(
        gravity=False,
        default_target_state=None,
        start_q=start_q,
        goal_q=goal_q,
        include_box_mocap=True,
    )
    mj_data.qpos[: len(start_q)] = start_q
    start_mocap_id = mj_model.body_mocapid[
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "start_marker")
    ]
    mj_data.mocap_quat[start_mocap_id, 0] = PHASE_APPROACH
    mujoco.mj_forward(mj_model, mj_data)

    phase_cb = make_sim_step_callback(mj_model)
    mjx_data = task.make_data()
    actions = jnp.zeros((policy.model.horizon, task.model.nu))

    @jax.jit
    def get_obs(mjx_d: mjx.Data) -> jax.Array:
        mjx_d = mjx.forward(task.model, mjx_d)
        return env.get_obs(mjx_d)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
            )
            obs = get_obs(mjx_data)
            rng, policy_rng = jax.random.split(rng)
            actions = jit_policy(actions, obs, policy_rng)
            mj_data.ctrl[:] = actions[0]

            mujoco.mj_step(mj_model, mj_data)
            phase_cb(mj_model, mj_data, None)
            viewer.sync()

            elapsed = time.time() - t0
            dt = mj_model.opt.timestep
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPC for 2D satellite locomanipulation (fixed layout)."
    )
    subparsers = parser.add_subparsers(dest="task", help="What to do (choose one)")
    train_p = subparsers.add_parser("train", help="Train (and save) a generative policy")
    train_p.add_argument(
        "--num-envs",
        type=int,
        default=64,
        metavar="N",
        help="Parallel training rollouts (vmap width).",
    )
    train_p.add_argument(
        "--num-videos",
        type=int,
        default=0,
        metavar="N",
        help="TensorBoard RGB videos per iteration (0 = headless).",
    )
    subparsers.add_parser("test", help="Test saved policy interactively (with grasp callback)")
    subparsers.add_parser(
        "sample",
        help="Bootstrap predictive sampling with the policy (interactive; not MPPI)",
    )
    args = parser.parse_args()

    env, ctrl, plan_horizon, num_knots = _make_env_and_ctrl()
    save_file = "/tmp/sat_2d_locomanip_policy.pkl"

    if args.task == "train":
        net = DenoisingMLP(
            action_size=env.task.model.nu,
            observation_size=env.observation_size,
            horizon=num_knots,
            hidden_layers=[128, 128],
            rngs=nnx.Rngs(0),
        )
        policy = train(
            env,
            ctrl,
            net,
            num_policy_samples=16,
            log_dir="/tmp/gpc_sat_2d_locomanip",
            num_iters=30,
            num_envs=args.num_envs,
            num_epochs=50,
            checkpoint_every=5,
            num_videos=args.num_videos,
        )
        policy.save(save_file)
        print(f"Saved policy to {save_file}")

    elif args.task == "test":
        print(f"Loading policy from {save_file}")
        policy = Policy.load(save_file)
        _test_locomanip_interactive(env, policy, inference_timestep=0.02)

    elif args.task == "sample":
        policy = Policy.load(save_file)
        bctrl = BootstrappedPredictiveSampling(
            policy,
            env.get_obs,
            inference_timestep=0.02,
            num_policy_samples=4,
            task=env.task,
            num_samples=16,
            noise_level=sat2d_config.MPPI_NOISE_LEVEL,
            plan_horizon=plan_horizon,
            num_knots=num_knots,
            spline_type=sat2d_config.MPPI_SPLINE_TYPE,
        )
        mj_model = env.task.mj_model
        mj_data = mujoco.MjData(mj_model)
        run_sampling(bctrl, mj_model, mj_data, frequency=50)

    else:
        parser.print_help()
