"""GPC training env for fixed 2D satellite pick-and-place (imports planarsim only)."""

from __future__ import annotations

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx

from planarsim.common.collision_formulation import CollisionFormulation
from planarsim.common.rrt_planner import CollisionChecker
from planarsim.sat_2d import config as sat2d_config
from planarsim.sat_2d.kinematics import compute_nominal_config, fk_ee_transform
from planarsim.sat_2d.locomanip_task import (
    PlanarSatelliteLocomanipTask,
    PHASE_APPROACH,
    PHASE_GRASPED,
    compute_grasp_ee_target,
    compute_place_ee_target,
)
from planarsim.sat_2d.obstacles import WORLD_OBJS as WORLD_OBJS_2D
from planarsim.sat_2d.satellite_plant import create_satellite_plant

from gpc.envs.base import SimulatorState, TrainingEnv


def build_fixed_locomanip_task() -> PlanarSatelliteLocomanipTask:
    """IK + task for current ``sat_2d_config`` layout (same as interactive locomanip)."""
    start_q = sat2d_config.START_Q.copy()
    goal_q = sat2d_config.GOAL_Q.copy()

    ik_model, _ = create_satellite_plant(
        gravity=False,
        default_target_state=None,
        start_q=start_q,
        goal_q=goal_q,
        include_box_mocap=False,
    )

    col_geom_ids = []
    col_geom_radii = []
    for geom_name in sat2d_config.COL_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            col_geom_ids.append(geom_id)
            col_geom_radii.append(ik_model.geom_size[geom_id][0])

    obs_bounds = []
    for name in WORLD_OBJS_2D:
        geom_id = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            continue
        pos = ik_model.geom_pos[geom_id]
        size = ik_model.geom_size[geom_id]
        obs_bounds.append((pos - size, pos + size))

    obs_min = np.array([b[0] for b in obs_bounds]) if obs_bounds else np.zeros((0, 3))
    obs_max = np.array([b[1] for b in obs_bounds]) if obs_bounds else np.zeros((0, 3))

    ik_data = mujoco.MjData(ik_model)
    robot_checker = CollisionChecker(
        mj_model=ik_model,
        mj_data=ik_data,
        col_geom_ids=col_geom_ids,
        col_geom_radii=np.array(col_geom_radii),
        obs_min=obs_min,
        obs_max=obs_max,
    )
    self_collision = CollisionFormulation(
        ik_model, sat2d_config.COL_GEOM_NAMES, world_obj_names=[]
    )

    def ik_self_collision_cost(q: np.ndarray) -> float:
        ik_data.qpos[: len(q)] = q
        mujoco.mj_forward(ik_model, ik_data)
        sphere_pos = ik_data.geom_xpos[self_collision.col_geom_ids]
        return float(self_collision.self_collision_cost(sphere_pos))

    grasp_ee_x, grasp_ee_y, grasp_ee_yaw = compute_grasp_ee_target(
        sat2d_config.BOX_START_POSE
    )
    place_ee_x, place_ee_y, place_ee_yaw = compute_place_ee_target(
        sat2d_config.BOX_GOAL_POSE
    )

    grasp_q = compute_nominal_config(
        grasp_ee_x,
        grasp_ee_y,
        grasp_ee_yaw,
        initial_guess=start_q,
        joint_limits=sat2d_config.JOINT_LIMITS,
        collision_checker=robot_checker.check_collision,
        collision_cost_fn=ik_self_collision_cost,
    )
    place_q = compute_nominal_config(
        place_ee_x,
        place_ee_y,
        place_ee_yaw,
        initial_guess=grasp_q,
        joint_limits=sat2d_config.JOINT_LIMITS,
        collision_checker=robot_checker.check_collision,
        collision_cost_fn=ik_self_collision_cost,
        posture_weight=sat2d_config.IK_POSTURE_WEIGHT_PLACE,
    )

    return PlanarSatelliteLocomanipTask(
        impl="jax",
        start_q=start_q,
        goal_q=goal_q,
        world_obj_names=WORLD_OBJS_2D,
        grasp_q=grasp_q,
        place_q=place_q,
        include_box_mocap=True,
    )


def _locomanip_post_step(
    data: mjx.Data,
    start_mocap_idx: int,
    box_mocap_idx: int,
    grasp_dx: jax.Array,
    grasp_dy: jax.Array,
    grasp_tol: jax.Array,
    align_threshold: jax.Array,
    grasp_off_x: jax.Array,
    grasp_off_y: jax.Array,
    grasp_yaw_off: jax.Array,
    box_start_xyyaw: jax.Array,
) -> mjx.Data:
    """After physics: grasp phase bit + rigid box attach (matches interactive locomanip)."""
    ee_x, ee_y, ee_yaw = fk_ee_transform(data.qpos[:7])

    bx = data.mocap_pos[box_mocap_idx, 0]
    by = data.mocap_pos[box_mocap_idx, 1]
    qw_b = data.mocap_quat[box_mocap_idx, 0]
    qz_b = data.mocap_quat[box_mocap_idx, 3]
    byaw = 2.0 * jnp.arctan2(qz_b, qw_b)

    gx = bx + grasp_dx * jnp.cos(byaw) - grasp_dy * jnp.sin(byaw)
    gy = by + grasp_dx * jnp.sin(byaw) + grasp_dy * jnp.cos(byaw)
    dist = jnp.hypot(ee_x - gx, ee_y - gy)

    face_nx = -jnp.sin(byaw)
    face_ny = jnp.cos(byaw)
    ee_c = jnp.cos(ee_yaw)
    ee_s = jnp.sin(ee_yaw)
    alignment = -(ee_c * face_nx + ee_s * face_ny)

    phase = data.mocap_quat[start_mocap_idx, 0]
    grasp_ok = (dist < grasp_tol) & (alignment >= align_threshold)
    phase_new = jnp.where(phase > 0.5, jnp.where(grasp_ok, PHASE_GRASPED, phase), phase)

    in_grasped = phase_new < 0.5

    bx0, by0, byaw0 = box_start_xyyaw[0], box_start_xyyaw[1], box_start_xyyaw[2]
    box_x_att = ee_x + grasp_off_x * ee_c - grasp_off_y * ee_s
    box_y_att = ee_y + grasp_off_x * ee_s + grasp_off_y * ee_c
    box_yaw_att = ee_yaw + grasp_yaw_off

    pos_x = jnp.where(in_grasped, box_x_att, bx0)
    pos_y = jnp.where(in_grasped, box_y_att, by0)
    pos_z = jnp.asarray(0.025, dtype=data.mocap_pos.dtype)

    yaw_box = jnp.where(in_grasped, box_yaw_att, byaw0)
    qw_box = jnp.cos(yaw_box * 0.5)
    qz_box = jnp.sin(yaw_box * 0.5)

    mocap_pos = (
        data.mocap_pos.at[box_mocap_idx, 0]
        .set(pos_x)
        .at[box_mocap_idx, 1]
        .set(pos_y)
        .at[box_mocap_idx, 2]
        .set(pos_z)
    )
    mocap_quat = (
        data.mocap_quat.at[start_mocap_idx, 0]
        .set(phase_new)
        .at[box_mocap_idx, 0]
        .set(qw_box)
        .at[box_mocap_idx, 1]
        .set(0.0)
        .at[box_mocap_idx, 2]
        .set(0.0)
        .at[box_mocap_idx, 3]
        .set(qz_box)
    )
    return data.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)


class Locomanip2DEnv(TrainingEnv):
    """Fixed-layout locomanip; phase and box mocap updated each step after ``mjx.step``."""

    def __init__(self, task: PlanarSatelliteLocomanipTask, episode_length: int) -> None:
        super().__init__(task=task, episode_length=episode_length)
        mj = task.mj_model
        start_bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "start_marker")
        box_bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "box_mocap")
        if start_bid < 0 or box_bid < 0:
            raise RuntimeError(
                "Locomanip2DEnv requires start_marker and box_mocap bodies "
                "(build task with include_box_mocap=True)."
            )
        self._start_mocap_idx = int(mj.body_mocapid[start_bid])
        self._box_mocap_idx = int(mj.body_mocapid[box_bid])

        self._start_q_jnp = jnp.asarray(sat2d_config.START_Q, dtype=jnp.float32)
        bx, by, byaw = sat2d_config.BOX_START_POSE
        self._box_start_xyyaw = jnp.array([bx, by, byaw], dtype=jnp.float32)
        gdx, gdy, _ = sat2d_config.BOX_GRASP_POINT_OFFSET
        self._grasp_dx = jnp.asarray(gdx, dtype=jnp.float32)
        self._grasp_dy = jnp.asarray(gdy, dtype=jnp.float32)
        self._grasp_tol = jnp.asarray(sat2d_config.GRASP_TOL, dtype=jnp.float32)
        self._align_threshold = jnp.asarray(sat2d_config.GRASP_ALIGN_THRESHOLD, dtype=jnp.float32)
        ox, oy, _ = sat2d_config.GRASP_EE_OFFSET
        self._grasp_off_x = jnp.asarray(ox, dtype=jnp.float32)
        self._grasp_off_y = jnp.asarray(oy, dtype=jnp.float32)
        self._grasp_yaw_off = jnp.asarray(sat2d_config.GRASP_EE_YAW_OFFSET, dtype=jnp.float32)

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        del rng
        nq = int(self.task.model.nq)
        nv = int(self.task.model.nv)
        qpos = data.qpos.at[:7].set(self._start_q_jnp)
        if nq > 7:
            qpos = qpos.at[7:].set(data.qpos[7:])
        qvel = data.qvel.at[:7].set(0.0)
        if nv > 7:
            qvel = qvel.at[7:].set(data.qvel[7:])
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros_like(data.ctrl))

        bx0, by0, byaw0 = self._box_start_xyyaw[0], self._box_start_xyyaw[1], self._box_start_xyyaw[2]
        qw_box = jnp.cos(byaw0 * 0.5)
        qz_box = jnp.sin(byaw0 * 0.5)
        mocap_pos = (
            data.mocap_pos.at[self._box_mocap_idx, 0]
            .set(bx0)
            .at[self._box_mocap_idx, 1]
            .set(by0)
            .at[self._box_mocap_idx, 2]
            .set(0.025)
        )
        mocap_quat = (
            data.mocap_quat.at[self._start_mocap_idx, 0]
            .set(PHASE_APPROACH)
            .at[self._box_mocap_idx, 0]
            .set(qw_box)
            .at[self._box_mocap_idx, 1]
            .set(0.0)
            .at[self._box_mocap_idx, 2]
            .set(0.0)
            .at[self._box_mocap_idx, 3]
            .set(qz_box)
        )
        return data.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)

    def step(self, state: SimulatorState, action: jax.Array) -> SimulatorState:
        def advance(s: SimulatorState):
            d = mjx.step(self.task.model, s.data.replace(ctrl=action))
            d = _locomanip_post_step(
                d,
                self._start_mocap_idx,
                self._box_mocap_idx,
                self._grasp_dx,
                self._grasp_dy,
                self._grasp_tol,
                self._align_threshold,
                self._grasp_off_x,
                self._grasp_off_y,
                self._grasp_yaw_off,
                self._box_start_xyyaw,
            )
            d = mjx.forward(self.task.model, d)
            return s.replace(data=d, t=s.t + 1)

        next_state = jax.lax.cond(
            self.episode_over(state),
            lambda _: self._reset_state(state),
            lambda _: advance(state),
            operand=None,
        )
        next_state = jax.lax.cond(
            self.goal_reached(next_state),
            lambda _: self._update_goal(next_state),
            lambda _: next_state,
            operand=None,
        )
        return next_state

    def get_obs(self, data: mjx.Data) -> jax.Array:
        phase = data.mocap_quat[self._start_mocap_idx, 0]
        return jnp.concatenate([data.qpos[:7], data.qvel[:7], jnp.array([phase])])

    @property
    def observation_size(self) -> int:
        return 15
