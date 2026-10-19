"""Gymnasium environment: TCP position reaching for the UFACTORY arms
(roadmap 0.0.5: xArm 6; roadmap 0.0.7: lite6 / xarm5 / xarm6 / xarm7 / uf850).

Task: drive the robot TCP (``attachment_site``) to a randomly sampled target
point using joint-position increment actions on top of the model's position
servo actuators. The environment follows the standard Gymnasium API and only
requires mujoco + numpy at runtime; gymnasium itself is optional (a minimal
fallback base keeps random rollouts possible in environments without it,
e.g. the py314 project default), while training additionally needs
stable-baselines3 (see requirements-rl.txt).

MDP summary (docs/plan_0.0.5_rl.md; n = robot joint count):
    action   Box(-1, 1, (n,))  joint increments, |dq| <= delta per step
    obs      float32 (3+3n,)   TCP-target delta (3), normalized joints (n),
                               scaled joint velocities (n), last action (n)
    reward   potential shaping on distance, success bonus (5 mm threshold,
             aligned with the ufactory_genesis standard), step/action/limit costs
    episode  200 control steps x 10 physics steps x 2 ms = 4 s sim time

Per-robot reach-workspace profiles (target box, shoulder, radial band) are
calibrated by scripts/measure_reach_workspace.py and frozen in ROBOT_PROFILES.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

try:  # gymnasium is optional: core env logic must stay importable without it
    import gymnasium as gym
    from gymnasium import spaces

    _ENV_BASE = gym.Env
except ImportError:  # pragma: no cover - exercised on py314 (no gymnasium)
    gym = None
    spaces = None
    _ENV_BASE = object

_ROOT = Path(__file__).resolve().parent.parent.parent
TCP_SITE = "attachment_site"
TARGET_BODY = "target_mocap"


@dataclass(frozen=True)
class RobotProfile:
    """Frozen reach-workspace calibration for one robot (0.0.7 M0)."""

    robot: str
    n_joints: int
    target_low: tuple   # target sampling box, base frame (m)
    target_high: tuple
    shoulder: tuple     # radial pre-filter origin (joint2 anchor at q=0)
    reach_min: float    # radial band covering the box (m)
    reach_max: float


# Values produced by scripts/measure_reach_workspace.py (FK cloud + IK
# reachability >= 90%). The xArm 6 entry keeps the production 0.0.5 box;
# its reach_max was raised 0.55 -> 0.56 so the radial band covers every box
# corner (the far bottom corner is 0.5586 m from the shoulder).
ROBOT_PROFILES: dict[str, RobotProfile] = {
    "lite6": RobotProfile("lite6", 6,
                          (0.102, -0.2389, 0.0986), (0.2553, 0.2389, 0.4031),
                          (0.0, 0.0, 0.2435), 0.0918, 0.4228),
    "xarm5": RobotProfile("xarm5", 5,
                          (0.1386, -0.3631, 0.155), (0.3914, 0.3631, 0.5834),
                          (0.0, 0.0, 0.267), 0.1248, 0.6827),
    "xarm6": RobotProfile("xarm6", 6,
                          (0.25, -0.25, 0.05), (0.45, 0.25, 0.35),
                          (0.0, 0.0, 0.267), 0.15, 0.56),
    "xarm7": RobotProfile("xarm7", 7,
                          (0.199, -0.3954, 0.2102), (0.4231, 0.3954, 0.5341),
                          (0.0, 0.0, 0.267), 0.1791, 0.7015),
    "uf850": RobotProfile("uf850", 6,
                          (0.1956, -0.4869, 0.2071), (0.5182, 0.4869, 0.735),
                          (0.0, 0.0, 0.364), 0.1761, 0.8823),
}

# xArm 6 production defaults, kept as module constants for compatibility
TARGET_LOW = np.array(ROBOT_PROFILES["xarm6"].target_low)
TARGET_HIGH = np.array(ROBOT_PROFILES["xarm6"].target_high)

# Reward shaping constants (docs/plan_0.0.5_rl.md §3)
_K_SHAPE = 2.0 / 0.01     # per meter of distance reduction (= 2.0 per cm)
_SUCCESS_BONUS = 10.0
_STEP_COST = 0.02
_ACTION_COST = 0.001
_LIMIT_COST = 0.01        # per joint closer than 2% of its range to a limit
_LIMIT_MARGIN = 0.02

_QVEL_SCALE = 3.0         # rad/s -> obs units


def _obs_bounds(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Observation bounds: TCP-target delta (m), normalized joints, scaled
    joint velocities, last action. Observations are clipped to these bounds."""
    low = np.array([-2.0] * 3 + [-1.5] * n + [-5.0] * n + [-1.0] * n,
                   dtype=np.float32)
    high = -low
    high[3 + 2 * n:] = 1.0
    return low, high


class _BoxStub:
    """Minimal stand-in for gymnasium.spaces.Box (no gymnasium installed)."""

    def __init__(self, low, high, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), self.shape).copy()

    def sample(self):
        rng = np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)


def load_reach_model(model_path: str | Path, lite_visuals: bool = True) -> mujoco.MjModel:
    """Load a reach scene, optionally dropping the high-poly visual meshes.

    With ``lite_visuals`` the visual-only mesh geoms (contype=conaffinity=0) and
    their meshes are removed and the collision meshes are shown in their place
    (group 2). Moving links have explicit inertias in the MJCF, so kinematics
    and joint-space dynamics are unchanged (only the fixed, jointless base body's
    geom-derived mass differs, which has no effect on the arm), while memory drops from ~1.3 GB to ~0.1 GB per
    process -- this is what makes many parallel training envs affordable.
    """
    if not lite_visuals:
        return mujoco.MjModel.from_xml_path(str(model_path))
    spec = mujoco.MjSpec.from_file(str(model_path))
    mesh_type = mujoco.mjtGeom.mjGEOM_MESH
    visual = [g for g in spec.geoms
              if g.type == mesh_type and g.contype == 0 and g.conaffinity == 0]
    visual_meshes = {g.meshname for g in visual}
    for g in visual:
        spec.delete(g)
    kept = {g.meshname for g in spec.geoms if g.type == mesh_type}
    for m in list(spec.meshes):
        if m.name in visual_meshes and m.name not in kept:
            spec.delete(m)
    for g in spec.geoms:
        if g.type == mesh_type:
            g.group = 2
            g.rgba = [0.9, 0.9, 0.9, 1.0]
    return spec.compile()


def _resolve_actuators(model: mujoco.MjModel, n: int) -> np.ndarray:
    """Actuator ids of the arm: names act1..actN when present, else the first
    N actuators in model order (lite6 ships unnamed actuators)."""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{i + 1}")
           for i in range(n)]
    if all(a >= 0 for a in ids):
        return np.array(ids)
    if model.nu < n:
        raise ValueError(f"scene must define at least {n} actuators")
    return np.arange(n)


class ReachEnv(_ENV_BASE):
    """UFACTORY arm TCP reaching environment (Gymnasium API)."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, model_path: str | Path | None = None, *,
                 robot: str = "xarm6",
                 render_mode: str | None = None,
                 max_steps: int = 200, decimation: int = 10, delta: float = 0.05,
                 success_dist: float = 0.005,
                 target_low: np.ndarray | None = None,
                 target_high: np.ndarray | None = None,
                 q0_noise: float = 0.2,
                 validate_targets: bool = True,
                 lite_visuals: bool | None = None):
        if robot not in ROBOT_PROFILES:
            raise ValueError(f"unknown robot {robot!r}; "
                             f"expected one of {sorted(ROBOT_PROFILES)}")
        if render_mode is not None and render_mode != "human":
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        for name, value in (("max_steps", max_steps), ("decimation", decimation),
                            ("delta", delta), ("success_dist", success_dist)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(q0_noise) or q0_noise < 0:
            raise ValueError("q0_noise must be finite and non-negative")

        profile = ROBOT_PROFILES[robot]
        self.robot = robot
        self._n = profile.n_joints
        self.render_mode = render_mode
        self._max_steps = int(max_steps)
        self._decimation = int(decimation)
        self._delta = float(delta)
        self._success_dist = float(success_dist)
        self._target_low = (np.asarray(profile.target_low, dtype=float)
                            if target_low is None
                            else np.asarray(target_low, dtype=float))
        self._target_high = (np.asarray(profile.target_high, dtype=float)
                             if target_high is None
                             else np.asarray(target_high, dtype=float))
        self._shoulder = np.asarray(profile.shoulder, dtype=float)
        self._reach_min = float(profile.reach_min)
        self._reach_max = float(profile.reach_max)
        self._q0_noise = float(q0_noise)
        self._validate_targets = bool(validate_targets)

        if model_path is None:
            model_path = _ROOT / "models" / robot / "scene_reach.xml"
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"scene model not found: {model_path}")
        if lite_visuals is None:  # full meshes only when a human is watching
            lite_visuals = render_mode is None
        self.model = load_reach_model(model_path, lite_visuals=lite_visuals)
        self.data = mujoco.MjData(self.model)
        # dedicated instance for reachability IK so reset never perturbs env state
        self._ik_data = mujoco.MjData(self.model)

        self._joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}")
             for i in range(self._n)])
        if (self._joint_ids < 0).any():
            raise ValueError(f"scene must define joints joint1..joint{self._n}")
        self._act_ids = _resolve_actuators(self.model, self._n)
        self._qpos_adr = self.model.jnt_qposadr[self._joint_ids]
        self._qvel_adr = self.model.jnt_dofadr[self._joint_ids]
        self._q_lo = self.model.jnt_range[self._joint_ids, 0].copy()
        self._q_hi = self.model.jnt_range[self._joint_ids, 1].copy()
        self._q_mid = 0.5 * (self._q_lo + self._q_hi)
        self._q_half = 0.5 * (self._q_hi - self._q_lo)

        self._site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if self._site_id < 0:
            raise ValueError(f"scene must define site {TCP_SITE!r}")
        self._mocap_id = int(self.model.body(TARGET_BODY).mocapid[0])

        self._rng = np.random.default_rng()
        self._step_count = 0
        self._last_action = np.zeros(self._n, dtype=np.float32)
        self._target = np.zeros(3)
        self._prev_dist = np.inf
        self._viewer = None

        obs_low, obs_high = _obs_bounds(self._n)
        if spaces is not None:  # gymnasium installed: standard spaces
            self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (self._n,), dtype=np.float32)
        else:  # fallback: keep shapes/dtypes available for non-gym rollouts
            self.observation_space = _BoxStub(obs_low, obs_high, (3 + 3 * self._n,),
                                              np.float32)
            self.action_space = _BoxStub(-1.0, 1.0, (self._n,), np.float32)

    # ------------------------------------------------------------------ state
    @property
    def n_joints(self) -> int:
        return self._n

    @property
    def target_position(self) -> np.ndarray:
        return self._target.copy()

    @property
    def step_count(self) -> int:
        return self._step_count

    def tcp_position(self) -> np.ndarray:
        return self.data.site_xpos[self._site_id].copy()

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._qpos_adr].copy()

    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.tcp_position() - self._target))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        q0 = self._sample_q0()
        self._target = self._sample_target(options)
        self.data.qpos[self._qpos_adr] = q0
        self.data.qvel[:] = 0.0
        self.data.ctrl[self._act_ids] = q0
        self.data.mocap_pos[self._mocap_id] = self._target
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._last_action = np.zeros(self._n, dtype=np.float32)
        self._prev_dist = self.distance_to_target()
        obs = self._get_obs()
        info = {"target": self._target.copy(), "dist": self._prev_dist,
                "success": False, "steps": 0}
        if self.render_mode == "human":
            self._ensure_viewer()
            self._viewer.sync()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(self._n)
        if not np.all(np.isfinite(action)):
            raise ValueError("action must be finite")
        action = np.clip(action, -1.0, 1.0)

        q_cmd = np.clip(self.data.qpos[self._qpos_adr] + self._delta * action,
                        self._q_lo, self._q_hi)
        self.data.ctrl[self._act_ids] = q_cmd
        for _ in range(self._decimation):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        dist = self.distance_to_target()
        reward = self._reward(self._prev_dist, dist, action)
        self._prev_dist = dist

        success = dist < self._success_dist
        terminated = bool(success)
        truncated = bool(self._step_count >= self._max_steps and not success)

        self._last_action = action.astype(np.float32)
        obs = self._get_obs()
        info = {"target": self._target.copy(), "dist": dist,
                "success": success, "steps": self._step_count}
        if self.render_mode == "human":
            self._viewer.sync()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._ensure_viewer()
            self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # -------------------------------------------------------------- internals
    def _ensure_viewer(self):
        if self._viewer is None:
            from src.viewer import PassiveViewer
            self._viewer = PassiveViewer(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        rel = np.clip(self.tcp_position() - self._target, -2.0, 2.0)
        q = self.data.qpos[self._qpos_adr]
        q_norm = np.clip((q - self._q_mid) / self._q_half, -1.5, 1.5)
        qd = np.clip(self.data.qvel[self._qvel_adr] / _QVEL_SCALE, -5.0, 5.0)
        return np.concatenate([rel, q_norm, qd, self._last_action]).astype(np.float32)

    def _reward(self, prev_dist: float, dist: float, action: np.ndarray) -> float:
        reward = _K_SHAPE * (prev_dist - dist)
        reward -= _STEP_COST
        reward -= _ACTION_COST * float(action @ action)
        q = self.data.qpos[self._qpos_adr]
        near_limit = ((q <= self._q_lo + _LIMIT_MARGIN * (self._q_hi - self._q_lo))
                      | (q >= self._q_hi - _LIMIT_MARGIN * (self._q_hi - self._q_lo)))
        reward -= _LIMIT_COST * float(np.sum(near_limit))
        return reward

    def _sample_q0(self) -> np.ndarray:
        """Ready pose (all-zero) plus uniform noise in +/- q0_noise rad."""
        q0 = self._rng.uniform(-self._q0_noise, self._q0_noise, self._n)
        return np.clip(q0, self._q_lo, self._q_hi)

    def _sample_target(self, options: dict | None) -> np.ndarray:
        if options is not None and "target" in options:
            target = np.asarray(options["target"], dtype=float).reshape(3)
            if not self._is_reachable(target):
                raise ValueError(f"requested target is not reachable: {target}")
            return target
        for _ in range(100):
            target = self._rng.uniform(self._target_low, self._target_high)
            if self._is_reachable(target):
                return target
        raise RuntimeError("could not sample a reachable target "
                           f"in [{self._target_low}, {self._target_high}]")

    def _is_reachable(self, target: np.ndarray) -> bool:
        reach = float(np.linalg.norm(target - self._shoulder))
        if not (self._reach_min <= reach <= self._reach_max):
            return False
        if not self._validate_targets:
            return True
        from src.ik import solve_ik
        q_ref = self.data.qpos[self._qpos_adr].copy()
        _, converged = solve_ik(self.model, self._ik_data, TCP_SITE, target, None,
                                q_init=q_ref, pos_tol=2e-3, max_iter=80)
        self._ik_data.qpos[:] = self.data.qpos  # keep scratch state neutral
        return bool(converged)


# Compatibility alias (docs/plan_0.0.7_rl.md): removed at 0.1.0.
XArm6ReachEnv = ReachEnv
