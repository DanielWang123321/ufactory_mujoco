#!/usr/bin/env python3
"""Five-arm dynamics, collision and monitored motion validation.

    ufactory-mujoco dynamics --headless
    ufactory-mujoco dynamics --robot all --headless \
        --samples 32 --seed 0 --report build/dynamics/batch6.json

All modes execute home -> ready -> target -> ready. No robot SDK is required.
The xArm6 additionally retains the historical 3 kg payload checks.
"""

import argparse
from dataclasses import dataclass
import platform
import sys
from pathlib import Path

import mujoco
import numpy as np

from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.dynamics import ArmDynamics
from ufactory_mujoco.model_paths import MODELS_ROOT
from ufactory_mujoco.robots import ARMS, resolve_arm_scene
from ufactory_mujoco.sim_robot import SimRobot, _quintic
from ufactory_mujoco.trajectory import plan_ptp_quintic
from ._reporting import write_report

VALIDATION_VERSION = "batch6-v1"
THRESHOLDS = {
    "mass_asymmetry": 1e-12,              # strict <
    "mass_min_eigenvalue": 0.0,          # strict >
    "coriolis_at_rest_nm": 1e-9,         # strict <
    "id_fd_round_trip_rad_s2": 1e-8,     # strict <
    "penetration_m": 1e-6,               # <=
    "joint_limit_excess_rad": 1e-6,      # <=
    "arrival_error_rad": 0.01,           # strict <
    "collision_fixture_penetration_m": 0.001,  # strict >
    "payload_gravity_error_nm": 1e-6,    # <=
    "payload_reference_relative_error": 0.05,  # <=
    "feedforward_utilization": 1.0,     # <=
}
SETTLE_SECONDS = 0.3
XARM6 = MODELS_ROOT / "xarm6" / "xarm6.xml"
SCENE = MODELS_ROOT / "xarm6" / "scene.xml"
READY = np.array([0.0, -0.8, -1.2, 1.0, 0.3, 0.0])
TARGET = np.array([0.5, -1.2, -1.0, 0.8, 0.6, 0.4])
PAYLOAD = (3.0, [0.0, 0.0, 0.021])
REAL_J2, REAL_J3 = -5.46, -21.10  # Historical reference, no new hardware run.
REAL_TOL = THRESHOLDS["payload_reference_relative_error"]

# Fixed fixtures; never regenerate these from the current calculation.
PAYLOAD_GRAVITY_BASELINE = (
    0.0, -5.457423426494502, -21.096553945585573,
    -1.5295099439534692, -1.570599127710176, 0.0066876600139051945,
)
PRESETS = {
    "lite6": (np.deg2rad([0.0, 9.9, 31.8, 0.0, 21.9, 0.0]),
              [0.3, -0.8, 0.3, 0.2, 0.5, -0.3]),
    "xarm5": ([0.0] * 5, [0.3, -0.7, -0.3, 0.2, 0.6]),
    "xarm6": (READY, TARGET),
    "xarm7": ([0.0] * 7, [0.3, -0.7, 0.2, 0.5, 0.2, 0.2, -0.3]),
    "uf850": ([0.0] * 6, [0.3, -0.7, -0.3, 0.2, 0.4, -0.3]),
}
# First >1 mm penetration in each arm's independent seed-0, 32-pose sample.
COLLISION_POSES = {
    "lite6": [4.4912779532244995, -2.4421365994120947, 3.8039540832087884,
              -4.075834722372415, 1.5719836474992492, 0.5210174489137813],
    "xarm5": [-4.075834722372415, 1.5261273360880185, -1.6967320630183966,
              -0.24399488248358114, -0.9715417572861691],
    "xarm6": [1.4499731572157986, -0.46543364612869365, 0.1804877813170438,
              6.042359584490842, 1.621323856490987, 1.8907284401063622],
    "xarm7": [1.721112608322577, -0.9384678630531413, -5.768301128278682,
              -0.12390299981069924, 3.93667286848152, 2.7198016036837824,
              1.3400256798851737],
    "uf850": [4.4912779532244995, -2.149079088088024, -1.0972818286577795,
              -4.075834722372415, 1.5719836474992492, 0.5210174489137813],
}


class ValidationFailure(RuntimeError):
    """A failed or incomplete validation; stops this arm's motion."""


@dataclass(frozen=True)
class MotionSegment:
    name: str
    goal: np.ndarray
    duration: float


def motion_segments(ready, target) -> tuple[MotionSegment, ...]:
    return (
        MotionSegment("deploy", np.asarray(ready), 1.5),
        MotionSegment("outbound", np.asarray(target), 3.0),
        MotionSegment("return", np.asarray(ready), 3.0),
    )


def command_phases(home, segments, dt):
    """The exact move_joints quintic command samples, plus constant holds."""
    start = np.asarray(home, dtype=float)
    for segment in segments:
        steps = int(np.ceil(segment.duration / dt))
        times = np.minimum(np.arange(1, steps + 1) * dt, segment.duration)
        yield segment.name, _quintic(times, start, segment.goal, segment.duration)
        yield segment.name + "_settle", np.tile(
            segment.goal, (int(np.ceil(SETTLE_SECONDS / dt)), 1))
        start = segment.goal


def make_checker(scene) -> CollisionChecker:
    # Do not ignore world contacts involving any moving link.
    return CollisionChecker(scene, ignore_pairs=[("link_base", "world")])


def finite_array(value, shape, name) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValidationFailure(f"{name}: expected finite array of shape {shape}")
    return array


def pose_metrics(checker: CollisionChecker, q) -> dict:
    q = finite_array(q, (checker.n_arm_qpos,), "joint position")
    contacts = checker._contacts(q)
    if any(not np.isfinite(distance) for _, _, distance in contacts):
        raise ValidationFailure("non-finite contact distance")
    distance = min((d for _, _, d in contacts), default=None)
    colliding = [(a, b, d) for a, b, d in contacts if d < 0]
    return {
        "min_distance_m": distance,
        "penetration_m": max(0.0, -distance) if distance is not None else 0.0,
        "joint_limit_excess_rad": float(max(
            0.0, np.max(checker.q_lo - q), np.max(q - checker.q_hi))),
        "in_collision": bool(colliding),
        "self_collision": any(a != 0 and b != 0 for a, b, _ in colliding),
        "floor_collision": any(a == 0 or b == 0 for a, b, _ in colliding),
    }


def safety_failures(metrics) -> list[str]:
    failures = []
    if metrics["penetration_m"] > THRESHOLDS["penetration_m"]:
        failures.append(f"penetration {metrics['penetration_m']:.9g} m")
    if metrics["joint_limit_excess_rad"] > THRESHOLDS["joint_limit_excess_rad"]:
        failures.append(f"joint limit excess {metrics['joint_limit_excess_rad']:.9g} rad")
    return failures


def check_dynamics(dyn: ArmDynamics, poses, rng) -> dict:
    """Rigid-body identities at named and random poses, with no added load."""
    n = dyn.n_arm
    records, failures = [], []
    for name, q in poses:
        record = {"pose": name, "q": q.tolist()}
        records.append(record)
        try:
            matrix = finite_array(dyn.mass_matrix(q), (n, n), "mass matrix")
            gravity = finite_array(dyn.gravity(q), (n,), "gravity")
            coriolis = finite_array(dyn.coriolis(q, np.zeros(n)), (n,), "Coriolis")
            qd, qdd = rng.uniform(-1, 1, n), rng.uniform(-2, 2, n)
            tau = finite_array(dyn.inverse_dynamics(q, qd, qdd), (n,), "inverse dynamics")
            recovered = finite_array(dyn.forward_dynamics(q, qd, tau), (n,), "forward dynamics")
            record.update(
                mass_asymmetry=float(np.max(np.abs(matrix - matrix.T))),
                mass_min_eigenvalue=float(np.min(np.linalg.eigvalsh(matrix))),
                gravity_nm=gravity.tolist(),
                coriolis_at_rest_nm=float(np.max(np.abs(coriolis))),
                id_fd_round_trip_rad_s2=float(np.max(np.abs(recovered - qdd))),
            )
            for metric in ("mass_asymmetry", "coriolis_at_rest_nm", "id_fd_round_trip_rad_s2"):
                if not record[metric] < THRESHOLDS[metric]:
                    failures.append(f"{name}: {metric} = {record[metric]:.9g}")
            if not record["mass_min_eigenvalue"] > 0:
                failures.append(f"{name}: mass matrix is not positive definite")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    valid = [record for record in records if "mass_asymmetry" in record]
    return {
        "success": not failures,
        "pose_count": len(poses),
        "evaluated_pose_count": len(valid),
        "mass_asymmetry": max((r["mass_asymmetry"] for r in valid), default=None),
        "mass_min_eigenvalue": min((r["mass_min_eigenvalue"] for r in valid), default=None),
        "coriolis_at_rest_nm": max((r["coriolis_at_rest_nm"] for r in valid), default=None),
        "id_fd_round_trip_rad_s2": max((r["id_fd_round_trip_rad_s2"] for r in valid), default=None),
        "poses": records,
        "failures": failures,
    }


def check_feedforward(dyn: ArmDynamics, q, dt: float) -> dict:
    tau = finite_array(dyn.feedforward_traj(q, dt), (len(q), dyn.n_arm), "feedforward")
    ranges = finite_array(dyn.model.actuator_forcerange, (dyn.n_arm, 2), "actuator ranges")
    if (not np.all(dyn.model.actuator_forcelimited)
            or np.any(ranges[:, 0] >= 0) or np.any(ranges[:, 1] <= 0)):
        raise ValidationFailure("expected bounded, signed actuator force ranges")
    minimum, maximum = tau.min(axis=0), tau.max(axis=0)
    utilization = float(np.max(np.where(tau >= 0, tau / ranges[:, 1], tau / ranges[:, 0])))
    passed = bool(np.all(minimum >= ranges[:, 0]) and np.all(maximum <= ranges[:, 1]))
    return {
        "success": passed, "samples": len(q), "dt_s": dt,
        "min_tau_nm": minimum.tolist(), "max_tau_nm": maximum.tolist(),
        "peak_tau_nm": np.max(np.abs(tau), axis=0).tolist(),
        "actuator_force_ranges_nm": ranges.tolist(), "utilization": utilization,
        "failures": [] if passed else ["feedforward exceeds actuator force range"],
    }


def check_collision_fixtures(checker, robot, ready, random_q) -> dict:
    safe = pose_metrics(checker, ready)
    collision = pose_metrics(checker, COLLISION_POSES[robot])
    failures = []
    if safe["in_collision"] or safe["joint_limit_excess_rad"] > 0:
        failures.append("safe fixture is colliding or outside joint limits")
    if collision["penetration_m"] <= THRESHOLDS["collision_fixture_penetration_m"]:
        failures.append("collision fixture has no penetration greater than 1 mm")
    if collision["joint_limit_excess_rad"] > 0:
        failures.append("collision fixture is outside joint limits")
    classified = [{"sample": i, "q": q.tolist(), **pose_metrics(checker, q)}
                  for i, q in enumerate(random_q)]
    return {
        "success": not failures,
        "safe_fixture": {"q": ready.tolist(), **safe},
        "collision_fixture": {"q": COLLISION_POSES[robot], **collision},
        "random_poses": classified,
        "random_collision_count": sum(r["in_collision"] for r in classified),
        "failures": failures,
    }


def preflight_motion(checker, home, segments) -> dict:
    """Query every command at the physical timestep without advancing time."""
    report = {
        "success": False, "completed": False, "samples_checked": 0,
        "max_penetration_m": 0.0, "max_joint_limit_excess_rad": 0.0,
        "phases": [], "failures": [],
    }
    phases = [("initial", np.asarray([home]))]
    phases.extend(command_phases(home, segments, checker.model.opt.timestep))
    for name, commands in phases:
        phase = {"name": name, "samples_checked": 0, "completed": False}
        report["phases"].append(phase)
        for index, q in enumerate(commands):
            report["samples_checked"] += 1
            phase["samples_checked"] += 1
            try:
                metrics = pose_metrics(checker, q)
                report["max_penetration_m"] = max(report["max_penetration_m"], metrics["penetration_m"])
                report["max_joint_limit_excess_rad"] = max(
                    report["max_joint_limit_excess_rad"], metrics["joint_limit_excess_rad"])
                failures = safety_failures(metrics)
            except Exception as exc:
                failures = [f"{type(exc).__name__}: {exc}"]
            if failures:
                report["failures"] = [f"{name} sample {index}: {reason}" for reason in failures]
                report["failed_q"] = np.asarray(q).tolist()
                return report
        phase["completed"] = True
    report.update(success=True, completed=True)
    return report


def empty_motion_report() -> dict:
    return {
        "success": False, "completed": False, "steps": 0,
        "observations": 0, "simulated_seconds": 0.0,
        "max_penetration_m": 0.0, "max_joint_limit_excess_rad": 0.0,
        "max_arrival_error_rad": 0.0, "warnings": {}, "phases": [], "failures": [],
    }


class MonitoredSimRobot(SimRobot):
    """Inspect the integrated state after *every* SimRobot physics step.

    mj_step contacts describe the state before integration. Re-querying the
    independent checker is required to inspect the new qpos, including the
    final step of each deployment, move, settle and return.
    """

    def __init__(self, scene, checker, *, viewer=False):
        super().__init__(scene, viewer=viewer)
        self.checker = checker
        self.motion_report = empty_motion_report()
        self._start_time = float(self.data.time)
        self.phase = None

    def _check_state(self):
        for name in ("qpos", "qvel", "qacc", "act", "ctrl", "qfrc_bias", "qfrc_actuator"):
            if not np.all(np.isfinite(getattr(self.data, name))):
                raise ValidationFailure(f"non-finite state: {name}")
        if not np.isfinite(self.data.time):
            raise ValidationFailure("non-finite simulation time")
        warnings = {mujoco.mjtWarning(i).name: int(stat.number)
                    for i, stat in enumerate(self.data.warning) if stat.number}
        self.motion_report["warnings"] = warnings
        if warnings:
            raise ValidationFailure(f"MuJoCo warning: {warnings}")
        if self._viewer is not None and not self._viewer.is_running():
            raise ValidationFailure("viewer closed before validation completed")

    def inspect_position(self):
        metrics = pose_metrics(self.checker, self.get_joint_positions())
        report = self.motion_report
        report["observations"] += 1
        report["max_penetration_m"] = max(report["max_penetration_m"], metrics["penetration_m"])
        report["max_joint_limit_excess_rad"] = max(
            report["max_joint_limit_excess_rad"], metrics["joint_limit_excess_rad"])
        failures = safety_failures(metrics)
        if failures:
            raise ValidationFailure("; ".join(failures))

    def _step(self):
        self._check_state()  # Detect invalid input before MuJoCo can auto-reset it.
        previous_time = float(self.data.time)
        super()._step()
        report = self.motion_report
        report["steps"] += 1
        report["simulated_seconds"] = float(self.data.time) - self._start_time
        if self.phase is not None:
            self.phase["steps"] += 1
        self._check_state()
        if abs(self.data.time - previous_time - self.model.opt.timestep) > 1e-10:
            raise ValidationFailure("physics time did not advance by one timestep")
        self.inspect_position()


def execute_motion(robot: MonitoredSimRobot, segments) -> dict:
    report = robot.motion_report
    try:
        robot._check_state()
        robot.inspect_position()
        for segment in segments:
            for settle in (False, True):
                name = segment.name + ("_settle" if settle else "")
                duration = SETTLE_SECONDS if settle else segment.duration
                phase = {"name": name, "steps": 0, "completed": False}
                report["phases"].append(phase)
                robot.phase = phase
                start_time = float(robot.data.time)
                if settle:
                    robot.settle(duration)
                else:
                    robot.move_joints(segment.goal, duration, wait=False)
                robot._check_state()
                elapsed = float(robot.data.time) - start_time
                minimum_steps = int(np.ceil(duration / robot.model.opt.timestep))
                if (phase["steps"] < minimum_steps or elapsed < duration - 1e-10
                        or elapsed > duration + robot.model.opt.timestep + 1e-10):
                    raise ValidationFailure(f"{name}: motion ended early or exceeded duration")
                if settle:
                    error = float(np.max(np.abs(robot.get_joint_positions() - segment.goal)))
                    phase["arrival_error_rad"] = error
                    report["max_arrival_error_rad"] = max(report["max_arrival_error_rad"], error)
                    if not error < THRESHOLDS["arrival_error_rad"]:
                        raise ValidationFailure(f"{name}: arrival error {error:.9g} rad")
                phase["completed"] = True
        report.update(success=True, completed=True)
    except Exception as exc:
        phase_name = robot.phase["name"] if robot.phase is not None else "initial"
        report["failures"].append(f"{phase_name}: {type(exc).__name__}: {exc}")
    return report


def check_payload_regression() -> dict:
    dyn = ArmDynamics(XARM6, payload=PAYLOAD, site=ARMS["xarm6"].tcp_site)
    gravity = finite_array(dyn.gravity(READY), (6,), "payload gravity")
    baseline = np.asarray(PAYLOAD_GRAVITY_BASELINE)
    error = float(np.max(np.abs(gravity - baseline)))
    references = np.array([REAL_J2, REAL_J3])
    reference_error = np.abs(gravity[1:3] - references) / np.abs(references)
    # Keep the original 3 kg ready->target quintic feedforward demonstration.
    trajectory = plan_ptp_quintic(READY, TARGET, max_vel=1.0, max_acc=2.0, dt=0.01)
    feedforward = check_feedforward(dyn, trajectory.q, 0.01)
    failures = []
    if error > THRESHOLDS["payload_gravity_error_nm"]:
        failures.append(f"fixed six-axis gravity baseline error {error:.9g} Nm")
    if np.any(reference_error > REAL_TOL):
        failures.append("historical J2/J3 reference exceeds 5% tolerance")
    failures.extend(feedforward["failures"])
    return {
        "success": not failures, "mass_kg": PAYLOAD[0], "com_m": PAYLOAD[1],
        "q": READY.tolist(), "gravity_nm": gravity.tolist(),
        "baseline_gravity_nm": list(PAYLOAD_GRAVITY_BASELINE),
        "max_baseline_error_nm": error,
        "historical_j2_j3_nm": references.tolist(),
        "historical_relative_errors": reference_error.tolist(),
        "new_hardware_validation": False,
        "feedforward": feedforward, "failures": failures,
    }


def empty_robot_report(robot) -> dict:
    arm = ARMS.get(robot)
    return {
        "robot": robot, "success": False, "completed": False,
        "dof": arm.dof if arm is not None else None,
        "tcp_site": arm.tcp_site if arm is not None else None,
        "model_path": str(MODELS_ROOT / robot / f"{robot}.xml"),
        "scene_path": str(MODELS_ROOT / robot / "scene.xml"),
        "dynamics": None, "feedforward": None, "collision": None,
        "payload": None, "preflight": None, "motion": empty_motion_report(),
        "failures": [],
    }


def run_checks(robot="xarm6", samples=32, seed=0, *, viewer=False) -> dict:
    """One independent, reproducible validation, including actual motion."""
    report = empty_robot_report(robot)
    try:
        arm = ARMS[robot]
        scene = resolve_arm_scene(robot, None, "scene")
        model_path = scene.parent / f"{arm.key}.xml"
        report.update(dof=arm.dof, tcp_site=arm.tcp_site,
                      model_path=str(model_path), scene_path=str(scene),
                      sampling={"samples": samples, "seed": seed})
        checker = make_checker(scene)
        dyn = ArmDynamics(model_path, site=arm.tcp_site)
        report.update(timestep_s=float(checker.model.opt.timestep),
                      joint_limits_rad=np.column_stack((checker.q_lo, checker.q_hi)).tolist())
        if (dyn.n_arm != arm.dof or checker.n_arm_qpos != arm.dof
                or dyn.model.nu != arm.dof or dyn.model.nv != arm.dof):
            raise ValidationFailure("model DOFs do not match the arm registry")
        if mujoco.mj_name2id(dyn.model, mujoco.mjtObj.mjOBJ_SITE, arm.tcp_site) < 0:
            raise ValidationFailure("registry TCP site is missing")
        home = checker.model.key("home").qpos[:arm.dof].copy()
        ready, target = (np.asarray(q, dtype=float) for q in PRESETS[robot])
        rng = np.random.default_rng(seed)
        random_q = rng.uniform(checker.q_lo, checker.q_hi, (samples, arm.dof))
        poses = [("home", home), ("ready", ready), ("target", target)]
        poses.extend((f"random_{i:03d}", q) for i, q in enumerate(random_q))
        report["dynamics"] = check_dynamics(dyn, poses, rng)
        report["collision"] = check_collision_fixtures(checker, robot, ready, random_q)
        segments = motion_segments(ready, target)
        report["preflight"] = preflight_motion(checker, home, segments)
        commands = np.vstack([home, *(q for _, q in command_phases(
            home, segments, checker.model.opt.timestep))])
        report["feedforward"] = check_feedforward(dyn, commands, checker.model.opt.timestep)
        if robot == "xarm6":
            report["payload"] = check_payload_regression()
        for name in ("dynamics", "collision", "preflight", "feedforward", "payload"):
            section = report[name]
            if section is not None:
                report["failures"].extend(f"{name}: {reason}" for reason in section["failures"])
        if report["failures"]:
            report["motion"]["failures"].append("motion not started: static checks or preflight failed")
            return report
        with MonitoredSimRobot(scene, checker, viewer=viewer) as simulation:
            report["motion"] = simulation.motion_report
            if not np.array_equal(simulation.get_joint_positions(), home):
                raise ValidationFailure("initial simulation pose differs from preflight home")
            execute_motion(simulation, segments)
        report["failures"].extend(f"motion: {reason}" for reason in report["motion"]["failures"])
        report["completed"] = report["motion"]["completed"]
        report["success"] = bool(report["completed"] and not report["failures"])
    except KeyboardInterrupt:
        report["interrupted"] = True
        report["failures"].append("validation interrupted")
    except Exception as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
    return report


def run_validation(robot="xarm6", samples=32, seed=0, *, headless=True) -> dict:
    if robot not in (*ARMS, "all"):
        raise ValueError(f"unknown robot: {robot}")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    names = tuple(ARMS) if robot == "all" else (robot,)
    report = {
        "schema_version": 1, "validation_version": VALIDATION_VERSION,
        "environment": {
            "python": platform.python_version(), "executable": sys.executable,
            "mujoco": mujoco.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
        },
        "sampling": {"samples": samples, "seed": seed},
        "thresholds": dict(THRESHOLDS), "headless": headless,
        "success": False, "completed": False, "robots": [],
    }
    interrupted = False
    for name in names:
        result = empty_robot_report(name)
        if interrupted:
            result["failures"].append("not started: validation interrupted")
        else:
            try:
                result = run_checks(name, samples, seed, viewer=not headless)
                interrupted = result.get("interrupted", False)
            except KeyboardInterrupt:
                interrupted = True
                result["failures"].append("validation interrupted")
            except Exception as exc:
                result["failures"].append(f"{type(exc).__name__}: {exc}")
        report["robots"].append(result)
    report["completed"] = all(r["completed"] for r in report["robots"])
    report["success"] = report["completed"] and all(r["success"] for r in report["robots"])
    return report


def json_safe(value):
    """Represent unavailable/non-finite measurements as null, never NaN/Infinity."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def integer_argument(value, *, positive=False):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < (1 if positive else 0):
        raise argparse.ArgumentTypeError("must be positive" if positive else "must be nonnegative")
    return number


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="ufactory-mujoco dynamics", description=__doc__)
    parser.add_argument("--robot", choices=(*ARMS, "all"), default="xarm6")
    parser.add_argument("--headless", action="store_true", help="validate without a viewer")
    parser.add_argument("--samples", type=lambda s: integer_argument(s, positive=True), default=32)
    parser.add_argument("--seed", type=integer_argument, default=0)
    parser.add_argument("--report", type=Path, help="write a strict JSON validation report")
    args = parser.parse_args(argv)
    report = run_validation(args.robot, args.samples, args.seed, headless=args.headless)
    for result in report["robots"]:
        motion = result["motion"]
        print(f"{result['robot']}: {'PASS' if result['success'] else 'FAIL'}; "
              f"steps={motion['steps']}, time={motion['simulated_seconds']:.3f} s, "
              f"penetration={motion['max_penetration_m']:.3g} m, "
              f"arrival={motion['max_arrival_error_rad']:.5f} rad")
        dynamics = result["dynamics"]
        if dynamics is not None and dynamics["evaluated_pose_count"]:
            print(f"  M: asymmetry {dynamics['mass_asymmetry']:.1e}, "
                  f"min eigenvalue {dynamics['mass_min_eigenvalue']:.6f}; "
                  f"C(0) {dynamics['coriolis_at_rest_nm']:.1e} Nm; "
                  f"ID/FD {dynamics['id_fd_round_trip_rad_s2']:.1e} rad/s^2")
        if result["feedforward"] is not None:
            print(f"  bare-arm feedforward utilization: {result['feedforward']['utilization']:.1%}")
        if result["payload"] is not None:
            gravity = result["payload"]["gravity_nm"]
            print(f"  3 kg payload: J2 {gravity[1]:.10f}, J3 {gravity[2]:.10f} Nm "
                  f"(historical reference {REAL_J2}, {REAL_J3})")
            feedforward = result["payload"]["feedforward"]
            print(f"  3 kg feedforward peaks: {np.round(feedforward['peak_tau_nm'], 3).tolist()} Nm; "
                  f"utilization {feedforward['utilization']:.1%}")
        for reason in result["failures"]:
            print(f"  {reason}")
    if args.report is not None:
        try:
            write_report(args.report, json_safe(report))
            print(f"Report: {args.report}")
        except (OSError, TypeError, ValueError) as exc:
            print(f"Report failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    raise SystemExit(0 if report["success"] and report["completed"] else 1)


if __name__ == "__main__":
    main()
