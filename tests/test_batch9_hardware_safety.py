"""Batch 9 offline unit tests: XArmRobot, calibration SHA-256, FK/IK check, sdk-sim, and safety gate."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import yaml

from scripts import compare_dynamics_xarm6, compare_torque_xarm6
from ufactory_mujoco import (
    CalibrationError,
    CollisionChecker,
    JointTrajectory,
    KinematicsCalibration,
    MotionApproval,
    RobotInterface,
    SafetyGateError,
    XArmRobot,
    apply_calibration_to_model,
    authorize_real_motion,
    fk_pose,
    load_kinematics_calibration,
    run_sdk_simulation,
    validate_fk_ik_with_controller,
)
from ufactory_mujoco.calibration import (
    flange_m_quat_to_tcp_mm_rpy,
    pose_flange_to_tcp,
    pose_tcp_to_flange,
    tcp_mm_rpy_to_flange_m_quat,
)
from ufactory_mujoco.cli import main as cli_main
from ufactory_mujoco.model_paths import MODELS_ROOT

ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = ROOT / "calibrations" / "xarm6_kinematics_L56A01.yaml"
SN = "XI130505L56A01"


class FakeXArmAPI:
    """Deterministic fake xArm SDK controller for offline safety and sdk-sim tests."""

    def __init__(self, ip: str = "192.168.1.60", *, dof: int = 6, sn: str = SN):
        self.ip = ip
        self.axis = dof
        self.sn = sn
        self.connected = True
        self.state = 5
        self.mode = 0
        self.error_code = 0
        self.warn_code = 0
        self.is_simulation_robot = False
        self.tcp_load = [0.82, [0.0, 0.0, 48.0]]
        self.tcp_offset = [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]

        self.physical_q = np.array([0.0, -0.5, -0.4, 0.0, 0.9, 0.0, 0.0], dtype=float)
        self.virtual_q = self.physical_q.copy()
        self.tau = np.array([0.0, -8.1, -15.6, -0.15, -1.59, 0.0, 0.0], dtype=float)
        self.gripper_pos = 850.0
        self.calls: list[tuple[str, tuple]] = []

        # Calibrated reference model for accurate FK responses
        self._model = mujoco.MjModel.from_xml_path(str(MODELS_ROOT / "xarm6" / "xarm6.xml"))
        self._data = mujoco.MjData(self._model)
        if CALIB_PATH.is_file():
            calib = load_kinematics_calibration(CALIB_PATH, robot="xarm6", serial_number=SN)
            apply_calibration_to_model(self._model, calib, self._data)

    @property
    def motion_calls(self) -> list[tuple[str, tuple]]:
        motion_names = {
            "motion_enable",
            "set_mode",
            "set_state",
            "set_servo_angle",
            "set_servo_angle_j",
            "set_servo_cartesian",
            "set_position",
            "set_gripper_enable",
            "set_gripper_position",
            "clean_error",
            "clean_warn",
        }
        return [c for c in self.calls if c[0] in motion_names]

    def get_robot_sn(self):
        self.calls.append(("get_robot_sn", ()))
        return 0, self.sn

    def get_err_warn_code(self):
        self.calls.append(("get_err_warn_code", ()))
        return 0, (self.error_code, self.warn_code)

    def get_state(self):
        self.calls.append(("get_state", ()))
        return 0, self.state

    def get_servo_angle(self, is_radian: bool = True):
        del is_radian
        self.calls.append(("get_servo_angle", ()))
        q = self.virtual_q if self.is_simulation_robot else self.physical_q
        return 0, q.tolist()

    def get_joint_states(self, is_radian: bool = True):
        del is_radian
        self.calls.append(("get_joint_states", ()))
        q = self.virtual_q if self.is_simulation_robot else self.physical_q
        return 0, (q.tolist(), [0.0] * 7, self.tau.tolist())

    def get_joints_torque(self):
        self.calls.append(("get_joints_torque", ()))
        return 0, self.tau.tolist()

    def get_forward_kinematics(self, angles, input_is_radian: bool = True, return_is_radian: bool = True):
        del input_is_radian, return_is_radian
        self.calls.append(("get_forward_kinematics", (tuple(angles),)))
        q = np.asarray(angles[:6], dtype=float)
        pos_m, quat = fk_pose(self._model, self._data, "attachment_site", q)
        tcp_pose = flange_m_quat_to_tcp_mm_rpy(pos_m, quat, self.tcp_offset)
        return 0, tcp_pose.tolist()

    def get_inverse_kinematics(self, pose, input_is_radian: bool = True, return_is_radian: bool = True):
        del input_is_radian, return_is_radian
        self.calls.append(("get_inverse_kinematics", (tuple(pose),)))
        return 0, self.physical_q.tolist()

    def get_position(self, is_radian: bool = True):
        del is_radian
        self.calls.append(("get_position", ()))
        q = (self.virtual_q if self.is_simulation_robot else self.physical_q)[:6]
        _, tcp_pose = self.get_forward_kinematics(q)
        return 0, tcp_pose

    def motion_enable(self, enable: bool = True):
        self.calls.append(("motion_enable", (bool(enable),)))
        return 0

    def set_mode(self, mode: int = 0):
        self.calls.append(("set_mode", (int(mode),)))
        self.mode = int(mode)
        return 0

    def set_state(self, state: int = 0):
        self.calls.append(("set_state", (int(state),)))
        self.state = 2 if int(state) == 0 else int(state)
        return 0

    def set_simulation_robot(self, on_off: bool):
        self.calls.append(("set_simulation_robot", (bool(on_off),)))
        self.is_simulation_robot = bool(on_off)
        if on_off:
            self.virtual_q = self.physical_q.copy()
        return 0

    def set_servo_angle(self, angle=None, speed=0, acc=0, mvacc=0, wait=True, is_radian=True):
        del speed, acc, mvacc, wait, is_radian
        q = np.asarray(angle, dtype=float)
        self.calls.append(("set_servo_angle", (tuple(q.tolist()),)))
        if self.is_simulation_robot:
            self.virtual_q[: len(q)] = q
        else:
            self.physical_q[: len(q)] = q
        return 0

    def set_servo_angle_j(self, angles=None, speed=0, mvacc=0, mvtime=0, is_radian=True):
        del speed, mvacc, mvtime, is_radian
        q = np.asarray(angles, dtype=float)
        self.calls.append(("set_servo_angle_j", (tuple(q.tolist()),)))
        if self.is_simulation_robot:
            self.virtual_q[: len(q)] = q
        else:
            self.physical_q[: len(q)] = q
        return 0

    def get_gripper_position(self):
        self.calls.append(("get_gripper_position", ()))
        return 0, self.gripper_pos

    def set_gripper_enable(self, enable: bool):
        self.calls.append(("set_gripper_enable", (bool(enable),)))
        return 0

    def set_gripper_position(self, pos: float, wait: bool = True):
        del wait
        self.calls.append(("set_gripper_position", (float(pos),)))
        self.gripper_pos = float(pos)
        return 0

    def disconnect(self):
        self.calls.append(("disconnect", ()))
        self.connected = False


class CalibrationAndTransformTests(unittest.TestCase):
    def test_load_valid_calibration_and_patch_model(self):
        calib = load_kinematics_calibration(CALIB_PATH, robot="xarm6", serial_number=SN)
        self.assertIsInstance(calib, KinematicsCalibration)
        self.assertEqual(calib.robot_key, "xarm6")
        self.assertEqual(calib.serial_number, SN)
        self.assertEqual(len(calib.sha256), 64)

        model = mujoco.MjModel.from_xml_path(str(MODELS_ROOT / "xarm6" / "xarm6.xml"))
        data = mujoco.MjData(model)
        q_test = np.array([0.2, -0.6, -0.9, 0.4, 0.5, -0.1])
        pos_before, _ = fk_pose(model, data, "attachment_site", q_test)
        apply_calibration_to_model(model, calib, data)
        pos_after, _ = fk_pose(model, data, "attachment_site", q_test)
        # Per-unit calibration changes FK by a few millimeters
        diff_mm = float(np.linalg.norm(pos_after - pos_before) * 1000.0)
        self.assertGreater(diff_mm, 0.1)
        self.assertLess(diff_mm, 15.0)

    def test_calibration_rejects_mismatched_sn_robot_or_schema(self):
        with self.assertRaisesRegex(CalibrationError, "serial_number"):
            load_kinematics_calibration(CALIB_PATH, robot="xarm6", serial_number="XI130505WRONG99")
        with self.assertRaisesRegex(CalibrationError, "robot_key"):
            load_kinematics_calibration(CALIB_PATH, robot="lite6", serial_number=SN)

        raw = yaml.safe_load(CALIB_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            bad_path = Path(tmp) / "bad.yaml"
            bad = copy.deepcopy(raw)
            bad["schema_version"] = 2
            bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
            with self.assertRaisesRegex(CalibrationError, "schema_version"):
                load_kinematics_calibration(bad_path)

            bad = copy.deepcopy(raw)
            bad["units"]["position"] = "mm"
            bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
            with self.assertRaisesRegex(CalibrationError, "units"):
                load_kinematics_calibration(bad_path)

            bad = copy.deepcopy(raw)
            del bad["joints"]["joint6"]
            bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
            with self.assertRaisesRegex(CalibrationError, "joint set mismatch"):
                load_kinematics_calibration(bad_path)

    def test_flange_tcp_pose_round_trip(self):
        flange = np.array([310.5, -25.0, 280.2, -3.1, 0.05, -0.4])
        offset = np.array([0.0, 0.0, 172.0, 0.0, 0.0, 0.0])
        tcp = pose_flange_to_tcp(flange, offset)
        recovered = pose_tcp_to_flange(tcp, offset)
        np.testing.assert_allclose(recovered[:3], flange[:3], atol=1e-6)

        pos_m, quat = tcp_mm_rpy_to_flange_m_quat(tcp, offset)
        tcp_again = flange_m_quat_to_tcp_mm_rpy(pos_m, quat, offset)
        np.testing.assert_allclose(tcp_again[:3], tcp[:3], atol=1e-6)


class XArmRobotAndSafetyGateTests(unittest.TestCase):
    def test_read_only_observations_never_send_motion_commands(self):
        fake = FakeXArmAPI()
        with XArmRobot(
            robot="xarm6",
            gripper="gripper_g2",
            calibration=CALIB_PATH,
            read_only=True,
            arm_api=fake,
        ) as robot:
            self.assertIsInstance(robot, RobotInterface)
            self.assertEqual(robot.n_arm, 6)
            self.assertTrue(robot.read_only)
            self.assertFalse(robot.motion_unlocked)
            q = robot.get_joint_positions()
            self.assertEqual(q.shape, (6,))
            pos, quat = robot.get_tcp_pose()
            self.assertEqual(pos.shape, (3,))
            self.assertEqual(quat.shape, (4,))
            q2, qd, tau = robot.get_joint_states()
            np.testing.assert_allclose(q, q2)
            self.assertEqual(qd.shape, (6,))
            self.assertEqual(tau.shape, (6,))
            state = robot.read_state()
            self.assertEqual(state["serial_number"], SN)
            self.assertEqual(fake.motion_calls, [])

            # Every write/motion method must raise SafetyGateError and emit zero motion SDK calls
            for call in (
                lambda: robot.set_joint_positions(np.zeros(6)),
                lambda: robot.move_joints(np.zeros(6), duration=1.0),
                lambda: robot.move_tcp(pos, quat, duration=1.0),
                lambda: robot.execute_trajectory(
                    JointTrajectory(np.array([0.0, 0.1]), np.zeros((2, 6)), np.zeros((2, 6)), np.zeros((2, 6)))
                ),
                lambda: robot.set_gripper(0.5),
                lambda: robot.set_gripper_gap(0.022),
                lambda: robot.open_gripper(),
                lambda: robot.close_gripper(),
            ):
                with self.assertRaises(SafetyGateError):
                    call()
            self.assertEqual(fake.motion_calls, [])

    def test_motion_approval_cannot_be_forged_and_enforces_all_gates(self):
        fake = FakeXArmAPI()
        with self.assertRaises(SafetyGateError):
            MotionApproval(
                robot="xarm6",
                serial_number=SN,
                calibration_sha256="abc",
                config_sha256="def",
                program_sha256="ghi",
                cycles=1,
                issued_at_unix_s=0.0,
            )

        # Missing calibration rejected
        with self.assertRaisesRegex(SafetyGateError, "calibration"):
            authorize_real_motion(
                fake, robot="xarm6", calibration=None, preflight_ok=True, confirm_real=True, cycles=1
            )
        # Missing confirm_real rejected
        with self.assertRaisesRegex(SafetyGateError, "--confirm-real"):
            authorize_real_motion(
                fake, robot="xarm6", calibration=CALIB_PATH, preflight_ok=True, confirm_real=False, cycles=1
            )
        # Failed preflight rejected
        with self.assertRaisesRegex(SafetyGateError, "preflight"):
            authorize_real_motion(
                fake, robot="xarm6", calibration=CALIB_PATH, preflight_ok=False, confirm_real=True, cycles=1
            )
        # Multi-cycle cap rejected
        with self.assertRaisesRegex(SafetyGateError, "single-cycle"):
            authorize_real_motion(
                fake, robot="xarm6", calibration=CALIB_PATH, preflight_ok=True, confirm_real=True, cycles=2
            )

        # Active controller error or STOP state rejected without auto-clearing
        fake.error_code = 19
        with self.assertRaisesRegex(SafetyGateError, "error_code=19"):
            authorize_real_motion(
                fake, robot="xarm6", calibration=CALIB_PATH, preflight_ok=True, confirm_real=True, cycles=1
            )
        fake.error_code = 0
        fake.state = 4
        with self.assertRaisesRegex(SafetyGateError, "STOP"):
            authorize_real_motion(
                fake, robot="xarm6", calibration=CALIB_PATH, preflight_ok=True, confirm_real=True, cycles=1
            )
        fake.state = 5
        self.assertEqual(fake.motion_calls, [])

        # Valid approval unlocks motion
        approval = authorize_real_motion(
            fake, robot="xarm6", calibration=CALIB_PATH, preflight_ok=True, confirm_real=True, cycles=1
        )
        with XArmRobot(
            robot="xarm6",
            gripper="gripper_g2",
            calibration=CALIB_PATH,
            read_only=False,
            approval=approval,
            arm_api=fake,
        ) as robot:
            self.assertTrue(robot.motion_unlocked)
            target_q = np.array([0.1, -0.6, -0.5, 0.1, 0.8, -0.1])
            robot.move_joints(target_q, duration=1.0, wait=True)
            np.testing.assert_allclose(robot.get_joint_positions(), target_q)
            robot.set_gripper_gap(0.022)
            self.assertAlmostEqual(robot.get_gripper_gap(), 0.022, places=3)


class KinematicsAndSdkSimValidationTests(unittest.TestCase):
    def test_fk_ik_validation_with_controller_is_read_only_and_passes(self):
        fake = FakeXArmAPI()
        report = validate_fk_ik_with_controller(fake, robot="xarm6", calibration=CALIB_PATH)
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["serial_number"], SN)
        self.assertLess(report["max_fk_pos_error_mm"], 1e-3)
        self.assertLess(report["max_ik_pos_error_mm"], 1.0)
        self.assertEqual(fake.motion_calls, [])

    def test_sdk_simulation_replays_stream_restores_mode_and_checks_zero_physical_drift(self):
        fake = FakeXArmAPI()
        physical_before = fake.physical_q[:6].copy()
        checker = CollisionChecker(MODELS_ROOT / "xarm6" / "scene.xml", ignore_pairs=[("link_base", "world")])
        shadow = np.linspace(
            np.zeros(6),
            np.array([0.0, -0.8, -1.2, 1.0, 0.3, 0.0]),
            8,
        )
        evidence = run_sdk_simulation(
            fake,
            robot="xarm6",
            calibration=CALIB_PATH,
            shadow_joint_stream_rad=shadow,
            checker=checker,
            preflight_ok=True,
            cycles=1,
            step_dt_s=0.0,
        )
        self.assertTrue(evidence.passed, evidence.failures)
        self.assertFalse(fake.is_simulation_robot)
        self.assertEqual(evidence.samples, 8)
        self.assertEqual(evidence.physical_joint_drift_rad, 0.0)
        np.testing.assert_array_equal(fake.physical_q[:6], physical_before)

    def test_sdk_simulation_always_cleans_up_on_midstream_error(self):
        fake = FakeXArmAPI()

        def fail_servo_j(*args, **kwargs):
            del args, kwargs
            return 9

        fake.set_servo_angle_j = fail_servo_j
        checker = CollisionChecker(MODELS_ROOT / "xarm6" / "scene.xml", ignore_pairs=[("link_base", "world")])
        shadow = np.zeros((4, 6))
        with self.assertRaisesRegex(SafetyGateError, "set_servo_angle_j failed"):
            run_sdk_simulation(
                fake,
                robot="xarm6",
                calibration=CALIB_PATH,
                shadow_joint_stream_rad=shadow,
                checker=checker,
                preflight_ok=True,
                cycles=1,
                step_dt_s=0.0,
            )
        self.assertFalse(fake.is_simulation_robot)


class CliAndDynamicsGateTests(unittest.TestCase):
    def _run_cli(self, args: list[str]) -> int:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                return int(cli_main(args))
            except SystemExit as exc:
                return int(exc.code)

    def test_cli_rejects_missing_calibration_unconfirmed_real_and_multicycle(self):
        fake = FakeXArmAPI()

        def connect_fake(_ip: str):
            fake.connected = True
            return fake

        with patch("ufactory_mujoco.xarm_robot._connect_xarm_sdk", side_effect=connect_fake):
            # Missing calibration
            self.assertEqual(
                self._run_cli(["pick-place", "--mode", "sdk-sim", "--ip", "192.168.1.60"]),
                1,
            )
            # Multi-cycle cap (cycles=2) rejected in sdk-sim and real
            self.assertEqual(
                self._run_cli([
                    "pick-place", "--mode", "sdk-sim", "--ip", "192.168.1.60",
                    "--calibration", str(CALIB_PATH), "--cycles", "2",
                ]),
                1,
            )
            # Real mode without --confirm-real rejected without moving physical joints
            self.assertEqual(
                self._run_cli([
                    "pick-place", "--mode", "real", "--ip", "192.168.1.60",
                    "--calibration", str(CALIB_PATH),
                ]),
                1,
            )
            self.assertEqual(fake.motion_calls, [])

    def test_dynamics_scripts_enforce_confirm_real_and_allow_readonly_no_move(self):
        fake = FakeXArmAPI()

        def make_fake(*_a, **_kw):
            fake.connected = True
            return fake

        wrapper = types.ModuleType("xarm.wrapper")
        wrapper.XArmAPI = make_fake
        with patch.dict(sys.modules, {"xarm": types.ModuleType("xarm"), "xarm.wrapper": wrapper}):
            # Read-only --no-move succeeds without motion calls
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(
                    compare_torque_xarm6.main(["--no-move", "--calibration", str(CALIB_PATH)]),
                    0,
                )
                compare_dynamics_xarm6.main(["--no-move", "--calibration", str(CALIB_PATH)])
            self.assertEqual(fake.motion_calls, [])

            # Moving without --confirm-real is rejected with zero motion calls
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(
                    compare_torque_xarm6.main(["--calibration", str(CALIB_PATH)]),
                    1,
                )
                with self.assertRaises(SystemExit) as exc:
                    compare_dynamics_xarm6.main(["--calibration", str(CALIB_PATH)])
                self.assertEqual(exc.exception.code, 1)
            self.assertEqual(fake.motion_calls, [])


if __name__ == "__main__":
    unittest.main()

