"""Pick-and-place, box-packing, and preflight collision/detour tests."""

from pathlib import Path
import unittest
import importlib
import mujoco
import numpy as np
from ufactory_mujoco.config import ConfigError, load_task_config, resolve_arm_scene
from ufactory_mujoco.preflight import GRIP_DOWN_QUAT, tcp_task_heights
import tempfile
from ufactory_mujoco.planning import CollisionChecker
from ufactory_mujoco.preflight import (
    PHASES,
    _CHECK_RES,
    make_checker,
    plan_segment,
    preflight_box_pack,
    preflight_pick_place,
)
from ufactory_mujoco.sim_robot import SimRobot
from ufactory_mujoco.preflight import box_task_heights
from ufactory_mujoco import MODELS_ROOT

# ==========================================================================
# From test_pick_place.py
# ==========================================================================

pick_place = importlib.import_module("ufactory_mujoco.tasks.pick_place")

ROOT = Path(__file__).resolve().parent.parent

MODELS_ROOT = MODELS_ROOT

SCENE = resolve_arm_scene("xarm6", "gripper_g2", "pick_place")

class PickPlaceSceneTest(unittest.TestCase):
    def test_scene_props(self):
        """scene_pick_place.xml: 30 mm / 17 g cube free body + visual-only target marker."""
        m = mujoco.MjModel.from_xml_path(str(SCENE))
        cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.assertGreaterEqual(cube_id, 0)
        geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.assertAlmostEqual(m.body_mass[cube_id], 0.017, places=6)
        np.testing.assert_allclose(m.geom_size[geom_id], [0.015, 0.015, 0.015], atol=1e-12)
        # cube is a free body
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        self.assertEqual(int(m.jnt_type[jnt_id]), int(mujoco.mjtJoint.mjJNT_FREE))
        # target marker: visual only, at (0.30, 0.30, 0.015)
        marker_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_marker")
        self.assertGreaterEqual(marker_id, 0)
        self.assertEqual(m.geom_contype[marker_id], 0)
        self.assertEqual(m.geom_conaffinity[marker_id], 0)
        np.testing.assert_allclose(m.geom_pos[marker_id], [0.30, 0.30, 0.015], atol=1e-12)

    def test_simrobot_init_keeps_cube_at_authored_pose(self):
        """Keyframe state init must keep the free-joint cube at its authored qpos0 pose."""
        from ufactory_mujoco import SimRobot

        with SimRobot(SCENE, viewer=False) as robot:
            cube_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            np.testing.assert_allclose(robot.data.body(cube_id).xpos,
                                       [0.30, 0.0, 0.015], atol=1e-9)

    def test_resolve_scene(self):
        config = load_task_config(preset=("pick_place", "xarm6"))
        self.assertEqual(config.scene_path, SCENE)
        with self.assertRaises(ConfigError):
            load_task_config(preset=("pick_place", "xarm6"), overrides={"executor": "servo"})

class IkJointLimitTest(unittest.TestCase):
    def test_solve_ik_respects_joint_limits(self):
        """DLS IK must clamp hinge/slide joints to their ranges (numpy-scalar enum
        comparison silently skipped the clamp before, allowing multi-turn escapes)."""
        from ufactory_mujoco.kinematics import solve_ik

        m = mujoco.MjModel.from_xml_path(str(SCENE))
        d = mujoco.MjData(m)
        q, ok = solve_ik(m, d, "attachment_site", np.array([0.30, 0.0, 0.45]),
                         GRIP_DOWN_QUAT, q_init=np.zeros(m.nq))
        self.assertTrue(ok)
        for j in range(m.njnt):
            if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                adr = m.jnt_qposadr[j]
                lo, hi = m.jnt_range[j]
                # multi-turn equivalents (e.g. -396 deg for -36 deg) must be folded into range
                self.assertGreaterEqual(q[adr], lo - 1e-9, f"joint {m.joint(j).name} below range")
                self.assertLessEqual(q[adr], hi + 1e-9, f"joint {m.joint(j).name} above range")

class PickPlaceEndToEndTest(unittest.TestCase):
    def test_open_gripper_refuses_lift(self):
        """Fingers start open and clear of the cube, so the lift gate must refuse."""
        from ufactory_mujoco import SimRobot
        from ufactory_mujoco.grippers import GraspError

        with SimRobot(SCENE, viewer=False) as robot:
            cube_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            z_before = float(robot.data.body(cube_id).xpos[2])
            with self.assertRaises(GraspError):
                robot.require_bilateral_contact("cube")
            self.assertAlmostEqual(float(robot.data.body(cube_id).xpos[2]), z_before)

    def test_pick_place_headless(self):
        """Full pick-and-place sequence: cube lifted, placed within 50 mm, home within 10 mm."""
        from ufactory_mujoco import SimRobot

        config = load_task_config(preset=("pick_place", "xarm6"))
        with SimRobot(config.scene_path, viewer=False) as robot:
            results = pick_place.run_pick_place(robot, config, cycles=1)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(np.isfinite(r["place_error_mm"]))
        self.assertTrue(r["cube_lifted"], "cube was not lifted after closing the gripper")
        self.assertLess(r["place_error_mm"], 50.0, "place_error_mm above the 50 mm success threshold")
        self.assertLess(r["home_drift_mm"], 10.0, "home_drift_mm above the 10 mm success threshold")
        self.assertEqual(r["robot"], "xarm6")
        self.assertEqual(r["executor"], "servo_cartesian")
        self.assertLess(r["place_error_mm"], 15.0)

class MultiArmPickPlaceTests(unittest.TestCase):
    ARMS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")

    def _config(self, robot, executor):
        return load_task_config(preset=("pick_place", robot), overrides={"executor": executor})

    def test_overlays_resolve_and_lite6_gap_fits(self):
        for robot in self.ARMS:
            with self.subTest(robot=robot):
                config = self._config(robot, "servo_cartesian")
                self.assertEqual(config.scene, "pick_place")
                self.assertTrue(config.scene_path.is_file())
                self.assertEqual(config.scene_path.name, "scene_pick_place.xml")
        lite6 = self._config("lite6", "servo_cartesian")
        task = lite6.pick_place
        self.assertGreaterEqual(task.grasp_gap_m, 0.020)
        self.assertLessEqual(task.grasp_gap_m, 0.038)
        self.assertLess(task.grasp_gap_m, 2 * task.cube_half_m)
        self.assertLess(2 * task.cube_half_m, 0.038)

    def test_grasp_and_place_ik(self):
        from ufactory_mujoco import SimRobot

        for robot_name in self.ARMS:
            config = self._config(robot_name, "servo_j")
            task = config.pick_place
            with self.subTest(robot=robot_name):
                with SimRobot(config.scene_path, viewer=False) as robot:
                    heights, q_home = tcp_task_heights(robot, task)
                    for xy, z in (
                        (task.cube_xy, heights["grasp_z"]),
                        (task.target_xy, heights["lift_z"]),
                    ):
                        q, ok = robot.solve_tcp(
                            [xy[0], xy[1], z], GRIP_DOWN_QUAT, q_init=q_home)
                        self.assertTrue(ok, f"IK failed at {xy} z={z}")
                        self.assertEqual(len(q), robot.model.nq)

    def test_headless_both_executors(self):
        from ufactory_mujoco import SimRobot

        for robot_name in self.ARMS:
            for executor in ("servo_cartesian", "servo_j"):
                with self.subTest(robot=robot_name, executor=executor):
                    config = self._config(robot_name, executor)
                    with SimRobot(config.scene_path, viewer=False) as robot:
                        results = pick_place.run_pick_place(robot, config, cycles=1)
                    self.assertEqual(len(results), 1)
                    report = results[0]
                    self.assertEqual(report["robot"], robot_name)
                    self.assertEqual(report["executor"], executor)
                    self.assertTrue(report["cube_lifted"])
                    self.assertLess(report["place_error_mm"], 50.0)
                    self.assertLess(report["home_drift_mm"], 10.0)

def _cube_in_tcp(robot) -> np.ndarray:
    pos, quat = robot.get_tcp_pose()
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, quat)
    return rot.reshape(3, 3).T @ (robot.data.body("cube").xpos - pos)

class GraspHoldTests(unittest.TestCase):
    """The jaw stalls at the cube surface and the held cube does not creep."""

    @staticmethod
    def _grasp_and_hold(robot_name: str, hold_s: float = 2.0) -> tuple[float, float, float]:
        """Grasp and lift the preset cube, then hold still.

        Returns the deepest finger-cube penetration (m), the measured jaw gap (m) and how
        far the cube moved relative to the TCP during the hold (m).
        """
        from ufactory_mujoco.grippers import identify_gripper
        from ufactory_mujoco.preflight import _close_until_contact
        from ufactory_mujoco.tasks import _common

        config = load_task_config(preset=("pick_place", robot_name))
        task = config.pick_place
        with SimRobot(config.scene_path, viewer=False) as robot:
            mover = _common.Mover(robot, config.executor, task.cart_v_m_s, task.cart_a_m_s2,
                                  np.deg2rad(90.0), 3.5)
            h, q_seed = tcp_task_heights(robot, task)
            x, y = task.cube_xy
            robot.open_gripper(wait=True)
            robot.move_joints(q_seed[:robot.n_arm], duration=3.0)
            for start, end in (("lift_z", "pre_grasp_z"), ("pre_grasp_z", "grasp_z")):
                mover.move(np.array([x, y, h[end]]), mover.duration(h[start] - h[end]),
                           robot.data.qpos.copy())
            _close_until_contact(robot, "cube", task.grasp_gap_m)

            m, d = robot.model, robot.data
            spec = identify_gripper(m)
            cube = m.body("cube").id
            fingers = {m.body(spec.left_body).id, m.body(spec.right_body).id}
            depth = 0.0
            for i in range(d.ncon):
                contact = d.contact[i]
                pair = {int(m.geom_bodyid[contact.geom1]), int(m.geom_bodyid[contact.geom2])}
                if cube in pair and pair & fingers:
                    depth = max(depth, -float(contact.dist))
            gap = robot.get_gripper_gap(measured=True)

            mover.move(np.array([x, y, h["lift_z"]]), mover.duration(h["lift_z"] - h["grasp_z"]),
                       robot.data.qpos.copy())
            before = _cube_in_tcp(robot)
            robot.settle(hold_s)
            drift = float(np.linalg.norm(_cube_in_tcp(robot) - before))
        return depth, gap, drift

    def test_task_scenes_enable_noslip(self):
        for task_name in ("pick_place", "box_pack"):
            for robot_name in ("xarm6", "lite6"):
                with self.subTest(task=task_name, robot=robot_name):
                    config = load_task_config(preset=(task_name, robot_name))
                    model = mujoco.MjModel.from_xml_path(str(config.scene_path))
                    self.assertGreater(model.opt.noslip_iterations, 0)

    def test_g2_pads_stop_at_cube_surface_and_hold_still(self):
        depth, gap, drift = self._grasp_and_hold("xarm6")
        self.assertLess(depth, 1e-3, "pads sank more than 1 mm into the 30 mm cube")
        self.assertGreater(gap, 0.028, "the jaw closed far past the 30 mm cube")
        self.assertLess(drift, 5e-4, "the held cube moved more than 0.5 mm relative to the TCP")

    def test_lite6_held_cube_does_not_creep(self):
        _, _, drift = self._grasp_and_hold("lite6")
        self.assertLess(drift, 5e-4, "the held cube moved more than 0.5 mm relative to the TCP")

class PickPlaceTimingTest(unittest.TestCase):
    """The arm descends to the grasp without stopping and lifts soon after the jaw closes."""

    def test_descent_has_no_stop_and_lift_follows_closing(self):
        from unittest import mock
        from ufactory_mujoco.preflight import _close_until_contact

        for executor in ("servo_cartesian", "servo_j"):
            with self.subTest(executor=executor):
                config = load_task_config(preset=("pick_place", "xarm6"),
                                          overrides={"executor": executor})
                times, tcp, marks = [], [], {}

                def timed_close(robot, obj, gap):
                    marks["close"] = robot.data.time
                    gap = _close_until_contact(robot, obj, gap)
                    marks["lift"] = robot.data.time
                    return gap

                with SimRobot(config.scene_path, viewer=False) as robot:
                    h, _ = tcp_task_heights(robot, config.pick_place)
                    above = np.array([*config.pick_place.cube_xy, h["lift_z"]])

                    def record():
                        times.append(robot.data.time)
                        tcp.append(robot.get_tcp_pose()[0])

                    with robot.step_hook(record), \
                            mock.patch.object(pick_place, "_close_until_contact", timed_close):
                        results = pick_place.run_pick_place(robot, config, cycles=1)
                self.assertTrue(results[0]["cube_lifted"])

                t, p = np.array(times), np.array(tcp)
                arrived = t[np.argmax(np.linalg.norm(p - above, axis=1) < 2e-3)]
                window = (t >= arrived) & (t <= marks["close"])
                speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(t)
                still = np.r_[False, speed < 2e-3] & window
                longest = run = 0
                for flag in still:
                    run = run + 1 if flag else 0
                    longest = max(longest, run)
                dt = float(np.median(np.diff(t)))
                self.assertLess(longest * dt, 0.2, "the arm stopped on its way down to the cube")
                self.assertLess(marks["lift"] - marks["close"], 1.0,
                                "the lift started more than 1 s after the jaw began to close")

# ==========================================================================
# From test_box_pack.py
# ==========================================================================

box_pack = importlib.import_module("ufactory_mujoco.tasks.box_pack")

ROOT = Path(__file__).resolve().parent.parent

MODELS_ROOT = MODELS_ROOT

ARMS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")

def _config(robot, executor=None, overrides=None):
    merged = dict(overrides or {})
    if executor:
        merged["executor"] = executor
    return load_task_config(preset=("box_pack", robot), overrides=merged or None)

class BoxPackConfigTest(unittest.TestCase):
    def test_default_section(self):
        config = load_task_config()
        task = config.box_pack
        self.assertEqual(len(task.feed_xy), 3)
        self.assertEqual(len(task.drop_xy), 3)
        self.assertEqual(task.speed_scale, 1.0)
        self.assertEqual(task.release_above_rim_m, 0.05)

    def test_overlays_resolve(self):
        for robot in ARMS:
            with self.subTest(robot=robot):
                config = _config(robot)
                self.assertEqual(config.scene, "box_pack")
                self.assertTrue(config.scene_path.is_file())
                self.assertEqual(config.scene_path.name, "scene_box_pack.xml")
                self.assertEqual(len(config.box_pack.feed_xy), 3)

    def test_invalid_values(self):
        cases = (
            "box_pack:\n  speed_scale: 0\n",
            "box_pack:\n  feed_xy: []\n",
            "box_pack:\n  feed_xy: [[0.3, 0.0, 0.1]]\n",
            "box_pack:\n  drop_xy: [[0.3, 0.3]]\n",
            "box_pack:\n  bogus: 1\n",
        )
        for text in cases:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                overlay = Path(tmp) / "overlay.yaml"
                overlay.write_text(text, encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_task_config(overlay)

class BoxPackSceneTest(unittest.TestCase):
    def test_scene_props_mirror_yaml(self):
        for robot in ARMS:
            with self.subTest(robot=robot):
                config = _config(robot)
                task = config.box_pack
                m = mujoco.MjModel.from_xml_path(str(config.scene_path))
                box_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pack_box")
                self.assertGreaterEqual(box_id, 0)
                np.testing.assert_allclose(m.body_pos[box_id][:2], task.box_center_xy,
                                           atol=1e-12)
                for i, xy in enumerate(task.feed_xy):
                    body = f"cube_{i}"
                    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
                    self.assertGreaterEqual(body_id, 0)
                    geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                                f"{body}_geom")
                    np.testing.assert_allclose(m.geom_size[geom_id],
                                               [task.cube_half_m] * 3, atol=1e-12)
                    self.assertAlmostEqual(float(m.body_mass[body_id]),
                                           task.cube_mass_kg, places=9)
                    jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, body)
                    self.assertEqual(int(m.jnt_type[jnt]), int(mujoco.mjtJoint.mjJNT_FREE))
                    adr = int(m.jnt_qposadr[jnt])
                    np.testing.assert_allclose(m.qpos0[adr:adr + 2], xy, atol=1e-12)
                # box geoms mirror the YAML dimensions
                hx, hy = task.box_inner_half_xy
                w, f, rim = task.box_wall_m, task.box_floor_m, task.box_rim_z_m
                floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box_floor")
                np.testing.assert_allclose(m.geom_size[floor], [hx + w, hy + w, f / 2],
                                           atol=1e-12)
                np.testing.assert_allclose(m.geom_pos[floor], [0, 0, f / 2], atol=1e-12)
                wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box_wall_px")
                np.testing.assert_allclose(
                    m.geom_size[wall], [w / 2, hy + w, (rim - f) / 2], atol=1e-12)
                np.testing.assert_allclose(
                    m.geom_pos[wall], [hx + w / 2, 0, (rim + f) / 2], atol=1e-12)

    def test_lite6_feed_stands(self):
        config = _config("lite6")
        m = mujoco.MjModel.from_xml_path(str(config.scene_path))
        for i, xy in enumerate(config.box_pack.feed_xy):
            geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"feed_stand_{i}")
            self.assertGreaterEqual(geom, 0)
            np.testing.assert_allclose(m.geom_pos[geom][:2], xy, atol=1e-12)

class BoxPackPreflightTest(unittest.TestCase):
    def test_dry_run_all_phases_ok_no_detour(self):
        from ufactory_mujoco import SimRobot
        from ufactory_mujoco.preflight import preflight_box_pack

        pack_phases = ("接近", "抓取", "抬起", "运送", "释放", "回位")

        config = _config("xarm6")
        with SimRobot(config.scene_path, viewer=False) as robot:
            report = preflight_box_pack(robot, config, cycles=3)
            self.assertEqual(robot.data.time, 0.0)
        self.assertTrue(report.ok)
        self.assertEqual(len(report.phases), 3 * len(pack_phases))
        for i in range(3):
            names = [p.name for p in report.phases[i * 6:(i + 1) * 6]]
            self.assertEqual(names, [f"{name}#{i + 1}" for name in pack_phases])
        self.assertTrue(all(not seg.detour for seg in report.segments.values()))

    def test_drop_outside_box_fails_without_stepping(self):
        from ufactory_mujoco import SimRobot
        from ufactory_mujoco.preflight import preflight_box_pack

        config = _config("xarm6", overrides={
            "box_pack": {"drop_xy": [[0.30, 0.42], [0.30, 0.30], [0.345, 0.30]]},
        })
        with SimRobot(config.scene_path, viewer=False) as robot:
            report = preflight_box_pack(robot, config, cycles=1)
            self.assertEqual(robot.data.time, 0.0)
        self.assertFalse(report.ok)
        phase = next(p for p in report.phases if p.name == "释放#1")
        self.assertEqual(phase.status, "fail")
        self.assertEqual(phase.reason, "out_of_box")

    def test_unreachable_feed_fails_without_stepping(self):
        from ufactory_mujoco import SimRobot
        from ufactory_mujoco.preflight import preflight_box_pack

        config = _config("xarm6", overrides={
            "box_pack": {"feed_xy": [[1.5, 0.0], [0.30, -0.18], [0.39, -0.18]]},
        })
        with SimRobot(config.scene_path, viewer=False) as robot:
            report = preflight_box_pack(robot, config, cycles=1)
            self.assertEqual(robot.data.time, 0.0)
        self.assertFalse(report.ok)
        phase = next(p for p in report.phases if p.name == "接近#1")
        self.assertEqual(phase.reason, "ik")

    def test_unreachable_box_center_fails_without_stepping(self):
        from ufactory_mujoco import SimRobot
        from ufactory_mujoco.preflight import preflight_box_pack

        config = _config("xarm6", overrides={"box_pack": {"box_center_xy": [1.5, 0.0]}})
        with SimRobot(config.scene_path, viewer=False) as robot:
            report = preflight_box_pack(robot, config, cycles=1)
            self.assertEqual(robot.data.time, 0.0)
        self.assertFalse(report.ok)
        self.assertTrue(all(p.status == "fail" and p.reason == "ik"
                            for p in report.phases if not p.name.startswith("抓取")))

class BoxPackSimTest(unittest.TestCase):
    def test_xarm6_single_cycle_free_fall(self):
        from ufactory_mujoco import SimRobot

        config = _config("xarm6")
        task = config.box_pack
        with SimRobot(config.scene_path, viewer=False) as robot:
            results = box_pack.run_box_pack(robot, config, cycles=1)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(r["cube_lifted"])
        self.assertTrue(r["cube_rested"])
        self.assertTrue(r["in_box"])
        self.assertGreaterEqual(r["bounds_margin_mm"], -1.0)
        self.assertLess(r["home_drift_mm"], 10.0)
        # released at rim + 50 mm (cube bottom), then fell into the box
        release_bottom = r["released_z"] - task.cube_half_m
        self.assertAlmostEqual(release_bottom,
                               task.box_rim_z_m + task.release_above_rim_m, delta=0.01)
        self.assertLess(r["rest_xyz"][2], r["released_z"] - 0.03)

    def test_five_arms_headless(self):
        from ufactory_mujoco import SimRobot

        for robot_name in ARMS:
            with self.subTest(robot=robot_name):
                config = _config(robot_name)
                with SimRobot(config.scene_path, viewer=False) as robot:
                    results = box_pack.run_box_pack(robot, config, cycles=1)
                self.assertEqual(len(results), 1)
                r = results[0]
                self.assertTrue(r["cube_lifted"])
                self.assertTrue(r["in_box"])
                self.assertLess(r["home_drift_mm"], 10.0)

    def test_servo_j_executor(self):
        from ufactory_mujoco import SimRobot

        for robot_name in ARMS:
            with self.subTest(robot=robot_name):
                config = _config(robot_name, executor="servo_j")
                with SimRobot(config.scene_path, viewer=False) as robot:
                    results = box_pack.run_box_pack(robot, config, cycles=1)
                self.assertEqual(len(results), 1)
                r = results[0]
                self.assertTrue(r["cube_lifted"])
                self.assertTrue(r["in_box"])
                self.assertLess(r["home_drift_mm"], 10.0)

    def test_multi_cycle_all_in_box(self):
        from ufactory_mujoco import SimRobot

        config = _config("xarm6")
        with SimRobot(config.scene_path, viewer=False) as robot:
            results = box_pack.run_box_pack(robot, config, cycles=3)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(r["cube_lifted"])
            self.assertTrue(r["in_box"])
            self.assertLess(r["home_drift_mm"], 10.0)

    def test_speed_scale_shortens_motion(self):
        from ufactory_mujoco import SimRobot

        times = []
        for speed in (1.0, 2.0):
            config = _config("xarm6", overrides={"box_pack": {"speed_scale": speed}})
            with SimRobot(config.scene_path, viewer=False) as robot:
                results = box_pack.run_box_pack(robot, config, cycles=1)
                times.append(robot.data.time)
            self.assertTrue(results[0]["in_box"])
        self.assertLess(times[1], times[0] * 0.8)

# ==========================================================================
# From test_preflight.py
# ==========================================================================

ROOT = Path(__file__).resolve().parent.parent

OBSTACLES = MODELS_ROOT / "xarm6" / "scene_obstacles.xml"

def _key(model, name):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    return model.key(key_id).qpos.copy()

class PreflightTests(unittest.TestCase):
    def test_joint_limit_fails_without_stepping(self):
        config = load_task_config(preset=("pick_place", "xarm6"))
        with SimRobot(config.scene_path, viewer=False) as robot:
            checker = make_checker(config.scene_path)
            q0 = robot.get_joint_positions()
            q1 = checker.q_hi + 0.2
            _, reason = plan_segment(q0, q1, checker)
            self.assertEqual(reason, "joint_limit")
            self.assertEqual(robot.data.time, 0.0)

    def test_penetrating_goal_fails(self):
        checker = CollisionChecker(OBSTACLES, safety_margin=0.0, query_margin=0.02)
        model = checker.model
        start = _key(model, "start")
        goal = _key(model, "goal")
        blocked = None
        for t in np.linspace(0.0, 1.0, 21):
            q = (1.0 - t) * start + t * goal
            if checker.in_collision(q):
                blocked = q
                break
        self.assertIsNotNone(blocked)
        _, reason = plan_segment(blocked, blocked, checker)
        self.assertEqual(reason, "collision")

    def test_obstacle_segment_detours(self):
        checker = CollisionChecker(OBSTACLES, safety_margin=0.0, query_margin=0.02)
        model = checker.model
        start = _key(model, "start")
        goal = _key(model, "goal")
        path, reason = plan_segment(start, goal, checker, seed=0)
        self.assertIsNone(reason)
        self.assertGreater(path.shape[0], 2)
        for q in path:
            self.assertFalse(checker.in_collision(q))

    def test_free_segment_is_the_endpoints(self):
        config = load_task_config(preset=("pick_place", "xarm6"))
        checker = make_checker(config.scene_path)
        with SimRobot(config.scene_path, viewer=False) as robot:
            q0 = robot.get_joint_positions()
        q1 = q0.copy()
        q1[5] = np.clip(q1[5] + 0.05, checker.q_lo[5], checker.q_hi[5])
        path, reason = plan_segment(q0, q1, checker)
        self.assertIsNone(reason)
        self.assertEqual(path.shape[0], 2)
        np.testing.assert_allclose(path[0], q0)
        np.testing.assert_allclose(path[1], q1)

    def test_xarm6_dry_run_phases(self):
        import importlib
        pick_place = importlib.import_module("ufactory_mujoco.tasks.pick_place")
        config = load_task_config(preset=("pick_place", "xarm6"))
        with SimRobot(config.scene_path, viewer=False) as robot:
            report = pick_place.run_pick_place(robot, config, dry_run=True)
            self.assertEqual(robot.data.time, 0.0)
        self.assertTrue(report.ok)
        self.assertEqual([phase.name for phase in report.phases], list(PHASES))
        self.assertTrue(all(phase.status == "ok" for phase in report.phases))
        self.assertTrue(all(not segment.detour for segment in report.segments.values()))

    def test_xarm7_preflight_idempotent_and_matches_execution_geometry(self):
        self.assertAlmostEqual(_CHECK_RES, 0.02)

        pp_cfg = load_task_config(preset=("pick_place", "xarm7"))
        with SimRobot(pp_cfg.scene_path, viewer=False) as robot:
            plan1 = preflight_pick_place(robot, pp_cfg)
            plan2 = preflight_pick_place(robot, pp_cfg)
            self.assertTrue(plan1.ok)
            self.assertTrue(plan2.ok)
            q_home_1 = plan1.segments["return_home"].waypoints[-1]
            q_home_2 = plan2.segments["return_home"].waypoints[-1]
            np.testing.assert_allclose(q_home_1, q_home_2, atol=1e-12)
            # Perturb eval_data to simulate mid-run state before execution geometry query
            robot.eval_data.qpos[:7] = np.array([0.4, -0.5, 0.3, 1.1, -0.2, 0.8, -0.4])
            _, q_exec_pp_full = tcp_task_heights(robot, pp_cfg.pick_place)
            np.testing.assert_allclose(q_exec_pp_full[:7], q_home_1, atol=1e-12)

        bp_cfg = load_task_config(preset=("box_pack", "xarm7"))
        with SimRobot(bp_cfg.scene_path, viewer=False) as robot:
            bp1 = preflight_box_pack(robot, bp_cfg)
            bp2 = preflight_box_pack(robot, bp_cfg)
            self.assertTrue(bp1.ok)
            self.assertTrue(bp2.ok)
            np.testing.assert_allclose(
                bp1.segments["approach_deploy#1"].waypoints[-1],
                bp2.segments["approach_deploy#1"].waypoints[-1],
                atol=1e-12,
            )
            _, q_bp_ref = box_task_heights(robot, bp_cfg.box_pack)
            robot.eval_data.qpos[:7] = np.array([-0.3, -0.4, -0.2, 0.9, 0.3, 0.7, 0.2])
            _, q_exec_bp = box_task_heights(robot, bp_cfg.box_pack)
            np.testing.assert_allclose(q_exec_bp[:7], q_bp_ref[:7], atol=1e-12)

if __name__ == "__main__":
    unittest.main()
