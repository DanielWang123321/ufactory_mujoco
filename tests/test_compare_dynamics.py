"""Offline regressions for recording payload metadata and replay selection."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import compare_dynamics_xarm6 as comparison
from ufactory_mujoco.dynamics import ArmDynamics

ROOT = Path(__file__).resolve().parent.parent
METADATA = {"version": 1, "payload": {"mass_kg": 1.7, "com_m": [0.002, -0.003, 0.021],
                                     "frame": "attachment_site"}}


class RecordedPayloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.csv_path = Path(self.tmp.name) / "recording.csv"
        self.meta_path = self.csv_path.with_suffix(".meta.json")

    def test_save_preserves_csv_columns_and_payload_in_meters(self):
        rows = np.arange(3 * 19, dtype=float).reshape(3, 19)
        payload = (1.7, np.array([0.002, -0.003, 0.021]))
        self.assertEqual(comparison.save_recording(self.csv_path, rows, payload), self.meta_path)
        with self.csv_path.open(newline="") as f:
            self.assertEqual(next(csv.reader(f)), ["t"] + [f"q{j}" for j in range(6)]
                             + [f"qd{j}" for j in range(6)] + [f"tau{j}" for j in range(6)])
        np.testing.assert_array_equal(np.loadtxt(self.csv_path, delimiter=",", skiprows=1), rows)
        self.assertEqual(json.loads(self.meta_path.read_text()), METADATA)
        self.assertEqual(comparison.load_payload(self.csv_path), (1.7, [0.002, -0.003, 0.021]))

    def test_explicit_payload_overrides_metadata_and_converts_mm(self):
        for metadata in (json.dumps(METADATA), "{malformed", None):
            with self.subTest(metadata=metadata):
                if metadata is None:
                    self.meta_path.unlink()
                else:
                    self.meta_path.write_text(metadata)
                self.assertEqual(comparison.load_payload(self.csv_path, [2.3, 10, -20, 30]),
                                 (2.3, [0.01, -0.02, 0.03]))
                self.assertEqual(comparison.load_payload(self.csv_path, [0, 0, 0, 0]),
                                 (0.0, [0, 0, 0]))

    def test_zero_payload_is_restored(self):
        comparison.save_recording(self.csv_path, np.zeros((3, 19)), (0, [0.01, 0, 0.02]))
        self.assertEqual(comparison.load_payload(self.csv_path), (0, [0.01, 0, 0.02]))

    def test_legacy_csv_without_payload_errors_before_replay(self):
        self.csv_path.write_text(",".join(comparison.CSV_HEADER) + "\n")
        with self.assertRaisesRegex(ValueError, "legacy CSV requires --payload"):
            comparison.load_payload(self.csv_path)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), patch.object(comparison, "replay") as replay:
            with self.assertRaises(SystemExit) as exc:
                comparison.main(["--replay", str(self.csv_path)])
            self.assertEqual(exc.exception.code, 2)
            replay.assert_not_called()
        self.assertIn("requires --payload", stderr.getvalue())

    def test_invalid_metadata_is_rejected(self):
        invalid = ["{", "[]", "null", "{}"]
        for version in (2, True, "1", None):
            metadata = copy.deepcopy(METADATA)
            metadata["version"] = version
            invalid.append(json.dumps(metadata))
        for key, values in {
            "mass_kg": (-1, float("nan"), float("inf"), True, "1.7", None, 10 ** 400),
            "com_m": ([0, 0], [0, 0, 0, 0], [[0, 0, 0]], [0, float("nan"), 0],
                      [0, 0, float("inf")], [0, "1", 0], [True, 0, 0], None),
            "frame": ("world", "flange", None),
        }.items():
            for value in values:
                metadata = copy.deepcopy(METADATA)
                metadata["payload"][key] = value
                invalid.append(json.dumps(metadata))
            metadata = copy.deepcopy(METADATA)
            del metadata["payload"][key]
            invalid.append(json.dumps(metadata))
        for payload in (None, [], {}):
            invalid.append(json.dumps({"version": 1, "payload": payload}))
        for raw in invalid:
            with self.subTest(metadata=raw):
                self.meta_path.write_text(raw)
                with self.assertRaisesRegex(ValueError, "invalid payload metadata"):
                    comparison.load_payload(self.csv_path)

    def test_invalid_explicit_payload_is_rejected_without_robot_import(self):
        for args in (("-1", "0", "0", "0"), ("nan", "0", "0", "0"),
                     ("1", "0", "inf", "0")):
            with self.subTest(payload=args), contextlib.redirect_stderr(io.StringIO()):
                with patch.dict(sys.modules, {"xarm": None, "xarm.wrapper": None}):
                    with self.assertRaises(SystemExit) as exc:
                        comparison.main(["--payload", *args])
                    self.assertEqual(exc.exception.code, 2)

    def test_recording_main_saves_effective_robot_or_override_payload(self):
        arm = MagicMock()
        arm.tcp_load = [3.0, [0, 0, 21]]
        arm.get_err_warn_code.return_value = (0, (0, 0))
        arm.get_joint_states.return_value = (0, ([0] * 6, [0] * 6, [0] * 6))
        arm.get_joints_torque.return_value = (0, [0] * 6)
        wrapper = types.ModuleType("xarm.wrapper")
        wrapper.XArmAPI = MagicMock(return_value=arm)
        rows = np.zeros((3, 19))
        for args, expected in (([], (3.0, [0, 0, 0.021])),
                               (["--payload", "2", "10", "20", "30"], (2.0, [0.01, 0.02, 0.03])),
                               (["--payload", "0", "0", "0", "0"], (0.0, [0, 0, 0]))):
            with self.subTest(args=args), contextlib.redirect_stdout(io.StringIO()):
                with patch.dict(sys.modules, {"xarm": types.ModuleType("xarm"), "xarm.wrapper": wrapper}), \
                        patch.object(comparison, "OUT_DIR", Path(self.tmp.name)), \
                        patch.object(comparison, "record", return_value=rows), \
                        patch.object(comparison, "replay") as replay, \
                        patch.object(comparison.time, "strftime", return_value="offline"):
                    comparison.main(args)
                    self.assertEqual(comparison.load_payload(Path(self.tmp.name) / "run_offline.csv"), expected)
                    self.assertEqual(replay.call_args.args[1], expected)
                    arm.disconnect.assert_called()

    def test_offline_main_replay_recovers_and_overrides_payload(self):
        cases = ((METADATA, None, (1.7, [0.002, -0.003, 0.021])),
                 (METADATA, [2.3, 10, -20, 30], (2.3, [0.01, -0.02, 0.03])),
                 ({"version": 1, "payload": {"mass_kg": 0, "com_m": [0, 0, 0],
                                             "frame": "attachment_site"}}, None, (0, [0, 0, 0])),
                 (METADATA, [0, 0, 0, 0], (0, [0, 0, 0])),
                 (None, [3, 0, 0, 21], (3, [0, 0, 0.021])),
                 ("invalid JSON", [0, 0, 0, 0], (0, [0, 0, 0])))
        original_replay = comparison.replay
        for metadata, explicit, expected in cases:
            with self.subTest(metadata=metadata, explicit=explicit):
                dyn = ArmDynamics(comparison.MODEL, payload=expected)
                q = np.array([0.1, -0.8, -1.2, 1, 0.3, 0.1])
                rows = np.column_stack((np.arange(20) * 0.01, np.tile(q, (20, 1)),
                                        np.zeros((20, 6)), np.tile(dyn.gravity(q), (20, 1))))
                comparison.save_recording(self.csv_path, rows, expected)
                if metadata is None:
                    self.meta_path.unlink()
                else:
                    self.meta_path.write_text(metadata if isinstance(metadata, str) else json.dumps(metadata))
                args = ["--replay", str(self.csv_path)]
                if explicit is not None:
                    args += ["--payload", *map(str, explicit)]
                summaries = []

                def replay_offline(rows, payload, label):
                    self.assertEqual(payload, expected)
                    summary = original_replay(rows, payload, label)
                    summaries.append(summary)

                with patch.dict(sys.modules, {"xarm": None, "xarm.wrapper": None}), \
                        patch.object(comparison, "replay", side_effect=replay_offline), \
                        contextlib.redirect_stdout(io.StringIO()):
                    comparison.main(args)
                self.assertEqual(len(summaries), 1)
                np.testing.assert_allclose(np.array(summaries[0])[:, 1:4], 0, rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
