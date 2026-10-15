"""Headless smoke tests for the simulation topic examples (roadmap 0.0.6 M4)."""

import importlib
import sys
import unittest
from unittest.mock import patch

kinematics_demo = importlib.import_module("examples.simulation.07_kinematics")
dynamics_demo = importlib.import_module("examples.simulation.08_dynamics")


class TopicExampleSmokeTests(unittest.TestCase):
    def test_kinematics_demo_headless_passes(self):
        with patch.object(sys, "argv", ["07_kinematics.py", "--headless"]):
            with self.assertRaises(SystemExit) as ctx:
                kinematics_demo.main()
        self.assertEqual(ctx.exception.code, 0, "kinematics demo reported FAIL")

    def test_dynamics_demo_headless_passes(self):
        with patch.object(sys, "argv", ["08_dynamics.py", "--headless"]):
            with self.assertRaises(SystemExit) as ctx:
                dynamics_demo.main()
        self.assertEqual(ctx.exception.code, 0, "dynamics demo reported FAIL")


if __name__ == "__main__":
    unittest.main()
