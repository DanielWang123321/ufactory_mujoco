# Security Policy

ufactory_mujoco is research software for simulation, with an optional gated interface to a real xArm. It is not safety-certified.

## Supported Versions

Security fixes are made on the latest release only (see [CHANGELOG.md](CHANGELOG.md)).

## Reporting a Vulnerability

Report privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability**. Do not open a public issue for a problem that could cause unintended robot motion, code execution, a bypass of the configuration or calibration checks, or exposure of credentials.

Include the affected version or commit, a minimal reproduction and the impact. Leave out real robot IP addresses, serial numbers and calibration files.

## Trust Boundaries

**Configuration files.** Task configs and calibration files are read with `yaml.safe_load`. Task configs reject unknown fields.

**Models.** `UFACTORY_MUJOCO_MODELS` replaces the bundled model tree. MJCF and mesh files are parsed by MuJoCo; use model trees only from sources you trust.

**Real robot.** `XArmRobot` is read-only by default. Motion requires a per-unit calibration file whose serial number matches the connected robot, the controller checks in `ufactory_mujoco/safety.py`, and `--confirm-real`. `sdk-sim` runs in the controller's built-in simulator and does not move the arm. A way to move a real arm without these checks is a vulnerability.

Operators remain responsible for physical guarding, emergency-stop access, payload and tool validation, and speed limits.
