# Simulation Examples (`examples/simulation`)

[English](README.md) | [中文](README_zh.md)

The examples need only the core install ([installation](../../README.md#installation)). The commands below open the 3D viewer; on macOS start them with `mjpython` instead of `python`. Add `--headless` to run without a viewer, for example on CI or a server: the scripts then print their checks, and `01` prints a model summary. `--help` lists the options of every script.

## Examples

| Script | What it shows | Models |
| :--- | :--- | :--- |
| `01_load_and_view.py` | Any model in the viewer | all |
| `02_joint_control.py` | Quintic joint moves; bare `xarm6` runs the 20 dynamics-check poses | with an arm |
| `03_cartesian_ik.py` | Straight line and circle tracked with DLS IK | `xarm6` |
| `04_gripper_control.py` | Close, open, half open; Gripper G2, Bio Gripper G2, and Robotiq 2F-85/2F-140 take `--speed-percent` and `--force-percent` | with a jaw gripper |
| `05_pick_place.py` | Cube pick-and-place with preflight | `--robot`, five arms |
| `06_trajectory_planning.py` | Quintic point-to-point, C2 spline, Cartesian line, feedforward torques | `xarm6` |
| `07_kinematics.py` | FK, Jacobian against finite differences, velocity IK, manipulability | `xarm6` |
| `08_dynamics.py` | Dynamics checks, then a monitored home → ready → target → ready motion | `--robot`, five bare arms |
| `09_obstacle_avoidance.py` | RRT-Connect around a pillar and an overhead board | `xarm6` |
| `10_box_pack.py` | Cubes dropped into a box | `--robot`, five arms |
| `11_force_torque_sensor.py` | AI1500 force/torque readings at 200 Hz | with `ft_sensor` |

## Running

```bash
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper
python examples/simulation/02_joint_control.py --model lite6
python examples/simulation/03_cartesian_ik.py
python examples/simulation/04_gripper_control.py --model xarm6_with_bio_gripper --speed-percent 50
python examples/simulation/05_pick_place.py --robot lite6
python examples/simulation/06_trajectory_planning.py
python examples/simulation/07_kinematics.py
python examples/simulation/08_dynamics.py --robot uf850
python examples/simulation/09_obstacle_avoidance.py
python examples/simulation/10_box_pack.py --robot xarm7 --cycles 3
python examples/simulation/11_force_torque_sensor.py --model xarm6_with_ft_sensor_and_vacuum_gripper
```

- `05`, `08` and `10` run the same code as `ufactory-mujoco pick-place`, `dynamics` and `box-pack`. `--robot` selects the arm (`lite6`, `xarm5`, `xarm6`, `xarm7`, `uf850`); `05` and `10` mount the Lite 6 gripper on `lite6` and Gripper G2 on the others. `--dry-run` runs only the preflight, and `--report <file>` writes a JSON report.
- Static torque checks aligned with `ufactory_genesis` have their own commands: `dynamics-sim-check` in simulation, `dynamics-hardware-check` on a real arm, and `dynamics-report-compare` to compare two reports.

## Models

`--model` takes a model name or an MJCF path. `ufactory-mujoco export-model --list` prints every name. Combinations are named `<arm>_with_<parts>`:

| Arm | `<parts>` |
| :--- | :--- |
| `xarm5`, `xarm6`, `xarm7`, `uf850` | `gripper` (Gripper G2), `bio_gripper`, `robotiq_2f_85`, `robotiq_2f_140`, `vacuum_gripper`, `d435`, `gripper_and_d435`, `ft_sensor`, `ft_sensor_and_gripper`, `ft_sensor_and_vacuum_gripper`, `base_plate` |
| `xarm6` only | `gripper_and_d405`, `gripper_and_gemini_305`, `gripper_and_zed_x_nano`, `robotiq_2f_85_and_d435`, `robotiq_2f_140_and_d435` |
| `lite6` | `gripper` (Lite 6 gripper), `vacuum_gripper`, `d435`, `gripper_and_d435`, `base_plate` |

Standalone parts open in `01`: the grippers `gripper_g2`, `bio_gripper_g2`, `robotiq_2f_85`, `robotiq_2f_140`, `lite6_gripper`, `vacuum_gripper` and `lite6_vacuum_gripper`; `robotiq_mounting` (11 mm xArm-to-Robotiq flange adapter); the cameras `d405`, `d435`, `gemini_305` and `zed_x_nano`; `camera_stand` and `camera_stand_with_<camera>`; `ft_sensor`; and the base plates `xarm_base_plate`, `uf850_base_plate` and `lite6_base_plate`. `04` also runs the five standalone jaw grippers, and `11` runs `ft_sensor`. A script exits with code `2` when it cannot run the given model.

## Start Pose

- `home` sets every joint to zero on every model.
- `ready` points the tool down. On xArm and UF850 models with a tool, camera or F/T sensor it puts the Gripper G2 TCP at `(300, 0, 300)` mm, and the other tools use the same joint angles. Lite 6 models use the factory pose `[0, 9.9, 31.8, 0, 21.9, 0]°`. Bare xArm and UF850 arms, also on a base plate, keep `ready` at zero.
- Models whose mounted tool reaches below the floor at zero (xArm 5/6/7 combinations and `uf850` with `robotiq_2f_140`) also have `init`, equal to `ready`.
- The examples start from `init` when the model has it, otherwise from `home`; `01 --pose` selects another keyframe. `05` and `10` start every arm with the TCP at `(300, 0, 300)` mm pointing down. In `05` this is straight above the cube, which every arm carries from `(300, 0)` to `(300, 300)` mm.
- The standalone jaw grippers have `open` and `close` keyframes; the other standalone parts have none.

## Real Robot

Hardware modes need `pip install xarm-python-sdk` and a calibration file exported from the arm. They refuse to run when the file's serial number or model does not match the connected controller. Replace the IP with your controller's.

```bash
# 1. Export this arm's calibration to calibrations/<arm>_kinematics_<SN>.yaml
ufactory-mujoco export-calibration 192.168.1.60 --output-dir calibrations

# 2. Read-only checks: the arm does not move
ufactory-mujoco kinematics-check --robot xarm6 --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
python examples/simulation/08_dynamics.py --robot xarm6 --mode real-read-only --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
python examples/simulation/11_force_torque_sensor.py --ip 192.168.1.60 --robot xarm6 --gripper ft_sensor

# 3. Replay in the controller's built-in simulator: the arm does not move
python examples/simulation/05_pick_place.py --robot xarm6 --mode sdk-sim --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml

# 4. Real motion: clear the workspace and keep the E-stop within reach
python examples/simulation/05_pick_place.py --robot xarm6 --mode real --confirm-real --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
dynamics-hardware-check --robot xarm6 --ip 192.168.1.60 --confirm-real --require-reference
```

- `--mode real` runs a single cycle. It starts only with `--confirm-real`, a controller without errors, and a preflight that passes from the measured joint pose. In `05` and `10` it also drives the gripper.
- The hardware dynamics checks limit joint speed to 40 °/s. Arms whose calibration export is skipped (older serial numbers, the xArm 5) run them with `--nominal-model`.
- When a tool is mounted, pass `--gripper <name>` to `08` so that its preflight includes the tool.

## Viewer Controls

Press `F1` in the viewer for the full list.

- **Left drag**: rotate. **Right drag**: pan. **Scroll**: zoom.
- **Double-click**: select a body. **Ctrl + left / right drag**: apply a torque / force to it.
- **Space**: pause. **Backspace**: reset. Both work only in `01`; the other examples step the physics themselves.
- **C** / **F**: contact points / forces. **0–5**: geom groups (`2` visual meshes, `3` collision meshes). Turn on site group `4` in the right panel to see mounting sites and screw holes.
- **Tab** / **Shift + Tab**: show or hide the left / right panel.
