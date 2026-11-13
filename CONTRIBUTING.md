# Contributing to ufactory_mujoco

Bug reports, fixes and improvements are welcome. For a larger change (a new model, a new task), open an issue first so the approach can be agreed before you invest time.

## Development Setup

Python 3.12–3.14. The meshes are large, so clone with `--depth 1`.

```bash
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git && cd ufactory_mujoco
python3.13 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"
python -m pip install ruff==0.17.1
```

Dependencies are declared only in `pyproject.toml`. The `assets` extra is needed only for the mesh and MJCF regeneration scripts in `scripts/`.

`requirements-lock.txt` (MuJoCo 4.0.0) pins the versions CI tests with. To reproduce a CI failure, install it first: `pip install -r requirements-lock.txt -e ".[test]"`.

## Checks

```bash
ruff check .
python -m pytest tests
```

- Tests that need an optional package are skipped when it is missing.
- The full suite loads every model: one run peaks at about 6.9 GB of memory and takes about five and a half minutes on a 28-thread desktop. Run one `pytest` process at a time.
- While iterating, run the files that cover your change, for example `python -m pytest tests/test_joint_control.py`.
- Examples must keep running with `--headless`. The commands CI runs are in the `Example smoke tests` step of [.github/workflows/ci.yml](.github/workflows/ci.yml).

CI runs these jobs on every pull request:

| Job | Checks |
| :--- | :--- |
| `lint` | `ruff check .` with ruff 0.17.1 |
| `test` | the test suite and the headless example smoke tests on Python 3.12, 3.13 and 3.14 with MuJoCo 4 |
| `build` | sdist and wheel, `twine check`, `scripts/check_distribution.py` |
| `installed` | the wheel installed into a clean environment on Python 3.12, 3.13 and 3.14, checked by `scripts/accept_installed.py` |

### Release Acceptance

Maintainers run a local acceptance before pushing a release. It covers Python 3.12, 3.13 and 3.14, each with MuJoCo 4.x; MuJoCo 3 is not tested. In each environment, one at a time:

- the full `pytest` suite and the CI example smoke tests;
- every `models/*/*.xml` compiles without errors or warnings;
- zero drift: the 30 task reports (`pick-place` and `box-pack` variants for the five arms, written with `--report`) are byte-identical to the previous release in the same environment, unless the change is meant to alter them;
- sdist and wheel built from the tree, `twine check --strict`, `scripts/check_distribution.py`, and the wheel installed into a clean virtual environment and checked by `scripts/accept_installed.py`.

Commits that are not pushed do not need this acceptance.

## Changes That Need Extra Care

**Models and meshes.** The arm, end-effector, camera and combination MJCF files under `models/` are written by `scripts/build_all_mjcf.py` and `scripts/build_camera_mjcf.py`, and the meshes by `scripts/convert_glb_to_obj.py`, `scripts/fix_g2_base_meshes.py` and `scripts/make_double_sided_objs.py`, in that order. Visual meshes keep the full upstream tessellation; do not decimate them. Change the script and regenerate; do not edit generated files by hand. The task scenes (`scene_reach.xml`, `scene_pick_place.xml`, `scene_box_pack.xml`, `scene_obstacles.xml`) are maintained by hand. `models/gripper_g2/source/` is an upstream snapshot checked by hash and must not be modified.

**Real robot.** Tests and CI never connect to a robot. A change to `ufactory_mujoco/xarm_robot.py`, `ufactory_mujoco/safety.py` or a hardware mode must keep the read-only default and the `--confirm-real` gate. Say in the pull request how you tested it (`sdk-sim` or a real arm). Do not add the controller address, serial number or calibration file of your own robot.

**Third-party assets.** A new asset keeps its upstream license. Record its source in `NOTICE` and add the license text to `licenses/`.

## Documentation

- `README.md` and `README_zh.md` describe the same content, as does each `README.md` / `README_zh.md` pair under `examples/`. Update both languages.
- Add user-visible changes to `CHANGELOG.md`.
- Measured numbers (accuracy, timing, sizes) state the Python and MuJoCo versions they were measured with. When behaviour changes, measure again instead of copying old numbers.

## Commits and Pull Requests

- Commit messages follow `type(scope): summary`, with the types `feat`, `fix`, `docs`, `test`, `build`, `chore`, `refactor` and `perf`. Example: `fix(examples): keep 02_joint_control demo poses off the floor`.
- Keep one logical change per commit. A generator change and the files it regenerates belong in the same commit.
- A pull request must pass every CI job. Describe what changed, how you tested it, and any change in simulation or benchmark results.

## Security and Conduct

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Participation in this project is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Contributions are released under the repository's MIT license (see [LICENSE](LICENSE)).
