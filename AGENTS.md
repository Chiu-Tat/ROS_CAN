# Repository Guidelines

## Project Structure & Module Organization

This repository is a ROS catkin workspace for CAN-based electromagnetic robot arm control. Source packages live under `src/`: `ICRA2026` for the main closed-loop controller, `case1_omni_current_playback`, `case2_dual_magnet_trajectory`, `case4_three_sphere_playback`, `case6_single_sphere_closed_loop`, and `case7_dual_microsphere_closed_loop` for experiments, `basic_control` for shared UI/control utilities, and `calibration` for calibration routines. Each package uses `package.xml`, `CMakeLists.txt`, and Python/UI/data files in `scripts/`.

Generated outputs such as `devel/` and `build/` should not be edited by hand. Keep CSV, JSON, PNG, and `.ui` experiment assets with the package that consumes them.

## Build, Test, and Development Commands

- `catkin_make`: build all packages.
- `catkin build`: alternative catkin-tools build, if installed.
- `source devel/setup.bash`: load this workspace into the current shell after building.
- `rosrun ICRA2026 openmv_tracker.py`: run the OpenMV angle publisher.
- `rosrun ICRA2026 closed_loop_control.py`: run the main closed-loop controller.

Run commands from the workspace root unless a script requires otherwise.

## Coding Style & Naming Conventions

Python scripts should target Python 3 and ROS Noetic. Use 4-space indentation, `snake_case` for functions, variables, and new file names, and `CamelCase` only for classes. Prefer names such as `openmv_tracker.py`; avoid new files with spaces in their names. Keep hardware constants, CAN IDs, topic names, and trajectory filenames near the top of the script or in clearly named configuration sections.

## Testing Guidelines

No automated test suite is currently defined. Before submitting changes, run `catkin_make` and perform a ROS smoke test for the affected node. For controller changes, verify topics with `rostopic list`, `rostopic echo /arm/theta1`, and `rostopic echo /arm/theta2`. For plot or CSV scripts, ensure outputs are regenerated intentionally and do not overwrite important experiment data.

## Commit & Pull Request Guidelines

The existing history uses short commit messages such as `Readme` and `First commit`; keep future messages short but more specific, for example `Update case 6 controller gains` or `Add OpenMV tracker calibration notes`.

Pull requests should describe the affected package, hardware or dataset used for validation, commands run, and any ROS topics, CAN devices, or generated assets involved. Include screenshots or plots for UI, plotting, or trajectory-result changes.

## Security & Configuration Tips

Do not commit local machine paths, device-specific secrets, or private calibration data unless they are required for reproducible experiments. Document required CAN interface setup and camera assumptions in the relevant package README or script header.
