# Repository Guidelines

## Project Structure & Module Organization

This repository is a ROS catkin workspace for CAN-based electromagnetic robot control. Active source packages live under `src/`: `case_study1_single_fixed_magnet` for revised-paper Case Study 1 gamepad/keyboard experiments, `case8_two_dof_laser_arm_control` (retain until the user explicitly requests archiving), `basic_control` for shared UI/control utilities, and `calibration` for calibration routines. Each package uses `package.xml`, `CMakeLists.txt`, and Python/UI/data files in `scripts/`. The Case Study 1 reusable numerical model lives under `src/cs1_experiment/` inside its package.

Historical packages (`ICRA2026` and Cases 1, 2, 4, 6, and 7) live intact under `archive/ros_packages/`, outside the active catkin source space. See `archive/README.md` before restoring a package to `src/`. Keep future retired packages there with their experiment assets, and update the archive index. New case studies belong in `src/`.

Generated outputs such as `devel/` and `build/` should not be edited by hand. Keep CSV, JSON, PNG, and `.ui` experiment assets with the package that consumes them.

## Build, Test, and Development Commands

- `catkin_make`: build all packages.
- `catkin build`: alternative catkin-tools build, if installed.
- `source devel/setup.bash`: load this workspace into the current shell after building.
- `rosrun case8_two_dof_laser_arm_control openmv_tracker.py`: run the OpenMV laser spot publisher.
- `rosrun case8_two_dof_laser_arm_control main2.py`: run the current experiment's control GUI.
- `roslaunch case_study1_single_fixed_magnet preview.launch`: preview the new gamepad experiment without CAN or camera hardware.
- `rosrun case_study1_single_fixed_magnet main.py`: choose Single (1), Gate (2), or Triangle (3) fixed magnets before launching the shared UI.
- `python -m unittest discover -s src/case_study1_single_fixed_magnet/tests -v`: check the Case Study 1 model, TFW guard, calibration, and heading continuity.

Run commands from the workspace root unless a script requires otherwise.

For Case Study 1, read its package README before hardware operation. The panel supports selectable gamepad or W/S/A/D keyboard input; switching sources disarms, and keyboard focus loss clears held keys. Camera tracking uses the cylindrical magnet's undirected line axis with a manually initialized magnetic heading; prolonged detection loss requires reinitialization. Preserve the imported TFW/model pairing and its coordinate frame. Use `.venv` (Python 3.8) for ROS work. Keep real-hardware runs separate from preview testing. The CS1 free magnet has a physical green coating, 11 mm length and 4 mm diameter; detect its LAB color and verify calibrated rotated-box dimensions. Initialize N anywhere in the open EM +y half-plane (0 < beta < pi); the camera measures beta and the UI asks only for polarity confirmation, not a numerical angle. Save the measured initial pose, and preserve unwrapped heading over full rotations. Short segmentation misses (at most 120 ms) publish invalid feedback and no pose while retaining the polarity branch briefly; prolonged loss or ambiguity requires reinitialization. The Case Study 1 camera pane receives JPEG previews over the same USB JSON stream as pose data; preserve bounded chunked parsing and keep video failures independent from pose validity. The world/TFW pane must use measured pose feedback during physical runs. CS1 driver reset/current-mode/zero actions must disarm and cancel pending arming. Keep the single CAN owner in the controller; measured coil feedback must use the ten legacy 0x01B..0x0AB IDs and be shown separately from commands, with missing/stale readings explicit. Preview must not open CAN or fabricate measured zero currents. CS1 arming prepares UI video and data recording before nonzero holding commands. Each trial stays under results/.pending until an explicit Save/Delete decision; Save preserves a complete trial folder and introduction, while Delete removes only that trial. Do not introduce duplicate run logs outside its folder. Keep pre-run calibration/initial references separate. Recording workers must not block Qt/control; errors must stop control and preserve recoverable partial recordings. Camera image and pose arrival ages are separate; do not imply hardware synchronization.

## OpenMV Workflow for New Case Studies

CS1 layouts share control code and camera calibration but use matching case
configs, fixed-dipole sums and TFW provenance. Gate/Triangle configs/assets live
under `config/layouts/` and `data/layouts/`; their trials go under corresponding
result subfolders. Keep their hardware-validation flags false until validated.
Choose the layout before launch; do not hot-swap a live model/TFW or alter Single
tuning when working on another layout. Closing the main UI ends its launch only
after pending trial decisions. Camera code is shared; no reflash is needed merely
to select another layout.

CS1 pose holding is experimental after uncontrolled hardware motion. Runtime
`PoseHoldController` captures an arm reference, ramps the coordinated current
vector from zero, and holds that reference until movement commands advance it.
Keep `pose_hold.hardware_validated` false pending physical commissioning; a user
request to open the UI or enable CAN/feedback does not authorize clearing that
flag. The old forward-force TFW does not certify lateral feedback corrections.
Never describe software tests, a slew-limited ramp, or current feedback alone as
proof of physical stationary holding. Preserve the maximized UI and calibrated
100 mm square camera view alongside the TFW.

For all new case studies, edit camera-side MicroPython code in VS Code and use `mpremote` over USB to upload it as the board's `main.py` and restart the camera in software. Keep the camera source in the consuming package (prefer `scripts/openmv_code.py`), separate from the host-side ROS tracker (`scripts/openmv_tracker.py`). Provide upload/restart and serial-monitor commands or VS Code tasks with a configurable camera port. Routine development should not require copying code into OpenMV IDE or unplugging/replugging the USB cable.

Pause hardware control and release the camera serial port from the ROS tracker, OpenMV IDE, or any monitor before uploading; only one process should own the port at a time. Release the upload/monitor connection before resuming ROS tracking, and support tracker reconnection after camera resets. Verify board/firmware compatibility before treating the workflow as hardware-tested. OpenMV IDE remains optional for image preview and threshold tuning.

## Coding Style & Naming Conventions

Python scripts should target Python 3 and ROS Noetic. Use 4-space indentation, `snake_case` for functions, variables, and new file names, and `CamelCase` only for classes. Prefer names such as `openmv_tracker.py`; avoid new files with spaces in their names. Keep hardware constants, CAN IDs, topic names, and trajectory filenames near the top of the script or in clearly named configuration sections.

## Testing Guidelines

No automated test suite is currently defined. Before submitting changes, run `catkin_make` and perform a ROS smoke test for the affected node when its hardware is available. For Case 8 controller changes, verify topics with `rostopic list`, `rostopic echo /laser/spot`, and `rostopic echo /laser/fps`. For restored ICRA2026 experiments, use `/arm/theta1` and `/arm/theta2`. For plot or CSV scripts, ensure outputs are regenerated intentionally and do not overwrite important experiment data.

## Commit & Pull Request Guidelines

The existing history uses short commit messages such as `Readme` and `First commit`; keep future messages short but more specific, for example `Update case 6 controller gains` or `Add OpenMV tracker calibration notes`.

Pull requests should describe the affected package, hardware or dataset used for validation, commands run, and any ROS topics, CAN devices, or generated assets involved. Include screenshots or plots for UI, plotting, or trajectory-result changes.

## Security & Configuration Tips

Do not commit local machine paths, device-specific secrets, or private calibration data unless they are required for reproducible experiments. Document required CAN interface setup and camera assumptions in the relevant package README or script header.
