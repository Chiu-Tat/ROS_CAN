# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A ROS (Noetic, Python 3) catkin workspace for controlling an electromagnetic
manipulation system: a **10-coil electromagnetic array** drives magnetically
actuated objects (robot-arm joints, microspheres, a laser arm), with closed-loop
feedback from an **OpenMV camera** over serial and coil commands sent over a
**CAN bus** (`socketcan`, channel `can0`). Coils are addressed by device index
0–9; currents are in milliamperes, clamped to ±15000 mA (±15 A).

## Build & run

```bash
source /opt/ros/noetic/setup.bash
source .venv/bin/activate    # Python 3.8, matching ROS Noetic and Qwt
catkin_make                 # build all packages (run from workspace root)
source devel/setup.bash     # load workspace into the shell after every build
```

Active packages under `src/` are `case_study1_single_fixed_magnet`, `case8_two_dof_laser_arm_control`,
`basic_control`, and `calibration`. Historical experiments (`ICRA2026` and
Cases 1, 2, 4, 6, and 7) are preserved under `archive/ros_packages/` and excluded
from the active build. See `archive/README.md` for restoration instructions.
A typical case runs as **two nodes**:

```bash
rosrun <pkg> openmv_tracker.py   # reads OpenMV serial, publishes feedback topic(s)
rosrun <pkg> main2.py            # Qt GUI; its OnlineCheck button starts closed-loop control
```

Case Study 1 instead uses `roslaunch case_study1_single_fixed_magnet preview.launch`
for a hardware-free preview, or its documented `experiment.launch` workflow.
It has separate gamepad, line-tracker, controller, and TFW-panel nodes plus a
catkin Python module (`cs1_experiment`). The cylinder line axis is directed by
the user-supplied initial magnetic heading; tracking loss requires reinitializing.
Read its README for the paper coordinate frame, model-in TFW limitations, and
camera calibration before hardware operation. Retain Case 8 until instructed.

`main.py` / `main2.py` are the GUI entry points (`main2.py` is the current one for
the newer cases). `catkin_install_python` in each `CMakeLists.txt` marks the
scripts installable, but day-to-day you run them straight from `scripts/` via
`rosrun`. There is **no automated test suite** — validate changes with
`catkin_make` plus a ROS smoke test (`rostopic list`, `rostopic echo <topic>`).

## Architecture

The cases are **near-clones that diverge in the controller and the feedback
topic/units**, not a shared library. Each package's `scripts/` directory is
self-contained and carries its own copy of the common modules — editing one
package's copy does **not** affect the others.

Per-package `scripts/` modules (consistent across cases):

- **`can_lib.py`** — low-level CAN layer. `CANBusHandler` wraps `python-can`
  (send/recv/close); `CANMessageDefs` holds the device/feedback CAN IDs and
  current limits; `CANMessageBuilder` builds reset / mode-set / init /
  feedback-rate / set-current frames; `can_to_current` / `current_to_can`
  convert the 2-byte two's-complement payload. Currents are clamped here.
- **`basic_control.py`** — **generated** from `basic_control.ui` by the PyQt5 UI
  compiler (`Ui_MainWindow`). Do not hand-edit; regenerate from the `.ui` file.
- **`main2.py` / `main.py`** — `QtWidgets.QMainWindow` that owns the
  `CANBusHandler`, wires the 10 per-coil set buttons / dials / feedback labels,
  runs a background thread (`listen_for_feedback`) to poll CAN feedback frames,
  and instantiates the case's controller when OnlineCheck is pressed. CAN params
  come from ROS params (`~can_channel`, `~can_bustype`, `~verbose_can`).
- **`closed_loop_control.py`** — the actual control law (`ClosedLoop*Controller`).
  Subscribes to the camera feedback topic, runs `_control_loop` on its own
  `threading.Thread` gated by `control_running`, loads a `trajectory_*.json`,
  computes a 10-coil current vector (the newer cases build a field map and solve
  it by damped least squares), applies current/step limits, and sends frames
  through the GUI's `send_device_current` callback. Writes per-run CSVs to
  `scripts/results/`.
- **`openmv_tracker.py`** — reads the OpenMV camera over pyserial, converts raw
  pixel centers to metric workspace coordinates, and publishes the feedback
  topic(s). `OpenMV_code.py` / `openmv_code.py` is the script that runs **on the
  camera**, not in ROS.

### What differs between cases (check these first)

- **Feedback topic and message type.** Early cases publish joint angles as
  `std_msgs/Float32` on `/arm/theta1`, `/arm/theta2`. Case 7 uses
  `/sphere/red`, `/sphere/green`; Case 8 uses `geometry_msgs/PointStamped` on
  `/laser/spot` (+ `/laser/fps`).
- **Units.** Trackers may publish pixels or meters; controllers take a
  `~position_scale` param (e.g. `_position_scale:=0.001`) to reconcile.
- **Controller gains and physics.** Each case tunes its own gains for its force
  law — e.g. Case 7's soft-magnetic gains must stay in the micro-Newton range
  (`Kp≈1e-6`); values near 1 saturate the coils. Read the per-case `README.md`
  (cases 7 and 8 have detailed ones) before touching control parameters.

`trajectory_*.json` files store both the waypoints and the controller settings
for a run; the controller reads its tunables from there, so changing behavior
often means editing the JSON, not the code.

`archive/ros_packages/ICRA2026` is the original/reference package and contains several
work-in-progress controller variants (`closed_loop_control copy*.py`,
`... kazhu*.py`); its reference entry point is plain `closed_loop_control.py`.

## Conventions

- Python 3 / ROS Noetic, 4-space indent, `snake_case` (functions/vars/files),
  `CamelCase` only for classes. Avoid spaces in new file names (existing files
  like `Case_2_control single.py` predate this).
- Keep hardware constants (CAN IDs, topic names, current limits, trajectory
  filenames) near the top of the script or in a clearly named settings block.
- When changing one case, do not assume the edit propagates — apply it to each
  package's copy that needs it.
- `build/` and `devel/` are generated; never edit by hand. Don't overwrite
  existing experiment data/plots in `scripts/results/` unintentionally.
- Commit messages are short; make them specific (e.g. `Update case 6 controller
  gains`, not `Update`).
