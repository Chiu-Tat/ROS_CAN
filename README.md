# ROS_CAN

ROS Noetic workspace for electromagnetic robot control via CAN bus. Active experiments include the revised paper's Case Study 1 gamepad-controlled magnet and Case 8's two-DOF laser manipulator. Earlier arm and microsphere experiments are preserved in the archive.

## Overview

This project provides control software for an electromagnetic manipulation system consisting of:
- **10-coil electromagnetic array** generating rotating magnetic fields
- **Two-DOF laser manipulator** with magnetically actuated joints
- **OpenMV camera** for real-time laser spot tracking
- **CAN bus** for motor/actuator communication

## Packages

| Package | Description |
|---------|-------------|
| `case_study1_single_fixed_magnet` | Revised-paper Case Study 1: gamepad forward/backward/turn commands, one fixed magnet, and heading-dependent TFW guard |
| `case8_two_dof_laser_arm_control` | Two-DOF laser manipulator, camera feedback, and analytical or learned-model control |
| `basic_control` | Basic control interface and utilities |
| `calibration` | Calibration routines |

Historical packages (`ICRA2026` and Cases 1, 2, 4, 6, and 7) are stored under
[`archive/ros_packages/`](archive/ros_packages/). They are excluded from the
active build. See [the archive index](archive/README.md) for descriptions and
restoration instructions.

## Requirements

- ROS (tested with ROS Noetic)
- Python 3.8 using the local `.venv` (see [ENVIRONMENT.md](ENVIRONMENT.md))
- Numerical, serial/CAN, and Qt dependencies listed in [requirements.txt](requirements.txt)
- OpenMV (for vision-based joint tracking)
- CAN interface (for motor control)

## Build

```bash
cd /path/to/ROS_CAN
source /opt/ros/noetic/setup.bash
source .venv/bin/activate
catkin_make
# or
catkin build
```

## Usage

For the new single-fixed-magnet experiment, see its
[setup and operation guide](src/case_study1_single_fixed_magnet/README.md).
Start a preview with `roslaunch case_study1_single_fixed_magnet preview.launch`.
This opens no CAN interface or camera. The Case 8 commands follow below.

1. **Source the workspace:**
   ```bash
   source devel/setup.bash
   ```

2. **Run the vision tracker** (publishes the laser spot to `/laser/spot`):
   ```bash
   rosrun case8_two_dof_laser_arm_control openmv_tracker.py
   ```

3. **Run the control GUI:**
   ```bash
   rosrun case8_two_dof_laser_arm_control main2.py
   ```

See [Case 8 instructions](src/case8_two_dof_laser_arm_control/README.md) for
calibration and operation, and [the learned-model workflow](src/case8_two_dof_laser_arm_control/README_data_driven.md)
for data collection and training. For new case studies, edit camera code in
VS Code and use `mpremote` for USB uploads and software restarts. Release the
camera serial port from the tracker or IDE before uploading.

## Key Topics

- `/laser/spot` (PointStamped) — Calibrated laser spot position in meters
- `/laser/fps` (Float32) — OpenMV frame rate

## Project Structure

```
ROS_CAN/
├── src/
│   ├── CMakeLists.txt
│   ├── case_study1_single_fixed_magnet/    # Revised-paper gamepad experiment
│   ├── case8_two_dof_laser_arm_control/    # Current experiment
│   ├── basic_control/                     # Basic control utilities
│   └── calibration/                       # Calibration tools
├── archive/
│   ├── CATKIN_IGNORE
│   ├── README.md                          # Historical package index
│   └── ros_packages/                      # ICRA2026 and Cases 1, 2, 4, 6, 7
├── .venv/
├── ENVIRONMENT.md
├── requirements.txt
└── README.md
```

## Author

Da ZHAO — CUHK

## License

TODO
