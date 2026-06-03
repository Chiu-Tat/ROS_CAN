# ROS_CAN

ROS workspace for electromagnetic robot arm control via CAN bus. Implements closed-loop control of a 3-DOF electromagnetic robot arm using rotating magnetic fields, with joint feedback from an OpenMV camera.

## Overview

This project provides control software for an electromagnetic manipulation system consisting of:
- **10-coil electromagnetic array** generating rotating magnetic fields
- **3-DOF robot arm** with magnetically actuated joints
- **OpenMV camera** for real-time joint angle tracking
- **CAN bus** for motor/actuator communication

## Packages

| Package | Description |
|---------|-------------|
| `ICRA2026` | ICRA 2026 submission: closed-loop control with PID and predefined motion sequences |
| `case1_omni_current_playback` | Case 1: omnidirectional current playback |
| `case2_dual_magnet_trajectory` | Case 2: dual-magnet trajectory control |
| `case4_three_sphere_playback` | Case 4: three-sphere current playback and tracking |
| `case6_single_sphere_closed_loop` | Case 6: single-sphere closed-loop trajectory control |
| `case7_dual_microsphere_closed_loop` | Case 7: dual-microsphere closed-loop control |
| `basic_control` | Basic control interface and utilities |
| `calibration` | Calibration routines |

## Requirements

- ROS (tested with ROS Noetic)
- Python 3
- NumPy
- OpenMV (for vision-based joint tracking)
- CAN interface (for motor control)

## Build

```bash
cd /path/to/ROS_CAN
catkin_make
# or
catkin build
```

## Usage

1. **Source the workspace:**
   ```bash
   source devel/setup.bash
   ```

2. **Run the vision tracker** (publishes joint angles to `/arm/theta1`, `/arm/theta2`):
   ```bash
   rosrun ICRA2026 openmv_tracker.py
   ```

3. **Run the closed-loop controller:**
   ```bash
   rosrun ICRA2026 closed_loop_control.py
   ```

## Key Topics

- `/arm/theta1` (Float32) — Joint 1 angle (degrees), from OpenMV tracker
- `/arm/theta2` (Float32) — Joint 2 angle (degrees), from OpenMV tracker

## Project Structure

```
ROS_CAN/
├── src/
│   ├── ICRA2026/      # Main control package
│   ├── case1_omni_current_playback/        # Case 1: omnidirectional current playback
│   ├── case2_dual_magnet_trajectory/       # Case 2: dual-magnet trajectory control
│   ├── case4_three_sphere_playback/        # Case 4: three-sphere current playback
│   ├── case6_single_sphere_closed_loop/    # Case 6: single-sphere closed-loop control
│   ├── case7_dual_microsphere_closed_loop/ # Case 7: dual-microsphere closed-loop control
│   ├── basic_control/                      # Basic control utilities
│   └── calibration/                        # Calibration tools
└── README.md
```

## Author

Da ZHAO — CUHK

## License

TODO
