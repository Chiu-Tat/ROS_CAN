# Archived ROS packages

Previous CS1 workspace data is also preserved under
`tfw/cs1_single_fixed_cb70fe3f55a8/` (28,231 model-in boxes and original
provenance). The active experiment now uses the fine corrected-`dBy/dy` TFW
with 197,085 boxes. This data archive is separate from the ROS packages below.

These historical experiments retain their original package names, scripts,
camera code, UI files, calibration files, trajectories, and recorded data.
They remain in this Git repository for reference and reuse. They are outside
the active `src/` tree and excluded from normal catkin builds and package
discovery. `CATKIN_IGNORE` also prevents catkin discovery if the archive is
searched recursively.

| Package | Purpose |
| --- | --- |
| `ICRA2026` | Original electromagnetic arm controller, kinematics, and OpenMV joint tracking |
| `case1_omni_current_playback` | Omnidirectional current playback |
| `case2_dual_magnet_trajectory` | Dual-magnet trajectory control |
| `case4_three_sphere_playback` | Three-sphere current playback and tracking |
| `case6_single_sphere_closed_loop` | Single-sphere closed-loop trajectory control |
| `case7_dual_microsphere_closed_loop` | Dual-microsphere tracking and closed-loop control |

## Restore an experiment

From the workspace root, move the whole package back to `src/` and rebuild.
For example:

```bash
mv archive/ros_packages/case7_dual_microsphere_closed_loop src/
source /opt/ros/noetic/setup.bash
source .venv/bin/activate
catkin_make --force-cmake
source devel/setup.bash
rospack find case7_dual_microsphere_closed_loop
```

Update this index when restoring or archiving a package. To archive it again,
move the whole package back to `archive/ros_packages/`, rebuild, and re-source
the workspace. Never leave a duplicate package name in `src/`.

Archived scripts are historical code, not newly validated hardware runs.
Some retain original absolute machine paths and device assumptions. Check
CAN configuration, camera setup, and input/output paths before running an
experiment. Preserve the original datasets; choose new output filenames.

The legacy `src/basic_control/scripts/main copy.py` playback variant can read
the archived Case 1 `coil_currents.csv` without restoring Case 1. Its
`~playback_csv` ROS parameter can select a different dataset.
