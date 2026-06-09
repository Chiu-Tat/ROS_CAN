# Case 8: Two-DOF Laser Manipulator Spot Feedback

This case study mirrors
`/home/dz/Documents/DecaPulse_py/two_dof_manipulator_spot_feedback_control.html`.
It controls a laser mounted on a two-DOF manipulator. The OpenMV camera measures
the laser spot on the observation plane, and the controller uses the measured
spot with the known arm model to command magnetic fields at the two embedded
magnets.

## Topics

- Laser spot input: `/laser/spot`
- OpenMV FPS: `/laser/fps`

## Model

- Base position: `[-0.01571, 0.0, 0.06064] m`
- Observation plane: `z = -0.026 m`
- Link lengths: `0.02364 m`, `0.016 m`
- Laser extension: `0.012 m`
- Yaw range: `+/-150 deg`
- Pitch range: `+/-72 deg`
- Default field strength: `4 mT`
- Default current limit: `+/-15 A`
- Target spot speed: `2 mm/s`

The ten-coil dipole parameters are copied from the referenced HTML page. At
each control step, the node builds a `6 x 10` field map for the two magnet
positions, solves a damped least-squares current vector, applies current and
step limits, and sends the ten CAN current commands.

## Target And Camera Calibration

The five measured star points and matching OpenMV pixels are stored in
`scripts/fiver_corner_star_point_coordinate.txt`. The target trajectory uses
the real coordinates from that file in pentagram order:

```text
1 -> 3 -> 5 -> 2 -> 4 -> 1
```

The OpenMV tracker fits an affine pixel-to-world transform from the same file
at startup and publishes `/laser/spot` in meters. The current calibration fit
has about `0.49 mm` RMS error on the five measured points. Override the file
with `_calibration_points_file:=...`, disable file loading with
`_use_calibration_points_file:=false`, or override individual affine
coefficients with ROS params.

The target path is arc-length resampled at `2 mm/s` with `dt_s = 0.025 s`, so
successive target points are spaced by about `0.05 mm`. The controller uses
time-based reference mode for this trajectory.

Before the timed path starts, the controller now holds the first star point
until the measured laser spot is within `4 mm`. This avoids chasing a moving
reference when the laser starts far from the first point.

While holding point 1 at startup, the requested magnetic fields are explicitly
projected into the `y-z` plane (`Bx = 0`) before solving coil currents. For the
nominal point-1 pose this matches the arm geometry: the two magnet-aligned field
directions are already approximately `[0, -0.64, 0.77]` and `[0, -0.66, 0.75]`.

To apply only this point-1 `y-z` plane field in a separate script:

```bash
rosrun case8_two_dof_laser_arm_control point1_yz_plane_field.py
```

By default it waits until `/laser/spot` is within `4 mm` of point 1, then sends
the solved coil currents until interrupted. Use `_wait_for_spot:=false` to
apply the field immediately, `_duration_s:=5.0` to apply for a fixed duration,
or `_zero_on_exit:=false` to leave the last currents active when the script
exits.

To only plot and verify the point-1 `y-z` plane magnetic field without sending
CAN commands:

```bash
rosrun case8_two_dof_laser_arm_control plot_point1_yz_plane_field.py
```

This writes `scripts/point1_yz_plane_magnetic_field.png`.

## Run

Build and source the workspace:

```bash
catkin_make
source devel/setup.bash
```

Run the OpenMV tracker and GUI:

```bash
rosrun case8_two_dof_laser_arm_control openmv_tracker.py
rosrun case8_two_dof_laser_arm_control main2.py
```

Use the GUI `OnlineCheck` button to start the closed-loop laser trajectory. The
trajectory is stored in `scripts/trajectory_case8_laser_spot.json`.

For a headless run without the Qt GUI:

```bash
rosrun case8_two_dof_laser_arm_control run_case8_headless.py
```

## Results

The controller writes per-run CSV files under `scripts/results/`:

- `closed_loop_results_case8_*.csv`
- `coil_currents_case8_*.csv`
- `laser_positions_case8_*.csv`

Plot the latest result:

```bash
rosrun case8_two_dof_laser_arm_control plot_case8_results.py
```

Plot the target trajectory:

```bash
rosrun case8_two_dof_laser_arm_control plot_target_trajectory.py
```
