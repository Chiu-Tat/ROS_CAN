# Case 7: Two Soft-Magnetic Iron Microspheres

This case study mirrors `~/Documents/DecaPulse_py/soft_microsphere_control.html`.
It controls two iron microspheres with one shared ten-coil current vector.

## Settings

- Microsphere 1 topic: `/sphere/red`
- Microsphere 2 topic: `/sphere/blue`
- Workspace: `x^2 + y^2 <= 0.07^2`
- Plane: `z = -0.006 m`
- Current limit: `+/- 15.0 A`
- Soft-magnetic force law: `F = k_mag grad(B^T B)`, `k_mag = 3e-4`
- Controller: `F_des = Kp * (target - position) - Kd * velocity`
- Gains: `Kp = 1.0 N/m`, `Kd = 0.1 N s/m`
- Drag: `0.03 N s/m`
- Time step: `0.025 s`
- Waypoint radius: `0.0025 m`

The active controller reads these values from
`scripts/trajectory_case7_two_microspheres.json`, which is copied from the
latest persisted controls for `~/Documents/DecaPulse_py/soft_microsphere_control.html`.

## Trajectories

`scripts/trajectory_case7_two_microspheres.json` stores the latest persisted
HTML controls:

- M1: `(0.000, -0.0635) -> (0.005, -0.0435) -> (0.030, -0.025) -> (0.034, 0.020) -> (0.005, 0.0435) -> (0.000, 0.0635)`
- M2: `(0.000, 0.0635) -> (-0.005, 0.0435) -> (-0.038, 0.020) -> (-0.034, -0.018) -> (-0.005, -0.0435) -> (0.000, -0.0635)`

Each raw path is sampled with the same Catmull-Rom rule used by the HTML page.

## Run

Build and source the workspace:

```bash
catkin_make
source devel/setup.bash
```

Run the OpenMV tracker and GUI:

```bash
rosrun case7_dual_microsphere_closed_loop openmv_tracker.py
rosrun case7_dual_microsphere_closed_loop main2.py
```

If the OpenMV tracker publishes pixels rather than calibrated meters, launch
`main2.py` with a calibrated scale:

```bash
rosrun case7_dual_microsphere_closed_loop main2.py _position_scale:=0.001
```
