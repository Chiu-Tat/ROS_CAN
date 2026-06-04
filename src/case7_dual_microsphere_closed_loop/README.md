# Case 7: Two Soft-Magnetic Iron Microspheres

This case study mirrors `~/Documents/DecaPulse_py/soft_microsphere_control.html`.
It controls two iron microspheres with one shared ten-coil current vector.

## Settings

- Microsphere 1 topic: `/sphere/red`
- Microsphere 2 topic: `/sphere/green`
- Workspace: `x^2 + y^2 <= 0.07^2`
- Plane: `z = -0.006 m`
- Current limit: `+/- 15.0 A`
- Soft-magnetic force law: `F = k_mag grad(B^T B)`, `k_mag = 3e-4`
- Controller: `F_des = Kp * (target - position) - Kd * velocity`
- Gains: `Kp = 1e-6 N/m`, `Kd = 6e-6 N s/m`
- Progress-mode movement force floor: `8e-5 N` while the active target is outside the target radius
- Stuck breakaway force: `4e-4 N` only while measured speed is below `1.5e-4 m/s`
- Drag compensation for stuck motion: `drag_N_s_per_m * target_settle_speed_m_per_s`, defaulting to `1.0 * 3e-4 = 3e-4 N`
- Current slew limit: `1.5 A` per control update
- Drag: `1.0 N s/m` effective value for the sticky silicone-oil setup
- Time step: `0.025 s`
- Waypoint radius: `0.0025 m`

The active controller reads these values from
`scripts/trajectory_case7_two_microspheres.json`. Keep the controller gains in
the micro-Newton range; values near `Kp = 1` or `Kd = 0.1` request forces much
larger than the coil model can produce and will saturate the currents.

## Trajectories

`scripts/trajectory_case7_two_microspheres.json` stores the latest persisted
HTML controls:

- M1: `(0.000, -0.0635) -> (0.005, -0.0435) -> (0.030, -0.025) -> (0.034, 0.020) -> (0.005, 0.0435) -> (0.000, 0.0635)`
- M2: `(0.000, 0.0635) -> (-0.005, 0.0435) -> (-0.038, 0.020) -> (-0.034, -0.018) -> (-0.005, -0.0435) -> (0.000, -0.0635)`

Each raw path is sampled with the same Catmull-Rom rule used by the HTML page.
The closed-loop controller uses progress-based target advancement by default,
so the run is not forced to finish in 20 s. It estimates each sphere's nearest
path progress and aims a short lookahead target instead of waiting on a stale
discrete sample. The two trajectories use coupled progress by default, so the
faster microsphere cannot run to the end while the slower microsphere is still
midway. The active target uses a 12-sample path lookahead and target advancement
is capped to 2 samples per control update. Target progress pauses when either
sphere is more than `6 mm` from its active target, preventing endpoint runaway
after noisy detections or overshoot.

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
