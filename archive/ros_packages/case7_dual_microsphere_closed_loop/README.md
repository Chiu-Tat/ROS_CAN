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

## Joystick teleoperation (`main_joystick.py`)

`main_joystick.py` is an alternative GUI that drives the microspheres with an
Xbox controller instead of a stored trajectory. It reuses the same soft-magnetic
field/force model and current solver as the trajectory controller, but takes the
desired movement *direction* from the right stick. It still needs the tracker
running, because the closed-loop force depends on each sphere's measured
position.

Because both spheres share one ten-coil field, the controller (a) maps full
stick deflection straight to the push force `joystick_force_N` (with light
velocity damping) and (b) **weights the commanded sphere(s) above any held
sphere** in the solver so your command is not diluted by the hold constraint and
the solver does not over-fight to null the held sphere (which is what jitters
it). In single-target mode the unselected sphere holds the position it had when
you switched to it; in `both` mode the two spheres receive the same push, so they
move together.

> The soft-magnetic force is *quadratic* in coil current, so the solver's
> current-smoothness penalty is **off by default**: any appreciable value
> exceeds the tiny per-step force gains near zero current and stalls the solver,
> leaving every coil at ~0 A. Use `~solver_smoothness` only with care.

```bash
rosrun case7_dual_microsphere_closed_loop openmv_tracker.py
rosrun case7_dual_microsphere_closed_loop main_joystick.py
```

Then in the GUI press **Reset -> Mode Set -> Online Check**, and:

- **Right stick** moves the selected microsphere(s).
- **A button** switches the controlled target: `red -> green -> both -> red`.
  The active target is printed on the terminal on every switch.

The joystick is read straight from `/dev/input/jsX` (no pygame / ros-joy / evdev
dependency). The gamepad is auto-detected; devices with no buttons (e.g. the
laptop accelerometer) are skipped. Tunables are ROS params:

```bash
rosrun case7_dual_microsphere_closed_loop main_joystick.py \
  _joystick_force_N:=6e-4 _joy_deadzone:=0.15 _joy_invert_y:=true
```

- `~joy_device` (default `""` = auto) — force a specific `/dev/input/jsX`.
- `~joy_axis_x`, `~joy_axis_y` (default `3`, `4`) — right-stick axis indices.
- `~joy_button_cycle` (default `0` = A) — target-switch button index.
- `~joy_deadzone` (default `0.15`) — stick deadzone fraction.
- `~joy_invert_x` / `~joy_invert_y` (default `false` / `true`) — flip an axis if
  the sphere moves opposite to the stick (defaults map stick-up to `+y`).
- `~joystick_force_N` (default `6e-4`) — push force magnitude at full deflection,
  clamped to `max_desired_force_N` (raise both to push harder).
- `~move_damping_N_s_per_m` (default `0.1`) — velocity damping on a driven
  sphere; raise it if the sphere overshoots or oscillates.
- `~move_weight` / `~hold_weight` (default `1.0` / `0.3`) — solver priority of a
  commanded vs a held sphere. Raise `~hold_weight` toward `~move_weight` if the
  held sphere drifts; lower it if the held sphere jitters or holding fights your
  command too hard.
- `~solver_smoothness` (default off) — penalty on per-step current changes. Off
  by default because the quadratic force law makes any non-trivial value stall
  the solver near zero current; leave it off unless you understand the risk.
- `~solver_regularization` (default = JSON value) — penalty on total current.

### Tuning the feel

| Symptom | Knob |
| --- | --- |
| Driven sphere too weak / slow | raise `~joystick_force_N` (and JSON `max_desired_force_N`) |
| Held sphere drifts toward the moving one | raise `~hold_weight` |
| Held sphere jitters / jumps | lower `~hold_weight` (trades a little drift for less jitter) |
| Driven sphere overshoots / oscillates | raise `~move_damping_N_s_per_m` |

Physics settings (`k_mag`, `z_plane_m`, current limits, `kd`, slew limits) are
loaded from `trajectory_case7_two_microspheres.json` so the field model matches
the tuned trajectory case; if the file is missing it falls back to defaults.
