# Case 8 — Data-Driven Laser Spot Control

A model-free replacement for the analytic field-fit controller. Instead of a
hand-derived magnetics model, we **learn** the mapping from coil currents to the
laser spot from real measurements, then invert that learned model online to
follow a trajectory.

```
   collect_data.py  ->  train_model.py  ->  evaluate_model.py  ->  nn_closed_loop_control.py
   (currents,spot)      forward model        error in mm           learned-Jacobian servo
      CSV in data/        .npz in models/      plots in models/      results in results/
```

## Why this design

- **Learn the forward model `f: currents(10) -> spot(2)`.** This is well-posed:
  10 inputs, 2 outputs, one spot per current vector.
- **Do *not* learn the inverse directly.** `spot -> currents` is one-to-many (the
  arm has 2 DOF but 10 coils), so a network would average conflicting solutions
  and produce currents that miss the target.
- **Control by inverting `f` online with feedback.** At each step we take the
  network's input Jacobian `J = df/dI` (the local sensitivity of the spot to the
  currents) and step the currents along the damped pseudo-inverse of `J` toward
  the *measured* error. Because the error uses the real camera spot, a constant
  model bias is rejected automatically — the model only has to get the local
  *direction* right, not the absolute spot.

The math is verified in `mlp_model.py` (run it directly: analytic Jacobian vs
finite differences) and the servo was simulated to reject a 2 mm plant bias down
to sub-mm tracking.

## Files

| File | Role |
|------|------|
| `mlp_model.py` | Pure-NumPy MLP (forward, training grads, analytic input Jacobian, save/load). Shared by all stages. |
| `collect_data.py` | ROS node: sweep static currents, record `/laser/spot`, write `data/case8_dataset_*.csv`. |
| `train_model.py` | Train forward model with Adam + early stopping. Writes `models/forward_model_*.npz` (+ `_latest`) and a loss curve. |
| `evaluate_model.py` | Report error in mm + scatter/histogram plots. |
| `nn_closed_loop_control.py` | ROS node: deploy the model as a learned-Jacobian visual servo. Writes `results/nn_closed_loop_case8_*.csv` + plot. |

No new dependencies — everything uses the already-installed numpy/matplotlib.
(If you later want PyTorch for training, keep the same `.npz` layout and
deployment loads it unchanged.)

## Step 0 — Build

```bash
cd ~/Documents/CUHK/ROS_CAN
catkin_make
source devel/setup.bash
```

## Step 1 — Collect data

Start the camera tracker (publishes `/laser/spot` in meters), then the collector:

```bash
rosrun case8_two_dof_laser_arm_control openmv_tracker.py
rosrun case8_two_dof_laser_arm_control collect_data.py _num_samples:=2000
```

### Why a "drifting walk", not a global sweep

The laser spot is visible only for a **tiny region of current space** — your real
runs span the whole workspace with just ~0.1–0.3 A. If you sample the full
±15 A box, the spot flies off-camera on almost every sample, the tracker
publishes nothing, and the row is wasted (in offline simulation, global ±15 A
keeps only ~24% of samples in view; the seeded walk keeps ~92%).

So `collect_data.py` defaults to `_sampling:=walk`, which is exactly the
approach you described:

1. **Seed** at a current vector known to keep the spot in view.
2. **Perturb** the currents slightly and measure.
3. If still visible → adopt it as the new center (the walk *drifts* across the
   visible region) and let the step grow a little to probe the edge.
4. If lost → shrink the step, return to the last good center, and re-seed after
   repeated losses.

This automatically stays inside the visible set and traces its boundary, so you
never have to hand-tune a current range.

### Picking the seed

| `_seed_mode` | How the seed is chosen |
|--------------|------------------------|
| `model` (default) | Analytic solution for `_seed_target_x/_y` (defaults to star point 1). Same solver as `plot_point1_yz_plane_field.py`. Labels are still real measurements, so the learned model is *not* limited by this analytic fit. |
| `manual` | You supply `_seed_currents:="i0,i1,...,i9"` — the most reliable option; find a visible set by nudging the sliders in `main2.py`. |
| `zero` | Start from 0 A (the arm's rest pose). |

If the run logs "Seed spot NOT visible", switch to a `manual` seed or change
`_seed_target_x/_y`.

Key parameters (all ROS params, prefix `_`):

| Param | Default | Meaning |
|-------|---------|---------|
| `num_samples` | 2000 | number of current vectors to try |
| `sampling` | `walk` | `walk` (seeded drift) or `global` (LHS/random box) |
| `seed_mode` | `model` | `model` / `manual` / `zero` (see above) |
| `seed_currents` | "" | manual seed, 10 comma-separated Amps (overrides `seed_mode`) |
| `walk_sigma_A` | 0.03 | base per-coil step — **keep small**, the spot is sensitive |
| `walk_sigma_min_A` / `walk_sigma_max_A` | 0.01 / 0.3 | step adaptation bounds |
| `reseed_after_losses` | 8 | consecutive off-camera samples before re-seeding |
| `current_range_A` | 15 | hard safety clip only (the walk stays well inside it) |
| `settle_time_s` | 0.6 | wait after reaching currents before measuring |
| `measure_time_s` | 0.4 | window to average the spot over |
| `slew_step_A` | 0.05 | per-step current ramp (safety) |
| `zero_between` | false | return to 0 A between samples (reduces hysteresis) |
| `position_scale` | 1.0 | set if the tracker publishes pixels not meters |

The CSV has `I0_A..I9_A, spot_x_m, spot_y_m, spot_std_*, n_samples, detected`.
Rows where the laser wasn't detected are marked `detected=0` and skipped in
training (they harmlessly document the visible boundary). Collection is
crash-safe (each row is flushed immediately). Aim for **≥1000–2000 detected
samples**; collect a second file to use as a held-out test set.

> Tip: do a short run first (`_num_samples:=150`) and open the CSV — confirm most
> rows have `detected=1` and that `spot_x_m`/`spot_y_m` cover the region your
> trajectory visits. If coverage is narrow, raise `_walk_sigma_max_A` a little or
> run more samples; if many rows are `detected=0`, lower `_walk_sigma_A`.

## Step 2 — Train

```bash
roscd case8_two_dof_laser_arm_control/scripts   # or: cd .../scripts
python3 train_model.py --epochs 4000
# multiple files / bigger net:
python3 train_model.py --data data/case8_dataset_*.csv --hidden 64 64 --epochs 4000
```

Writes `models/forward_model_<timestamp>.npz`, updates
`models/forward_model_latest.npz`, and saves `training_loss_<timestamp>.png`.
Watch the printed **val RMS (mm)** — that is your model's spot-prediction error.

## Step 3 — Evaluate

```bash
python3 evaluate_model.py                                  # scores all data
python3 evaluate_model.py --data data/case8_dataset_HELDOUT.csv   # honest test set
```

Prints RMS / median / 90th-pct / max error in mm and writes
`eval_scatter.png` (measured vs predicted) and `eval_error_hist.png`. If RMS is
much worse on held-out data than during training, collect more / wider data or
shrink the network (`--hidden 32 32`, raise `--weight-decay`).

## Step 4 — Deploy

```bash
rosrun case8_two_dof_laser_arm_control openmv_tracker.py
rosrun case8_two_dof_laser_arm_control nn_closed_loop_control.py
```

The node warm-starts the currents by inverting the model for the first target,
holds there until the spot arrives (within `start_tolerance_m`), then runs the
timed trajectory from `trajectory_case8_laser_spot.json`. It writes
`results/nn_closed_loop_case8_*.csv` and a target-vs-actual plot, and logs the
tracking **RMS in mm** on stop.

Tuning params (ROS params):

| Param | Default | Meaning |
|-------|---------|---------|
| `model` | `models/forward_model_latest.npz` | model to deploy |
| `servo_kp` | 0.6 | feedback gain — raise for faster tracking, lower if it oscillates |
| `servo_damping` | 1e-4 | Jacobian-inverse damping (raise near singular poses) |
| `nullspace_beta` | 0.02 | pull redundant currents toward zero |
| `current_limit_A` | 1.0 | hard clamp per coil |
| `max_current_step_A` | 0.08 | per-step slew limit |

Start conservative (`servo_kp:=0.3`), confirm stable tracking, then increase.

## Troubleshooting

- **"No /laser/spot"** → the tracker isn't running or publishes a different
  topic/units. Check `rostopic echo /laser/spot`; set `_position_scale` if pixels.
- **Model accurate (Step 3) but tracking poor (Step 4)** → most likely `servo_kp`
  too high (oscillation) or too low (lag), or the trajectory leaves the region
  you collected data in. Re-collect to cover the full path.
- **Currents saturate at the limit** → the target spot is outside the reachable
  set you sampled, or `nullspace_beta` is too small. Re-check data coverage.
- **Verify the Jacobian** any time: `python3 mlp_model.py` (self-test must pass).
