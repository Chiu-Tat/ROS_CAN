#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case 8 data-driven step 4: deploy the learned model to track a trajectory.

The trained network is the *forward* model f(currents) -> spot. We control by
inverting it online with measured feedback (a learned-Jacobian visual servo):

    J = df/dI at the current command          (2x10, from the network)
    e = target_spot - measured_spot
    dI = Kp * J^+ e  -  beta * nullspace(J) @ I   (pull redundant currents to 0)

Because the error `e` uses the *real* measured spot (not the model's
prediction), steady-state error is driven down even if the model has a constant
bias - the model only needs to supply a locally correct sensitivity J.

Run:
    rosrun case8_two_dof_laser_arm_control openmv_tracker.py
    rosrun case8_two_dof_laser_arm_control nn_closed_loop_control.py \
        _model:=$(rospack find case8_two_dof_laser_arm_control)/scripts/models/forward_model_latest.npz
"""

import glob
import json
import os
import sys
import time

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped

from can_lib import CANBusHandler, CANMessageBuilder
from mlp_model import MLP

NUM_COILS = 10
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- trajectory helpers (same sampling as the analytic controller) --------
def catmull_rom_path(raw_path, samples_per_segment):
    if len(raw_path) < 2:
        return raw_path.copy()
    path = []
    for seg in range(len(raw_path) - 1):
        p0 = raw_path[max(seg - 1, 0)]
        p1 = raw_path[seg]
        p2 = raw_path[seg + 1]
        p3 = raw_path[min(seg + 2, len(raw_path) - 1)]
        for s in range(samples_per_segment):
            t = s / float(samples_per_segment)
            t2, t3 = t * t, t * t * t
            path.append(0.5 * (2.0 * p1 + (-p0 + p2) * t
                               + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                               + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3))
    path.append(raw_path[-1])
    return np.array(path, dtype=float)


def resample_by_speed(path, speed, dt):
    if speed <= 0.0 or len(path) < 2:
        return path
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    spacing = speed * dt
    if total <= 0.0 or spacing <= 0.0:
        return path
    count = max(2, int(round(total / spacing)) + 1)
    d = np.linspace(0.0, total, count)
    return np.column_stack((np.interp(d, cum, path[:, 0]), np.interp(d, cum, path[:, 1])))


def invert_model(model, target, i0, iters=150, damping=1e-4, reg=1e-5, limit=15.0):
    """Gauss-Newton inversion: find currents whose predicted spot ~= target.

    Used once to warm-start the command so the servo doesn't begin from zero.
    Minimizes ||f(I) - target||^2 + reg*||I||^2 with current limits.
    """
    I = np.clip(np.asarray(i0, dtype=float), -limit, limit)
    for _ in range(iters):
        r = model.predict(I) - target            # (2,)
        J = model.input_jacobian(I)              # (2,10)
        # solve (J^T J + (damping+reg) I) dI = -(J^T r + reg I)
        H = J.T @ J + (damping + reg) * np.eye(NUM_COILS)
        g = J.T @ r + reg * I
        dI = np.linalg.solve(H, -g)
        I = np.clip(I + dI, -limit, limit)
        if np.linalg.norm(dI) < 1e-6:
            break
    return I


class NNController(object):
    def __init__(self):
        rospy.init_node("case8_nn_controller", anonymous=False)

        model_path = rospy.get_param("~model", os.path.join(SCRIPT_DIR, "models", "forward_model_latest.npz"))
        self.model = MLP.load(model_path)
        rospy.loginfo("Loaded forward model: %s (layers %s)", model_path, self.model.layer_sizes)

        traj_path = self._select_trajectory()
        self.spot_topic = rospy.get_param("~spot_topic", "/laser/spot")
        self.position_scale = float(rospy.get_param("~position_scale", 1.0))

        # control gains / limits
        self.dt_s = float(rospy.get_param("~control_dt_s", 0.025))
        self.kp = float(rospy.get_param("~servo_kp", 0.3))
        self.servo_damping = float(rospy.get_param("~servo_damping", 1e-3))
        self.nullspace_beta = float(rospy.get_param("~nullspace_beta", 0.0))
        self.current_limit_A = float(rospy.get_param("~current_limit_A", 15.0))
        self.max_current_step_A = float(rospy.get_param("~max_current_step_A", 0.05))
        self.start_tolerance_m = float(rospy.get_param("~start_tolerance_m", 0.004))
        self.target_radius_m = float(rospy.get_param("~target_radius_m", 0.0025))
        self.num_loops = int(rospy.get_param("~num_loops", 0))   # 0 = loop forever
        self.loop_count = 0
        self.pos_filter_alpha = float(rospy.get_param("~position_filter_alpha", 0.3))
        # anti-swing extras
        self.feedforward = bool(rospy.get_param("~feedforward", True))
        self.current_smoothing = float(rospy.get_param("~current_smoothing", 0.5))
        self.error_deadband_m = float(rospy.get_param("~error_deadband_m", 0.0003))
        self._dI_filt = np.zeros(NUM_COILS)

        # trajectory-following method: "pursuit" (pure pursuit) or "time"
        self.reference_mode = str(rospy.get_param("~reference_mode", "pursuit")).lower()
        self.lookahead_m = float(rospy.get_param("~lookahead_m", 0.006))
        self.max_lead_m = float(rospy.get_param("~max_lead_m", 0.012))
        self.progress_search_window = int(rospy.get_param("~progress_search_window", 60))
        self.progress_backtrack = int(rospy.get_param("~progress_backtrack", 10))
        self.progress_idx = 0
        self._refu = 0.0
        self._progu = 0.0
        self._prevc = 0

        self.trajectory = self._load_trajectory(traj_path)
        rospy.loginfo("Trajectory: %d points, dt=%.3f s", len(self.trajectory), self.dt_s)

        self.can_handler = CANBusHandler(
            channel=rospy.get_param("~can_channel", "can0"),
            bustype=rospy.get_param("~can_bustype", "socketcan"),
            verbose_send=False,
        )
        self.currents = np.zeros(NUM_COILS)

        self._spot = None
        self._spot_time = None
        rospy.Subscriber(self.spot_topic, PointStamped, self._spot_cb)

        self.results_dir = os.path.join(SCRIPT_DIR, "results")
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        self.log = []

    def _select_trajectory(self):
        """Pick the trajectory file.

        Priority: explicit `~trajectory` (full path) > `~shape` shorthand
        (trajectory_case8_<shape>.json) > interactive menu of the available
        trajectory_case8_*.json files. Falls back to the first file when not
        attached to a terminal (e.g. launched from a .launch file).
        """
        explicit = rospy.get_param("~trajectory", "")
        if explicit:
            return explicit
        shape = rospy.get_param("~shape", "")
        if shape:
            return os.path.join(SCRIPT_DIR, "trajectory_case8_%s.json" % shape)

        files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "trajectory_case8_*.json")))
        if not files:
            raise SystemExit("No trajectory_case8_*.json files found in %s" % SCRIPT_DIR)
        if len(files) == 1 or not sys.stdin.isatty():
            rospy.loginfo("Using trajectory: %s", os.path.basename(files[0]))
            return files[0]

        print("\nAvailable trajectories:")
        for i, f in enumerate(files):
            label = os.path.basename(f)
            try:
                with open(f) as fh:
                    label += "  -  " + json.load(fh).get("description", "")
            except Exception:
                pass
            print("  [%d] %s" % (i, label))
        while True:
            sel = raw_input("Select trajectory number [0]: ").strip() if sys.version_info[0] < 3 \
                else input("Select trajectory number [0]: ").strip()
            if sel == "":
                return files[0]
            if sel.isdigit() and int(sel) < len(files):
                return files[int(sel)]
            print("Invalid selection, try again.")

    def _load_trajectory(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        settings = data.get("settings", {})
        raw = np.array(data["laser_spot"]["raw_path"], dtype=float)
        dense = catmull_rom_path(raw, int(settings.get("samples_per_segment", 160)))
        self.target_speed_m_per_s = float(settings.get("target_speed_m_per_s", 0.002))
        traj = resample_by_speed(dense, self.target_speed_m_per_s, self.dt_s)
        # cumulative arc length along the resampled path (for look-ahead lookup)
        seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        self.cumlen = np.concatenate(([0.0], np.cumsum(seg)))
        self.path_len = float(self.cumlen[-1])
        return traj

    def _spot_cb(self, msg):
        raw = np.array([msg.point.x, msg.point.y]) * self.position_scale
        if self._spot is None:
            self._spot = raw
        else:
            self._spot = self._spot + self.pos_filter_alpha * (raw - self._spot)
        self._spot_time = time.time()

    # --- CAN ----------------------------------------------------------------
    def _send_currents(self, currents_A):
        for idx in range(NUM_COILS):
            current_mA = int(np.clip(currents_A[idx] * 1000.0, -15000, 15000))
            can_id, data = CANMessageBuilder.set_device_current(idx, current_mA)
            self.can_handler.send_message(can_id, data)
        self.currents = np.asarray(currents_A, dtype=float)

    def _init_coils(self):
        can_id, data = CANMessageBuilder.reset_all()
        self.can_handler.send_message(can_id, data)
        rospy.sleep(0.3)
        can_id, data = CANMessageBuilder.mode_set()
        self.can_handler.send_message(can_id, data)
        rospy.sleep(0.5)
        can_id, data = CANMessageBuilder.initialize()
        self.can_handler.send_message(can_id, data)
        rospy.sleep(0.3)

    def _apply_step_limited(self, desired):
        desired = np.clip(desired, -self.current_limit_A, self.current_limit_A)
        delta = np.clip(desired - self.currents, -self.max_current_step_A, self.max_current_step_A)
        self._send_currents(self.currents + delta)

    # --- control law --------------------------------------------------------
    def _servo_update(self, target, target_prev=None):
        """One learned-Jacobian visual-servo step toward `target`.

        dI = J^+ ( v_ff + Kp * deadband(e) )  -  beta * N @ I,  then EMA-smoothed.

        The feedforward term v_ff = target - target_prev moves the spot *with*
        the reference so the proportional term only has to correct residual
        error. With a small Kp this stops the loop from bang-banging the slew
        limit (the cause of the sharp swings). A soft deadband keeps measurement
        and model noise from driving needless current motion.
        """
        J = self.model.input_jacobian(self.currents)             # (2,10)
        e = target - self._spot                                  # (2,)

        # soft deadband: subtract a small dead-zone radius, keep direction
        e_norm = np.linalg.norm(e)
        if e_norm > 1e-12:
            e = e * max(0.0, e_norm - self.error_deadband_m) / e_norm

        # velocity feedforward from the reference (zero while holding a point)
        v_ff = (target - target_prev) if (self.feedforward and target_prev is not None) \
            else np.zeros(2)

        JJt = J @ J.T + self.servo_damping * np.eye(2)
        Jpinv = J.T @ np.linalg.solve(JJt, np.eye(2))            # (10,2)
        dI = Jpinv @ (v_ff + self.kp * e)
        # null-space term keeps redundant currents small/unique
        N = np.eye(NUM_COILS) - Jpinv @ J
        dI -= self.nullspace_beta * (N @ self.currents)

        # EMA smoothing of the command increment to suppress sharp swings
        a = self.current_smoothing
        self._dI_filt = (1.0 - a) * self._dI_filt + a * dI
        self._apply_step_limited(self.currents + self._dI_filt)

    # --- pursuit (clock-paced look-ahead, closed-loop in speed) -------------
    def _reset_pursuit(self):
        """Reset the pursuit reference/progress state (call when tracking starts)."""
        self._refu = 0.0       # clock-paced reference position, in index units
        self._progu = 0.0      # unwrapped progress of the measured spot, index units
        self._prevc = 0        # previous closest index (for unwrapping)
        self.progress_idx = 0  # window center for the closest-point search
        self.loop_count = 0

    def _pursuit_step(self):
        """One pursuit step.

        Pace is set by a *clock-paced* reference that advances one resampled-index
        per tick (== target_speed, since the path is resampled at that speed). We
        aim a look-ahead "carrot" ahead of that reference and drive the spot to it
        with proportional feedback through the damped learned Jacobian.

        Critically, the reference is clamped to lead the measured spot by at most
        `max_lead_m`; it is never pulled forward by the spot. So the speed does
        NOT depend on the model's (possibly wrong) Jacobian magnitude: if the
        model over-estimates sensitivity the error simply grows until the spot
        keeps up, and if the spot truly cannot keep up the reference waits.
        """
        n = len(self.trajectory)
        spacing = self.path_len / max(1, n - 1)
        la = max(1, int(round(self.lookahead_m / spacing)))
        ml = max(1, int(round(self.max_lead_m / spacing)))
        spot = self._spot

        # windowed, forward-biased closest-point search around last progress
        offs = np.arange(-self.progress_backtrack, self.progress_search_window + 1)
        cand = (self.progress_idx + offs) % n
        closest = int(cand[int(np.argmin(np.linalg.norm(self.trajectory[cand] - spot, axis=1)))])
        # unwrap progress across the closed-path seam
        delta = ((closest - self._prevc + n // 2) % n) - n // 2
        self._progu += delta
        self._prevc = closest
        self.progress_idx = closest

        # clock-paced reference: +1 index/tick, but never more than ml ahead of spot
        self._refu = min(self._refu + 1.0, self._progu + ml)
        carrot = self.trajectory[int(round(self._refu + la)) % n]

        # proportional visual servo toward the carrot
        e = carrot - spot
        J = self.model.input_jacobian(self.currents)
        JJt = J @ J.T + self.servo_damping * np.eye(2)
        Jpinv = J.T @ np.linalg.solve(JJt, np.eye(2))
        dI = Jpinv @ (self.kp * e)

        a = self.current_smoothing
        self._dI_filt = (1.0 - a) * self._dI_filt + a * dI
        self._apply_step_limited(self.currents + self._dI_filt)

        # lap counting from unwrapped progress
        laps = int(self._progu // n)
        if laps > self.loop_count:
            self.loop_count = laps
            rospy.loginfo("Completed loop %d of %s", self.loop_count,
                          self.num_loops if self.num_loops > 0 else "inf")
        return carrot, closest

    def run(self):
        rospy.loginfo("Waiting for /laser/spot...")
        t0 = time.time()
        while self._spot is None and not rospy.is_shutdown():
            if time.time() - t0 > 10.0:
                rospy.logerr("No /laser/spot. Is openmv_tracker.py running?")
                return
            time.sleep(0.05)

        self._init_coils()

        # warm-start the command by inverting the model for the first target
        warm = invert_model(self.model, self.trajectory[0], np.zeros(NUM_COILS),
                            limit=self.current_limit_A)
        rospy.loginfo("Warm-start currents |I|=%.3f A", np.linalg.norm(warm))
        self._apply_step_limited(warm)

        # hold first point until the spot arrives, then run timed reference
        started = False
        idx = 0
        start_time = time.time()
        rate = rospy.Rate(1.0 / self.dt_s)
        while not rospy.is_shutdown():
            if self._spot is None:
                rate.sleep()
                continue
            if not started:
                target = self.trajectory[0]
                self._servo_update(target)                       # hold: no feedforward
                if np.linalg.norm(target - self._spot) <= self.start_tolerance_m:
                    started = True
                    start_time = time.time()
                    self._reset_pursuit()
                    rospy.loginfo("Reached start point; following path (mode=%s)",
                                  self.reference_mode)
            elif self.reference_mode == "pursuit":
                target, idx = self._pursuit_step()
                if self.num_loops > 0 and self.loop_count >= self.num_loops:
                    rospy.loginfo("All %d loop(s) complete", self.num_loops)
                    break
            else:
                prev = self.trajectory[idx]
                idx += 1
                if idx >= len(self.trajectory):
                    # path is closed (start == end): wrap for continuous tracking
                    idx = 0
                    self.loop_count += 1
                    rospy.loginfo("Completed loop %d of %s", self.loop_count,
                                  self.num_loops if self.num_loops > 0 else "inf")
                    if self.num_loops > 0 and self.loop_count >= self.num_loops:
                        rospy.loginfo("All %d loop(s) complete", self.num_loops)
                        break
                target = self.trajectory[idx]
                self._servo_update(target, target_prev=prev)

            # cross-track error: distance to the closest path point (trajectory[idx]
            # is the closest point in pursuit mode, the target in time mode)
            ref = self.trajectory[idx] if started else target
            err = float(np.linalg.norm(ref - self._spot))
            self.log.append([time.time() - start_time, ref[0], ref[1],
                             self._spot[0], self._spot[1], err, idx]
                            + list(self.currents))
            rospy.loginfo_throttle(1.0, "loop=%d idx=%d err=%.4f m |I|=%.3f A",
                                   self.loop_count, idx, err, np.linalg.norm(self.currents))
            rate.sleep()

        self._send_currents(np.zeros(NUM_COILS))
        self._save_results()

    def _save_results(self):
        if not self.log:
            return
        run_id = time.strftime("%Y%m%d_%H%M%S")
        arr = np.array(self.log)
        header = (["Time_s", "Target_X_m", "Target_Y_m", "Spot_X_m", "Spot_Y_m",
                   "Error_m", "Target_Index"] + ["I%d_A" % i for i in range(NUM_COILS)])
        csv_path = os.path.join(self.results_dir, "nn_closed_loop_case8_%s.csv" % run_id)
        with open(csv_path, "w") as f:
            f.write(",".join(header) + "\n")
            for row in arr:
                f.write(",".join("%.8g" % v for v in row) + "\n")
        rms_mm = float(np.sqrt(np.mean(arr[:, 5] ** 2))) * 1000.0
        rospy.loginfo("Saved %s  (tracking RMS = %.2f mm)", csv_path, rms_mm)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(self.trajectory[:, 0] * 1000, self.trajectory[:, 1] * 1000,
                    "k--", lw=1, label="target path")
            ax.plot(arr[:, 3] * 1000, arr[:, 4] * 1000, "r-", lw=1.2, label="actual spot")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_aspect("equal", "box")
            ax.set_title("Case 8 NN control (RMS = %.2f mm)" % rms_mm)
            ax.legend()
            fig.tight_layout()
            png = os.path.join(self.results_dir, "nn_closed_loop_case8_%s.png" % run_id)
            fig.savefig(png, dpi=130)
            rospy.loginfo("Saved %s", png)
        except Exception as exc:
            rospy.logwarn("Could not plot result: %s", exc)

    def shutdown(self):
        try:
            self._send_currents(np.zeros(NUM_COILS))
            self.can_handler.close()
        except Exception:
            pass


if __name__ == "__main__":
    controller = NNController()
    rospy.on_shutdown(controller.shutdown)
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
