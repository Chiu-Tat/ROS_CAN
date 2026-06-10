#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case 8 data-driven step 1: collect (coil currents -> laser spot) pairs.

Why not sample the whole current box: the laser spot is only visible for a tiny
region of current space (the real runs span the whole workspace with only
~0.1-0.3 A). Random/LHS sampling over +/-15 A would throw the spot off-camera on
almost every sample - the OpenMV tracker then publishes nothing and the row is
useless.

Default strategy is a SEEDED DRIFTING WALK (matches the suggested approach):
  1. Start from a current vector that is known to keep the spot in view
     (the analytic solution for a reachable target spot, a manual seed you found
     with main2.py, or zero current).
  2. Perturb the currents by a small random step and measure.
  3. If the spot is still visible -> keep it as the new center (the walk drifts
     across the visible region) and let the step grow a little to probe outward.
  4. If the spot is lost -> shrink the step and return to the last good center;
     after repeated losses, re-seed.
This keeps almost every sample inside the visible set and naturally maps out its
boundary ("change the current slightly until the spot is lost").

Run the OpenMV tracker first so /laser/spot is published in meters:
    rosrun case8_two_dof_laser_arm_control openmv_tracker.py
    rosrun case8_two_dof_laser_arm_control collect_data.py _num_samples:=2000

Useful params:
    _sampling:=walk|global         (default walk)
    _seed_mode:=model|manual|zero  (default model; manual uses _seed_currents)
    _seed_currents:="0.1,-0.05,..." (10 comma-separated Amps, overrides seed_mode)
    _seed_target_x:=0.0 _seed_target_y:=0.05963   (target spot for the model seed)
    _walk_sigma_A:=0.03            (base per-coil step; keep small, spot is sensitive)
    _walk_sigma_max_A:=0.3 _walk_sigma_min_A:=0.01
"""

import os
import time

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped

from can_lib import CANBusHandler, CANMessageBuilder

NUM_COILS = 10


class DataCollector(object):
    def __init__(self):
        rospy.init_node("case8_data_collector", anonymous=False)

        # --- general sampling parameters ---------------------------------
        self.num_samples = int(rospy.get_param("~num_samples", 2000))
        self.sampling = rospy.get_param("~sampling", "walk")       # walk | global
        self.current_range_A = float(rospy.get_param("~current_range_A", 15))  # hard clip only
        self.strategy = rospy.get_param("~strategy", "lhs")        # global: lhs | random
        self.seed = int(rospy.get_param("~seed", 0))
        self.rng = np.random.default_rng(self.seed)
        self.zero_between = bool(rospy.get_param("~zero_between", False))

        # --- seed (for the walk) -----------------------------------------
        self.trajectory_path = rospy.get_param(
            "~trajectory",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory_case8_laser_spot.json"))
        self.settings, default_seed_target = self._load_settings_and_point1()
        self.seed_mode = rospy.get_param("~seed_mode", "model")    # model | manual | zero
        self.seed_currents_str = rospy.get_param("~seed_currents", "")
        self.seed_target = np.array([
            float(rospy.get_param("~seed_target_x", default_seed_target[0])),
            float(rospy.get_param("~seed_target_y", default_seed_target[1])),
        ])

        # --- walk step adaptation ----------------------------------------
        self.walk_sigma_A = float(rospy.get_param("~walk_sigma_A", 0.03))
        self.walk_sigma_min_A = float(rospy.get_param("~walk_sigma_min_A", 0.01))
        self.walk_sigma_max_A = float(rospy.get_param("~walk_sigma_max_A", 0.3))
        self.walk_grow = float(rospy.get_param("~walk_grow", 1.2))
        self.walk_shrink = float(rospy.get_param("~walk_shrink", 0.5))
        self.reseed_after_losses = int(rospy.get_param("~reseed_after_losses", 8))

        # --- timing / safety ---------------------------------------------
        self.settle_time_s = float(rospy.get_param("~settle_time_s", 0.6))
        self.measure_time_s = float(rospy.get_param("~measure_time_s", 0.4))
        self.slew_step_A = float(rospy.get_param("~slew_step_A", 0.05))
        self.slew_dt_s = float(rospy.get_param("~slew_dt_s", 0.02))
        self.spot_timeout_s = float(rospy.get_param("~spot_timeout_s", 1.0))
        self.spot_topic = rospy.get_param("~spot_topic", "/laser/spot")
        self.position_scale = float(rospy.get_param("~position_scale", 1.0))

        # --- output -------------------------------------------------------
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.data_dir = rospy.get_param("~data_dir", default_dir)
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        run_id = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.data_dir, "case8_dataset_%s.csv" % run_id)

        # --- CAN ----------------------------------------------------------
        self.can_handler = CANBusHandler(
            channel=rospy.get_param("~can_channel", "can0"),
            bustype=rospy.get_param("~can_bustype", "socketcan"),
            verbose_send=False,
        )
        self.current_command = np.zeros(NUM_COILS)

        # --- spot subscription -------------------------------------------
        self._latest_spot = None
        self._latest_spot_time = None
        self.spot_sub = rospy.Subscriber(self.spot_topic, PointStamped, self._spot_cb)

        rospy.loginfo("Case 8 collector: %d samples, sampling=%s, clip=+/-%.2f A",
                      self.num_samples, self.sampling, self.current_range_A)
        rospy.loginfo("Writing dataset to %s", self.csv_path)

    # --- config loading --------------------------------------------------
    def _load_settings_and_point1(self):
        import json
        try:
            with open(self.trajectory_path, "r") as f:
                data = json.load(f)
            point1 = np.array(data["laser_spot"]["raw_path"][0], dtype=float)
            return data.get("settings", {}), point1
        except Exception as exc:
            rospy.logwarn("Could not load trajectory %s: %s", self.trajectory_path, exc)
            return {}, np.array([0.0, 0.05963])

    # --- ROS callbacks ----------------------------------------------------
    def _spot_cb(self, msg):
        self._latest_spot = np.array([msg.point.x, msg.point.y]) * self.position_scale
        self._latest_spot_time = time.time()

    # --- CAN helpers ------------------------------------------------------
    def _send_currents(self, currents_A):
        for idx in range(NUM_COILS):
            current_mA = int(np.clip(currents_A[idx] * 1000.0, -15000, 15000))
            can_id, data = CANMessageBuilder.set_device_current(idx, current_mA)
            self.can_handler.send_message(can_id, data)
        self.current_command = np.asarray(currents_A, dtype=float)

    def _ramp_to(self, target_A):
        """Slew-limited ramp from the current command to target, for safety."""
        target_A = np.clip(np.asarray(target_A, dtype=float),
                           -self.current_range_A, self.current_range_A)
        while not rospy.is_shutdown():
            delta = target_A - self.current_command
            if float(np.max(np.abs(delta))) <= self.slew_step_A:
                self._send_currents(target_A)
                return
            step = np.clip(delta, -self.slew_step_A, self.slew_step_A)
            self._send_currents(self.current_command + step)
            time.sleep(self.slew_dt_s)

    def _init_coils(self):
        rospy.loginfo("Initializing coils (reset, mode set, zero current)...")
        can_id, data = CANMessageBuilder.reset_all()
        self.can_handler.send_message(can_id, data)
        rospy.sleep(0.3)
        can_id, data = CANMessageBuilder.mode_set()
        self.can_handler.send_message(can_id, data)
        rospy.sleep(0.5)
        can_id, data = CANMessageBuilder.initialize()
        self.can_handler.send_message(can_id, data)
        rospy.sleep(0.3)

    def _send_zero(self):
        self._send_currents(np.zeros(NUM_COILS))

    # --- seed -------------------------------------------------------------
    def _make_seed(self):
        """Return a current vector expected to keep the spot in view."""
        if self.seed_currents_str.strip():
            vals = [float(v) for v in self.seed_currents_str.split(",")]
            if len(vals) == NUM_COILS:
                rospy.loginfo("Using manual seed currents")
                return np.clip(np.array(vals), -self.current_range_A, self.current_range_A)
            rospy.logwarn("seed_currents needs %d values; got %d. Falling back.", NUM_COILS, len(vals))

        if self.seed_mode == "zero":
            return np.zeros(NUM_COILS)

        # model seed: analytic currents for a reachable target spot (see
        # plot_point1_yz_plane_field.py). Labels still come from real
        # measurements, so the learned model is not limited by this analytic fit.
        try:
            from closed_loop_control import ClosedLoopLaserSpotController, TwoDOFLaserModel
            model = TwoDOFLaserModel()
            solver = object.__new__(ClosedLoopLaserSpotController)
            solver.model = model
            solver.settings = {
                "field_strength_T": float(self.settings.get("field_strength_T", 0.004)),
                "current_limit_A": float(self.settings.get("current_limit_A", 15.0)),
                "current_smoothing": float(self.settings.get("current_smoothing", 0.35)),
                "current_regularization": float(self.settings.get("current_regularization", 1e-9)),
            }
            yaw, pitch = model.inverse_laser_target(self.seed_target[0], self.seed_target[1])
            kin = model.forward_kinematics(yaw, pitch)
            fields = ClosedLoopLaserSpotController._magnet_aligned_fields(solver, kin)
            sol = ClosedLoopLaserSpotController._solve_currents_for_fields(solver, kin, fields, np.zeros(NUM_COILS))
            seed = np.clip(sol["currents"], -self.current_range_A, self.current_range_A)
            rospy.loginfo("Model seed for target (%.3f, %.3f): |I|=%.3f A",
                          self.seed_target[0], self.seed_target[1], np.linalg.norm(seed))
            return seed
        except Exception as exc:
            rospy.logwarn("Model seed failed (%s). Using zero seed.", exc)
            return np.zeros(NUM_COILS)

    # --- measurement ------------------------------------------------------
    def _measure_spot(self):
        """Average the spot over measure_time_s. Returns (mean, std, n, ok).

        ok=False means the tracker published nothing fresh = laser off-camera.
        """
        samples = []
        end = time.time() + self.measure_time_s
        last_seen = None
        while time.time() < end and not rospy.is_shutdown():
            if self._latest_spot_time is not None and self._latest_spot_time != last_seen:
                if time.time() - self._latest_spot_time <= self.spot_timeout_s:
                    samples.append(self._latest_spot.copy())
                    last_seen = self._latest_spot_time
            time.sleep(0.005)
        if len(samples) == 0:
            return np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), 0, False
        arr = np.array(samples)
        return arr.mean(axis=0), arr.std(axis=0), len(arr), True

    def _probe(self, currents):
        """Ramp to currents, settle, and measure the spot."""
        self._ramp_to(currents)
        time.sleep(self.settle_time_s)
        return self._measure_spot()

    # --- CSV --------------------------------------------------------------
    def _open_csv(self):
        header = (["index"] + ["I%d_A" % i for i in range(NUM_COILS)]
                  + ["spot_x_m", "spot_y_m", "spot_std_x_m", "spot_std_y_m",
                     "n_samples", "detected", "unix_time"])
        f = open(self.csv_path, "w")
        f.write(",".join(header) + "\n")
        f.flush()
        return f

    def _write_row(self, f, index, mean, std, n, ok):
        row = ([index] + list(self.current_command)
               + [mean[0], mean[1], std[0], std[1], n, int(ok), time.time()])
        f.write(",".join("%.8g" % v for v in row) + "\n")
        f.flush()  # crash-safe: keep partial datasets

    # --- global sampling (kept as an option) ------------------------------
    def _global_samples(self):
        n, d, r = self.num_samples, NUM_COILS, self.current_range_A
        if self.strategy == "random":
            return self.rng.uniform(-r, r, size=(n, d))
        samples = np.empty((n, d))
        for j in range(d):
            cuts = (np.arange(n) + self.rng.random(n)) / n
            samples[:, j] = self.rng.permutation(cuts)
        return (2.0 * samples - 1.0) * r

    def _run_global(self, f):
        n_ok = 0
        for i, currents in enumerate(self._global_samples()):
            if rospy.is_shutdown():
                break
            mean, std, n, ok = self._probe(currents)
            self._write_row(f, i, mean, std, n, ok)
            n_ok += int(ok)
            if self.zero_between:
                self._ramp_to(np.zeros(NUM_COILS))
            if i % 25 == 0 or not ok:
                rospy.loginfo("[%d/%d] detected=%s spot=(%.4f, %.4f) ok_total=%d",
                              i + 1, self.num_samples, ok, mean[0], mean[1], n_ok)
        return n_ok

    # --- seeded drifting walk (default) -----------------------------------
    def _run_walk(self, f):
        seed = self._make_seed()
        mean, std, n, ok = self._probe(seed)
        if not ok:
            rospy.logwarn("Seed spot NOT visible. Provide a known-good _seed_currents "
                          "(find one with main2.py) or adjust _seed_target_x/y.")
        center = self.current_command.copy()
        sigma = self.walk_sigma_A
        self._write_row(f, 0, mean, std, n, ok)
        n_ok = int(ok)
        losses = 0

        for i in range(1, self.num_samples):
            if rospy.is_shutdown():
                break
            propose = np.clip(center + self.rng.normal(0.0, sigma, NUM_COILS),
                              -self.current_range_A, self.current_range_A)
            mean, std, n, ok = self._probe(propose)
            self._write_row(f, i, mean, std, n, ok)

            if ok:
                center = propose                       # drift into the visible region
                sigma = min(sigma * self.walk_grow, self.walk_sigma_max_A)
                losses = 0
                n_ok += 1
            else:
                sigma = max(sigma * self.walk_shrink, self.walk_sigma_min_A)
                losses += 1
                self._ramp_to(center)                  # back to last good currents
                if losses >= self.reseed_after_losses:
                    rospy.loginfo("Re-seeding after %d consecutive losses", losses)
                    center, sigma, losses = seed.copy(), self.walk_sigma_A, 0
                    self._ramp_to(center)

            if self.zero_between and ok:
                self._ramp_to(np.zeros(NUM_COILS))
            if i % 25 == 0 or not ok:
                rospy.loginfo("[%d/%d] detected=%s sigma=%.3f spot=(%.4f, %.4f) ok_total=%d",
                              i + 1, self.num_samples, ok, sigma, mean[0], mean[1], n_ok)
        return n_ok

    # --- main loop --------------------------------------------------------
    def run(self):
        rospy.loginfo("Waiting for first /laser/spot message...")
        t0 = time.time()
        while self._latest_spot is None and not rospy.is_shutdown():
            if time.time() - t0 > 10.0:
                rospy.logerr("No /laser/spot received. Is openmv_tracker.py running?")
                return
            time.sleep(0.1)

        self._init_coils()
        f = self._open_csv()
        try:
            if self.sampling == "global":
                n_ok = self._run_global(f)
            else:
                n_ok = self._run_walk(f)
        finally:
            f.close()

        self._send_zero()
        rospy.loginfo("Collection done: %d/%d detected. Saved %s",
                      n_ok, self.num_samples, self.csv_path)

    def shutdown(self):
        try:
            self._send_zero()
            self.can_handler.close()
        except Exception:
            pass


if __name__ == "__main__":
    collector = DataCollector()
    rospy.on_shutdown(collector.shutdown)
    try:
        collector.run()
    except rospy.ROSInterruptException:
        pass
