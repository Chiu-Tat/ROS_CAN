#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Closed-loop controller for Case 7: two soft-magnetic iron microspheres.

This controller mirrors the settings and force-to-current method in
~/Documents/DecaPulse_py/soft_microsphere_control.html:
  - two microspheres share one ten-coil current vector
  - particles are constrained to z = -0.006 m
  - force law is F = k_mag grad(B^T B)
  - currents are solved by bounded coordinate search with current and jump penalties
"""

import csv
import json
import math
import os
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped


DEFAULT_SETTINGS = {
    "workspace_radius_m": 0.07,
    "z_plane_m": -0.006,
    "current_limit_A": 15.0,
    "k_mag": 3e-4,
    "drag_N_s_per_m": 6e-6,
    "dt_s": 0.025,
    "kp_N_per_m": 1e-6,
    "kd_N_s_per_m": 6e-6,
    "target_radius_m": 0.0025,
    "path_lookahead_index": 8,
    "time_based_reference": True,
    "force_regularization": 1e-16,
    "smoothness": 1e-16,
}
MU_OVER_4PI = 1e-7
EPS = 1e-9
STALE_POSE_TIMEOUT = 0.5


COIL_PARAMS = np.array([
    [-13.04945069, -4.41557229, 6.47376799, 0.12129096, 0.00466922, -0.0174842],
    [-5.10083416, 13.54294901, 7.85474539, 0.05834654, -0.11165548, -0.01850546],
    [4.05088788, 14.23365818, 6.44760956, -0.05903076, -0.11020417, -0.01488244],
    [13.89011305, -0.06092074, 4.77365608, -0.12306086, -0.00085745, -0.01378161],
    [11.44363813, -9.40543896, 4.46367162, -0.06806179, 0.1024875, -0.01397152],
    [-9.00577939, -12.78905365, 5.98650851, 0.06473315, 0.10618968, -0.0151172],
    [0.92820081, 8.54965337, 8.72298349, -0.00381254, -0.08845466, -0.08874662],
    [8.7302819, -4.90773115, 7.00109937, -0.07977306, 0.04481733, -0.08536032],
    [-7.68962762, -6.83258326, 8.12112247, 0.07498008, 0.04542436, -0.08696975],
    [2.35614001, -1.11370036, 14.00304846, -0.00722183, 0.00029277, -0.12482979],
])


class ClosedLoopTrajectoryController:
    def __init__(
        self,
        can_handler,
        send_device_current_callback,
        topic_m1="/sphere/red",
        topic_m2="/sphere/green",
        position_scale=1.0,
        reference_duration_s=None,
        control_dt_s=None,
    ):
        self.can_handler = can_handler
        self.send_current = send_device_current_callback
        self.position_scale = position_scale
        self.reference_duration_s = reference_duration_s
        self.control_dt_s = control_dt_s

        self.control_running = False
        self.control_thread = None
        self.lock = threading.Lock()

        self.trajectory_data = None
        self.currents = np.zeros(10)
        self.settings = DEFAULT_SETTINGS.copy()
        self.states = {
            "m1": self._empty_state(),
            "m2": self._empty_state(),
        }
        self.results_data = []
        self.coil_currents_data = []
        self.position_trace_data = []
        self.recording_positions = False
        self.control_start_time = None
        self.run_id = None
        self.results_saved = False

        self.subscribers = {
            "m1": rospy.Subscriber(topic_m1, PointStamped, lambda msg: self._sphere_callback(msg, "m1")),
            "m2": rospy.Subscriber(topic_m2, PointStamped, lambda msg: self._sphere_callback(msg, "m2")),
        }
        rospy.loginfo("Case 7 subscribed to %s and %s", topic_m1, topic_m2)

    def _empty_state(self):
        return {
            "position": None,
            "velocity": np.zeros(2),
            "last_position": None,
            "last_time": None,
            "target_index": int(self.settings["path_lookahead_index"]),
            "desired_force": np.zeros(2),
            "actual_force": np.zeros(2),
        }

    def _sphere_callback(self, msg, key):
        now = time.time()
        pos = np.array([msg.point.x, msg.point.y], dtype=float) * self.position_scale
        with self.lock:
            state = self.states[key]
            if state["position"] is not None and state["last_time"] is not None:
                dt = now - state["last_time"]
                if dt > 1e-4:
                    state["velocity"] = (pos - state["position"]) / dt
            state["last_position"] = state["position"]
            state["position"] = pos
            state["last_time"] = now
            if self.recording_positions and self.control_start_time is not None:
                self._record_position_snapshot_locked(now, key)

    def prepare_trajectory(self, filepath):
        rospy.loginfo("Loading Case 7 trajectory from %s", filepath)
        with open(filepath, "r") as f:
            data = json.load(f)

        self.settings = self._read_controller_settings(data.get("settings", {}))
        if self.control_dt_s is not None:
            self.settings["dt_s"] = min(0.2, max(0.001, float(self.control_dt_s)))

        path_m1 = self._load_reference_path(data["microsphere_1"])
        path_m2 = self._load_reference_path(data["microsphere_2"])
        self.trajectory_data = {
            "m1": path_m1,
            "m2": path_m2,
            "settings": self.settings.copy(),
        }
        duration_s = self.reference_duration()
        rospy.loginfo(
            "Loaded Case 7 trajectories: M1=%d points, M2=%d points, dt=%.4f s, duration=%.2f s, limit=%.3f A, z=%.4f m",
            len(self.trajectory_data["m1"]),
            len(self.trajectory_data["m2"]),
            self.settings["dt_s"],
            duration_s,
            self.settings["current_limit_A"],
            self.settings["z_plane_m"],
        )

    def reference_duration(self):
        if self.trajectory_data is None:
            return 0.0
        points = max(len(self.trajectory_data["m1"]), len(self.trajectory_data["m2"]))
        return max(points - 1, 0) * self.settings["dt_s"]

    def _load_reference_path(self, sphere_data):
        if self.reference_duration_s is None:
            return np.array(sphere_data["path"], dtype=float)

        dt_s = self.settings["dt_s"]
        point_count = int(round(float(self.reference_duration_s) / dt_s)) + 1
        point_count = max(2, point_count)

        if "raw_path" in sphere_data:
            return self._catmull_rom_path(np.array(sphere_data["raw_path"], dtype=float), point_count)

        return self._resample_path(np.array(sphere_data["path"], dtype=float), point_count)

    def _catmull_rom_path(self, raw_path, point_count):
        if len(raw_path) < 2:
            return raw_path.copy()

        segment_count = len(raw_path) - 1
        samples_per_segment = int(round((point_count - 1) / float(segment_count)))
        samples_per_segment = max(1, samples_per_segment)
        path = []

        for segment in range(segment_count):
            p0 = raw_path[max(segment - 1, 0)]
            p1 = raw_path[segment]
            p2 = raw_path[segment + 1]
            p3 = raw_path[min(segment + 2, len(raw_path) - 1)]
            for sample in range(samples_per_segment):
                t = sample / float(samples_per_segment)
                path.append(self._catmull_rom_point(p0, p1, p2, p3, t))

        path.append(raw_path[-1])
        path = np.array(path, dtype=float)
        if len(path) == point_count:
            return path
        return self._resample_path(path, point_count)

    def _catmull_rom_point(self, p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    def _resample_path(self, path, point_count):
        if len(path) <= 1:
            return path.copy()
        old_t = np.linspace(0.0, 1.0, len(path))
        new_t = np.linspace(0.0, 1.0, point_count)
        x = np.interp(new_t, old_t, path[:, 0])
        y = np.interp(new_t, old_t, path[:, 1])
        return np.column_stack((x, y))

    def _read_controller_settings(self, raw_settings):
        settings = DEFAULT_SETTINGS.copy()
        for key in settings:
            if key not in raw_settings:
                continue
            settings[key] = raw_settings[key]

        settings["workspace_radius_m"] = max(1e-6, float(settings["workspace_radius_m"]))
        settings["current_limit_A"] = max(0.05, float(settings["current_limit_A"]))
        settings["k_mag"] = max(0.0, float(settings["k_mag"]))
        settings["drag_N_s_per_m"] = max(1e-9, float(settings["drag_N_s_per_m"]))
        settings["dt_s"] = min(0.2, max(0.001, float(settings["dt_s"])))
        settings["kp_N_per_m"] = max(0.0, float(settings["kp_N_per_m"]))
        settings["kd_N_s_per_m"] = max(0.0, float(settings["kd_N_s_per_m"]))
        settings["target_radius_m"] = max(0.0002, float(settings["target_radius_m"]))
        settings["path_lookahead_index"] = max(0, int(settings["path_lookahead_index"]))
        settings["time_based_reference"] = bool(settings["time_based_reference"])
        settings["force_regularization"] = max(0.0, float(settings["force_regularization"]))
        settings["smoothness"] = max(0.0, float(settings["smoothness"]))
        settings["z_plane_m"] = float(settings["z_plane_m"])
        return settings

    def start_control(self):
        if self.control_running:
            return
        if self.trajectory_data is None:
            rospy.logerr("No Case 7 trajectory loaded")
            return

        with self.lock:
            for key in self.states:
                self.states[key]["target_index"] = int(self.settings["path_lookahead_index"])

        self.currents = np.zeros(10)
        self.results_data = []
        self.coil_currents_data = []
        self.position_trace_data = []
        self.control_start_time = time.time()
        self.run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.control_start_time))
        self._clear_previous_results()
        self.recording_positions = True
        self.results_saved = False
        self.control_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        rospy.loginfo("Case 7 two-microsphere control started, run_id=%s", self.run_id)

    def stop_control(self):
        if not self.control_running:
            return

        self.control_running = False
        if self.control_thread and threading.current_thread() != self.control_thread:
            self.control_thread.join(timeout=2.0)

        for i in range(10):
            self.send_current(i, 0)
        self.recording_positions = False
        self._save_results()
        rospy.loginfo("Case 7 control stopped")

    def _control_loop(self):
        if not self._wait_for_initial_pose(timeout=5.0):
            self.control_running = False
            self.recording_positions = False
            self._send_zero_currents()
            self._save_results()
            return

        start_time = self.control_start_time or time.time()
        period = self.settings["dt_s"]

        while self.control_running and not rospy.is_shutdown():
            loop_start = time.time()
            elapsed = loop_start - start_time

            with self.lock:
                states = {
                    key: {
                        "position": value["position"].copy() if value["position"] is not None else None,
                        "velocity": value["velocity"].copy(),
                        "last_time": value["last_time"],
                        "target_index": value["target_index"],
                    }
                    for key, value in self.states.items()
                }

            if not self._states_are_fresh(states):
                self._send_zero_currents()
                time.sleep(period)
                continue

            force_targets = []
            target_points = {}
            all_done = elapsed >= self.reference_duration()

            for key in ("m1", "m2"):
                path = self.trajectory_data[key]
                position = states[key]["position"]
                if self.settings["time_based_reference"]:
                    target_index = self._time_based_target_index(elapsed, path)
                else:
                    target_index = self._advance_target(path, position, states[key]["target_index"])
                target = path[min(target_index, len(path) - 1)]
                desired_force = self._desired_force(position, states[key]["velocity"], target)
                force_targets.append({"position": position, "desired_force": desired_force})
                target_points[key] = target

                with self.lock:
                    self.states[key]["target_index"] = target_index
                    self.states[key]["desired_force"] = desired_force

                if not self.settings["time_based_reference"] and target_index < len(path) - 1:
                    all_done = False

            solution = self._solve_currents_for_targets(force_targets, self.currents)
            self.currents = solution["currents"]

            for index, key in enumerate(("m1", "m2")):
                actual_force = np.array(solution["forces"][index][:2])
                with self.lock:
                    self.states[key]["actual_force"] = actual_force

            self._apply_currents(self.currents)
            self._record_step(elapsed, states, target_points, force_targets, solution)

            rospy.loginfo_throttle(
                1.0,
                "Case 7 err=%.3e N, |I|=%.3f A, M1=(%.4f, %.4f), M2=(%.4f, %.4f)",
                solution["error"],
                np.linalg.norm(self.currents),
                states["m1"]["position"][0],
                states["m1"]["position"][1],
                states["m2"]["position"][0],
                states["m2"]["position"][1],
            )

            if all_done:
                rospy.loginfo("Case 7 trajectories complete")
                break

            sleep_time = period - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                rospy.logwarn_throttle(1.0, "Case 7 control loop overrun: %.4fs", -sleep_time)

        self.stop_control()

    def _record_position_snapshot_locked(self, stamp, updated_key):
        elapsed = stamp - self.control_start_time
        m1 = self.states["m1"]
        m2 = self.states["m2"]
        m1_pos = m1["position"]
        m2_pos = m2["position"]
        m1_vel = m1["velocity"]
        m2_vel = m2["velocity"]
        self.position_trace_data.append([
            elapsed,
            stamp,
            updated_key,
            m1_pos[0] if m1_pos is not None else "",
            m1_pos[1] if m1_pos is not None else "",
            m2_pos[0] if m2_pos is not None else "",
            m2_pos[1] if m2_pos is not None else "",
            m1_vel[0] if m1_pos is not None else "",
            m1_vel[1] if m1_pos is not None else "",
            m2_vel[0] if m2_pos is not None else "",
            m2_vel[1] if m2_pos is not None else "",
            m1["target_index"],
            m2["target_index"],
        ])

    def _wait_for_initial_pose(self, timeout):
        rospy.loginfo("Waiting for both microsphere positions...")
        start = time.time()
        while time.time() - start < timeout and not rospy.is_shutdown():
            with self.lock:
                ready = self.states["m1"]["position"] is not None and self.states["m2"]["position"] is not None
            if ready:
                return True
            time.sleep(0.05)
        rospy.logerr("Timed out waiting for /sphere/red and /sphere/green positions")
        return False

    def _states_are_fresh(self, states):
        now = time.time()
        for key, state in states.items():
            if state["position"] is None:
                rospy.logwarn_throttle(1.0, "Missing %s position", key)
                return False
            if state["last_time"] is None or now - state["last_time"] > STALE_POSE_TIMEOUT:
                rospy.logwarn_throttle(1.0, "Stale %s position", key)
                return False
        return True

    def _advance_target(self, path, position, target_index):
        target_index = min(max(int(target_index), 0), len(path) - 1)
        while target_index < len(path) - 1:
            target = path[target_index]
            if np.linalg.norm(target - position) >= self.settings["target_radius_m"]:
                break
            target_index += 1
        return target_index

    def _time_based_target_index(self, elapsed, path):
        index = int(round(elapsed / self.settings["dt_s"]))
        return min(max(index, 0), len(path) - 1)

    def _desired_force(self, position, velocity, target):
        error = target - position
        return self.settings["kp_N_per_m"] * error - self.settings["kd_N_s_per_m"] * velocity

    def _solve_currents_for_targets(self, targets, previous_currents):
        limit = self.settings["current_limit_A"]
        currents = np.clip(previous_currents.copy(), -limit, limit)
        best = self._score_currents(currents, targets, previous_currents)
        steps = [0.45 * limit, 0.22 * limit, 0.10 * limit, 0.045 * limit, 0.02 * limit]

        for step in steps:
            for _ in range(2):
                for coil_idx in range(10):
                    local_best = best
                    local_currents = currents
                    for direction in (-1.0, 1.0):
                        candidate = currents.copy()
                        candidate[coil_idx] = np.clip(candidate[coil_idx] + direction * step, -limit, limit)
                        candidate_score = self._score_currents(candidate, targets, previous_currents)
                        if candidate_score["score"] < local_best["score"]:
                            local_best = candidate_score
                            local_currents = candidate
                    currents = local_currents
                    best = local_best

        return {"currents": currents, "forces": best["forces"], "error": best["error"]}

    def _score_currents(self, currents, targets, previous_currents):
        force_error = 0.0
        combined_error = 0.0
        forces = []

        for target in targets:
            force = self._compute_soft_force(currents, target["position"])
            forces.append(force)
            diff = force[:2] - target["desired_force"]
            force_error += float(np.dot(diff, diff))
            combined_error += float(np.linalg.norm(diff))

        norm_penalty = float(np.dot(currents, currents))
        jump = currents - previous_currents
        smooth_penalty = float(np.dot(jump, jump))
        score = (
            force_error
            + self.settings["force_regularization"] * norm_penalty
            + self.settings["smoothness"] * smooth_penalty
        )

        return {"score": score, "forces": forces, "error": combined_error}

    def _compute_soft_force(self, currents, position):
        field = self._compute_field(currents, position)
        bx, by, bz = field["b"]
        bz_dz = -(field["bx_dx"] + field["by_dy"])

        return 2.0 * self.settings["k_mag"] * np.array([
            bx * field["bx_dx"] + by * field["bx_dy"] + bz * field["bx_dz"],
            bx * field["bx_dy"] + by * field["by_dy"] + bz * field["by_dz"],
            bx * field["bx_dz"] + by * field["by_dz"] + bz * bz_dz,
        ])

    def _compute_field(self, currents, position):
        total = {
            "b": np.zeros(3),
            "bx_dx": 0.0,
            "bx_dy": 0.0,
            "bx_dz": 0.0,
            "by_dy": 0.0,
            "by_dz": 0.0,
        }
        point = np.array([position[0], position[1], self.settings["z_plane_m"]], dtype=float)

        for params, current in zip(COIL_PARAMS, currents):
            contribution = self._coil_field_and_derivatives(params, current, point)
            total["b"] += contribution["b"]
            total["bx_dx"] += contribution["bx_dx"]
            total["bx_dy"] += contribution["bx_dy"]
            total["bx_dz"] += contribution["bx_dz"]
            total["by_dy"] += contribution["by_dy"]
            total["by_dz"] += contribution["by_dz"]

        return total

    def _coil_field_and_derivatives(self, params, current, point):
        m0, m1, m2, r0x, r0y, r0z = params
        dx = point[0] - r0x
        dy = point[1] - r0y
        dz = point[2] - r0z
        dot = m0 * dx + m1 * dy + m2 * dz
        r = math.sqrt(dx * dx + dy * dy + dz * dz) + EPS
        r3 = r ** 3
        r5 = r ** 5
        r7 = r ** 7
        c = MU_OVER_4PI * current

        bx = c * (3.0 * dx * dot / r5 - m0 / r3)
        by = c * (3.0 * dy * dot / r5 - m1 / r3)
        bz = c * (3.0 * dz * dot / r5 - m2 / r3)

        return {
            "b": np.array([bx, by, bz]),
            "bx_dx": c * (3.0 * dot / r5 + 6.0 * dx * m0 / r5 - 15.0 * dx * dx * dot / r7),
            "bx_dy": c * (3.0 * dx * m1 / r5 + 3.0 * m0 * dy / r5 - 15.0 * dx * dy * dot / r7),
            "bx_dz": c * (3.0 * dx * m2 / r5 + 3.0 * m0 * dz / r5 - 15.0 * dx * dz * dot / r7),
            "by_dy": c * (3.0 * dot / r5 + 3.0 * dy * m1 / r5 - 15.0 * dy * dy * dot / r7),
            "by_dz": c * (3.0 * dy * m2 / r5 + 3.0 * m1 * dz / r5 - 15.0 * dy * dz * dot / r7),
        }

    def _apply_currents(self, currents):
        current_limit = self.settings["current_limit_A"]
        currents = np.clip(currents, -current_limit, current_limit)
        for i, current in enumerate(currents):
            self.send_current(i, int(round(current * 1000.0)))

    def _send_zero_currents(self):
        for i in range(10):
            self.send_current(i, 0)

    def _record_step(self, elapsed, states, target_points, force_targets, solution):
        row = [
            elapsed,
            target_points["m1"][0],
            target_points["m1"][1],
            states["m1"]["position"][0],
            states["m1"]["position"][1],
            target_points["m2"][0],
            target_points["m2"][1],
            states["m2"]["position"][0],
            states["m2"]["position"][1],
            force_targets[0]["desired_force"][0],
            force_targets[0]["desired_force"][1],
            solution["forces"][0][0],
            solution["forces"][0][1],
            force_targets[1]["desired_force"][0],
            force_targets[1]["desired_force"][1],
            solution["forces"][1][0],
            solution["forces"][1][1],
            solution["error"],
        ]
        self.results_data.append(row)
        self.coil_currents_data.append([elapsed] + list(self.currents))

    def _results_dir(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    def _clear_previous_results(self):
        output_dir = self._results_dir()
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as exc:
            rospy.logerr("Failed to create Case 7 results directory: %s", exc)
            return

        prefixes = (
            "closed_loop_results_case7",
            "coil_currents_case7",
            "sphere_positions_case7",
            "trajectory_comparison_case7",
        )
        deleted = 0
        for filename in os.listdir(output_dir):
            if not filename.endswith((".csv", ".png")):
                continue
            if not filename.startswith(prefixes):
                continue
            path = os.path.join(output_dir, filename)
            try:
                os.remove(path)
                deleted += 1
            except Exception as exc:
                rospy.logwarn("Failed to delete old Case 7 result %s: %s", path, exc)

        if deleted:
            rospy.loginfo("Deleted %d old Case 7 result file(s)", deleted)

    def _save_results(self):
        if self.results_saved:
            return
        self.results_saved = True

        output_dir = self._results_dir()
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as exc:
            rospy.logerr("Failed to create Case 7 results directory: %s", exc)
            output_dir = os.path.dirname(os.path.abspath(__file__))

        run_id = self.run_id or time.strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, "closed_loop_results_case7_%s.csv" % run_id)
        currents_path = os.path.join(output_dir, "coil_currents_case7_%s.csv" % run_id)
        positions_path = os.path.join(output_dir, "sphere_positions_case7_%s.csv" % run_id)

        try:
            with open(results_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Time(s)",
                    "M1_Desired_X(m)", "M1_Desired_Y(m)", "M1_Actual_X(m)", "M1_Actual_Y(m)",
                    "M2_Desired_X(m)", "M2_Desired_Y(m)", "M2_Actual_X(m)", "M2_Actual_Y(m)",
                    "M1_Desired_Fx(N)", "M1_Desired_Fy(N)", "M1_Actual_Fx(N)", "M1_Actual_Fy(N)",
                    "M2_Desired_Fx(N)", "M2_Desired_Fy(N)", "M2_Actual_Fx(N)", "M2_Actual_Fy(N)",
                    "Solver_Error(N)",
                ])
                writer.writerows(self.results_data)
            rospy.loginfo("Saved Case 7 results to %s", results_path)
        except Exception as exc:
            rospy.logerr("Failed to save Case 7 results: %s", exc)

        try:
            with open(currents_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Time(s)"] + [f"Current_Coil_{i + 1}(A)" for i in range(10)])
                writer.writerows(self.coil_currents_data)
            rospy.loginfo("Saved Case 7 currents to %s", currents_path)
        except Exception as exc:
            rospy.logerr("Failed to save Case 7 currents: %s", exc)

        try:
            with open(positions_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Time(s)",
                    "Unix_Time(s)",
                    "Updated_Sphere",
                    "M1_Red_X(m)",
                    "M1_Red_Y(m)",
                    "M2_Green_X(m)",
                    "M2_Green_Y(m)",
                    "M1_Red_Vx(m/s)",
                    "M1_Red_Vy(m/s)",
                    "M2_Green_Vx(m/s)",
                    "M2_Green_Vy(m/s)",
                    "M1_Target_Index",
                    "M2_Target_Index",
                ])
                writer.writerows(self.position_trace_data)
            rospy.loginfo("Saved Case 7 sphere positions to %s", positions_path)
        except Exception as exc:
            rospy.logerr("Failed to save Case 7 sphere positions: %s", exc)
