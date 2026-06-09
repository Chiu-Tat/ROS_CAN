#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Closed-loop controller for Case 8: two-DOF laser manipulator spot tracking.

The controller mirrors the field-fit method in
/home/dz/Documents/DecaPulse_py/two_dof_manipulator_spot_feedback_control.html:
  - the laser spot is measured on the observation plane
  - the measured spot corrects an internal yaw/pitch estimate
  - desired fields are aligned with the two arm magnet moment directions
  - one ten-coil current vector is solved by damped least squares
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
    "workspace_radius_m": 0.08,
    "dt_s": 0.025,
    "current_limit_A": 15.0,
    "field_strength_T": 0.004,
    "use_joint_torque_field": True,
    "joint_field_sign": 1.0,
    "joint_field_gain_T_per_rad": 0.012,
    "joint_field_damping_T_s_per_rad": 0.012,
    "yaw_field_sign": -1.0,
    "pitch_field_sign": 1.0,
    "yaw_field_gain_T_per_rad": 0.025,
    "pitch_field_gain_T_per_rad": 0.10,
    "yaw_field_damping_T_s_per_rad": 0.018,
    "pitch_field_damping_T_s_per_rad": 0.006,
    "field_filter_alpha": 0.15,
    "current_smoothing": 0.35,
    "current_regularization": 1e-9,
    "spot_kp_x": 0.03,
    "spot_kp_y": 0.12,
    "spot_kd": 0.06,
    "pose_correction_gain": 0.65,
    "pose_command_gain": 1.0,
    "max_pose_command_step_deg": 0.35,
    "max_pose_command_step_yaw_deg": 0.25,
    "max_pose_command_step_pitch_deg": 1.2,
    "max_current_step_A": 0.08,
    "position_filter_alpha": 0.12,
    "velocity_filter_alpha": 0.08,
    "max_measured_speed_m_per_s": 0.025,
    "min_velocity_dt_s": 0.01,
    "target_radius_m": 0.0025,
    "hold_start_until_close": True,
    "start_tolerance_m": 0.004,
    "start_yz_plane_field": True,
    "path_lookahead_index": 8,
    "max_target_advance_per_step": 2,
    "progress_search_window_index": 45,
    "progress_backtrack_index": 12,
    "max_tracking_error_m": 0.012,
    "samples_per_segment": 36,
    "target_speed_m_per_s": 0.002,
    "time_based_reference": True,
}

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

MU_OVER_4PI = 1e-7
EPS = 1e-12
STALE_SPOT_TIMEOUT = 0.5


def deg_to_rad(deg):
    return deg * math.pi / 180.0


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def unit(vector, fallback):
    norm = float(np.linalg.norm(vector))
    if norm > EPS:
        return vector / norm
    return np.array(fallback, dtype=float)


class TwoDOFLaserModel(object):
    def __init__(self):
        self.base_x = -0.01571
        self.base_y = 0.0
        self.base_z = 0.06064
        self.observation_plane_z = -0.026
        self.link1 = 0.02364
        self.link2 = 0.016
        self.laser_extension = 0.012
        self.min_yaw = deg_to_rad(-150.0)
        self.max_yaw = deg_to_rad(150.0)
        self.min_pitch = deg_to_rad(-72.0)
        self.max_pitch = deg_to_rad(72.0)
        self.base_tilt_y = deg_to_rad(50.0)
        self.base_rotation_z = deg_to_rad(90.0)
        self.magnet_moment = 0.095

        base_frame = {
            "x": np.array([math.cos(self.base_tilt_y), 0.0, -math.sin(self.base_tilt_y)]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([math.sin(self.base_tilt_y), 0.0, math.cos(self.base_tilt_y)]),
        }
        self.base_frame = {
            key: self.rotate_global_z(value, self.base_rotation_z)
            for key, value in base_frame.items()
        }
        self.base_position = self.rotate_global_z(
            np.array([self.base_x, self.base_y, self.base_z], dtype=float),
            self.base_rotation_z,
        )

    def rotate_global_z(self, vector, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([
            c * vector[0] - s * vector[1],
            s * vector[0] + c * vector[1],
            vector[2],
        ], dtype=float)

    def clamp_yaw(self, yaw):
        return clamp(normalize_angle(yaw), self.min_yaw, self.max_yaw)

    def clamp_pitch(self, pitch):
        return clamp(pitch, self.min_pitch, self.max_pitch)

    def clamp_pose(self, yaw, pitch):
        return self.clamp_yaw(yaw), self.clamp_pitch(pitch)

    def forward_kinematics(self, yaw, pitch):
        yaw, pitch = self.clamp_pose(yaw, pitch)
        yaw_dir = unit(
            self.base_frame["x"] * math.cos(yaw) + self.base_frame["y"] * math.sin(yaw),
            [1.0, 0.0, 0.0],
        )
        pitch_axis = unit(np.cross(self.base_frame["z"], yaw_dir), [1.0, 0.0, 0.0])
        base = self.base_position
        joint2 = base + yaw_dir * self.link1
        end_dir = unit(
            yaw_dir * math.cos(pitch) + self.base_frame["z"] * -math.sin(pitch),
            [1.0, 0.0, 0.0],
        )
        magnet1 = base + yaw_dir * (self.link1 * 0.48)
        magnet2 = joint2 + end_dir * (self.link2 * 0.52)
        tip = joint2 + end_dir * self.link2
        laser_end = tip + end_dir * self.laser_extension
        if end_dir[2] < -1e-6:
            t = (self.observation_plane_z - tip[2]) / end_dir[2]
        else:
            t = 0.0
        spot = tip + end_dir * t if t > 0.0 else laser_end
        return {
            "base": base,
            "joint2": joint2,
            "tip": tip,
            "laser_end": laser_end,
            "spot": spot,
            "magnet1": magnet1,
            "magnet2": magnet2,
            "magnet1_moment_dir": -yaw_dir,
            "magnet2_moment_dir": -end_dir,
            "yaw_dir": yaw_dir,
            "pitch_axis": pitch_axis,
            "end_dir": end_dir,
        }

    def inverse_laser_target(self, x, y):
        from_base = np.array([x, y, self.observation_plane_z], dtype=float) - self.base_position
        local_x = float(np.dot(from_base, self.base_frame["x"]))
        local_y = float(np.dot(from_base, self.base_frame["y"]))
        local_z = float(np.dot(from_base, self.base_frame["z"]))
        yaw = self.clamp_yaw(math.atan2(local_y, local_x))
        run = max(math.hypot(local_x, local_y) - self.link1, 0.012)
        pitch = self.clamp_pitch(math.atan2(-local_z, run))
        return yaw, pitch

    def spot_jacobian(self, yaw, pitch, delta=deg_to_rad(0.6)):
        center = self.forward_kinematics(yaw, pitch)["spot"]
        yaw_plus = self.forward_kinematics(self.clamp_yaw(yaw + delta), pitch)["spot"]
        pitch_plus = self.forward_kinematics(yaw, self.clamp_pitch(pitch + delta))["spot"]
        return np.array([
            [(yaw_plus[0] - center[0]) / delta, (pitch_plus[0] - center[0]) / delta],
            [(yaw_plus[1] - center[1]) / delta, (pitch_plus[1] - center[1]) / delta],
        ], dtype=float)


class ClosedLoopLaserSpotController(object):
    def __init__(
        self,
        can_handler,
        send_device_current_callback,
        spot_topic="/laser/spot",
        position_scale=1.0,
        control_dt_s=None,
    ):
        self.can_handler = can_handler
        self.send_current = send_device_current_callback
        self.position_scale = position_scale
        self.control_dt_s = control_dt_s
        self.model = TwoDOFLaserModel()

        self.control_running = False
        self.control_thread = None
        self.lock = threading.Lock()
        self.settings = DEFAULT_SETTINGS.copy()
        self.trajectory = None
        self.currents = np.zeros(10)
        self.measured_spot = None
        self.filtered_velocity = np.zeros(2)
        self.last_spot = None
        self.last_time = None
        self.model_yaw = 0.0
        self.model_pitch = 0.0
        self.model_yaw_rate = 0.0
        self.model_pitch_rate = 0.0
        self.last_pose_update_time = None
        self.command_yaw = 0.0
        self.command_pitch = 0.0
        self.target_index = 0
        self.path_start_time = None
        self.last_command_fields = None
        self.results_data = []
        self.coil_currents_data = []
        self.spot_trace_data = []
        self.recording_positions = False
        self.control_start_time = None
        self.run_id = None
        self.results_saved = False

        self.subscriber = rospy.Subscriber(spot_topic, PointStamped, self._spot_callback)
        rospy.loginfo("Case 8 subscribed to %s", spot_topic)

    def _spot_callback(self, msg):
        now = time.time()
        raw_pos = np.array([msg.point.x, msg.point.y], dtype=float) * self.position_scale
        with self.lock:
            previous = self.measured_spot
            previous_time = self.last_time
            pos = raw_pos
            dt = None
            if previous is not None and previous_time is not None:
                dt = now - previous_time
                pos = self._filtered_position(previous, raw_pos, dt)
                if dt >= self.settings["min_velocity_dt_s"]:
                    measured_velocity = (pos - previous) / dt
                    self.filtered_velocity = self._filtered_velocity(self.filtered_velocity, measured_velocity)

            self.last_spot = previous
            self.measured_spot = pos
            self.last_time = now
            if self.recording_positions and self.control_start_time is not None:
                self.spot_trace_data.append([
                    now - self.control_start_time,
                    now,
                    pos[0],
                    pos[1],
                    self.filtered_velocity[0],
                    self.filtered_velocity[1],
                    self.target_index,
                ])

    def _filtered_position(self, previous, raw_position, dt):
        alpha = self.settings["position_filter_alpha"]
        filtered = previous + alpha * (raw_position - previous)
        max_step = self.settings["max_measured_speed_m_per_s"] * max(dt, self.settings["min_velocity_dt_s"])
        delta = filtered - previous
        norm = float(np.linalg.norm(delta))
        if max_step > 0.0 and norm > max_step:
            filtered = previous + delta * (max_step / norm)
        return filtered

    def _filtered_velocity(self, previous_velocity, measured_velocity):
        max_speed = self.settings["max_measured_speed_m_per_s"]
        speed = float(np.linalg.norm(measured_velocity))
        if max_speed > 0.0 and speed > max_speed:
            measured_velocity = measured_velocity * (max_speed / speed)
        alpha = self.settings["velocity_filter_alpha"]
        return previous_velocity + alpha * (measured_velocity - previous_velocity)

    def prepare_trajectory(self, filepath):
        rospy.loginfo("Loading Case 8 laser trajectory from %s", filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        self.settings = self._read_controller_settings(data.get("settings", {}))
        if self.control_dt_s is not None:
            self.settings["dt_s"] = min(0.2, max(0.001, float(self.control_dt_s)))
        raw_path = np.array(data["laser_spot"]["raw_path"], dtype=float)
        dense_path = self._catmull_rom_path(raw_path, self.settings["samples_per_segment"])
        self.trajectory = self._resample_path_by_target_speed(dense_path)
        path_length = self._path_length(self.trajectory)
        duration_s = max(len(self.trajectory) - 1, 0) * self.settings["dt_s"]
        nominal_speed = path_length / duration_s if duration_s > 0.0 else 0.0
        rospy.loginfo(
            "Loaded Case 8 laser path: %d points, dt=%.4f s, duration=%.2f s, target speed=%.4f m/s, nominal speed=%.4f m/s, limit=%.2f A, field=%.4g T",
            len(self.trajectory),
            self.settings["dt_s"],
            duration_s,
            self.settings["target_speed_m_per_s"],
            nominal_speed,
            self.settings["current_limit_A"],
            self.settings["field_strength_T"],
        )

    def _read_controller_settings(self, raw_settings):
        settings = DEFAULT_SETTINGS.copy()
        for key in settings:
            if key in raw_settings:
                settings[key] = raw_settings[key]
        settings["workspace_radius_m"] = max(0.001, float(settings["workspace_radius_m"]))
        settings["dt_s"] = min(0.2, max(0.001, float(settings["dt_s"])))
        settings["current_limit_A"] = max(0.05, float(settings["current_limit_A"]))
        settings["field_strength_T"] = max(0.0, float(settings["field_strength_T"]))
        settings["use_joint_torque_field"] = self._as_bool(settings["use_joint_torque_field"])
        settings["joint_field_sign"] = -1.0 if float(settings["joint_field_sign"]) < 0.0 else 1.0
        settings["joint_field_gain_T_per_rad"] = max(0.0, float(settings["joint_field_gain_T_per_rad"]))
        settings["joint_field_damping_T_s_per_rad"] = max(0.0, float(settings["joint_field_damping_T_s_per_rad"]))
        settings["yaw_field_sign"] = -1.0 if float(settings["yaw_field_sign"]) < 0.0 else 1.0
        settings["pitch_field_sign"] = -1.0 if float(settings["pitch_field_sign"]) < 0.0 else 1.0
        settings["yaw_field_gain_T_per_rad"] = max(0.0, float(settings["yaw_field_gain_T_per_rad"]))
        settings["pitch_field_gain_T_per_rad"] = max(0.0, float(settings["pitch_field_gain_T_per_rad"]))
        settings["yaw_field_damping_T_s_per_rad"] = max(0.0, float(settings["yaw_field_damping_T_s_per_rad"]))
        settings["pitch_field_damping_T_s_per_rad"] = max(0.0, float(settings["pitch_field_damping_T_s_per_rad"]))
        settings["field_filter_alpha"] = min(1.0, max(0.01, float(settings["field_filter_alpha"])))
        settings["current_smoothing"] = min(1.0, max(0.0, float(settings["current_smoothing"])))
        settings["current_regularization"] = max(0.0, float(settings["current_regularization"]))
        settings["spot_kp_x"] = max(0.0, float(settings["spot_kp_x"]))
        settings["spot_kp_y"] = max(0.0, float(settings["spot_kp_y"]))
        settings["spot_kd"] = max(0.0, float(settings["spot_kd"]))
        settings["pose_correction_gain"] = min(1.0, max(0.0, float(settings["pose_correction_gain"])))
        settings["pose_command_gain"] = max(0.0, float(settings["pose_command_gain"]))
        settings["max_pose_command_step_deg"] = max(0.0, float(settings["max_pose_command_step_deg"]))
        settings["max_pose_command_step_yaw_deg"] = max(0.0, float(settings["max_pose_command_step_yaw_deg"]))
        settings["max_pose_command_step_pitch_deg"] = max(0.0, float(settings["max_pose_command_step_pitch_deg"]))
        settings["max_current_step_A"] = max(0.0, float(settings["max_current_step_A"]))
        settings["position_filter_alpha"] = min(1.0, max(0.01, float(settings["position_filter_alpha"])))
        settings["velocity_filter_alpha"] = min(1.0, max(0.01, float(settings["velocity_filter_alpha"])))
        settings["max_measured_speed_m_per_s"] = max(0.0, float(settings["max_measured_speed_m_per_s"]))
        settings["min_velocity_dt_s"] = max(0.0, float(settings["min_velocity_dt_s"]))
        settings["target_radius_m"] = max(0.0002, float(settings["target_radius_m"]))
        settings["hold_start_until_close"] = self._as_bool(settings["hold_start_until_close"])
        settings["start_tolerance_m"] = max(settings["target_radius_m"], float(settings["start_tolerance_m"]))
        settings["start_yz_plane_field"] = self._as_bool(settings["start_yz_plane_field"])
        settings["path_lookahead_index"] = max(0, int(settings["path_lookahead_index"]))
        settings["max_target_advance_per_step"] = max(1, int(settings["max_target_advance_per_step"]))
        settings["progress_search_window_index"] = max(1, int(settings["progress_search_window_index"]))
        settings["progress_backtrack_index"] = max(0, int(settings["progress_backtrack_index"]))
        settings["max_tracking_error_m"] = max(settings["target_radius_m"], float(settings["max_tracking_error_m"]))
        settings["samples_per_segment"] = max(2, int(settings["samples_per_segment"]))
        settings["target_speed_m_per_s"] = max(0.0, float(settings["target_speed_m_per_s"]))
        settings["time_based_reference"] = self._as_bool(settings["time_based_reference"])
        return settings

    def _as_bool(self, value):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _catmull_rom_path(self, raw_path, samples_per_segment):
        if len(raw_path) < 2:
            return raw_path.copy()
        path = []
        for segment in range(len(raw_path) - 1):
            p0 = raw_path[max(segment - 1, 0)]
            p1 = raw_path[segment]
            p2 = raw_path[segment + 1]
            p3 = raw_path[min(segment + 2, len(raw_path) - 1)]
            for sample in range(samples_per_segment):
                t = sample / float(samples_per_segment)
                path.append(self._catmull_rom_point(p0, p1, p2, p3, t))
        path.append(raw_path[-1])
        return np.array(path, dtype=float)

    def _catmull_rom_point(self, p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    def _resample_path_by_target_speed(self, path):
        speed = self.settings["target_speed_m_per_s"]
        if speed <= 0.0 or len(path) < 2:
            return path

        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative[-1])
        spacing = speed * self.settings["dt_s"]
        if total_length <= 0.0 or spacing <= 0.0:
            return path

        point_count = max(2, int(round(total_length / spacing)) + 1)
        sample_distances = np.linspace(0.0, total_length, point_count)
        x = np.interp(sample_distances, cumulative, path[:, 0])
        y = np.interp(sample_distances, cumulative, path[:, 1])
        return np.column_stack((x, y))

    def _path_length(self, path):
        if len(path) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    def start_control(self):
        if self.control_running:
            return
        if self.trajectory is None:
            rospy.logerr("No Case 8 laser trajectory loaded")
            return
        if not self._wait_for_initial_spot(timeout=5.0):
            return

        with self.lock:
            spot = self.measured_spot.copy()
        self.model_yaw, self.model_pitch = self.model.inverse_laser_target(spot[0], spot[1])
        self.model_yaw_rate = 0.0
        self.model_pitch_rate = 0.0
        self.last_pose_update_time = time.time()
        self.command_yaw = self.model_yaw
        self.command_pitch = self.model_pitch
        self.target_index = 0 if self.settings["time_based_reference"] else int(self.settings["path_lookahead_index"])
        self.path_start_time = None
        self.last_command_fields = None
        self.currents = np.zeros(10)
        self.results_data = []
        self.coil_currents_data = []
        self.spot_trace_data = []
        self.control_start_time = time.time()
        self.run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.control_start_time))
        self._clear_previous_results()
        self.recording_positions = True
        self.results_saved = False
        self.control_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        rospy.loginfo("Case 8 laser spot control started, run_id=%s", self.run_id)

    def stop_control(self):
        if not self.control_running:
            return
        self.control_running = False
        if self.control_thread and threading.current_thread() != self.control_thread:
            self.control_thread.join(timeout=2.0)
        self._send_zero_currents()
        self.recording_positions = False
        self._save_results()
        rospy.loginfo("Case 8 laser control stopped")

    def _control_loop(self):
        start_time = self.control_start_time or time.time()
        if not self.settings["hold_start_until_close"]:
            self.path_start_time = start_time
        period = self.settings["dt_s"]
        while self.control_running and not rospy.is_shutdown():
            loop_start = time.time()
            elapsed = loop_start - start_time
            spot_state = self._latest_spot_state()
            if spot_state is None:
                self._send_zero_currents()
                time.sleep(period)
                continue

            spot = spot_state["position"]
            velocity = spot_state["velocity"]
            path_elapsed = 0.0 if self.path_start_time is None else loop_start - self.path_start_time
            target_index = self._target_index_for_spot(spot, path_elapsed)
            target = self.trajectory[min(target_index, len(self.trajectory) - 1)]
            if (
                self.settings["time_based_reference"]
                and self.settings["hold_start_until_close"]
                and self.path_start_time is None
            ):
                target_index = 0
                target = self.trajectory[0]
                start_error = float(np.linalg.norm(target - spot))
                if start_error <= self.settings["start_tolerance_m"]:
                    self.path_start_time = loop_start
                    rospy.loginfo("Case 8 start point reached; starting timed trajectory")
            self.target_index = target_index

            self._update_spot_pose_estimate(spot)
            actual_kin = self.model.forward_kinematics(self.model_yaw, self.model_pitch)
            self._advance_pose_command(spot, velocity, target, period)
            command_kin = self.model.forward_kinematics(self.command_yaw, self.command_pitch)
            if self.settings["use_joint_torque_field"]:
                command_fields = self._joint_torque_fields(actual_kin, self.command_yaw, self.command_pitch)
            else:
                command_fields = self._magnet_aligned_fields(command_kin)
            if self._should_use_start_yz_plane_field(target_index):
                command_fields = self._project_fields_to_yz_plane(command_fields)
            command_fields = self._filtered_command_fields(command_fields)
            solution = self._solve_currents_for_fields(actual_kin, command_fields, self.currents)
            currents = self._limit_current_step(solution["currents"], self.currents)
            if not np.allclose(currents, solution["currents"]):
                solution = self._solve_currents_for_fields(actual_kin, command_fields, currents)
                currents = self._limit_current_step(solution["currents"], self.currents)
            self.currents = currents
            self._apply_currents(self.currents)
            self._record_step(elapsed, spot, velocity, target, actual_kin, command_fields, solution)

            error = float(np.linalg.norm(target - spot))
            rospy.loginfo_throttle(
                1.0,
                "Case 8 laser error=%.4f m, yaw=%.1f deg, pitch=%.1f deg, |I|=%.2f A",
                error,
                self.model_yaw * 180.0 / math.pi,
                self.model_pitch * 180.0 / math.pi,
                np.linalg.norm(self.currents),
            )
            if target_index >= len(self.trajectory) - 1 and error <= self.settings["target_radius_m"]:
                rospy.loginfo("Case 8 laser trajectory complete")
                break

            sleep_time = period - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                rospy.logwarn_throttle(1.0, "Case 8 control loop overrun: %.4fs", -sleep_time)
        self.stop_control()

    def _wait_for_initial_spot(self, timeout):
        rospy.loginfo("Waiting for laser spot on /laser/spot...")
        start = time.time()
        while time.time() - start < timeout and not rospy.is_shutdown():
            with self.lock:
                ready = self.measured_spot is not None
            if ready:
                return True
            time.sleep(0.05)
        rospy.logerr("Timed out waiting for /laser/spot")
        return False

    def _latest_spot_state(self):
        with self.lock:
            if self.measured_spot is None or self.last_time is None:
                return None
            if time.time() - self.last_time > STALE_SPOT_TIMEOUT:
                rospy.logwarn_throttle(1.0, "Stale /laser/spot measurement")
                return None
            return {
                "position": self.measured_spot.copy(),
                "velocity": self.filtered_velocity.copy(),
                "last_time": self.last_time,
            }

    def _target_index_for_spot(self, spot, elapsed):
        if self.settings["time_based_reference"]:
            index = int(round(elapsed / self.settings["dt_s"]))
            return min(max(index, 0), len(self.trajectory) - 1)

        current = min(max(int(self.target_index), 0), len(self.trajectory) - 1)
        current_error = float(np.linalg.norm(self.trajectory[current] - spot))
        if current_error > self.settings["max_tracking_error_m"]:
            return current

        search_start = max(0, current - self.settings["progress_backtrack_index"])
        search_stop = min(len(self.trajectory) - 1, current + self.settings["progress_search_window_index"])
        window = self.trajectory[search_start:search_stop + 1]
        nearest_offset = int(np.argmin(np.linalg.norm(window - spot, axis=1)))
        nearest_index = search_start + nearest_offset
        proposed = nearest_index + self.settings["path_lookahead_index"]
        max_step = current + self.settings["max_target_advance_per_step"]
        return min(max(current, proposed), max_step, len(self.trajectory) - 1)

    def _update_spot_pose_estimate(self, measured_spot):
        previous_yaw = self.model_yaw
        previous_pitch = self.model_pitch
        previous_time = self.last_pose_update_time
        estimated_spot = self.model.forward_kinematics(self.model_yaw, self.model_pitch)["spot"]
        residual = np.array([measured_spot[0] - estimated_spot[0], measured_spot[1] - estimated_spot[1]])
        correction = self._solve_spot_dls(self.model.spot_jacobian(self.model_yaw, self.model_pitch), residual)
        self.model_yaw += correction[0] * self.settings["pose_correction_gain"]
        self.model_pitch += correction[1] * self.settings["pose_correction_gain"]
        self.model_yaw, self.model_pitch = self.model.clamp_pose(self.model_yaw, self.model_pitch)
        now = time.time()
        if previous_time is not None:
            dt = max(now - previous_time, 1e-3)
            yaw_rate = normalize_angle(self.model_yaw - previous_yaw) / dt
            pitch_rate = (self.model_pitch - previous_pitch) / dt
            self.model_yaw_rate = 0.7 * self.model_yaw_rate + 0.3 * yaw_rate
            self.model_pitch_rate = 0.7 * self.model_pitch_rate + 0.3 * pitch_rate
        self.last_pose_update_time = now

    def _advance_pose_command(self, measured_spot, velocity, target, dt):
        desired_spot_motion = np.array([
            self.settings["spot_kp_x"] * (target[0] - measured_spot[0]) - self.settings["spot_kd"] * velocity[0],
            self.settings["spot_kp_y"] * (target[1] - measured_spot[1]) - self.settings["spot_kd"] * velocity[1],
        ])
        dq = self._solve_spot_dls(self.model.spot_jacobian(self.model_yaw, self.model_pitch), desired_spot_motion)
        step_dt = max(dt, 0.012)
        command_step = dq * step_dt * self.settings["pose_command_gain"]
        fallback_max_step = self.settings["max_pose_command_step_deg"]
        yaw_max_step = deg_to_rad(self.settings["max_pose_command_step_yaw_deg"] or fallback_max_step)
        pitch_max_step = deg_to_rad(self.settings["max_pose_command_step_pitch_deg"] or fallback_max_step)
        if yaw_max_step > 0.0:
            command_step[0] = clamp(command_step[0], -yaw_max_step, yaw_max_step)
        if pitch_max_step > 0.0:
            command_step[1] = clamp(command_step[1], -pitch_max_step, pitch_max_step)
        self.command_yaw = self.model_yaw + command_step[0]
        self.command_pitch = self.model_pitch + command_step[1]
        self.command_yaw, self.command_pitch = self.model.clamp_pose(self.command_yaw, self.command_pitch)

    def _solve_spot_dls(self, jacobian, vector, damping=1e-7):
        jjt = np.dot(jacobian, jacobian.T)
        jjt[0, 0] += damping
        jjt[1, 1] += damping
        try:
            tmp = np.linalg.solve(jjt, vector)
        except np.linalg.LinAlgError:
            tmp = np.linalg.lstsq(jjt, vector, rcond=None)[0]
        return np.dot(jacobian.T, tmp)

    def _magnet_aligned_fields(self, kin):
        strength = self.settings["field_strength_T"]
        return [
            kin["magnet1_moment_dir"] * strength,
            kin["magnet2_moment_dir"] * strength,
        ]

    def _joint_torque_fields(self, actual_kin, command_yaw, command_pitch):
        yaw_error = normalize_angle(command_yaw - self.model_yaw)
        pitch_error = command_pitch - self.model_pitch
        yaw_field = self._field_for_axis_torque(
            actual_kin["magnet1_moment_dir"],
            self.model.base_frame["z"],
            yaw_error,
            self.model_yaw_rate,
            self.settings["yaw_field_sign"],
            self.settings["yaw_field_gain_T_per_rad"],
            self.settings["yaw_field_damping_T_s_per_rad"],
        )
        pitch_field = self._field_for_axis_torque(
            actual_kin["magnet2_moment_dir"],
            actual_kin["pitch_axis"],
            pitch_error,
            self.model_pitch_rate,
            self.settings["pitch_field_sign"],
            self.settings["pitch_field_gain_T_per_rad"],
            self.settings["pitch_field_damping_T_s_per_rad"],
        )
        return [yaw_field, pitch_field]

    def _field_for_axis_torque(self, moment_dir, axis, angle_error, angle_rate, sign, gain, damping):
        torque_direction = np.cross(axis, moment_dir)
        norm = float(np.linalg.norm(torque_direction))
        if norm <= EPS:
            return np.zeros(3)
        unit_direction = torque_direction / norm
        magnitude = sign * gain * angle_error - damping * angle_rate
        limit = self.settings["field_strength_T"]
        magnitude = clamp(magnitude, -limit, limit)
        return unit_direction * magnitude

    def _filtered_command_fields(self, command_fields):
        if self.last_command_fields is None:
            self.last_command_fields = [field.copy() for field in command_fields]
            return command_fields
        alpha = self.settings["field_filter_alpha"]
        filtered = []
        for previous, current in zip(self.last_command_fields, command_fields):
            value = previous + alpha * (current - previous)
            filtered.append(value)
        self.last_command_fields = [field.copy() for field in filtered]
        return filtered

    def _should_use_start_yz_plane_field(self, target_index):
        return (
            self.settings["start_yz_plane_field"]
            and self.settings["time_based_reference"]
            and int(target_index) == 0
            and self.path_start_time is None
        )

    def _project_fields_to_yz_plane(self, fields):
        projected_fields = []
        for field in fields:
            projected = np.array([0.0, field[1], field[2]], dtype=float)
            norm = float(np.linalg.norm(projected))
            desired_norm = float(np.linalg.norm(field))
            if norm > EPS and desired_norm > 0.0:
                projected *= desired_norm / norm
            projected_fields.append(projected)
        return projected_fields

    def _solve_currents_for_fields(self, kin, command_fields, previous_currents):
        targets = [
            {"position": kin["magnet1"], "field": command_fields[0]},
            {"position": kin["magnet2"], "field": command_fields[1]},
        ]
        rows = []
        desired = []
        for target in targets:
            for axis in range(3):
                rows.append([self._dipole_field(params, target["position"])[axis] for params in COIL_PARAMS])
                desired.append(target["field"][axis])
        a = np.array(rows, dtype=float)
        b = np.array(desired, dtype=float)
        ata = np.dot(a.T, a)
        atb = np.dot(a.T, b)
        smooth_penalty = self.settings["current_smoothing"] * 2e-9
        regularization = self.settings["current_regularization"]
        ata += np.eye(10) * (regularization + smooth_penalty)
        atb += smooth_penalty * previous_currents
        try:
            raw_currents = np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            raw_currents = np.linalg.lstsq(ata, atb, rcond=None)[0]

        limit = self.settings["current_limit_A"]
        max_abs = max(1e-12, float(np.max(np.abs(raw_currents))))
        if max_abs > limit:
            raw_currents = raw_currents * (limit / max_abs)
        currents = np.clip(raw_currents, -limit, limit)
        actual_fields = self._actual_fields(currents, targets)
        fit_error = math.sqrt(sum(float(np.dot(field - target["field"], field - target["field"]))
                                  for field, target in zip(actual_fields, targets)))
        return {"currents": currents, "actual_fields": actual_fields, "fit_error": fit_error}

    def _dipole_field(self, params, point):
        m = np.array(params[:3], dtype=float)
        r0 = np.array(params[3:], dtype=float)
        r = point - r0
        distance = max(float(np.linalg.norm(r)), 1e-9)
        mdotr = float(np.dot(m, r))
        r3 = distance ** 3
        r5 = distance ** 5
        return MU_OVER_4PI * (3.0 * r * mdotr / r5 - m / r3)

    def _actual_fields(self, currents, targets):
        fields = []
        for target in targets:
            field = np.zeros(3)
            for params, current in zip(COIL_PARAMS, currents):
                field += self._dipole_field(params, target["position"]) * current
            fields.append(field)
        return fields

    def _limit_current_step(self, desired_currents, previous_currents):
        max_step = self.settings["max_current_step_A"]
        if max_step <= 0.0:
            return desired_currents
        delta = np.clip(desired_currents - previous_currents, -max_step, max_step)
        return previous_currents + delta

    def _apply_currents(self, currents):
        limit = self.settings["current_limit_A"]
        for i, current in enumerate(np.clip(currents, -limit, limit)):
            self.send_current(i, int(round(current * 1000.0)))

    def _send_zero_currents(self):
        for i in range(10):
            self.send_current(i, 0)

    def _record_step(self, elapsed, spot, velocity, target, kin, command_fields, solution):
        actual_fields = solution["actual_fields"]
        self.results_data.append([
            elapsed,
            target[0],
            target[1],
            spot[0],
            spot[1],
            velocity[0],
            velocity[1],
            self.model_yaw,
            self.model_pitch,
            kin["spot"][0],
            kin["spot"][1],
            command_fields[0][0],
            command_fields[0][1],
            command_fields[0][2],
            actual_fields[0][0],
            actual_fields[0][1],
            actual_fields[0][2],
            command_fields[1][0],
            command_fields[1][1],
            command_fields[1][2],
            actual_fields[1][0],
            actual_fields[1][1],
            actual_fields[1][2],
            solution["fit_error"],
            self.target_index,
        ])
        self.coil_currents_data.append([elapsed] + list(self.currents))

    def _results_dir(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    def _clear_previous_results(self):
        output_dir = self._results_dir()
        os.makedirs(output_dir, exist_ok=True)
        prefixes = (
            "closed_loop_results_case8",
            "coil_currents_case8",
            "laser_positions_case8",
            "trajectory_comparison_case8",
        )
        deleted = 0
        for filename in os.listdir(output_dir):
            if not filename.endswith((".csv", ".png")):
                continue
            if not filename.startswith(prefixes):
                continue
            try:
                os.remove(os.path.join(output_dir, filename))
                deleted += 1
            except Exception as exc:
                rospy.logwarn("Failed to delete old Case 8 result %s: %s", filename, exc)
        if deleted:
            rospy.loginfo("Deleted %d old Case 8 result file(s)", deleted)

    def _save_results(self):
        if self.results_saved:
            return
        self.results_saved = True
        output_dir = self._results_dir()
        os.makedirs(output_dir, exist_ok=True)
        run_id = self.run_id or time.strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, "closed_loop_results_case8_%s.csv" % run_id)
        currents_path = os.path.join(output_dir, "coil_currents_case8_%s.csv" % run_id)
        positions_path = os.path.join(output_dir, "laser_positions_case8_%s.csv" % run_id)

        try:
            with open(results_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Time(s)",
                    "Target_X(m)", "Target_Y(m)", "Spot_X(m)", "Spot_Y(m)",
                    "Spot_Vx(m/s)", "Spot_Vy(m/s)", "Model_Yaw(rad)", "Model_Pitch(rad)",
                    "Model_Spot_X(m)", "Model_Spot_Y(m)",
                    "Cmd_B1_X(T)", "Cmd_B1_Y(T)", "Cmd_B1_Z(T)",
                    "Actual_B1_X(T)", "Actual_B1_Y(T)", "Actual_B1_Z(T)",
                    "Cmd_B2_X(T)", "Cmd_B2_Y(T)", "Cmd_B2_Z(T)",
                    "Actual_B2_X(T)", "Actual_B2_Y(T)", "Actual_B2_Z(T)",
                    "Field_Fit_Error(T)", "Target_Index",
                ])
                writer.writerows(self.results_data)
            rospy.loginfo("Saved Case 8 results to %s", results_path)
        except Exception as exc:
            rospy.logerr("Failed to save Case 8 results: %s", exc)

        try:
            with open(currents_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Time(s)"] + ["Current_Coil_%d(A)" % (i + 1) for i in range(10)])
                writer.writerows(self.coil_currents_data)
            rospy.loginfo("Saved Case 8 currents to %s", currents_path)
        except Exception as exc:
            rospy.logerr("Failed to save Case 8 currents: %s", exc)

        try:
            with open(positions_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Time(s)", "Unix_Time(s)", "Spot_X(m)", "Spot_Y(m)",
                    "Spot_Vx(m/s)", "Spot_Vy(m/s)", "Target_Index",
                ])
                writer.writerows(self.spot_trace_data)
            rospy.loginfo("Saved Case 8 laser spot trace to %s", positions_path)
        except Exception as exc:
            rospy.logerr("Failed to save Case 8 laser spot trace: %s", exc)
