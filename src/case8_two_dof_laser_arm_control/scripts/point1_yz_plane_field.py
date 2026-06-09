#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply a Case 8 startup magnetic field in the y-z plane at laser point 1.

This script is separate from the closed-loop trajectory controller. It:
  - loads point 1 from trajectory_case8_laser_spot.json
  - optionally waits until /laser/spot is within the point-1 tolerance
  - computes the two magnet positions from the point-1 arm pose
  - requests magnetic fields with Bx = 0 for both magnets
  - solves the ten-coil current vector and sends it over CAN
"""

import json
import math
import os
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped

from can_lib import CANBusHandler, CANMessageBuilder
from closed_loop_control import ClosedLoopLaserSpotController, TwoDOFLaserModel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAJECTORY_PATH = os.path.join(SCRIPT_DIR, "trajectory_case8_laser_spot.json")


class Point1YZPlaneField(object):
    def __init__(self):
        rospy.init_node("case8_point1_yz_plane_field", anonymous=False)
        self.lock = threading.Lock()
        self.latest_spot = None
        self.latest_spot_time = None

        self.trajectory_path = rospy.get_param("~trajectory", DEFAULT_TRAJECTORY_PATH)
        self.can_channel = rospy.get_param("~can_channel", "can0")
        self.can_bustype = rospy.get_param("~can_bustype", "socketcan")
        self.verbose_can = rospy.get_param("~verbose_can", False)
        self.wait_for_spot = rospy.get_param("~wait_for_spot", True)
        self.spot_topic = rospy.get_param("~spot_topic", "/laser/spot")
        self.spot_timeout_s = float(rospy.get_param("~spot_timeout_s", 30.0))
        self.publish_duration_s = float(rospy.get_param("~duration_s", 0.0))
        self.send_rate_hz = max(0.2, float(rospy.get_param("~send_rate_hz", 5.0)))

        self.data = self._load_trajectory()
        self.settings = self.data["settings"]
        self.point1 = np.array(self.data["laser_spot"]["raw_path"][0], dtype=float)
        self.field_strength_T = float(rospy.get_param("~field_strength_T", self.settings["field_strength_T"]))
        self.current_limit_A = float(rospy.get_param("~current_limit_A", self.settings["current_limit_A"]))
        self.current_smoothing = float(rospy.get_param("~current_smoothing", self.settings["current_smoothing"]))
        self.current_regularization = float(
            rospy.get_param("~current_regularization", self.settings["current_regularization"])
        )
        self.tolerance_m = float(rospy.get_param("~tolerance_m", self.settings.get("start_tolerance_m", 0.004)))

        self.model = TwoDOFLaserModel()
        self.solver = object.__new__(ClosedLoopLaserSpotController)
        self.solver.model = self.model
        self.solver.settings = {
            "field_strength_T": self.field_strength_T,
            "current_limit_A": self.current_limit_A,
            "current_smoothing": self.current_smoothing,
            "current_regularization": self.current_regularization,
        }

        self.can_handler = CANBusHandler(
            channel=self.can_channel,
            bustype=self.can_bustype,
            verbose_send=self.verbose_can,
        )
        self.spot_subscriber = None
        if self.wait_for_spot:
            self.spot_subscriber = rospy.Subscriber(self.spot_topic, PointStamped, self._spot_callback)

    def _load_trajectory(self):
        with open(self.trajectory_path, "r") as f:
            return json.load(f)

    def _spot_callback(self, msg):
        with self.lock:
            self.latest_spot = np.array([msg.point.x, msg.point.y], dtype=float)
            self.latest_spot_time = time.time()

    def _wait_until_spot_at_point1(self):
        if not self.wait_for_spot:
            return True
        rospy.loginfo(
            "Waiting for %s to be within %.1f mm of point 1 (%.5f, %.5f)",
            self.spot_topic,
            self.tolerance_m * 1000.0,
            self.point1[0],
            self.point1[1],
        )
        start = time.time()
        while not rospy.is_shutdown():
            with self.lock:
                spot = None if self.latest_spot is None else self.latest_spot.copy()
            if spot is not None:
                error = float(np.linalg.norm(spot - self.point1))
                rospy.loginfo_throttle(
                    1.0,
                    "Point 1 wait error: %.2f mm, spot=(%.5f, %.5f)",
                    error * 1000.0,
                    spot[0],
                    spot[1],
                )
                if error <= self.tolerance_m:
                    rospy.loginfo("Laser spot is at point 1; applying y-z plane field")
                    return True
            if self.spot_timeout_s > 0.0 and time.time() - start > self.spot_timeout_s:
                rospy.logerr("Timed out waiting for laser spot at point 1")
                return False
            rospy.sleep(0.05)
        return False

    def _point1_kinematics(self):
        yaw, pitch = self.model.inverse_laser_target(self.point1[0], self.point1[1])
        return yaw, pitch, self.model.forward_kinematics(yaw, pitch)

    def _yz_plane_fields(self, kin):
        fields = ClosedLoopLaserSpotController._magnet_aligned_fields(self.solver, kin)
        return ClosedLoopLaserSpotController._project_fields_to_yz_plane(self.solver, fields)

    def _solve_currents(self):
        yaw, pitch, kin = self._point1_kinematics()
        fields = self._yz_plane_fields(kin)
        solution = ClosedLoopLaserSpotController._solve_currents_for_fields(
            self.solver,
            kin,
            fields,
            np.zeros(10),
        )
        currents = np.clip(solution["currents"], -self.current_limit_A, self.current_limit_A)
        rospy.loginfo(
            "Point 1 pose yaw=%.2f deg pitch=%.2f deg",
            yaw * 180.0 / math.pi,
            pitch * 180.0 / math.pi,
        )
        for index, field in enumerate(fields, 1):
            rospy.loginfo(
                "B%d target y-z plane: [%.6f, %.6f, %.6f] T, |B|=%.3f mT",
                index,
                field[0],
                field[1],
                field[2],
                np.linalg.norm(field) * 1000.0,
            )
        rospy.loginfo(
            "Solved currents: max=%.3f A, norm=%.3f A, field fit error=%.3f mT",
            float(np.max(np.abs(currents))),
            float(np.linalg.norm(currents)),
            solution["fit_error"] * 1000.0,
        )
        return currents

    def _send_currents(self, currents):
        for index, current_A in enumerate(currents):
            current_mA = int(round(current_A * 1000.0))
            can_id, data = CANMessageBuilder.set_device_current(index, current_mA)
            self.can_handler.send_message(can_id, data)

    def _send_zero_currents(self):
        for index in range(10):
            can_id, data = CANMessageBuilder.set_device_current(index, 0)
            self.can_handler.send_message(can_id, data)

    def run(self):
        if not self._wait_until_spot_at_point1():
            return

        can_id, data = CANMessageBuilder.mode_set()
        self.can_handler.send_message(can_id, data)
        rospy.sleep(0.5)

        currents = self._solve_currents()
        period = 1.0 / self.send_rate_hz
        start = time.time()
        rospy.loginfo(
            "Applying point-1 y-z plane field. duration_s=%.2f; use Ctrl-C to stop when duration is 0.",
            self.publish_duration_s,
        )
        try:
            while not rospy.is_shutdown():
                self._send_currents(currents)
                if self.publish_duration_s > 0.0 and time.time() - start >= self.publish_duration_s:
                    break
                rospy.sleep(period)
        finally:
            if rospy.get_param("~zero_on_exit", True):
                self._send_zero_currents()
            self.can_handler.close()


if __name__ == "__main__":
    try:
        Point1YZPlaneField().run()
    except rospy.ROSInterruptException:
        pass
