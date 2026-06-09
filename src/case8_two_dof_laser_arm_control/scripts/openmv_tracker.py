#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenMV tracker for Case 8.

Publishes the laser spot as:
  /laser/spot -> geometry_msgs/PointStamped

The OpenMV script emits raw pixel centers. This node converts those pixels to
metric workspace coordinates before publishing.
"""

import json
import os
import threading
import time

import numpy as np
import rospy
import serial
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Header


DEFAULT_CALIBRATION = {
    # Affine least-squares fit from fiver_corner_star_point_coordinate.txt.
    # Real coordinates are in millimeters in the file and meters in ROS.
    "world_x_from_pixel_x": -0.00000890023737,
    "world_x_from_pixel_y": 0.000661677308,
    "world_x_offset": -0.0879107985,
    "world_y_from_pixel_x": 0.000653888149,
    "world_y_from_pixel_y": -0.00000465832823,
    "world_y_offset": -0.113985961,
}
DEFAULT_CALIBRATION_POINTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fiver_corner_star_point_coordinate.txt",
)


class OpenMVLaserTracker(object):
    def __init__(self):
        rospy.init_node("case8_openmv_laser_tracker", anonymous=False)

        self.requested_port = rospy.get_param("~port", "/dev/ttyACM0")
        self.serial_port = self.requested_port
        self.baudrate = rospy.get_param("~baudrate", 115200)
        self.frame_id = rospy.get_param("~frame_id", "openmv_camera")
        self.reconnect_delay_s = rospy.get_param("~reconnect_delay_s", 1.0)
        self.transient_read_fail_limit = max(1, int(rospy.get_param("~transient_read_fail_limit", 3)))
        self.suppress_usb_control_lines = rospy.get_param("~suppress_usb_control_lines", False)
        self.ser = None
        self.transient_read_failures = 0
        self.calibration = self._load_calibration()

        self.spot_pub = rospy.Publisher("/laser/spot", PointStamped, queue_size=10)
        self.fps_pub = rospy.Publisher("/laser/fps", Float32, queue_size=10)

        rospy.loginfo("Connecting to %s...", self.serial_port)
        time.sleep(3.0)
        if not self._connect_serial():
            rospy.logerr("Failed to open OpenMV serial port")
            rospy.signal_shutdown("Serial port error")
            return

        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()
        rospy.loginfo("Case 8 OpenMV laser tracker started")

    def _load_calibration(self):
        calibration = DEFAULT_CALIBRATION.copy()
        calibration_points_file = rospy.get_param("~calibration_points_file", DEFAULT_CALIBRATION_POINTS_FILE)
        use_calibration_file = rospy.get_param("~use_calibration_points_file", True)
        if use_calibration_file and calibration_points_file:
            try:
                calibration = self._fit_affine_calibration(calibration_points_file)
                rospy.loginfo("Loaded Case 8 pixel calibration from %s", calibration_points_file)
            except Exception as exc:
                rospy.logwarn(
                    "Could not load Case 8 calibration file %s: %s. Using built-in coefficients.",
                    calibration_points_file,
                    exc,
                )

        for key, value in DEFAULT_CALIBRATION.items():
            param_name = "~" + key
            if rospy.has_param(param_name):
                calibration[key] = float(rospy.get_param(param_name, value))
        return calibration

    def _fit_affine_calibration(self, path):
        points = self._read_calibration_points(path)
        if len(points) < 3:
            raise ValueError("at least 3 calibration points are required")

        pixel = np.array([[point["pixel_x"], point["pixel_y"]] for point in points], dtype=float)
        world = np.array([[point["real_x_mm"], point["real_y_mm"]] for point in points], dtype=float) / 1000.0
        design = np.column_stack((pixel[:, 0], pixel[:, 1], np.ones(len(pixel))))
        coef_x = np.linalg.lstsq(design, world[:, 0], rcond=None)[0]
        coef_y = np.linalg.lstsq(design, world[:, 1], rcond=None)[0]
        predicted = np.column_stack((np.dot(design, coef_x), np.dot(design, coef_y)))
        error_mm = (predicted - world) * 1000.0
        rms_mm = float(np.sqrt(np.mean(np.sum(error_mm ** 2, axis=1))))
        rospy.loginfo("Case 8 OpenMV affine calibration RMS error: %.3f mm from %d points", rms_mm, len(points))
        return {
            "world_x_from_pixel_x": float(coef_x[0]),
            "world_x_from_pixel_y": float(coef_x[1]),
            "world_x_offset": float(coef_x[2]),
            "world_y_from_pixel_x": float(coef_y[0]),
            "world_y_from_pixel_y": float(coef_y[1]),
            "world_y_offset": float(coef_y[2]),
        }

    def _read_calibration_points(self, path):
        points = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.lower().startswith("point no."):
                    continue
                parts = [part.strip() for part in line.split("|")]
                parts = [part for part in parts if part]
                if len(parts) < 5:
                    continue
                points.append({
                    "point_no": int(parts[0]),
                    "real_x_mm": float(parts[1]),
                    "real_y_mm": float(parts[2]),
                    "pixel_x": float(parts[3]),
                    "pixel_y": float(parts[4]),
                })
        return points

    def _connect_serial(self):
        if not self.requested_port:
            rospy.logwarn_throttle(2.0, "No OpenMV serial port configured.")
            return False
        self.serial_port = self.requested_port
        try:
            kwargs = {
                "port": self.serial_port,
                "baudrate": self.baudrate,
                "timeout": 1.0,
                "write_timeout": 1.0,
            }
            try:
                self.ser = serial.Serial(exclusive=True, **kwargs)
            except TypeError:
                self.ser = serial.Serial(**kwargs)
            if self.suppress_usb_control_lines:
                try:
                    self.ser.dtr = False
                    self.ser.rts = False
                except Exception:
                    pass
            self.ser.reset_input_buffer()
            self.transient_read_failures = 0
            rospy.loginfo("Opened serial port: %s", self.ser.name)
            return True
        except serial.SerialException as exc:
            self.ser = None
            rospy.logerr_throttle(2.0, "Failed to open OpenMV serial port %s: %s", self.serial_port, exc)
            return False

    def _close_serial(self):
        if self.ser is None:
            return
        try:
            self.ser.close()
        except Exception:
            pass
        self.ser = None

    def _reader(self):
        while not rospy.is_shutdown():
            try:
                if self.ser is None or not self.ser.is_open:
                    if not self._connect_serial():
                        rospy.sleep(self.reconnect_delay_s)
                        continue
                line = self.ser.readline()
                if not line:
                    rospy.logwarn_throttle(3.0, "No data from OpenMV on %s", self.serial_port)
                    continue
                text = line.decode("utf-8", errors="ignore").strip()
                if text:
                    self.transient_read_failures = 0
                    self._process_line(text)
            except (serial.SerialException, OSError) as exc:
                if "returned no data" in str(exc):
                    self.transient_read_failures += 1
                    if self.transient_read_failures < self.transient_read_fail_limit:
                        rospy.sleep(0.1)
                        continue
                rospy.logerr_throttle(2.0, "OpenMV serial read failed: %s. Reconnecting.", exc)
                self._close_serial()
                rospy.sleep(self.reconnect_delay_s)
            except Exception as exc:
                rospy.logerr_throttle(2.0, "Unexpected OpenMV reader error: %s", exc)
                rospy.sleep(1.0)

    def _process_line(self, line):
        try:
            data = json.loads(line)
        except (ValueError, json.JSONDecodeError) as exc:
            rospy.logwarn("Invalid JSON from OpenMV: %s (%s)", line[:80], exc)
            return

        if isinstance(data, (int, float)):
            self._publish_fps(float(data))
            return
        if not isinstance(data, dict):
            return
        if "fps" in data:
            try:
                self._publish_fps(float(data["fps"]))
            except (TypeError, ValueError):
                pass

        blob = data.get("spot") or data.get("laser") or data.get("red")
        if not isinstance(blob, dict) or "x" not in blob or "y" not in blob:
            return
        try:
            pixel_x = float(blob["x"])
            pixel_y = float(blob["y"])
        except (TypeError, ValueError):
            rospy.logwarn_throttle(2.0, "Ignoring invalid laser spot blob: %r", blob)
            return

        world_x, world_y = self._pixel_to_world(pixel_x, pixel_y)
        msg = PointStamped()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        msg.point.x = world_x
        msg.point.y = world_y
        msg.point.z = 0.0
        self.spot_pub.publish(msg)
        rospy.loginfo_throttle(
            0.5,
            "Detected laser spot pixel=(%.1f, %.1f) world=(%.5f, %.5f)",
            pixel_x,
            pixel_y,
            world_x,
            world_y,
        )

    def _publish_fps(self, fps):
        msg = Float32()
        msg.data = fps
        self.fps_pub.publish(msg)

    def _pixel_to_world(self, pixel_x, pixel_y):
        c = self.calibration
        world_x = (
            c["world_x_from_pixel_x"] * pixel_x +
            c["world_x_from_pixel_y"] * pixel_y +
            c["world_x_offset"]
        )
        world_y = (
            c["world_y_from_pixel_x"] * pixel_x +
            c["world_y_from_pixel_y"] * pixel_y +
            c["world_y_offset"]
        )
        return world_x, world_y

    def spin(self):
        rospy.spin()
        self._close_serial()


if __name__ == "__main__":
    try:
        node = OpenMVLaserTracker()
        node.spin()
    except rospy.ROSInterruptException:
        pass
