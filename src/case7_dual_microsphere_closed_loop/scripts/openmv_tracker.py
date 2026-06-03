#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenMV tracker for Case 7.

Publishes two planar microsphere positions:
  /sphere/red  -> geometry_msgs/PointStamped, microsphere 1
  /sphere/green -> geometry_msgs/PointStamped, microsphere 2

The OpenMV script emits raw pixel centers. This node converts those pixels to
metric Case 7 workspace coordinates before publishing.
"""

import json
import threading
import time

import rospy
import serial
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Header


DEFAULT_CALIBRATION = {
    # Affine transform fitted from:
    #   pixel (59, 137)  -> world (0, -0.0635)
    #   pixel (285, 137) -> world (0,  0.0635)
    #   pixel (170, 21)  -> world (-0.068, 0)
    "world_x_from_pixel_x": 0.0,
    "world_x_from_pixel_y": 0.000586206896552,
    "world_x_offset": -0.0803103448276,
    "world_y_from_pixel_x": 0.000561946902655,
    "world_y_from_pixel_y": -0.00000968873970095,
    "world_y_offset": -0.0953275099176,
}


class OpenMVTracker(object):
    def __init__(self):
        rospy.init_node("case7_openmv_sphere_tracker", anonymous=False)

        self.requested_port = rospy.get_param("~port", "/dev/ttyACM1")
        self.serial_port = self.requested_port
        self.baudrate = rospy.get_param("~baudrate", 115200)
        self.frame_id = rospy.get_param("~frame_id", "openmv_camera")
        self.reconnect_delay_s = rospy.get_param("~reconnect_delay_s", 1.0)
        self.ser = None
        self.calibration = {
            key: float(rospy.get_param("~" + key, value))
            for key, value in DEFAULT_CALIBRATION.items()
        }

        self.pub = {
            "red": rospy.Publisher("/sphere/red", PointStamped, queue_size=10),
            "green": rospy.Publisher("/sphere/green", PointStamped, queue_size=10),
            "fps": rospy.Publisher("/sphere/fps", Float32, queue_size=10),
        }

        rospy.loginfo("Connecting to %s...", self.serial_port)
        time.sleep(3.0)

        if not self._connect_serial():
            rospy.logerr("Failed to open OpenMV serial port")
            rospy.signal_shutdown("Serial port error")
            return

        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()
        rospy.loginfo("Case 7 OpenMV tracker started with calibrated metric output")

    def _detect_serial_port(self):
        if self.requested_port:
            return self.requested_port
        return None

    def _connect_serial(self):
        port = self._detect_serial_port()
        if port is None:
            rospy.logwarn_throttle(2.0, "No OpenMV serial port configured.")
            return False

        self.serial_port = port
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
            self.ser.reset_input_buffer()
            rospy.loginfo("Opened serial port: %s", self.ser.name)
            return True
        except serial.SerialException as exc:
            self.ser = None
            rospy.logerr_throttle(
                2.0,
                "Failed to open OpenMV serial port %s: %s. Close OpenMV IDE serial terminal or other readers if open.",
                self.serial_port,
                exc,
            )
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
                    continue
                text = line.decode("utf-8", errors="ignore").strip()
                if text:
                    self._process_line(text)
            except (serial.SerialException, OSError) as exc:
                rospy.logerr_throttle(
                    2.0,
                    "OpenMV serial read failed: %s. Reconnecting; make sure only this ROS node is using the camera port.",
                    exc,
                )
                self._close_serial()
                rospy.sleep(self.reconnect_delay_s)
            except Exception as exc:
                rospy.logerr_throttle(2.0, "Unexpected OpenMV serial reader error: %s", exc)
                rospy.sleep(1.0)

    def _process_line(self, line):
        try:
            data = json.loads(line)
        except (ValueError, json.JSONDecodeError) as exc:
            rospy.logwarn("Invalid JSON from OpenMV: %s (%s)", line[:80], exc)
            return

        now = rospy.Time.now()
        if "fps" in data:
            fps_msg = Float32()
            fps_msg.data = float(data["fps"])
            self.pub["fps"].publish(fps_msg)

        for color in ("red", "green"):
            blob = data.get(color)
            if not isinstance(blob, dict) or "x" not in blob or "y" not in blob:
                continue
            pixel_x = float(blob["x"])
            pixel_y = float(blob["y"])
            world_x, world_y = self._pixel_to_world(pixel_x, pixel_y)

            msg = PointStamped()
            msg.header = Header(stamp=now, frame_id=self.frame_id)
            msg.point.x = world_x
            msg.point.y = world_y
            msg.point.z = 0.0
            self.pub[color].publish(msg)

            rospy.loginfo_throttle(
                0.5,
                "Detected %s microsphere pixel=(%.1f, %.1f) world=(%.5f, %.5f)",
                color.upper(),
                pixel_x,
                pixel_y,
                msg.point.x,
                msg.point.y,
            )

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
        node = OpenMVTracker()
        node.spin()
    except rospy.ROSInterruptException:
        pass
