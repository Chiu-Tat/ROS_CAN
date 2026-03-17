#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROS 1 Node: openmv_sphere_tracker

Reads JSON data from OpenMV camera running Openmv_code.py via USB serial
(/dev/ttyACM*). Computes joint angles for a 2-DOF planar arm:

  - Joint 1 (base): fixed at (joint1_x, joint1_y) in image pixels
  - Green: second joint (end of link1)
  - Blue: end effector (end of link2)

  theta1: angle of link1 from horizontal (joint1 -> green)
  theta2: angle of link2 relative to link1 (green -> blue, relative to link1)

Publishes:
  /sphere/green   -> geometry_msgs/PointStamped
  /sphere/blue    -> geometry_msgs/PointStamped
  /arm/theta1     -> std_msgs/Float32 (rad)
  /arm/theta2     -> std_msgs/Float32 (rad)
  /sphere/fps     -> std_msgs/Float32
"""

import math
import rospy
import serial
import json
import threading
import glob
import time
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Header

class OpenMVTracker(object):
    def __init__(self):
        rospy.init_node('openmv_sphere_tracker', anonymous=False)

        # --- Parameters ---
        # Auto-detect port like Quick_test.py
        ports = glob.glob('/dev/ttyACM*')
        if not ports:
            rospy.logerr("No OpenMV device found! Check /dev/ttyACM*")
            rospy.signal_shutdown("No serial port found")
            return
        
        self.serial_port = rospy.get_param('~port', ports[0])
        self.baudrate = rospy.get_param('~baudrate', 115200)
        self.frame_id = rospy.get_param('~frame_id', 'openmv_camera')

        # 2-DOF arm geometry (image pixels). Joint1 = base, green = joint2, blue = end effector
        self.joint1_x = rospy.get_param('~joint1_x', 217.0)
        self.joint1_y = rospy.get_param('~joint1_y', 117.0)

        # Calibration offsets (degrees): subtracted so that physical 0° reads as 0°
        self.theta1_offset = rospy.get_param('~theta1_offset_deg', -4.4)
        self.theta2_offset = rospy.get_param('~theta2_offset_deg', 4.4)

        rospy.loginfo(f"Connecting to {self.serial_port}...")
        
        # Wait for boot (like Quick_test.py)
        time.sleep(3)

        # --- Publishers ---
        self.pub = {
            'green':  rospy.Publisher('/sphere/green',  PointStamped, queue_size=10),
            'blue':   rospy.Publisher('/sphere/blue',   PointStamped, queue_size=10),
            'theta1': rospy.Publisher('/arm/theta1',    Float32, queue_size=10),
            'theta2': rospy.Publisher('/arm/theta2',   Float32, queue_size=10),
            'fps':    rospy.Publisher('/sphere/fps',   Float32, queue_size=10)
        }

        # --- Serial ---
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=1.0
            )
            rospy.loginfo("Opened serial port: %s", self.ser.name)
        except serial.SerialException as e:
            rospy.logerr("Failed to open serial port: %s", e)
            rospy.signal_shutdown("Serial port error")
            return

        # --- Background thread ---
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

        rospy.loginfo("OpenMV Tracker node started.")

    def _reader(self):
        """Read serial data using readline() like Quick_test.py"""
        rospy.loginfo("Listening for OpenMV data...")
        while not rospy.is_shutdown():
            try:
                line = self.ser.readline()
                if line:
                    text = line.decode('utf-8', errors='ignore').strip()
                    if text:
                        self._process_line(text)
            except Exception as e:
                rospy.logerr("Serial read error: %s", e)
                rospy.sleep(1.0)

    def _process_line(self, line):
        """Process a line of JSON data from OpenMV"""
        try:
            data = json.loads(line)
        except (ValueError, json.JSONDecodeError) as e:
            rospy.logwarn("Invalid JSON: %s (error: %s)", line[:50], e)
            return

        now = rospy.Time.now()

        # --- FPS ---
        if 'fps' in data:
            fps_msg = Float32()
            fps_msg.data = float(data['fps'])
            self.pub['fps'].publish(fps_msg)

        # --- Green and blue (from Openmv_code.py) ---
        green = data.get('green')
        blue = data.get('blue')

        for color, blob in [('green', green), ('blue', blue)]:
            if blob is None:
                continue
            if not isinstance(blob, dict) or 'x' not in blob or 'y' not in blob:
                rospy.logwarn("Invalid blob data for %s: %s", color, blob)
                continue

            pt = PointStamped()
            pt.header = Header(stamp=now, frame_id=self.frame_id)
            pt.point.x = round(float(blob['x']), 4)
            pt.point.y = round(float(blob['y']), 4)
            pt.point.z = 0.0
            self.pub[color].publish(pt)

        # --- Joint angles (2-DOF arm: joint1 -> green -> blue) ---
        # Openmv_code sends (x, y) as raw pixels (cx, cy)
        if green is not None and blue is not None and isinstance(green, dict) and isinstance(blue, dict):
            if 'x' in green and 'y' in green and 'x' in blue and 'y' in blue:
                gx = float(green['x'])
                gy = float(green['y'])
                bx = float(blue['x'])
                by = float(blue['y'])

                # Arm extends leftward from joint1, so negate dx to set
                # 0° = horizontal (left), 90° = straight down (image y-down)
                dx1 = self.joint1_x - gx
                dy1 = gy - self.joint1_y
                theta1 = math.atan2(dy1, dx1) - math.radians(self.theta1_offset)

                # theta2: angle of link2 relative to link1 (green -> blue)
                dx2 = gx - bx
                dy2 = by - gy
                link2_angle = math.atan2(dy2, dx2)
                theta2 = link2_angle - theta1 - math.radians(self.theta2_offset)

                # Normalize to [-pi, pi]
                theta1 = math.atan2(math.sin(theta1), math.cos(theta1))
                theta2 = math.atan2(math.sin(theta2), math.cos(theta2))

                theta1_deg = math.degrees(theta1)
                theta2_deg = math.degrees(theta2)

                t1_msg = Float32()
                t1_msg.data = theta1_deg
                t2_msg = Float32()
                t2_msg.data = theta2_deg
                self.pub['theta1'].publish(t1_msg)
                self.pub['theta2'].publish(t2_msg)

                rospy.loginfo(
                    "Joint angles: theta1=%.1f deg, theta2=%.1f deg",
                    theta1_deg, theta2_deg
                )

    def spin(self):
        rospy.spin()
        self.ser.close()


if __name__ == '__main__':
    try:
        node = OpenMVTracker()
        node.spin()
    except rospy.ROSInterruptException:
        pass