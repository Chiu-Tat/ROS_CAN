#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROS 1 Node: openmv_sphere_tracker
Publishes:
  /sphere/red   -> geometry_msgs/PointStamped
  /sphere/green -> geometry_msgs/PointStamped
  /sphere/blue  -> geometry_msgs/PointStamped
  /sphere/fps   -> std_msgs/Float32
"""

import rospy
import serial
import json
import threading
import glob
import time
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Header
from collections import defaultdict

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

        rospy.loginfo(f"Connecting to {self.serial_port}...")
        
        # Wait for boot (like Quick_test.py)
        time.sleep(3)

        # --- Publishers ---
        self.pub = {
            'red':   rospy.Publisher('/sphere/red',   PointStamped, queue_size=10),
            'green': rospy.Publisher('/sphere/green', PointStamped, queue_size=10),
            'blue':  rospy.Publisher('/sphere/blue',  PointStamped, queue_size=10),
            'fps':   rospy.Publisher('/sphere/fps',   Float32, queue_size=10)
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

        # --- Each color ---
        # Process like Quick_test.py: check if color exists in data
        for color in ('red', 'green', 'blue'):
            blob = data.get(color)
            if blob is None:
                continue
            
            # Check if blob has required fields
            if not isinstance(blob, dict) or 'x' not in blob or 'y' not in blob:
                rospy.logwarn("Invalid blob data for %s: %s", color, blob)
                continue

            pt = PointStamped()
            pt.header = Header(stamp=now, frame_id=self.frame_id)
            pt.point.x = round(float(blob['x']), 4)  # Precision: 0.0001
            pt.point.y = 0.0  # Only x position needed
            pt.point.z = 0.0  # Only x position needed

            self.pub[color].publish(pt)

            # Log detection - only x position with precision 0.0001
            rospy.loginfo(
                "Detected %s sphere @ x=%.5f",
                color.upper(), pt.point.x
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