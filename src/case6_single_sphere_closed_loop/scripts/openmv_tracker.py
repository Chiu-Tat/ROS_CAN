#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROS 1 Node: openmv_line_tracker
Receives data from OpenMV camera detecting green lines.
Publishes:
  /sphere/green -> geometry_msgs/PoseStamped (with x, y position and rotation orientation)
  /sphere/fps   -> std_msgs/Float32
"""

import rospy
import serial
import json
import threading
import glob
import time
import math
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Float32, Header

class OpenMVTracker(object):
    def __init__(self):
        rospy.init_node('openmv_line_tracker', anonymous=False)

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
            'green': rospy.Publisher('/sphere/green', PoseStamped, queue_size=10),
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

    def _angle_to_quaternion(self, angle_deg):
        """
        Convert rotation angle (in degrees, 0-180) to quaternion.
        The angle represents rotation around z-axis (yaw).
        Args:
            angle_deg: Rotation angle in degrees (0-180)
        Returns:
            Quaternion representing rotation around z-axis
        """
        # Convert degrees to radians
        angle_rad = math.radians(angle_deg)
        
        # Create quaternion for rotation around z-axis
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(angle_rad / 2.0)
        q.w = math.cos(angle_rad / 2.0)
        
        return q

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

        # --- Green line (with x, y, rotation) ---
        green_data = data.get('green')
        if green_data is not None:
            if isinstance(green_data, dict) and 'x' in green_data and 'y' in green_data:
                pose = PoseStamped()
                pose.header = Header(stamp=now, frame_id=self.frame_id)
                pose.pose.position.x = round(float(green_data['x']), 6)  # Real-world X in meters
                pose.pose.position.y = round(float(green_data['y']), 6)  # Real-world Y in meters
                pose.pose.position.z = 0.0
                
                # Handle rotation if available
                if 'rotation' in green_data:
                    rotation_deg = float(green_data['rotation'])
                    pose.pose.orientation = self._angle_to_quaternion(rotation_deg)
                else:
                    # Default orientation (no rotation)
                    pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                
                self.pub['green'].publish(pose)
                
                # Log detection with x, y, and rotation
                if 'rotation' in green_data:
                    rospy.loginfo(
                        "Detected GREEN line @ x=%.6f, y=%.6f, rotation=%.2f deg",
                        pose.pose.position.x, pose.pose.position.y, rotation_deg
                    )
                else:
                    rospy.loginfo(
                        "Detected GREEN line @ x=%.6f, y=%.6f",
                        pose.pose.position.x, pose.pose.position.y
                    )
            else:
                rospy.logwarn("Invalid green data: %s", green_data)

    def spin(self):
        rospy.spin()
        self.ser.close()


if __name__ == '__main__':
    try:
        node = OpenMVTracker()
        node.spin()
    except rospy.ROSInterruptException:
        pass