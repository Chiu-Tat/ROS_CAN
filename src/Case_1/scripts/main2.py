#!/usr/bin/env python

import sys
import threading
import rospy
import csv
import time
import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.Qwt import QwtDial, QwtDialSimpleNeedle
from PyQt5.QtGui import QColor
from geometry_msgs.msg import PointStamped
from basic_control import Ui_MainWindow  # Import the generated GUI file
from can_lib import (
    CANBusHandler,
    CANMessageBuilder,
    CANMessageDefs,
    can_to_current
)

class CANApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(CANApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initialize ROS node
        rospy.init_node("can_gui_node", anonymous=True)

        # Initialize CAN bus handler
        self.can_handler = CANBusHandler(channel="can0", bustype="socketcan")

        # Connect buttons to functions
        self.ui.pushButton_Reset.clicked.connect(self.reset_devices)
        self.ui.pushButton_ModeSet.clicked.connect(self.mode_set)
        self.ui.pushButton_OnlineCheck.clicked.connect(self.online_check)
        self.ui.pushButton_Feedback.clicked.connect(self.set_feedback_rate)
        self.ui.pushButton_Stop.clicked.connect(self.stop_devices)

        # Connect individual set buttons for each device
        self.device_buttons = [
            (self.ui.pushButton_Set1, self.ui.lineEdit_Value1, 0),
            (self.ui.pushButton_Set2, self.ui.lineEdit_Value2, 1),
            (self.ui.pushButton_Set3, self.ui.lineEdit_Value3, 2),
            (self.ui.pushButton_Set4, self.ui.lineEdit_Value4, 3),
            (self.ui.pushButton_Set5, self.ui.lineEdit_Value5, 4),
            (self.ui.pushButton_Set6, self.ui.lineEdit_Value6, 5),
            (self.ui.pushButton_Set7, self.ui.lineEdit_Value7, 6),
            (self.ui.pushButton_Set8, self.ui.lineEdit_Value8, 7),
            (self.ui.pushButton_Set9, self.ui.lineEdit_Value9, 8),
            (self.ui.pushButton_Set10, self.ui.lineEdit_Value10, 9),
        ]
        for button, line_edit, device_idx in self.device_buttons:
            button.clicked.connect(lambda _, le=line_edit, idx=device_idx: self.set_device_value(le, idx))

        # Configure dials
        for dial in [self.ui.Dial_Value1, self.ui.Dial_Value2, self.ui.Dial_Value3, self.ui.Dial_Value4,
                     self.ui.Dial_Value5, self.ui.Dial_Value6, self.ui.Dial_Value7, self.ui.Dial_Value8,
                     self.ui.Dial_Value9, self.ui.Dial_Value10]:
            dial.setScale(-15.0, 15.0)   # Set scale range from -15 to 15
            dial.setOrigin(-90) 
            dial.setScaleArc(-160, 160)
            dial.setScaleMaxMajor(7)  # 5 major ticks
            dial.setScaleMaxMinor(10) # 10 minor ticks between majors
            dial.setReadOnly(True)    # Display only
            # Set a visible needle
            needle = QwtDialSimpleNeedle(QwtDialSimpleNeedle.Arrow, True, QColor("red"), QColor("gray"))
            dial.setNeedle(needle)
            dial.setValue(0.0)        # Initial value

        # Mapping for feedback labels and CAN IDs
        self.feedback_labels = {
            can_id: label for can_id, label in zip(
                CANMessageDefs.FEEDBACK_IDS,
                [self.ui.label_Value1, self.ui.label_Value2, self.ui.label_Value3,
                 self.ui.label_Value4, self.ui.label_Value5, self.ui.label_Value6,
                 self.ui.label_Value7, self.ui.label_Value8, self.ui.label_Value9,
                 self.ui.label_Value10]
            )
        }

        # Add dial mapping alongside the existing feedback labels mapping
        self.feedback_dials = {
            can_id: dial for can_id, dial in zip(
                CANMessageDefs.FEEDBACK_IDS,
                [self.ui.Dial_Value1, self.ui.Dial_Value2, self.ui.Dial_Value3,
                 self.ui.Dial_Value4, self.ui.Dial_Value5, self.ui.Dial_Value6,
                 self.ui.Dial_Value7, self.ui.Dial_Value8, self.ui.Dial_Value9,
                 self.ui.Dial_Value10]
            )
        }

        # Initialize current values
        self.current_values = {can_id: 0 for can_id in CANMessageDefs.FEEDBACK_IDS}

        # Thread for listening to CAN messages
        self.listener_thread = threading.Thread(target=self.listen_for_feedback, daemon=True)
        self.listening = False

        # CSV playback variables
        self.csv_thread = None
        self.csv_playing = False
        self.csv_data = []
        self.total_time = 22.5  # Default total time in seconds

        # Sphere position recording variables
        self.recording_spheres = False
        self.sphere_data = []  # List of (timestamp, red_x, red_y, green_x, green_y, blue_x, blue_y)
        self.sphere_data_lock = threading.Lock()  # Lock for thread-safe access
        self.recording_start_time = None
        
        # ROS subscribers for sphere positions (created after ROS node is initialized)
        # Wait a moment for ROS to be ready
        rospy.sleep(0.5)
        self.sphere_subscribers = {
            'red': rospy.Subscriber('/sphere/red', PointStamped, lambda msg: self.sphere_callback('red', msg)),
            'green': rospy.Subscriber('/sphere/green', PointStamped, lambda msg: self.sphere_callback('green', msg)),
            'blue': rospy.Subscriber('/sphere/blue', PointStamped, lambda msg: self.sphere_callback('blue', msg))
        }
        rospy.loginfo("Subscribed to sphere position topics: /sphere/red, /sphere/green, /sphere/blue")
        
        # Store latest sphere positions
        self.latest_sphere_positions = {
            'red': {'x': None, 'y': None, 'time': None},
            'green': {'x': None, 'y': None, 'time': None},
            'blue': {'x': None, 'y': None, 'time': None}
        }

    def reset_devices(self):
        """Send a CAN message to reset all devices."""
        can_id, data = CANMessageBuilder.reset_all()
        self.can_handler.send_message(can_id, data)

    def mode_set(self):
        """Send two CAN messages to set mode and initialize devices."""
        can_id, data = CANMessageBuilder.mode_set()
        self.can_handler.send_message(can_id, data)  # Mode choose - current mode
        rospy.sleep(0.5)  # Wait for 0.5 seconds
        can_id, data = CANMessageBuilder.initialize()
        self.can_handler.send_message(can_id, data)  # Set all device current values to 0

    def online_check(self):
        """Load and play CSV current values over time, and start recording sphere positions."""
        if self.csv_playing:
            rospy.loginfo("CSV playback already running")
            return
            
        # Load CSV data
        csv_file_path = "/home/dz/Documents/CUHK/ROS_CAN/src/Case_1/scripts/coil_currents_3.csv"
        if not os.path.exists(csv_file_path):
            rospy.logerr(f"CSV file not found: {csv_file_path}")
            return
            
        try:
            self.load_csv_data(csv_file_path)
            rospy.loginfo(f"Loaded {len(self.csv_data)} data points from CSV")
            
            # Start sphere position recording
            self.start_sphere_recording()
            
            # Start CSV playback thread
            self.csv_playing = True
            self.csv_thread = threading.Thread(target=self.play_csv_data, daemon=True)
            self.csv_thread.start()
            
        except Exception as e:
            rospy.logerr(f"Error loading CSV data: {e}")
    
    def load_csv_data(self, csv_file_path):
        """Load CSV data from file."""
        self.csv_data = []
        with open(csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Parse time and current values
                time_val = float(row['Time(s)'])
                currents = []
                for i in range(1, 11):  # Coils 1-10
                    current_key = f'Current_Coil_{i}(A)'
                    current_val = float(row[current_key])
                    # Convert from Amperes to milliamperes (device expects mA)
                    current_mA = int(current_val * 1000)
                    currents.append(current_mA)
                
                self.csv_data.append((time_val, currents))
    
    def play_csv_data(self):
        """Play CSV data in real-time according to timestamps."""
        if not self.csv_data:
            rospy.logwarn("No CSV data to play")
            return
            
        start_time = time.time()
        rospy.loginfo(f"Starting CSV playback for {self.total_time} seconds")
        
        for csv_time, currents in self.csv_data:
            if not self.csv_playing:
                break
                
            # Check if we've exceeded the total time limit
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.total_time:
                rospy.loginfo(f"Reached total time limit of {self.total_time} seconds")
                break
                
            # Wait until it's time to send this data point
            target_time = start_time + csv_time
            current_time = time.time()
            if current_time < target_time:
                time.sleep(target_time - current_time)
            
            # Send current values to all devices
            for i, current_mA in enumerate(currents):
                if i < len(CANMessageDefs.DEVICE_IDS):
                    self.send_device_current(i, current_mA)
            
            rospy.loginfo(f"Sent currents at time {csv_time:.1f}s: {currents}")
        
        # Stop all devices after playback
        rospy.loginfo("CSV playback finished, stopping devices")
        self.stop_devices()
        self.csv_playing = False
        
        # Stop sphere position recording and save to CSV
        self.stop_sphere_recording()
    
    def send_device_current(self, device_index, current_mA):
        """Send current value to a specific device."""
        try:
            can_id, data = CANMessageBuilder.set_device_current(device_index, current_mA)
            self.can_handler.send_message(can_id, data)
        except Exception as e:
            rospy.logerr(f"Error sending current to device {device_index}: {e}")
    
    def set_csv_total_time(self, total_time_seconds):
        """Set the total time for CSV playback."""
        self.total_time = float(total_time_seconds)
        rospy.loginfo(f"Set CSV playback total time to {self.total_time} seconds")        

    def set_feedback_rate(self):
        """Send a CAN message to set feedback rate and start feedback listening."""
        can_id, data = CANMessageBuilder.set_feedback_rate()
        self.can_handler.send_message(can_id, data)

        # Start listening for feedback messages
        if not self.listening:
            self.listening = True
            if not self.listener_thread.is_alive():
                self.listener_thread.start()

    def stop_devices(self):
        """Send a CAN message to stop all devices and stop CSV playback."""
        # Stop CSV playback if running
        if self.csv_playing:
            self.csv_playing = False
            rospy.loginfo("Stopped CSV playback")
        
        # Stop sphere recording if running
        if self.recording_spheres:
            self.stop_sphere_recording()
        
        can_id, data = CANMessageBuilder.initialize()
        self.can_handler.send_message(can_id, data)

    def set_device_value(self, line_edit, device_index):
        """Send a CAN message to set a specific device value."""
        value = int(line_edit.text())
        can_id, data = CANMessageBuilder.set_device_current(device_index, value)
        self.can_handler.send_message(can_id, data) 

    def listen_for_feedback(self):
        """Listen for CAN messages and update feedback labels."""
        while self.listening:
            # Receive a CAN message from the bus
            message = self.can_handler.receive_message(timeout=1)
            if message is not None:
                # Check if the CAN ID maps to a feedback label
                if message.arbitration_id in self.feedback_labels:
                    try:
                        # Convert the CAN data to a current value
                        current_value = can_to_current(message.data[:2])

                        # Store the current value
                        self.current_values[message.arbitration_id] = current_value

                        # Update the corresponding label
                        label = self.feedback_labels[message.arbitration_id]
                        QtCore.QMetaObject.invokeMethod(label, "setText", QtCore.Q_ARG(str, str(current_value)))

                        # Update the corresponding dial
                        dial = self.feedback_dials[message.arbitration_id]
                        QtCore.QMetaObject.invokeMethod(dial, "setValue",
                            QtCore.Q_ARG(float, float(current_value/1000.0)))  # Convert to Amperes
                        # Calculate sum of squares and update thermo
                        sum_squares = sum(3 * (value/1000.0) * (value/1000.0) for value in self.current_values.values())
                        QtCore.QMetaObject.invokeMethod(self.ui.Thermo, "setValue",
                            QtCore.Q_ARG(float, float(sum_squares)))
                    
                    except Exception as e:
                        rospy.logerr(f"Error processing CAN message: {e}")

    def sphere_callback(self, color, msg):
        """Callback for sphere position messages."""
        # Store latest position (always update, even when not recording)
        self.latest_sphere_positions[color]['x'] = msg.point.x
        self.latest_sphere_positions[color]['y'] = msg.point.y
        self.latest_sphere_positions[color]['time'] = time.time()
        
        if not self.recording_spheres:
            return
        
        # Record data point when recording
        elapsed_time = time.time() - self.recording_start_time
        self.record_sphere_data(elapsed_time)
        rospy.loginfo(f"Recorded {color} sphere position at t={elapsed_time:.3f}s: ({msg.point.x:.2f}, {msg.point.y:.2f})")

    def start_sphere_recording(self):
        """Start recording sphere positions."""
        with self.sphere_data_lock:
            self.sphere_data = []
            self.recording_spheres = True
            self.recording_start_time = time.time()
            # Reset latest positions
            for color in ['red', 'green', 'blue']:
                self.latest_sphere_positions[color] = {'x': None, 'y': None, 'time': None}
        rospy.loginfo("Started recording sphere positions")
        
        # Debug: Check if subscribers are connected
        for color, sub in self.sphere_subscribers.items():
            num_connections = sub.get_num_connections()
            rospy.loginfo(f"Sphere subscriber '{color}' has {num_connections} publisher(s)")

    def record_sphere_data(self, elapsed_time):
        """Record current sphere positions with timestamp."""
        with self.sphere_data_lock:
            red_x = self.latest_sphere_positions['red']['x']
            red_y = self.latest_sphere_positions['red']['y']
            green_x = self.latest_sphere_positions['green']['x']
            green_y = self.latest_sphere_positions['green']['y']
            blue_x = self.latest_sphere_positions['blue']['x']
            blue_y = self.latest_sphere_positions['blue']['y']
            
            # Only record if at least one sphere is detected
            if red_x is not None or green_x is not None or blue_x is not None:
                self.sphere_data.append((
                    elapsed_time,
                    red_x if red_x is not None else '',
                    red_y if red_y is not None else '',
                    green_x if green_x is not None else '',
                    green_y if green_y is not None else '',
                    blue_x if blue_x is not None else '',
                    blue_y if blue_y is not None else ''
                ))
                rospy.logdebug(f"Added data point: t={elapsed_time:.3f}s, red=({red_x}, {red_y}), green=({green_x}, {green_y}), blue=({blue_x}, {blue_y})")
            else:
                rospy.logdebug(f"No sphere detected, skipping data point at t={elapsed_time:.3f}s")

    def stop_sphere_recording(self):
        """Stop recording sphere positions and save to CSV file."""
        if not self.recording_spheres:
            return
        
        self.recording_spheres = False
        rospy.loginfo("Stopped recording sphere positions")
        
        # Save to CSV file
        csv_output_path = "/home/dz/Documents/CUHK/ROS_CAN/src/Case_1/scripts/sphere_positions.csv"
        try:
            with self.sphere_data_lock:
                with open(csv_output_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow([
                        'Time(s)',
                        'Red_X', 'Red_Y',
                        'Green_X', 'Green_Y',
                        'Blue_X', 'Blue_Y'
                    ])
                    # Write data
                    for row in self.sphere_data:
                        writer.writerow(row)
            
            rospy.loginfo(f"Saved {len(self.sphere_data)} sphere position data points to {csv_output_path}")
        except Exception as e:
            rospy.logerr(f"Error saving sphere positions to CSV: {e}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CANApp()
    window.show()
    sys.exit(app.exec_())