#!/usr/bin/env python

import sys
import threading
import rospy
import can
import csv
import time
import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.Qwt import QwtDial, QwtDialSimpleNeedle
from PyQt5.QtGui import QColor
from basic_control import Ui_MainWindow  # Import the generated GUI file

def can_to_current(data):
    """
    Convert CAN message data (2 bytes) to current value.
    """
    if len(data) != 2:
        raise ValueError("Data must be exactly 2 bytes")

    # Combine the two bytes into a single raw value
    raw_value = (data[0] << 8) | data[1]

    # If the raw value is negative in two's complement (16 bits), correct it
    if raw_value & 0x8000:  # Check the sign bit
        raw_value -= 0x10000

    return raw_value

def current_to_can(current):
    """
    Convert current value to CAN message data (2 bytes).
    """
    # Clamp the current value to the valid range
    # current = max(-32768, min(32767, current))
    current = max(-15000, min(15000, current))
    current &= 0xFFFF  # Mask to 16 bits
    data_0 = (current >> 8) & 0xFF # Get the high byte
    data_1 = current & 0xFF # Get the low byte

    return [data_0, data_1]

class CANApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(CANApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initialize ROS node
        rospy.init_node("can_gui_node", anonymous=True)

        # Initialize CAN bus
        self.bus = can.interface.Bus(channel="can0", bustype="socketcan")

        # Connect buttons to functions
        self.ui.pushButton_Reset.clicked.connect(self.reset_devices)
        self.ui.pushButton_ModeSet.clicked.connect(self.mode_set)
        self.ui.pushButton_OnlineCheck.clicked.connect(self.online_check)
        self.ui.pushButton_Feedback.clicked.connect(self.set_feedback_rate)
        self.ui.pushButton_Stop.clicked.connect(self.stop_devices)

        # Connect individual set buttons for each device
        self.device_buttons = [
            (self.ui.pushButton_Set1, self.ui.lineEdit_Value1, 0x013),
            (self.ui.pushButton_Set2, self.ui.lineEdit_Value2, 0x023),
            (self.ui.pushButton_Set3, self.ui.lineEdit_Value3, 0x033),
            (self.ui.pushButton_Set4, self.ui.lineEdit_Value4, 0x043),
            (self.ui.pushButton_Set5, self.ui.lineEdit_Value5, 0x053),
            (self.ui.pushButton_Set6, self.ui.lineEdit_Value6, 0x063),
            (self.ui.pushButton_Set7, self.ui.lineEdit_Value7, 0x073),
            (self.ui.pushButton_Set8, self.ui.lineEdit_Value8, 0x083),
            (self.ui.pushButton_Set9, self.ui.lineEdit_Value9, 0x093),
            (self.ui.pushButton_Set10, self.ui.lineEdit_Value10, 0x0A3),
        ]
        for button, line_edit, can_id in self.device_buttons:
            button.clicked.connect(lambda _, le=line_edit, cid=can_id: self.set_device_value(le, cid))

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
            0x01B: self.ui.label_Value1,
            0x02B: self.ui.label_Value2,
            0x03B: self.ui.label_Value3,
            0x04B: self.ui.label_Value4,
            0x05B: self.ui.label_Value5,
            0x06B: self.ui.label_Value6,
            0x07B: self.ui.label_Value7,
            0x08B: self.ui.label_Value8,
            0x09B: self.ui.label_Value9,
            0x0AB: self.ui.label_Value10,
        }

        # Add dial mapping alongside the existing feedback labels mapping
        self.feedback_dials = {
            0x01B: self.ui.Dial_Value1,
            0x02B: self.ui.Dial_Value2,
            0x03B: self.ui.Dial_Value3,
            0x04B: self.ui.Dial_Value4,
            0x05B: self.ui.Dial_Value5,
            0x06B: self.ui.Dial_Value6,
            0x07B: self.ui.Dial_Value7,
            0x08B: self.ui.Dial_Value8,
            0x09B: self.ui.Dial_Value9,
            0x0AB: self.ui.Dial_Value10,
        }

        # Initialize current values
        self.current_values = {
            0x01B: 0, 0x02B: 0, 0x03B: 0, 0x04B: 0, 0x05B: 0,
            0x06B: 0, 0x07B: 0, 0x08B: 0, 0x09B: 0, 0x0AB: 0
        }

        # Thread for listening to CAN messages
        self.listener_thread = threading.Thread(target=self.listen_for_feedback, daemon=True)
        self.listening = False

        # CSV playback variables
        self.csv_thread = None
        self.csv_playing = False
        self.csv_data = []
        self.total_time = 2.0  # Default total time in seconds
        
        # CAN IDs for setting device values (matching the device_buttons mapping)
        self.device_can_ids = [0x013, 0x023, 0x033, 0x043, 0x053, 0x063, 0x073, 0x083, 0x093, 0x0A3]

    def send_can_message(self, can_id, data):
        """Helper function to send a CAN message."""
        try:
            message = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
            self.bus.send(message)
            rospy.loginfo(f"Sent CAN message: ID={can_id}, Data={data}")
        except can.CanError as e:
            rospy.logerr(f"CAN error: {e}")
        except Exception as e:
            rospy.logerr(f"Failed to send CAN message: {e}")

    def reset_devices(self):
        """Send a CAN message to reset all devices."""
        can_id = 0x000  # CAN ID
        data = [0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55]  # CAN data
        self.send_can_message(can_id, data)

    def mode_set(self):
        """Send two CAN messages to set mode and initialize devices."""
        can_id = 0x001  # CAN ID
        data = [0x02, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55]  # CAN data
        self.send_can_message(can_id, data)  # Mode choose - current mode
        rospy.sleep(0.5) # Wait for 0.5 seconds
        can_id = 0x003  # CAN ID
        data = [0x13, 0x88, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55] # CAN data
        self.send_can_message(can_id, data)  # Set all device current values to 0

    def online_check(self):
        """Load and play CSV current values over time."""
        if self.csv_playing:
            rospy.loginfo("CSV playback already running")
            return
            
        # Load CSV data
        csv_file_path = "/home/dz/Documents/CUHK/ROS_CAN/src/Case_1/scripts/coil_currents.csv"
        if not os.path.exists(csv_file_path):
            rospy.logerr(f"CSV file not found: {csv_file_path}")
            return
            
        try:
            self.load_csv_data(csv_file_path)
            rospy.loginfo(f"Loaded {len(self.csv_data)} data points from CSV")
            
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
                if i < len(self.device_can_ids):
                    can_id = self.device_can_ids[i]
                    self.send_device_current(can_id, current_mA)
            
            rospy.loginfo(f"Sent currents at time {csv_time:.1f}s: {currents}")
        
        # Stop all devices after playback
        rospy.loginfo("CSV playback finished, stopping devices")
        self.stop_devices()
        self.csv_playing = False
    
    def send_device_current(self, can_id, current_mA):
        """Send current value to a specific device."""
        try:
            [data_0, data_1] = current_to_can(current_mA)
            data = [0x13, 0x88, data_0, data_1, 0x55, 0x55, 0x55, 0x55]
            self.send_can_message(can_id, data)
        except Exception as e:
            rospy.logerr(f"Error sending current to device {can_id}: {e}")
    
    def set_csv_total_time(self, total_time_seconds):
        """Set the total time for CSV playback."""
        self.total_time = float(total_time_seconds)
        rospy.loginfo(f"Set CSV playback total time to {self.total_time} seconds")        

    def set_feedback_rate(self):
        """Send a CAN message to set feedback rate and start feedback listening."""
        can_id = 0x00A  # CAN ID
        data = [0x0A, 0x00, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55]
        self.send_can_message(can_id, data)

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
        
        can_id = 0x003  # CAN ID
        data = [0x13, 0x88, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55]
        self.send_can_message(can_id, data)

    def set_device_value(self, line_edit, can_id):
        """Send a CAN message to set a specific device value."""
        value = int(line_edit.text())
        [data_0, data_1] = current_to_can(value)
        data = [0x13, 0x88, data_0, data_1, 0x55, 0x55, 0x55, 0x55]
        self.send_can_message(can_id, data) 

    def listen_for_feedback(self):
        """Listen for CAN messages and update feedback labels."""
        while self.listening:
            try:
                # Receive a CAN message from the bus
                message = self.bus.recv(timeout=1)
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
                            sum_squares = sum(3 * (value/1000.0) * (value/1000.0)  for value in self.current_values.values())
                            QtCore.QMetaObject.invokeMethod(self.ui.Thermo, "setValue",
                                QtCore.Q_ARG(float, float(sum_squares)))
                        
                        except Exception as e:
                            rospy.logerr(f"Error processing CAN message: {e}")
            except Exception as e:
                rospy.logerr(f"Error receiving CAN message: {e}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CANApp()
    window.show()
    sys.exit(app.exec_())