#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import threading

import rospy
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QIntValidator
from PyQt5.Qwt import QwtDialSimpleNeedle

from basic_control import Ui_MainWindow
from can_lib import CANBusHandler, CANMessageBuilder, CANMessageDefs, can_to_current
from closed_loop_control import ClosedLoopTrajectoryController


CASE7_REFERENCE_DURATION_S = 20.0
CASE7_CONTROL_DT_S = 0.025
CASE7_CURRENT_LIMIT_A = 15.0
CASE7_CURRENT_LIMIT_MA = int(CASE7_CURRENT_LIMIT_A * 1000.0)


class CANApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(CANApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        rospy.init_node("case7_can_gui_node", anonymous=True)
        self.can_handler = CANBusHandler(channel="can0", bustype="socketcan")

        self.ui.pushButton_Reset.clicked.connect(self.reset_devices)
        self.ui.pushButton_ModeSet.clicked.connect(self.mode_set)
        self.ui.pushButton_OnlineCheck.clicked.connect(self.online_check)
        self.ui.pushButton_Feedback.clicked.connect(self.set_feedback_rate)
        self.ui.pushButton_Stop.clicked.connect(self.stop_devices)

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

        self._configure_manual_current_inputs()
        self._configure_dials()
        self.feedback_labels = {
            can_id: label for can_id, label in zip(
                CANMessageDefs.FEEDBACK_IDS,
                [
                    self.ui.label_Value1, self.ui.label_Value2, self.ui.label_Value3,
                    self.ui.label_Value4, self.ui.label_Value5, self.ui.label_Value6,
                    self.ui.label_Value7, self.ui.label_Value8, self.ui.label_Value9,
                    self.ui.label_Value10,
                ],
            )
        }
        self.feedback_dials = {
            can_id: dial for can_id, dial in zip(
                CANMessageDefs.FEEDBACK_IDS,
                [
                    self.ui.Dial_Value1, self.ui.Dial_Value2, self.ui.Dial_Value3,
                    self.ui.Dial_Value4, self.ui.Dial_Value5, self.ui.Dial_Value6,
                    self.ui.Dial_Value7, self.ui.Dial_Value8, self.ui.Dial_Value9,
                    self.ui.Dial_Value10,
                ],
            )
        }

        self.current_values = {can_id: 0 for can_id in CANMessageDefs.FEEDBACK_IDS}
        self.listener_thread = threading.Thread(target=self.listen_for_feedback)
        self.listener_thread.daemon = True
        self.listening = False
        self.closed_loop_controller = None

    def _configure_dials(self):
        for dial in [
            self.ui.Dial_Value1, self.ui.Dial_Value2, self.ui.Dial_Value3, self.ui.Dial_Value4,
            self.ui.Dial_Value5, self.ui.Dial_Value6, self.ui.Dial_Value7, self.ui.Dial_Value8,
            self.ui.Dial_Value9, self.ui.Dial_Value10,
        ]:
            dial.setScale(-CASE7_CURRENT_LIMIT_A, CASE7_CURRENT_LIMIT_A)
            dial.setOrigin(-90)
            dial.setScaleArc(-160, 160)
            dial.setScaleMaxMajor(6)
            dial.setScaleMaxMinor(5)
            dial.setReadOnly(True)
            needle = QwtDialSimpleNeedle(QwtDialSimpleNeedle.Arrow, True, QColor("red"), QColor("gray"))
            dial.setNeedle(needle)
            dial.setValue(0.0)

    def _configure_manual_current_inputs(self):
        validator = QIntValidator(-CASE7_CURRENT_LIMIT_MA, CASE7_CURRENT_LIMIT_MA, self)
        for _, line_edit, _ in self.device_buttons:
            line_edit.setValidator(validator)
            line_edit.setPlaceholderText("+/-15000 mA")

    def reset_devices(self):
        can_id, data = CANMessageBuilder.reset_all()
        self.can_handler.send_message(can_id, data)

    def mode_set(self):
        can_id, data = CANMessageBuilder.mode_set()
        self.can_handler.send_message(can_id, data)
        rospy.sleep(0.5)
        can_id, data = CANMessageBuilder.initialize()
        self.can_handler.send_message(can_id, data)

    def online_check(self):
        if self.closed_loop_controller is not None and self.closed_loop_controller.control_running:
            rospy.loginfo("Case 7 control is already running")
            return

        if self.closed_loop_controller is None:
            self.closed_loop_controller = ClosedLoopTrajectoryController(
                can_handler=self.can_handler,
                send_device_current_callback=self.send_device_current,
                topic_m1=rospy.get_param("~topic_m1", "/sphere/red"),
                topic_m2=rospy.get_param("~topic_m2", "/sphere/green"),
                position_scale=rospy.get_param("~position_scale", 1.0),
                reference_duration_s=rospy.get_param("~reference_duration_s", CASE7_REFERENCE_DURATION_S),
                control_dt_s=rospy.get_param("~control_dt_s", CASE7_CONTROL_DT_S),
            )

        trajectory_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "trajectory_case7_two_microspheres.json",
        )
        if not os.path.exists(trajectory_file_path):
            rospy.logerr("Case 7 trajectory file not found: %s", trajectory_file_path)
            return

        try:
            self.closed_loop_controller.prepare_trajectory(trajectory_file_path)
            self.closed_loop_controller.start_control()
            rospy.loginfo("Case 7 two-microsphere closed-loop control started")
        except Exception as exc:
            rospy.logerr("Error starting Case 7 control: %s", exc)
            import traceback
            rospy.logerr(traceback.format_exc())

    def send_device_current(self, device_index, current_mA):
        try:
            can_id, data = CANMessageBuilder.set_device_current(device_index, current_mA)
            self.can_handler.send_message(can_id, data)
        except Exception as exc:
            rospy.logerr("Error sending current to device %d: %s", device_index, exc)

    def set_feedback_rate(self):
        can_id, data = CANMessageBuilder.set_feedback_rate()
        self.can_handler.send_message(can_id, data)
        if not self.listening:
            self.listening = True
            if not self.listener_thread.is_alive():
                self.listener_thread.start()

    def stop_devices(self):
        if self.closed_loop_controller is not None and self.closed_loop_controller.control_running:
            self.closed_loop_controller.stop_control()
        can_id, data = CANMessageBuilder.initialize()
        self.can_handler.send_message(can_id, data)

    def set_device_value(self, line_edit, device_index):
        value = int(line_edit.text())
        value = max(-CASE7_CURRENT_LIMIT_MA, min(CASE7_CURRENT_LIMIT_MA, value))
        can_id, data = CANMessageBuilder.set_device_current(device_index, value)
        self.can_handler.send_message(can_id, data)

    def listen_for_feedback(self):
        while self.listening and not rospy.is_shutdown():
            message = self.can_handler.receive_message(timeout=1)
            if message is None or message.arbitration_id not in self.feedback_labels:
                continue
            try:
                current_value = can_to_current(message.data[:2])
                self.current_values[message.arbitration_id] = current_value

                label = self.feedback_labels[message.arbitration_id]
                QtCore.QMetaObject.invokeMethod(label, "setText", QtCore.Q_ARG(str, str(current_value)))

                dial = self.feedback_dials[message.arbitration_id]
                QtCore.QMetaObject.invokeMethod(
                    dial,
                    "setValue",
                    QtCore.Q_ARG(float, float(current_value / 1000.0)),
                )

                sum_squares = sum(3.0 * (value / 1000.0) ** 2 for value in self.current_values.values())
                QtCore.QMetaObject.invokeMethod(
                    self.ui.Thermo,
                    "setValue",
                    QtCore.Q_ARG(float, float(sum_squares)),
                )
            except Exception as exc:
                rospy.logerr("Error processing CAN feedback: %s", exc)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CANApp()
    window.show()
    sys.exit(app.exec_())
