#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case 7 joystick teleoperation GUI.

Same Qt panel as main2.py (Reset / Mode Set / Online Check / Stop and the ten
per-coil controls), but Online Check starts joystick teleoperation instead of a
stored trajectory.

Workflow:
  1. rosrun case7_dual_microsphere_closed_loop openmv_tracker.py   (positions)
  2. rosrun case7_dual_microsphere_closed_loop main_joystick.py
  3. In the GUI: Reset -> Mode Set -> Online Check
  4. Right stick moves the selected microsphere(s).
     Press A to switch the controlled target: red -> green -> both -> red ...
     The active target is printed on the terminal.

Optional params (rosrun ... main_joystick.py _param:=value):
  ~joy_device (str, "" = auto-detect)   ~joy_axis_x (int, 3)   ~joy_axis_y (int, 4)
  ~joy_button_cycle (int, 0 = A)        ~joy_deadzone (float, 0.15)
  ~joy_invert_x (bool, false)           ~joy_invert_y (bool, true, so up = +y)
  ~joystick_force_N (float, 6e-4)       ~move_damping_N_s_per_m (float, 0.1)
  ~move_weight (float, 1.0)             ~hold_weight (float, 0.3)
  ~solver_smoothness (float, off)       ~solver_regularization (float, JSON value)
  ~hold_force_N (float, 2e-4)           ~hold_radius_m (float, 1.5 * target_radius)
  ~hold_deadband_m (float, 8e-4)        ~hold_velocity_deadband_m_per_s (float, 1e-4)
  ~hold_damping_N_s_per_m (float, 0.35)
  ~position_scale (float, 1.0)

Tuning guide (the knobs that matter most):
  - sphere too weak/slow  -> raise ~joystick_force_N (push force at full stick;
    it is clamped to the JSON max_desired_force_N, so raise that too if needed).
  - held sphere drifts    -> raise ~hold_weight toward ~move_weight.
  - held sphere jitters    -> lower ~hold_weight (the solver stops fighting to
    perfectly null the held sphere; trades a little drift for far less jitter).
  - sphere overshoots     -> raise ~move_damping_N_s_per_m.
  NOTE: ~solver_smoothness is off by default. The soft-magnetic force is
  quadratic in current, so a non-trivial smoothness penalty stalls the solver
  near zero current and the coils never energize. Leave it off unless you know
  what you are doing.
"""

import os
import sys
import threading

import rospy
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QIntValidator
from PyQt5.Qwt import QwtDialSimpleNeedle

from basic_control import Ui_MainWindow
from can_lib import CANBusHandler, CANMessageBuilder, CANMessageDefs, can_to_current
from joystick_field_control import JoystickFieldController
from joystick_input import JoystickReader


CASE7_CONTROL_DT_S = 0.025
CASE7_CURRENT_LIMIT_A = 15.0
CASE7_CURRENT_LIMIT_MA = int(CASE7_CURRENT_LIMIT_A * 1000.0)


class CANApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(CANApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        rospy.init_node("case7_joystick_gui_node", anonymous=True)
        self.can_handler = CANBusHandler(
            channel="can0",
            bustype="socketcan",
            verbose_send=rospy.get_param("~verbose_can", False),
        )

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

        # Build the joystick controller up front so target selection works and
        # persists even before Online Check is pressed.
        self.closed_loop_controller = JoystickFieldController(
            can_handler=self.can_handler,
            send_device_current_callback=self.send_device_current,
            topic_m1=rospy.get_param("~topic_m1", "/sphere/red"),
            topic_m2=rospy.get_param("~topic_m2", "/sphere/green"),
            position_scale=rospy.get_param("~position_scale", 1.0),
            control_dt_s=rospy.get_param("~control_dt_s", CASE7_CONTROL_DT_S),
            joystick_force_N=rospy.get_param("~joystick_force_N", 6e-4),
            move_damping_N_s_per_m=rospy.get_param("~move_damping_N_s_per_m", 0.1),
            move_weight=rospy.get_param("~move_weight", 1.0),
            hold_weight=rospy.get_param("~hold_weight", 0.3),
            # Default OFF: the soft-magnetic force is quadratic in current, so any
            # appreciable smoothness penalty exceeds the tiny per-step force gains
            # near zero current and stalls the solver (coils never energize).
            solver_smoothness=rospy.get_param("~solver_smoothness", None),
            solver_regularization=rospy.get_param("~solver_regularization", None),
            hold_force_N=rospy.get_param("~hold_force_N", 2e-4),
            hold_radius_m=rospy.get_param("~hold_radius_m", None),
            hold_deadband_m=rospy.get_param("~hold_deadband_m", 8e-4),
            hold_velocity_deadband_m_per_s=rospy.get_param("~hold_velocity_deadband_m_per_s", 1e-4),
            hold_damping_N_s_per_m=rospy.get_param("~hold_damping_N_s_per_m", 0.35),
        )

        self.joystick_reader = JoystickReader(
            axis_x=rospy.get_param("~joy_axis_x", 3),
            axis_y=rospy.get_param("~joy_axis_y", 4),
            button_cycle=rospy.get_param("~joy_button_cycle", 0),
            deadzone=rospy.get_param("~joy_deadzone", 0.15),
            invert_x=rospy.get_param("~joy_invert_x", False),
            invert_y=rospy.get_param("~joy_invert_y", True),
            device=rospy.get_param("~joy_device", "") or None,
            on_vector=self._on_joystick_vector,
            on_cycle=self._on_joystick_cycle,
            logger=rospy.loginfo,
        )
        self.joystick_reader.start()

        rospy.loginfo("Case 7 joystick GUI ready")
        print(
            "[joystick] Right stick = move selected microsphere; A = switch target "
            "(red -> green -> both). Press Reset, Mode Set, Online Check to begin.",
            flush=True,
        )

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
        if self.closed_loop_controller.control_running:
            rospy.loginfo("Case 7 joystick control is already running")
            return

        trajectory_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "trajectory_case7_two_microspheres.json",
        )
        # The JSON is only used for its calibrated physics settings here; if it is
        # missing the controller falls back to defaults.
        self.closed_loop_controller.load_settings(trajectory_file_path)

        try:
            self.closed_loop_controller.start_control()
            rospy.loginfo("Case 7 joystick teleoperation started")
        except Exception as exc:
            rospy.logerr("Error starting Case 7 joystick control: %s", exc)
            import traceback
            rospy.logerr(traceback.format_exc())

    def _on_joystick_vector(self, x, y):
        if self.closed_loop_controller.control_running:
            self.closed_loop_controller.set_command(x, y)

    def _on_joystick_cycle(self):
        self.closed_loop_controller.cycle_target()

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
        if self.closed_loop_controller.control_running:
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

    def closeEvent(self, event):
        try:
            self.joystick_reader.stop()
            if self.closed_loop_controller.control_running:
                self.closed_loop_controller.stop_control()
        finally:
            super(CANApp, self).closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CANApp()
    window.show()
    sys.exit(app.exec_())
