#!/usr/bin/env python
"""
Closed-loop controller for 3-DOF electromagnetic robot arm.

Uses PID control with joint-angle feedback from OpenMV camera
to follow a predefined motion sequence via rotating magnetic fields.

Joint feedback (from openmv_tracker.py):
  /arm/theta1  (Float32, degrees)  ->  alpha  (joint 1)
  /arm/theta2  (Float32, degrees)  ->  beta   (joint 2)

Author: Da ZHAO
Date: 2026
"""

import numpy as np
import rospy
import time
import threading
from std_msgs.msg import Float32
from Kinematics import robot_arm_kinematics

# ============================================================
# Control constants  (mirrored from exp_control.py)
# ============================================================
# Field amplitude at the active motor
FIELD_AMP_MOTOR1 = 0.03   # T
FIELD_AMP_MOTOR2 = 0.03   # T
FIELD_AMP_MOTOR3 = 0.03   # T
ROTATION_FREQ      = 2                                    # Hz
ANGULAR_VELOCITY   = 2 * np.pi * ROTATION_FREQ               # rad/s
GEAR_RATIO         = 50
MOTOR_OUTPUT_SPEED = 360.0 * ROTATION_FREQ / GEAR_RATIO      # deg/s at output
CTRL_FREQ          = 20                                      # Hz
CTRL_PERIOD        = 1.0 / CTRL_FREQ                         # s

ALPHA_START, BETA_START = 52.1, -33.2
GRIPPER_DURATION = 5.0  # s

_s1_end = 42.63
_s2_end = 14.2
_s3_end = -33.2

CONTROL_STAGES = [
    # (name, active_motor, alpha_i, alpha_f, beta_i, beta_f, open_loop_dur)
    ("Close gripper",  3, ALPHA_START, ALPHA_START, BETA_START, BETA_START,
     GRIPPER_DURATION),
    ("Motor1 52->43",  1, ALPHA_START, _s1_end, BETA_START, BETA_START,
     abs(_s1_end - ALPHA_START) / MOTOR_OUTPUT_SPEED),
    ("Motor2 -33->14", 2, _s1_end, _s1_end, BETA_START, _s2_end,
     abs(_s2_end - BETA_START) / MOTOR_OUTPUT_SPEED),
    ("Motor1 43->-33", 1, _s1_end, _s3_end, _s2_end, _s2_end,
     abs(_s3_end - _s1_end) / MOTOR_OUTPUT_SPEED),
    ("Open gripper",   3, _s3_end, _s3_end, _s2_end, _s2_end,
     GRIPPER_DURATION),
]

# 10-coil dipole parameters  [mx, my, mz, rx, ry, rz]
COIL_PARAMS = [
    np.array([-13.04945069, -4.41557229,  6.47376799,  0.12129096,  0.00466922, -0.0174842 ]),
    np.array([ -5.10083416, 13.54294901,  7.85474539,  0.05834654, -0.11165548, -0.01850546]),
    np.array([  4.05088788, 14.23365818,  6.44760956, -0.05903076, -0.11020417, -0.01488244]),
    np.array([ 13.89011305, -0.06092074,  4.77365608, -0.12306086, -0.00085745, -0.01378161]),
    np.array([ 11.44363813, -9.40543896,  4.46367162, -0.06806179,  0.1024875,  -0.01397152]),
    np.array([ -9.00577939,-12.78905365,  5.98650851,  0.06473315,  0.10618968, -0.0151172 ]),
    np.array([  0.92820081,  8.54965337,  8.72298349, -0.00381254, -0.08845466, -0.08874662]),
    np.array([  8.7302819,  -4.90773115,  7.00109937, -0.07977306,  0.04481733, -0.08536032]),
    np.array([ -7.68962762, -6.83258326,  8.12112247,  0.07498008,  0.04542436, -0.08696975]),
    np.array([  2.35614001, -1.11370036, 14.00304846, -0.00722183,  0.00029277, -0.12482979]),
]
NUM_COILS = len(COIL_PARAMS)

# PID gains – tune for your hardware
PID_KP = 2
PID_KI = 0.05
PID_KD = 0.3
ANGLE_TOLERANCE = 1.5  # deg – joint stage ends when |error| < this

_MU0_4PI = 1e-7  # mu_0 / (4*pi)


# ============================================================
# Magnetic-field computation  (pure numpy, no sympy)
# ============================================================
def build_field_matrix_single(px, py, pz):
    """
    2 x NUM_COILS actuation matrix  A  so that  [Bx, By] = A @ currents.
    Computes each coil's (Bx, By) contribution per unit current at one
    motor position, via the magnetic-dipole model.
    """
    A = np.zeros((2, NUM_COILS))
    for j, params in enumerate(COIL_PARAMS):
        m  = params[:3]
        r0 = params[3:]
        d  = np.array([px - r0[0], py - r0[1], pz - r0[2]])
        r2 = d @ d + 1e-18
        r  = np.sqrt(r2)
        r3 = r2 * r
        r5 = r2 * r3
        mdot = m @ d
        A[0, j] = _MU0_4PI * (3 * d[0] * mdot / r5 - m[0] / r3)
        A[1, j] = _MU0_4PI * (3 * d[1] * mdot / r5 - m[1] / r3)
    return A


def solve_currents(A, b_target, max_current_A=15.0):
    """Minimum-norm least-squares, clipped to hardware limits (Amperes)."""
    currents, _, _, _ = np.linalg.lstsq(A, b_target, rcond=None)
    return np.clip(currents, -max_current_A, max_current_A)


# ============================================================
# Controller class
# ============================================================
class ClosedLoopController:
    """
    Closed-loop controller for a 3-DOF electromagnetic robot arm.

    Subscribes to /arm/theta1 (alpha) and /arm/theta2 (beta) for
    joint-angle feedback from the OpenMV camera.  Uses PID control
    for joint stages and open-loop timed rotation for the gripper.
    """

    def __init__(self, send_device_current, device_can_ids):
        """
        Parameters
        ----------
        send_device_current : callable(can_id: int, current_mA: int)
            Send a current command to one coil via CAN.
        device_can_ids : list[int]
            CAN IDs for the 10 coils, in order.
        """
        self.send_device_current = send_device_current
        self.device_can_ids = device_can_ids

        self.running = False
        self._thread = None
        self.motor_phase = {1: 0.0, 2: 0.0, 3: 0.0}
        self.active_motor = None
        self.current_stage = 0

        # Joint feedback (degrees), initialised to the start pose
        self.meas_alpha = ALPHA_START
        self.meas_beta  = BETA_START

        self._sub_t1 = rospy.Subscriber(
            '/arm/theta1', Float32, self._cb_theta1)
        self._sub_t2 = rospy.Subscriber(
            '/arm/theta2', Float32, self._cb_theta2)

    # ---- ROS callbacks ------------------------------------------------
    def _cb_theta1(self, msg):
        self.meas_alpha = msg.data

    def _cb_theta2(self, msg):
        self.meas_beta = msg.data

    # ---- Public API ---------------------------------------------------
    def run_next_stage(self):
        """Run the next stage in a background thread. Call once per click."""
        if self.running:
            rospy.loginfo("A stage is still running")
            return
        if self._thread is not None and self._thread.is_alive():
            rospy.loginfo("Previous stage thread still active")
            return

        if self.current_stage >= len(CONTROL_STAGES):
            rospy.loginfo("All stages completed. Resetting to stage 0.")
            self.current_stage = 0
            self.motor_phase = {1: 0.0, 2: 0.0, 3: 0.0}
            return

        self.running = True
        self._thread = threading.Thread(
            target=self._run_single_stage, daemon=True)
        self._thread.start()

    def stop(self):
        """Request the control loop to stop (non-blocking)."""
        self.running = False

    def reset(self):
        """Reset stage counter and motor phases."""
        self.running = False
        self.current_stage = 0
        self.motor_phase = {1: 0.0, 2: 0.0, 3: 0.0}
        self.active_motor = None
        rospy.loginfo("Controller reset to stage 0")

    # ---- Internal: run one stage --------------------------------------
    def _run_single_stage(self):
        idx = self.current_stage
        name, motor, a_i, a_f, b_i, b_f, ol_dur = CONTROL_STAGES[idx]

        rospy.loginfo(
            f"--- Stage {idx}: {name} (motor {motor}) ---  "
            f"alpha={self.meas_alpha:.1f}, beta={self.meas_beta:.1f}")

        if motor == 3:
            self._stage_gripper(name, ol_dur)
        else:
            self._stage_joint(motor, a_f, b_f, timeout=ol_dur * 5)

        self.running = False
        self.current_stage += 1

        if self.current_stage < len(CONTROL_STAGES):
            next_name = CONTROL_STAGES[self.current_stage][0]
            rospy.loginfo(
                f"=== Finished stage {idx}: {name}. "
                f"Waiting for stage {self.current_stage}: {next_name} ... ===")
        else:
            rospy.loginfo(
                f"=== Finished stage {idx}: {name}. "
                f"All stages completed! ===")

    # ---- Gripper stage (open-loop timed rotation) ---------------------
    def _stage_gripper(self, name, duration):
        self.active_motor = 3
        rot_sign = 1.0 if "Close" in name else -1.0
        rate = rospy.Rate(CTRL_FREQ)
        t0 = time.time()
        while self.running and (time.time() - t0) < duration:
            self.motor_phase[3] += rot_sign * ANGULAR_VELOCITY * CTRL_PERIOD
            self._send_field_step()
            rate.sleep()

    # ---- Joint stage (closed-loop PID) --------------------------------
    def _stage_joint(self, active_motor, target_alpha, target_beta, timeout):
        self.active_motor = active_motor
        rate     = rospy.Rate(CTRL_FREQ)
        integral = 0.0
        prev_err = None
        t0       = time.time()
        log_ctr  = 0

        while self.running and (time.time() - t0) < timeout:
            alpha_fb = self.meas_alpha
            beta_fb  = self.meas_beta

            err = ((target_alpha - alpha_fb) if active_motor == 1
                   else (target_beta - beta_fb))

            if abs(err) < ANGLE_TOLERANCE:
                rospy.loginfo(
                    f"  Motor {active_motor} target reached "
                    f"(err={err:.2f} deg)")
                break

            # PID
            integral = np.clip(integral + err * CTRL_PERIOD, -100, 100)
            deriv = (0.0 if prev_err is None
                     else (err - prev_err) / CTRL_PERIOD)

            vel = np.clip(
                PID_KP * err + PID_KI * integral + PID_KD * deriv,
                -MOTOR_OUTPUT_SPEED,
                 MOTOR_OUTPUT_SPEED)

            field_omega = np.radians(vel) * GEAR_RATIO
            self.motor_phase[active_motor] += field_omega * CTRL_PERIOD

            self._send_field_step()

            prev_err = err
            log_ctr += 1
            if log_ctr % CTRL_FREQ == 0:
                rospy.loginfo(
                    f"  alpha={alpha_fb:.1f} beta={beta_fb:.1f}  "
                    f"err={err:.2f}  vel_cmd={vel:.1f} deg/s")
            rate.sleep()

    # ---- Core: compute B-field target at active motor -> solve -> CAN --
    def _send_field_step(self):
        alpha = self.meas_alpha
        beta  = self.meas_beta

        x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(
            alpha, beta)

        alpha_rad = np.radians(alpha)
        beta_rad  = np.radians(beta)
        th1 = alpha_rad - np.pi / 2
        th2 = th1 + beta_rad

        am = self.active_motor
        if am == 1:
            amp = FIELD_AMP_MOTOR1
            ga  = self.motor_phase[1]
            px, py, pz = x1, y1, z1
        elif am == 2:
            amp = FIELD_AMP_MOTOR2
            ga  = self.motor_phase[2] + th1
            px, py, pz = x2, y2, z2
        else:
            amp = FIELD_AMP_MOTOR3
            ga  = self.motor_phase[3] + th2
            px, py, pz = x3, y3, z3

        b_target = np.array([amp * np.cos(ga), amp * np.sin(ga)])

        A = build_field_matrix_single(px, py, pz)
        currents_A = solve_currents(A, b_target)

        for i, cur in enumerate(currents_A):
            self.send_device_current(
                self.device_can_ids[i], int(cur * 1000))
