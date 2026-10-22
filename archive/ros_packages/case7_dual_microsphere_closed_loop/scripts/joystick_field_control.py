#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joystick teleoperation controller for Case 7.

Instead of following a stored trajectory, this controller takes a desired
movement *direction* (from the gamepad right stick) and pushes the selected
microsphere(s) that way, reusing the exact soft-magnetic field/force model and
the bounded current solver from `ClosedLoopTrajectoryController`:

  - both spheres still share one ten-coil current vector,
  - particles are constrained to z = z_plane_m,
  - force law is F = k_mag grad(B^T B),
  - currents are solved by bounded coordinate search.

The desired force for a *selected* sphere is `joystick_force * stick_vector`
(with light velocity damping for stability); an *unselected* sphere is asked to
hold its captured position, so the solver tries to keep it still while moving the
others.

Two refinements keep teleoperation responsive and stable on the shared field:
  - the solver weights the *actively commanded* sphere(s) far above any *held*
    sphere, so your stick command is never diluted by the hold constraint, and
  - a real current-smoothness penalty stops the solver from hopping between
    near-equivalent ten-coil solutions every frame (which otherwise jitters the
    sphere that should stay put).

Target selection cycles red -> green -> both with the cycle button.
"""

import json
import threading
import time

import numpy as np
import rospy

from closed_loop_control import ClosedLoopTrajectoryController, DEFAULT_SETTINGS


# Selection cycle and the camera-topic key each name maps to (m1=red, m2=green).
TARGET_CYCLE = ("red", "green", "both")
TARGET_KEYS = {
    "red": ("m1",),
    "green": ("m2",),
    "both": ("m1", "m2"),
}


class JoystickFieldController(ClosedLoopTrajectoryController):
    def __init__(
        self,
        can_handler,
        send_device_current_callback,
        topic_m1="/sphere/red",
        topic_m2="/sphere/green",
        position_scale=1.0,
        control_dt_s=None,
        joystick_force_N=6e-4,
        move_damping_N_s_per_m=0.1,
        move_weight=1.0,
        hold_weight=0.3,
        solver_smoothness=None,
        solver_regularization=None,
        hold_force_N=None,
        hold_radius_m=None,
        hold_deadband_m=8e-4,
        hold_velocity_deadband_m_per_s=1e-4,
        hold_damping_N_s_per_m=0.35,
    ):
        super(JoystickFieldController, self).__init__(
            can_handler=can_handler,
            send_device_current_callback=send_device_current_callback,
            topic_m1=topic_m1,
            topic_m2=topic_m2,
            position_scale=position_scale,
            reference_duration_s=None,
            control_dt_s=control_dt_s,
        )
        self.joystick_force_N = max(0.0, float(joystick_force_N))
        # Light damping bleeds speed near the target without throttling the push.
        self.move_damping_N_s_per_m = max(0.0, float(move_damping_N_s_per_m))
        # Solver weights: the commanded sphere(s) must out-vote any held sphere.
        self.move_weight = max(1e-6, float(move_weight))
        self.hold_weight = max(0.0, float(hold_weight))
        # Optional teleop overrides for the shared-field solver penalties.
        self.solver_smoothness = (
            None if solver_smoothness is None else max(0.0, float(solver_smoothness))
        )
        self.solver_regularization = (
            None if solver_regularization is None else max(0.0, float(solver_regularization))
        )
        self.hold_force_N = None if hold_force_N is None else max(0.0, float(hold_force_N))
        self.hold_radius_m = None if hold_radius_m is None else max(1e-6, float(hold_radius_m))
        self.hold_deadband_m = max(0.0, float(hold_deadband_m))
        self.hold_velocity_deadband_m_per_s = max(0.0, float(hold_velocity_deadband_m_per_s))
        self.hold_damping_N_s_per_m = max(0.0, float(hold_damping_N_s_per_m))
        self.command_vec = np.zeros(2)
        self.hold_positions = {
            "m1": None,
            "m2": None,
        }
        self._target_index = 0

    # ------------------------------------------------------------------ setup
    def load_settings(self, filepath):
        """Reuse Case 7's calibrated physics settings (k_mag, z plane, limits)."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            self.settings = self._read_controller_settings(data.get("settings", {}))
            rospy.loginfo("Joystick controller loaded Case 7 settings from %s", filepath)
        except Exception as exc:
            self.settings = DEFAULT_SETTINGS.copy()
            rospy.logwarn(
                "Joystick controller could not load settings from %s (%s); using defaults",
                filepath,
                exc,
            )
        if self.control_dt_s is not None:
            self.settings["dt_s"] = min(0.2, max(0.001, float(self.control_dt_s)))
        self._apply_teleop_settings()

    def _apply_teleop_settings(self):
        """Tune the inherited settings for direct teleoperation feel.

        Trajectory following wants a heavily smoothed desired-force path; live
        joystick driving wants it to respond immediately. The per-coil current
        slew limit (`max_current_step_A`) still smooths the physical output, so
        relaxing the *desired-force* filter here costs no stability but removes
        the lag that made the stick feel sluggish.
        """
        self.settings["desired_force_filter_alpha"] = 0.7
        self.settings["max_desired_force_step_N"] = self.settings["max_desired_force_N"]
        if self.solver_smoothness is not None:
            # The current-jump penalty competes with the (quadratically small)
            # force gains near zero current; if it is comparable to the
            # force-error scale the solver can never energize the coils.
            force_scale = self.settings["max_desired_force_N"] ** 2
            if self.solver_smoothness > 0.01 * force_scale:
                rospy.logwarn(
                    "Case 7 joystick solver_smoothness=%.2e is large vs the force "
                    "scale (%.2e); this can stall current buildup. Consider <= %.2e.",
                    self.solver_smoothness,
                    force_scale,
                    0.01 * force_scale,
                )
            self.settings["smoothness"] = self.solver_smoothness
        if self.solver_regularization is not None:
            self.settings["force_regularization"] = self.solver_regularization

    # ------------------------------------------------------- joystick command
    def set_command(self, x, y):
        with self.lock:
            self.command_vec = np.array([float(x), float(y)], dtype=float)

    def current_target_name(self):
        with self.lock:
            return TARGET_CYCLE[self._target_index]

    def cycle_target(self):
        with self.lock:
            self._target_index = (self._target_index + 1) % len(TARGET_CYCLE)
            name = TARGET_CYCLE[self._target_index]
            self._refresh_hold_positions_locked(TARGET_KEYS[name])
        self._announce_target(name)
        return name

    def _refresh_hold_positions_locked(self, active_keys=()):
        active_keys = set(active_keys)
        for key, state in self.states.items():
            if key not in active_keys and state["position"] is not None:
                self.hold_positions[key] = state["position"].copy()

    def _announce_target(self, name):
        rospy.loginfo("Case 7 joystick controlled target = %s", name.upper())
        # Direct terminal print so the active target is always visible.
        print("[joystick] controlled target = %s" % name.upper(), flush=True)

    # ------------------------------------------------------------- run/stop
    def start_control(self):
        if self.control_running:
            rospy.loginfo("Case 7 joystick control is already running")
            return

        with self.lock:
            for key in self.states:
                self.states[key]["velocity"] = np.zeros(2)
                self.states[key]["last_desired_force"] = np.zeros(2)
                self.hold_positions[key] = None
            self.command_vec = np.zeros(2)

        self.currents = np.zeros(10)
        self.control_running = True
        self.control_thread = threading.Thread(target=self._joystick_control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        rospy.loginfo("Case 7 joystick control started")
        self._announce_target(self.current_target_name())

    def stop_control(self):
        if not self.control_running:
            return
        self.control_running = False
        if self.control_thread and threading.current_thread() != self.control_thread:
            self.control_thread.join(timeout=2.0)
        self._send_zero_currents()
        rospy.loginfo("Case 7 joystick control stopped")

    def _joystick_control_loop(self):
        if not self._wait_for_initial_pose(timeout=5.0):
            self.control_running = False
            self._send_zero_currents()
            return

        period = self.settings["dt_s"]
        command_hold_threshold = 1e-6
        with self.lock:
            for key, state in self.states.items():
                if state["position"] is not None:
                    self.hold_positions[key] = state["position"].copy()

        while self.control_running and not rospy.is_shutdown():
            loop_start = time.time()

            with self.lock:
                states = {
                    key: {
                        "position": value["position"].copy() if value["position"] is not None else None,
                        "velocity": value["velocity"].copy(),
                        "last_time": value["last_time"],
                    }
                    for key, value in self.states.items()
                }
                command = self.command_vec.copy()
                target_name = TARGET_CYCLE[self._target_index]
                hold_positions = {
                    key: value.copy() if value is not None else None
                    for key, value in self.hold_positions.items()
                }

            if not self._states_are_fresh(states):
                self._send_zero_currents()
                time.sleep(period)
                continue

            active_keys = TARGET_KEYS[target_name]
            command_magnitude = float(np.linalg.norm(command))
            moving_active_keys = active_keys if command_magnitude > command_hold_threshold else ()
            force_targets = []
            for key in ("m1", "m2"):
                position = states[key]["position"]
                if key in moving_active_keys:
                    desired_force = self._joystick_desired_force(command, states[key]["velocity"])
                    weight = self.move_weight
                else:
                    desired_force = self._hold_desired_force(
                        position,
                        states[key]["velocity"],
                        hold_positions[key],
                    )
                    weight = self.hold_weight
                desired_force = self._filtered_desired_force(key, desired_force)
                force_targets.append(
                    {"position": position, "desired_force": desired_force, "weight": weight}
                )

            solution = self._solve_currents_for_targets(force_targets, self.currents)
            limited_currents = self._limit_current_step(solution["currents"], self.currents)
            if not np.allclose(limited_currents, solution["currents"]):
                limited_score = self._score_currents(limited_currents, force_targets, self.currents)
                solution = {
                    "currents": limited_currents,
                    "forces": limited_score["forces"],
                    "error": limited_score["error"],
                }
            self.currents = solution["currents"]
            with self.lock:
                for key in moving_active_keys:
                    position = states[key]["position"]
                    if position is not None:
                        self.hold_positions[key] = position.copy()
            self._apply_currents(self.currents)

            rospy.loginfo_throttle(
                1.0,
                "Case 7 joystick target=%s cmd=(%.2f, %.2f) |I|=%.2f A err=%.2e N",
                target_name.upper(),
                command[0],
                command[1],
                float(np.linalg.norm(self.currents)),
                solution["error"],
            )

            sleep_time = period - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                rospy.logwarn_throttle(1.0, "Case 7 joystick loop overrun: %.4fs", -sleep_time)

        self.stop_control()

    def _joystick_desired_force(self, command, velocity):
        # Stick deflection maps straight to a push force in that direction; a
        # small velocity-damping term keeps it from overshooting. This is what
        # gives a crisp, "it follows my stick" feel, unlike a tiny target-speed
        # setpoint whose force collapses the moment the sphere starts to creep.
        max_force = self.settings["max_desired_force_N"]
        force = self.joystick_force_N * command - self.move_damping_N_s_per_m * velocity
        norm = float(np.linalg.norm(force))
        if max_force > 0.0 and norm > max_force:
            force = force * (max_force / norm)
        return force

    def _score_currents(self, currents, targets, previous_currents):
        # Same cost as the trajectory controller, but each target carries a
        # weight so the actively commanded sphere(s) out-vote a held sphere. The
        # smoothness term (raised for teleop) penalises hopping between
        # near-equivalent ten-coil solutions, which is what jitters a held sphere.
        force_error = 0.0
        combined_error = 0.0
        forces = []

        for target in targets:
            force = self._compute_soft_force(currents, target["position"])
            forces.append(force)
            diff = force[:2] - target["desired_force"]
            weight = float(target.get("weight", 1.0))
            force_error += weight * float(np.dot(diff, diff))
            combined_error += float(np.linalg.norm(diff))

        norm_penalty = float(np.dot(currents, currents))
        jump = currents - previous_currents
        smooth_penalty = float(np.dot(jump, jump))
        score = (
            force_error
            + self.settings["force_regularization"] * norm_penalty
            + self.settings["smoothness"] * smooth_penalty
        )

        return {"score": score, "forces": forces, "error": combined_error}

    def _hold_desired_force(self, position, velocity, hold_position):
        max_force = self.settings["max_desired_force_N"]
        hold_force = self.hold_force_N
        if hold_force is None:
            hold_force = 0.5 * self.joystick_force_N
        if max_force > 0.0:
            hold_force = min(hold_force, max_force)

        speed = float(np.linalg.norm(velocity))
        if hold_position is None:
            if speed <= self.hold_velocity_deadband_m_per_s:
                return np.zeros(2)
            force = -self.hold_damping_N_s_per_m * velocity
        else:
            hold_radius = self.hold_radius_m
            if hold_radius is None:
                hold_radius = 1.5 * self.settings["target_radius_m"]
            kp_hold = hold_force / hold_radius
            error = hold_position - position
            error_norm = float(np.linalg.norm(error))
            if error_norm <= self.hold_deadband_m:
                if speed <= self.hold_velocity_deadband_m_per_s:
                    return np.zeros(2)
                force = -self.hold_damping_N_s_per_m * velocity
            else:
                correction = error * ((error_norm - self.hold_deadband_m) / error_norm)
                force = kp_hold * correction - self.hold_damping_N_s_per_m * velocity

        norm = float(np.linalg.norm(force))
        if max_force > 0.0 and norm > max_force:
            force = force * (max_force / norm)
        return force
