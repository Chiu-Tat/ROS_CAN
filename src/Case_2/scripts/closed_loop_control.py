#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Closed-loop trajectory control for three magnetic objects (Red, Green, Blue)
Uses feedback from openmv_tracker to correct trajectory tracking
"""

import numpy as np
import pandas as pd
import rospy
import time
import threading
import csv
from scipy.optimize import minimize
from geometry_msgs.msg import PointStamped
import sys
import os
import importlib.util

# Import functions from Case_2_control.py
control_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Case_2_control.py")
spec = importlib.util.spec_from_file_location("Case_2_control", control_path)
control_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control_module)

# Import the functions we need
load_trajectory_results = control_module.load_trajectory_results
generate_trajectory_kinematics = control_module.generate_trajectory_kinematics
precompute_force_mapping_coils_only = control_module.precompute_force_mapping_coils_only
compute_magnet_interaction_forces = control_module.compute_magnet_interaction_forces
unconstrained_objective = control_module.unconstrained_objective
unconstrained_gradient = control_module.unconstrained_gradient
magnets_config = control_module.magnets_config
magnet_mass = control_module.magnet_mass
control_frequency = control_module.control_frequency
current_limit = control_module.current_limit

class ClosedLoopTrajectoryController:
    """Closed-loop controller for 2-magnet trajectory following (Red, Blue)"""
    
    def __init__(self, can_handler, send_device_current_callback, Kp=100.0, Ki=0.0, Kd=0.0):
        """
        Initialize closed-loop controller
        
        Args:
            can_handler: CAN bus handler for sending messages
            send_device_current_callback: Function to send current to device (device_index, current_mA)
            Kp: Proportional gain (default: 100.0 N/m)
            Ki: Integral gain (default: 0.0 N/(m*s))
            Kd: Derivative gain (default: 0.0 N*s/m)
        """
        self.can_handler = can_handler
        self.send_device_current = send_device_current_callback
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        # PID state for each magnet
        self.pid_states = {
            'magnet1': {'integral': 0.0, 'prev_error': 0.0, 'last_time': None}, # Red
            'magnet3': {'integral': 0.0, 'prev_error': 0.0, 'last_time': None}  # Blue
        }
        
        # Trajectory data
        self.trajectory_data = None
        self.kinematics = None
        
        # Control state
        self.control_running = False
        self.control_thread = None
        
        # Real-time positions
        self.latest_positions = {
            'magnet1': None, # Red
            'magnet3': None  # Blue
        }
        self.position_lock = threading.Lock()
        
        # ROS subscribers
        self.subscribers = {}
        
        # Position recording for CSV
        self.recording_positions = False
        self.position_data = []  # List of (time, desired_m1_y, actual_m1_y, desired_m3_y, actual_m3_y)
        self.position_data_lock = threading.Lock()
        self.recording_start_time = None
        
    def prepare_trajectory(self, trajectory_file_path, use_csv=False):
        """
        Load trajectory data and pre-compute kinematics
        
        Args:
            trajectory_file_path: Path to trajectory file
            use_csv: If True, load from CSV. If False, load from JSON using Case_2_control
        """
        rospy.loginfo(f"Loading trajectory from {trajectory_file_path}")
        
        if use_csv:
            self.kinematics = self._load_kinematics_from_csv(trajectory_file_path)
            times = self.kinematics['time']
            rospy.loginfo(f"Trajectory loaded from CSV: {len(times)} points, Duration: {times[-1]:.3f} seconds")
        else:
            self.trajectory_data = load_trajectory_results(trajectory_file_path)
            
            # Generate kinematics at control frequency
            dt = 1.0 / control_frequency
            rospy.loginfo(f"Generating trajectory kinematics at {control_frequency} Hz...")
            self.kinematics = generate_trajectory_kinematics(self.trajectory_data, dt)
            
            times = self.kinematics['time']
            rospy.loginfo(f"Trajectory loaded from JSON: {len(times)} points, Duration: {times[-1]:.3f} seconds")
            
        rospy.loginfo("Trajectory preparation complete")

    def _load_kinematics_from_csv(self, filepath):
        """
        Load kinematics directly from sampled CSV
        CSV columns: time, x, y, z, segment_type
        Mapping: x -> magnet1_y, y -> magnet2_y, z -> magnet3_y
        """
        try:
            df = pd.read_csv(filepath)
            
            # Extract time and positions
            times = df['time'].values
            
            # Magnet positions (mapped as requested)
            # x column -> magnet1 (Red)
            # y column -> magnet2 (Green)
            # z column -> magnet3 (Blue)
            pos_m1 = df['x'].values
            pos_m2 = df['y'].values
            pos_m3 = df['z'].values
            
            # Calculate velocities (gradient)
            vel_m1 = np.gradient(pos_m1, times)
            vel_m2 = np.gradient(pos_m2, times)
            vel_m3 = np.gradient(pos_m3, times)
            
            # Calculate accelerations (gradient of velocity)
            acc_m1 = np.gradient(vel_m1, times)
            acc_m2 = np.gradient(vel_m2, times)
            acc_m3 = np.gradient(vel_m3, times)
            
            return {
                'time': times,
                'position': {
                    'magnet1_y': pos_m1,
                    'magnet2_y': pos_m2,
                    'magnet3_y': pos_m3
                },
                'velocity': {
                    'magnet1_y': vel_m1,
                    'magnet2_y': vel_m2,
                    'magnet3_y': vel_m3
                },
                'acceleration': {
                    'magnet1_y': acc_m1,
                    'magnet2_y': acc_m2,
                    'magnet3_y': acc_m3
                }
            }
            
        except Exception as e:
            rospy.logerr(f"Error loading trajectory from CSV: {e}")
            raise e
    
    def _sphere_callback(self, msg, magnet_key):
        """Generic callback for sphere position updates"""
        with self.position_lock:
            # OpenMV tracker coordinate mapping:
            # Tracker X -> Control Y (based on single magnet experience)
            self.latest_positions[magnet_key] = {
                'y': msg.point.x,
                'time': time.time()
            }

    def get_desired_state(self, t):
        """
        Get desired position, velocity, and acceleration at time t for all magnets
        
        Args:
            t: Current time (seconds)
            
        Returns:
            dict with 'magnet1', 'magnet3' states
        """
        if self.kinematics is None:
            return None
        
        times = self.kinematics['time']
        
        # Find closest time index
        if t < times[0]:
            idx = 0
        elif t >= times[-1]:
            idx = len(times) - 1
        else:
            idx = np.searchsorted(times, t)
            if idx > 0 and abs(times[idx-1] - t) < abs(times[idx] - t):
                idx = idx - 1
        
        result = {}
        for i, key in enumerate(['magnet1', 'magnet3']):
            pos_key = f'{key}_y'
            result[key] = {
                'position': self.kinematics['position'][pos_key][idx],
                'velocity': self.kinematics['velocity'][pos_key][idx],
                'acceleration': self.kinematics['acceleration'][pos_key][idx],
                'time': times[idx]
            }
        return result
    
    def compute_pid_force(self, magnet_key, desired_state, actual_y, current_time):
        """
        Compute PID control force for a single magnet
        """
        pid = self.pid_states[magnet_key]
        
        # Theoretical force: F = ma
        F_theory = magnet_mass * desired_state['acceleration']
        
        # Error: desired - actual
        error = desired_state['position'] - actual_y
        
        # Time step
        if pid['last_time'] is None:
            dt = 0.0
        else:
            dt = current_time - pid['last_time']
        
        # PID terms
        p_term = self.Kp * error
        
        if dt > 0:
            pid['integral'] += error * dt
            # Optional anti-windup could go here
        i_term = self.Ki * pid['integral']
        
        d_term = 0.0
        if dt > 0:
            d_term = self.Kd * (error - pid['prev_error']) / dt
            
        # Update state
        pid['prev_error'] = error
        pid['last_time'] = current_time
        
        F_total = F_theory + p_term + i_term + d_term
        
        # Clamp total force to [-0.1, 0.1] N
        F_total = max(min(F_total, 1.0), -1.0)
        
        return F_total
    
    def control_loop(self):
        """Main control loop running in separate thread"""
        if self.kinematics is None:
            rospy.logerr("Trajectory not prepared. Call prepare_trajectory() first.")
            return
        
        rospy.loginfo("Starting closed-loop trajectory control (2 magnets: Red, Blue)...")
        
        start_time = time.time()
        times = self.kinematics['time']
        total_duration = times[-1]
        
        # Initial guess for optimization
        currents_guess = np.zeros(10)
        
        while self.control_running and not rospy.is_shutdown():
            loop_start = time.time()
            current_traj_time = loop_start - start_time
            
            if current_traj_time >= total_duration:
                rospy.loginfo(f"Trajectory completed after {current_traj_time:.3f} seconds")
                break
            
            # 1. Get Desired State
            desired_states = self.get_desired_state(current_traj_time)
            if desired_states is None:
                time.sleep(0.001)
                continue
                
            # 2. Get Actual Positions & Build Magnet State
            magnets_state_list = []
            actual_positions = {}
            
            with self.position_lock:
                for i, key in enumerate(['magnet1', 'magnet3']):
                    # Map key to config index (0=Red/Mag1, 1=Blue/Mag3)
                    
                    # Check if we have fresh data (e.g. within 0.5s)
                    has_data = False
                    if self.latest_positions[key] is not None:
                        age = loop_start - self.latest_positions[key]['time']
                        if age < 0.5:
                            has_data = True
                            actual_y = self.latest_positions[key]['y']
                    
                    if not has_data:
                        # Fallback to desired position
                        actual_y = desired_states[key]['position']
                        # rospy.logwarn_throttle(1.0, f"No feedback for {key}, using desired: {actual_y:.4f}")
                    
                    actual_positions[key] = actual_y
                    
                    # Build state dict for this magnet
                    magnets_state_list.append({
                        'X': magnets_config[i]['X'],
                        'Y': actual_y,
                        'Z': magnets_config[i]['Z'],
                        'm': magnets_config[i]['m'],
                        'alpha': magnets_config[i]['alpha'],
                        'beta': magnets_config[i]['beta']
                    })

            # 3. Compute Interaction Forces
            interaction_forces = compute_magnet_interaction_forces(magnets_state_list)
            
            # 4. Compute Total Desired Forces (PID included) & Required Coil Forces
            total_desired_forces = []
            for i, key in enumerate(['magnet1', 'magnet3']):
                f_pid = self.compute_pid_force(
                    key, 
                    desired_states[key], 
                    actual_positions[key], 
                    current_traj_time
                )
                total_desired_forces.append(f_pid)
            
            total_desired_forces = np.array(total_desired_forces)
            required_coil_forces = total_desired_forces - interaction_forces
            
            # 5. Optimization
            # Precompute coil mapping
            A_matrix = precompute_force_mapping_coils_only(magnets_state_list)
            
            result = minimize(
                lambda x: unconstrained_objective(x, A_matrix, required_coil_forces),
                currents_guess,
                method='L-BFGS-B',
                jac=lambda x: unconstrained_gradient(x, A_matrix, required_coil_forces),
                options={'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 50} # Reduced maxiter for realtime
            )
            
            if result.success:
                currents_guess = result.x # Warm start
                
            # 6. Send Currents
            for i, current_A in enumerate(result.x):
                current_mA = int(current_A * 1000)
                self.send_device_current(i, current_mA)
                
            # 7. Record positions to CSV
            if self.recording_positions:
                with self.position_data_lock:
                    self.position_data.append((
                        current_traj_time,
                        desired_states['magnet1']['position'],
                        actual_positions['magnet1'],
                        desired_states['magnet3']['position'],
                        actual_positions['magnet3']
                    ))
            
            # 8. Logging (Throttle to avoid flooding)
            if int(current_traj_time * 10) % 10 == 0: # ~Every 1s
                err1 = desired_states['magnet1']['position'] - actual_positions['magnet1']
                err3 = desired_states['magnet3']['position'] - actual_positions['magnet3']
                rospy.loginfo(
                    f"t={current_traj_time:.2f}s | Err(mm): R={err1*1000:.1f}, B={err3*1000:.1f} | "
                    f"IntF(N): {interaction_forces[0]:.1e}, {interaction_forces[1]:.1e}"
                )
            
            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / control_frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        rospy.loginfo("Closed-loop control stopped")
        # Stop currents
        for i in range(10):
            self.send_device_current(i, 0)
        self.control_running = False
        
        # Stop recording and save to CSV
        self.stop_position_recording()

    def start_control(self):
        """Start closed-loop control"""
        if self.control_running:
            return
        
        if self.kinematics is None:
            rospy.logerr("Trajectory not prepared.")
            return
            
        # Subscribe to spheres
        topics = {
            'magnet1': '/sphere/red',
            'magnet3': '/sphere/blue'
        }
        
        for key, topic in topics.items():
            if key not in self.subscribers:
                # Use default argument in lambda to capture key value
                cb = lambda msg, k=key: self._sphere_callback(msg, k)
                self.subscribers[key] = rospy.Subscriber(topic, PointStamped, cb)
                rospy.loginfo(f"Subscribed to {topic} for {key}")

        # Reset state
        with self.position_lock:
            for k in self.latest_positions:
                self.latest_positions[k] = None
        
        for k in self.pid_states:
            self.pid_states[k] = {'integral': 0.0, 'prev_error': 0.0, 'last_time': None}
            
        self.control_running = True
        
        # Start position recording
        self.start_position_recording()
        
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
    
    def start_position_recording(self):
        """Start recording magnet positions and time to CSV"""
        with self.position_data_lock:
            self.position_data = []
            self.recording_positions = True
            self.recording_start_time = time.time()
        rospy.loginfo("Started recording magnet positions to CSV")
    
    def stop_position_recording(self):
        """Stop recording and save magnet positions to CSV file"""
        if not self.recording_positions:
            return
        
        self.recording_positions = False
        rospy.loginfo("Stopped recording magnet positions")
        
        # Save to CSV file
        csv_output_path = "/home/dz/Documents/CUHK/ROS_CAN/src/Case_2/scripts/magnet_positions_recorded.csv"
        try:
            with self.position_data_lock:
                with open(csv_output_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow([
                        'Time(s)',
                        'Desired_Magnet1_Y(m)', 'Actual_Magnet1_Y(m)',
                        'Desired_Magnet3_Y(m)', 'Actual_Magnet3_Y(m)'
                    ])
                    # Write data
                    for row in self.position_data:
                        writer.writerow(row)
            
            rospy.loginfo(f"Saved {len(self.position_data)} magnet position data points to {csv_output_path}")
        except Exception as e:
            rospy.logerr(f"Error saving magnet positions to CSV: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
        
    def stop_control(self):
        """Stop closed-loop control"""
        self.control_running = False
        if self.control_thread is not None:
            self.control_thread.join(timeout=2.0)
        # Stop recording and save to CSV
        self.stop_position_recording()
