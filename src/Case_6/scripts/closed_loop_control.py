
import rospy
import numpy as np
import threading
import time
import json
import csv
import os
import sympy as sp
from scipy.optimize import minimize
from geometry_msgs.msg import PoseStamped, PointStamped, Point
import math

# Configuration matches simulation
CURRENT_LIMIT = 15  # Amperes
B_MAGNITUDE = 0.02  # Tesla
FORCE_LIMIT = 0.1  # Newton
MAGNET_MASS = 0.0015  # kg
CONTROL_FREQUENCY = 10  # Hz

# PID Gains (from simulation)
KP_POS = 0.2
KI_POS = 0.2
KD_POS = 0.6
KP_THETA = 0.01
KD_THETA = 0.01

# Velocity Filter
ALPHA_VEL = 0.05  # Very aggressive filter (mostly previous value)

# Soft Start
STARTUP_RAMP_TIME = 3.0  # Seconds to ramp up control authority

# Pure Pursuit
LOOKAHEAD_DISTANCE = 0.005

# Magnet configuration
MAGNET_CONFIG = {
    'm': 0.12,  # magnetic moment
    'Z': 0.0,   # height
}

FIXED_MAGNET_CONFIG = {
    'X': 0.02,
    'Y': 0.0,
    'Z': 0.0,
    'm': 0.145,
    'alpha': np.pi/2,
    'beta': np.pi/2
}

# Coil parameters (from simulation)
PARAMS_LIST = [
    np.array([-13.04945069, -4.41557229, 6.47376799, 0.12129096, 0.00466922, -0.0174842]),
    np.array([-5.10083416, 13.54294901, 7.85474539, 0.05834654, -0.11165548, -0.01850546]),
    np.array([4.05088788, 14.23365818, 6.44760956, -0.05903076, -0.11020417, -0.01488244]),
    np.array([13.89011305, -0.06092074, 4.77365608, -0.12306086, -0.00085745, -0.01378161]),
    np.array([11.44363813, -9.40543896, 4.46367162, -0.06806179, 0.1024875, -0.01397152]),
    np.array([-9.00577939, -12.78905365, 5.98650851, 0.06473315, 0.10618968, -0.0151172]),
    np.array([0.92820081, 8.54965337, 8.72298349, -0.00381254, -0.08845466, -0.08874662]),
    np.array([8.7302819, -4.90773115, 7.00109937, -0.07977306, 0.04481733, -0.08536032]),
    np.array([-7.68962762, -6.83258326, 8.12112247, 0.07498008, 0.04542436, -0.08696975]),
    np.array([2.35614001, -1.11370036, 14.00304846, -0.00722183, 0.00029277, -0.12482979])
]

class ClosedLoopTrajectoryController:
    def __init__(self, can_handler, send_device_current_callback, **kwargs):
        self.can_handler = can_handler
        self.send_current = send_device_current_callback
        
        self.control_running = False
        self.control_thread = None
        self.lock = threading.Lock()
        
        # State
        self.current_state = {
            'X': None, 'Y': None, 'theta': None, 'time': None
        }
        self.last_valid_state = None
        self.trajectory_data = None
        
        # Initialize Physics Model (SymPy)
        self._init_physics_model()
        
        # Results logging
        self.results_data = []
        self.coil_currents_data = []
        
        # Subscribe to OpenMV topic
        # Note: In main2.py, rospy.init_node is already called.
        self.pose_sub = rospy.Subscriber('/sphere/green', PoseStamped, self._pose_callback)
        self.lookahead_pub = rospy.Publisher('/control/lookahead_point', PointStamped, queue_size=1)
        
        # Integral Error State
        self.integral_error = {'x': 0.0, 'y': 0.0}
        
    def _init_physics_model(self):
        """Initialize SymPy lambdified functions for magnetic field calculations"""
        rospy.loginfo("Initializing physics model...")
        m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z = sp.symbols('m0 m1 m2 r0_0 r0_1 r0_2 X Y Z')
        mu0 = 4 * sp.pi * 1e-7
        
        dx = X - r0_0
        dy = Y - r0_1
        dz = Z - r0_2
        r_sq = dx**2 + dy**2 + dz**2
        r_safe = sp.sqrt(r_sq + 1.0e-5)
        
        dot_product = m0 * dx + m1 * dy + m2 * dz
        
        # B components
        model_Bx = (mu0 / (4 * sp.pi)) * (3 * dx * dot_product / r_safe**5 - m0 / r_safe**3)
        model_By = (mu0 / (4 * sp.pi)) * (3 * dy * dot_product / r_safe**5 - m1 / r_safe**3)
        model_Bz = (mu0 / (4 * sp.pi)) * (3 * dz * dot_product / r_safe**5 - m2 / r_safe**3)
        
        # Gradients
        grad_Bx_x = sp.diff(model_Bx, X)
        grad_Bx_y = sp.diff(model_Bx, Y)
        grad_By_x = sp.diff(model_By, X)
        grad_By_y = sp.diff(model_By, Y)
        
        # Lambdify
        self.dipole_model_Bx = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx, 'numpy')
        self.dipole_model_By = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By, 'numpy')
        self.dipole_model_Bz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bz, 'numpy')
        self.dipole_grad_Bx_x = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), grad_Bx_x, 'numpy')
        self.dipole_grad_Bx_y = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), grad_Bx_y, 'numpy')
        self.dipole_grad_By_x = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), grad_By_x, 'numpy')
        self.dipole_grad_By_y = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), grad_By_y, 'numpy')
        rospy.loginfo("Physics model initialized.")

    def _pose_callback(self, msg):
        """Update current state from ROS topic"""
        with self.lock:
            # Extract Yaw from Quaternion
            q = msg.pose.orientation
            # yaw = atan2(2(wz), 1-2(z^2)) or just standard conversion
            # Using simple calculation matching main2.py/simulation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # Or use the simplified 2*atan2(z, w) if axis is purely Z (from openmv_tracker)
            # But standard conversion is safer.
            
            self.current_state = {
                'X': msg.pose.position.x,
                'Y': msg.pose.position.y,
                'theta': yaw,
                'time': time.time()
            }
            self.last_valid_state = self.current_state.copy()

    def prepare_trajectory(self, filepath):
        """Load trajectory from JSON"""
        rospy.loginfo(f"Loading trajectory from {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        trajectory = np.array(data['trajectory'])
        self.trajectory_data = {
            'x': trajectory[:, 0],
            'y': trajectory[:, 1],
            'theta': trajectory[:, 2],
            'time': trajectory[:, 3]
        }
        rospy.loginfo(f"Trajectory loaded with {len(trajectory)} points.")

    def start_control(self):
        if self.control_running:
            return
            
        self.control_running = True
        self.results_data = []
        self.coil_currents_data = []
        
        # Reset Integral Error
        self.integral_error = {'x': 0.0, 'y': 0.0}
        
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        rospy.loginfo("Control thread started.")

    def stop_control(self):
        if not self.control_running:
            return
            
        self.control_running = False
        if self.control_thread and threading.current_thread() != self.control_thread:
            self.control_thread.join(timeout=2.0)
            
        # Reset currents
        for i in range(10):
            self.send_current(i, 0)
            
        self._save_results()
        rospy.loginfo("Control stopped and results saved.")

    def _control_loop(self):
        if self.trajectory_data is None:
            rospy.logerr("No trajectory loaded!")
            self.control_running = False
            return
            
        # Initial wait for pose
        rospy.loginfo("Waiting for pose data...")
        timeout = 5.0
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            with self.lock:
                if self.current_state['X'] is not None:
                    break
            time.sleep(0.1)
            
        if self.current_state['X'] is None:
            rospy.logerr("Timeout waiting for pose data.")
            self.control_running = False
            return
            
        # Initialization
        
        # Find closest point on trajectory to start from, instead of index 0
        # This prevents "racing" if we start mid-trajectory
        if self.trajectory_data is not None:
            last_traj_index = self._find_closest_point_index(self.current_state['X'], self.current_state['Y'])
            rospy.loginfo(f"Starting at trajectory index {last_traj_index}/{len(self.trajectory_data['x'])}")
        else:
            last_traj_index = 0
            
        n_points = len(self.trajectory_data['time'])
        start_time = time.time()
        loop_period = 1.0 / CONTROL_FREQUENCY
        
        # Previous state for velocity estimation
        prev_state = self.current_state.copy()
        prev_time = prev_state['time']
        
        currents_guess = np.zeros(10)
        
        while self.control_running and last_traj_index < n_points - 1:
            loop_start = time.time()
            elapsed_time = loop_start - start_time
            
            # 1. Get State
            with self.lock:
                state = self.current_state.copy()
                
            # DEBUG: Print current state to verify angle
            # rospy.loginfo_throttle(0.5, f"State: X={state['X']:.3f}, Y={state['Y']:.3f}, Theta={np.degrees(state['theta']):.1f} deg")
                
            # Check for stale data (timeout > 0.5s)
            if time.time() - state['time'] > 0.5:
                rospy.logwarn("Stale pose data! Holding...")
                # Safety: maybe set currents to zero? For now, continue but warn.
                pass
                
            # Velocity Estimation
            dt_meas = state['time'] - prev_time
            if dt_meas > 0.001:
                meas_vx = (state['X'] - prev_state['X']) / dt_meas
                meas_vy = (state['Y'] - prev_state['Y']) / dt_meas
                
                d_theta = state['theta'] - prev_state['theta']
                d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta))
                meas_omega = d_theta / dt_meas
            else:
                meas_vx, meas_vy, meas_omega = 0, 0, 0
                
            prev_state = state.copy()
            prev_time = state['time']
            
            # 2. Pure Pursuit
            lookahead_x, lookahead_y, last_traj_index = self._get_lookahead_point(
                self.trajectory_data['x'], self.trajectory_data['y'],
                state['X'], state['Y'],
                LOOKAHEAD_DISTANCE, last_traj_index
            )
            
            # Publish lookahead point
            lookahead_msg = PointStamped()
            lookahead_msg.header.stamp = rospy.Time.now()
            lookahead_msg.header.frame_id = "map" # or whatever frame your system uses
            lookahead_msg.point = Point(x=lookahead_x, y=lookahead_y, z=0.0)
            self.lookahead_pub.publish(lookahead_msg)
            
            dx_lookahead = lookahead_x - state['X']
            dy_lookahead = lookahead_y - state['Y']
            
            # Integral Error Update
            dt = dt_meas if dt_meas > 0 else loop_period
            self.integral_error['x'] += dx_lookahead * dt
            self.integral_error['y'] += dy_lookahead * dt
            
            # Anti-windup (simple clamping of integral term contribution)
            # Limit integral contribution to e.g. 50% of FORCE_LIMIT
            max_integral_force = FORCE_LIMIT * 0.5
            
            F_integral_x = KI_POS * self.integral_error['x']
            F_integral_y = KI_POS * self.integral_error['y']
            
            # Clamp integral force components
            if abs(F_integral_x) > max_integral_force:
                F_integral_x = np.sign(F_integral_x) * max_integral_force
                # Back-calculate accumulated error to prevent windup
                self.integral_error['x'] = F_integral_x / KI_POS
                
            if abs(F_integral_y) > max_integral_force:
                F_integral_y = np.sign(F_integral_y) * max_integral_force
                self.integral_error['y'] = F_integral_y / KI_POS

            F_control_x = KP_POS * dx_lookahead + F_integral_x - KD_POS * meas_vx
            F_control_y = KP_POS * dy_lookahead + F_integral_y - KD_POS * meas_vy
            
            # Soft Start Ramp
            if elapsed_time < STARTUP_RAMP_TIME:
                ramp_factor = elapsed_time / STARTUP_RAMP_TIME
                F_control_x *= ramp_factor
                F_control_y *= ramp_factor
            
            # Clamp Control Force
            F_control_mag = np.sqrt(F_control_x**2 + F_control_y**2)
            if F_control_mag > FORCE_LIMIT * 2:
                scale = (FORCE_LIMIT * 2) / F_control_mag
                F_control_x *= scale
                F_control_y *= scale
                
            # 3. Orientation Control
            # Change: Use Pure Pursuit heading (angle to lookahead point) instead of trajectory theta.
            # This is required because we constrained lateral force to 0. 
            # To correct position error, the magnet MUST face the target point.
            
            # ref_theta = self.trajectory_data['theta'][last_traj_index] # Old: Trajectory Tangent
            ref_theta = np.arctan2(dy_lookahead, dx_lookahead) # New: Pure Pursuit Heading
            
            err_theta = ref_theta - state['theta']
            err_theta = np.arctan2(np.sin(err_theta), np.cos(err_theta))
            
            # DEBUG: Print orientation errors
            rospy.loginfo_throttle(0.2, f"Theta: {np.degrees(state['theta']):.1f}, Ref: {np.degrees(ref_theta):.1f}, Err: {np.degrees(err_theta):.1f}")
            
            theta_correction = KP_THETA * err_theta - KD_THETA * meas_omega
            max_angle_dev = 60 * np.pi / 180.0
            theta_correction = np.clip(theta_correction, -max_angle_dev, max_angle_dev)
            
            target_heading = ref_theta + theta_correction
            
            # REMOVED Movement Constraint Check
            # The movement direction estimate is too noisy at low speeds and causes field flips.
            # We will rely purely on the Pure Pursuit heading (ref_theta) which is stable.
            
            target_Bx = B_MAGNITUDE * np.cos(target_heading)
            target_By = B_MAGNITUDE * np.sin(target_heading)
            
            # 4. Force Control (Pushing Only)
            # We ignore interaction forces as requested.
            # We also constrain the lateral force (body frame y) to 0.
            # Only apply force in the direction of the magnet's orientation (body frame x).
            
            theta_curr = state['theta']
            cos_th = np.cos(theta_curr)
            sin_th = np.sin(theta_curr)
            
            # Project PID control force onto the magnet's longitudinal axis (Body Frame X)
            f_push = F_control_x * cos_th + F_control_y * sin_th
            
            # Transform purely longitudinal force back to World Frame
            # F_world_x = f_push * cos(theta)
            # F_world_y = f_push * sin(theta)
            target_Fx_coil = f_push * cos_th
            target_Fy_coil = f_push * sin_th
            
            # 5. Optimization
            magnet_state_est = {
                'X': state['X'],
                'Y': state['Y'],
                'Z': MAGNET_CONFIG['Z'],
                'm': MAGNET_CONFIG['m'],
                'alpha': state['theta'],
                'beta': np.pi/2
            }
            mapping = self._precompute_field_and_force_mapping(magnet_state_est)
            
            if np.any(np.isnan(mapping['Bx'])):
                 rospy.logwarn("NaN in mapping, skipping step")
                 applied_currents = np.zeros(10)
            else:
                try:
                    res = minimize(
                        lambda x: self._unconstrained_objective(x, mapping, target_Bx, target_By, target_Fx_coil, target_Fy_coil),
                        currents_guess,
                        method='L-BFGS-B',
                        jac=lambda x: self._unconstrained_gradient(x, mapping, target_Bx, target_By, target_Fx_coil, target_Fy_coil),
                        options={'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 500}
                    )
                    applied_currents = res.x
                    currents_guess = applied_currents # Warm start
                except Exception as e:
                    rospy.logerr(f"Optimization failed: {e}")
                    applied_currents = np.zeros(10)

            # Debug output for optimization
            if np.linalg.norm(applied_currents) < 0.1:
                rospy.logwarn_throttle(1.0, f"Low current output: {np.linalg.norm(applied_currents):.4f} A. Targets: Fx={target_Fx_coil:.4f}, Fy={target_Fy_coil:.4f}, Bx={target_Bx:.4f}, By={target_By:.4f}")
            else:
                 rospy.loginfo_throttle(1.0, f"Currents: {np.array2string(applied_currents, precision=2, suppress_small=True)}")
            
            # 6. Apply Currents
            # Clamp to limits just in case
            applied_currents = np.clip(applied_currents, -CURRENT_LIMIT, CURRENT_LIMIT)
            
            # Send to devices (convert to mA)
            for i in range(10):
                self.send_current(i, int(applied_currents[i] * 1000))
                
            # 7. Log Data
            # Format: Time, Desired_X, Desired_Y, Desired_Theta, Actual_X, Actual_Y, Actual_Theta
            # Reference trajectory at current lookahead or time?
            # Plot script expects Desired to be the Reference trajectory point, usually.
            # We can log the lookahead point or the trajectory point at current index.
            # Let's log the trajectory point corresponding to current index
            ref_x = self.trajectory_data['x'][last_traj_index]
            ref_y = self.trajectory_data['y'][last_traj_index]
            ref_th = self.trajectory_data['theta'][last_traj_index]
            
            self.results_data.append([
                elapsed_time,
                ref_x, ref_y, ref_th,
                state['X'], state['Y'], state['theta']
            ])
            
            self.coil_currents_data.append([elapsed_time] + list(applied_currents))
            
            # Sleep to maintain frequency
            proc_duration = time.time() - loop_start
            sleep_time = loop_period - proc_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                rospy.logwarn_throttle(1.0, f"Control loop overrun: {proc_duration:.4f}s")
                
        self.stop_control()

    def _find_closest_point_index(self, x_robot, y_robot):
        # Brute force search for closest point index
        dists = (self.trajectory_data['x'] - x_robot)**2 + (self.trajectory_data['y'] - y_robot)**2
        return np.argmin(dists)

    def _get_lookahead_point(self, x_traj, y_traj, x_robot, y_robot, lookahead_dist, last_index):
        for i in range(last_index, len(x_traj)):
            dx = x_traj[i] - x_robot
            dy = y_traj[i] - y_robot
            dist = np.sqrt(dx**2 + dy**2)
            if dist >= lookahead_dist:
                return x_traj[i], y_traj[i], i
        return x_traj[-1], y_traj[-1], len(x_traj)-1

    def _calculate_interaction_force(self, moving_magnet, fixed_magnet):
        # Dipole-Dipole Interaction Force
        # Based on simulation code
        mu0 = 4 * np.pi * 1e-7
        
        r_vec = np.array([
            moving_magnet['X'] - fixed_magnet['X'],
            moving_magnet['Y'] - fixed_magnet['Y'],
            moving_magnet['Z'] - fixed_magnet['Z']
        ])
        dist = np.linalg.norm(r_vec)
        
        if dist < 0.005: dist = 0.005 # clamp
        
        r_hat = r_vec / dist
        
        # Helper to get moment vector
        def get_m_vec(mag_state):
            m = mag_state['m']
            a = mag_state['alpha']
            b = mag_state['beta']
            return np.array([
                m * np.sin(b) * np.cos(a),
                m * np.sin(b) * np.sin(a),
                m * np.cos(b)
            ])
            
        m1 = get_m_vec(moving_magnet)
        m2 = get_m_vec(fixed_magnet)
        
        m1_dot_r = np.dot(m1, r_hat)
        m2_dot_r = np.dot(m2, r_hat)
        m1_dot_m2 = np.dot(m1, m2)
        
        F = (3 * mu0 / (4 * np.pi * dist**4)) * (
            m1_dot_r * m2 +
            m2_dot_r * m1 +
            m1_dot_m2 * r_hat -
            5 * m1_dot_r * m2_dot_r * r_hat
        )
        
        # Clamp
        f_mag = np.linalg.norm(F)
        if f_mag > 5.0:
            F = F / f_mag * 5.0
            
        return F

    def _precompute_field_and_force_mapping(self, magnet_state):
        n_coils = 10
        Bx_vec = np.zeros(n_coils)
        By_vec = np.zeros(n_coils)
        Bz_vec = np.zeros(n_coils)
        Fx_vec = np.zeros(n_coils)
        Fy_vec = np.zeros(n_coils)
        
        x = magnet_state['X']
        y = magnet_state['Y']
        z = magnet_state['Z']
        m_mag = magnet_state['m']
        alpha = magnet_state['alpha']
        beta = magnet_state['beta']
        
        mx = m_mag * np.sin(beta) * np.cos(alpha)
        my = m_mag * np.sin(beta) * np.sin(alpha)
        
        # DEBUG: Check magnet vector components
        if np.random.random() < 0.05: # Throttle log
             rospy.loginfo(f"Magnet: alpha={alpha:.2f}, beta={beta:.2f} => mx={mx:.4f}, my={my:.4f}")
        
        for i in range(n_coils):
            params = PARAMS_LIST[i]
            m0, m1, m2, r00, r01, r02 = params
            
            Bx_vec[i] = self.dipole_model_Bx(m0, m1, m2, r00, r01, r02, x, y, z)
            By_vec[i] = self.dipole_model_By(m0, m1, m2, r00, r01, r02, x, y, z)
            Bz_vec[i] = self.dipole_model_Bz(m0, m1, m2, r00, r01, r02, x, y, z)
            
            dBx_dx = self.dipole_grad_Bx_x(m0, m1, m2, r00, r01, r02, x, y, z)
            dBy_dx = self.dipole_grad_By_x(m0, m1, m2, r00, r01, r02, x, y, z)
            dBx_dy = self.dipole_grad_Bx_y(m0, m1, m2, r00, r01, r02, x, y, z)
            dBy_dy = self.dipole_grad_By_y(m0, m1, m2, r00, r01, r02, x, y, z)
            
            Fx_vec[i] = mx * dBx_dx + my * dBy_dx
            Fy_vec[i] = mx * dBx_dy + my * dBy_dy
            
        return {
            'Bx': Bx_vec, 'By': By_vec, 'Bz': Bz_vec,
            'Fx': Fx_vec, 'Fy': Fy_vec
        }

    def _unconstrained_objective(self, currents, mapping, target_Bx, target_By, target_Fx, target_Fy, penalty_weight=1e12):
        """Objective function with penalty method"""
        # Original objective: minimize power
        original_obj = np.sum(currents**2)
        
        # Magnetic field constraint penalties
        predicted_Bx = mapping['Bx'] @ currents
        predicted_By = mapping['By'] @ currents
        predicted_Bz = mapping['Bz'] @ currents
        
        # Weight field error higher to ensure B_magnitude is maintained (0.02T vs 0.1N)
        # Scaling factor to balance terms: (0.1/0.02)^2 = 25. Use 100 to prioritize field.
        field_weight = 200.0
        # Added Bz constraint (target Bz = 0)
        field_error = field_weight * ((predicted_Bx - target_Bx)**2 + (predicted_By - target_By)**2 + predicted_Bz**2)
        
        # Force constraint penalties
        predicted_Fx = mapping['Fx'] @ currents
        predicted_Fy = mapping['Fy'] @ currents
        force_error = (predicted_Fx - target_Fx)**2 + (predicted_Fy - target_Fy)**2
        
        # Current limit penalty
        current_violation = np.maximum(0, np.abs(currents) - CURRENT_LIMIT)
        current_penalty = np.sum(current_violation**2)
        
        return original_obj + penalty_weight * (field_error + force_error + current_penalty)

    def _unconstrained_gradient(self, currents, mapping, target_Bx, target_By, target_Fx, target_Fy, penalty_weight=1e12):
        """Analytical gradient"""
        # Gradient of original objective
        grad_original = 2 * currents
        
        field_weight = 200.0
        
        # Gradient of field constraint penalties
        predicted_Bx = mapping['Bx'] @ currents
        predicted_By = mapping['By'] @ currents
        predicted_Bz = mapping['Bz'] @ currents
        
        grad_field = 2 * penalty_weight * field_weight * (
            (predicted_Bx - target_Bx) * mapping['Bx'] +
            (predicted_By - target_By) * mapping['By'] +
            predicted_Bz * mapping['Bz']
        )
        
        # Gradient of force constraint penalties
        predicted_Fx = mapping['Fx'] @ currents
        predicted_Fy = mapping['Fy'] @ currents
        grad_force = 2 * penalty_weight * (
            (predicted_Fx - target_Fx) * mapping['Fx'] +
            (predicted_Fy - target_Fy) * mapping['Fy']
        )
        
        # Gradient of current limit penalty
        current_violation = np.maximum(0, np.abs(currents) - CURRENT_LIMIT)
        grad_current = 2 * penalty_weight * current_violation * np.sign(currents)
        
        return grad_original + grad_field + grad_force + grad_current

    def _save_results(self):
        # 1. Save trajectory results
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'closed_loop_results.csv')
        try:
            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Time(s)', 'Desired_X(m)', 'Desired_Y(m)', 'Desired_Theta(rad)',
                    'Actual_X(m)', 'Actual_Y(m)', 'Actual_Theta(rad)'
                ])
                writer.writerows(self.results_data)
            rospy.loginfo(f"Saved results to {out_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save results: {e}")
            
        # 2. Save currents (optional but good for debugging)
        curr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coil_currents_case6_closed_loop.csv')
        try:
            with open(curr_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['Time(s)'] + [f'Current_Coil_{i+1}(A)' for i in range(10)]
                writer.writerow(header)
                writer.writerows(self.coil_currents_data)
            rospy.loginfo(f"Saved currents to {curr_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save currents: {e}")

