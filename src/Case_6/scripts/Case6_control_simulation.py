import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import sympy as sp

# Configuration
current_limit = 15  # Amperes
B_magnitude = 0.02  # Tesla - magnetic field magnitude
force_limit = 0.1  # Newton - maximum pushing force
magnet_mass = 0.0015  # kg (example mass)
control_frequency = 10  # Hz

# Physics & Control Parameters
rotational_inertia = 5e-7  # kg*m^2 (Increased for numerical stability)
linear_damping = 2.0       # N/(m/s) - viscous drag (Increased for stability)
angular_damping = 5e-4     # Nm/(rad/s) (Increased for stability)

# PID Gains (Modified for Pure Pursuit)
Kp_pos = 15.0  # Reduced from 40.0 for stability
Kd_pos = 3.0   # Damping
Kp_theta = 0.5 # Reduced from 1.0
Kd_theta = 0.05 # Damping for orientation

# Pure Pursuit Parameters
lookahead_distance = 0.005  # 0.5 cm lookahead

# Magnet configuration
magnet_config = {
    'm': 0.12,  # magnetic moment
    'Z': 0.0,    # height (on the plane)
}

# Fixed magnet configuration (stationary magnet)
fixed_magnet_config = {
    'X': 0.02,
    'Y': 0.0,
    'Z': 0.0,
    'm': 0.145,
    'alpha': np.pi/2,
    'beta': np.pi/2
}

# List of fitted parameters for each coil (dipole model parameters)
params_list = [
    np.array([-13.04945069, -4.41557229, 6.47376799, 0.12129096, 0.00466922, -0.0174842]),  # coil 1
    np.array([-5.10083416, 13.54294901, 7.85474539, 0.05834654, -0.11165548, -0.01850546]),  # coil 2
    np.array([4.05088788, 14.23365818, 6.44760956, -0.05903076, -0.11020417, -0.01488244]),  # coil 3
    np.array([13.89011305, -0.06092074, 4.77365608, -0.12306086, -0.00085745, -0.01378161]), # coil 4
    np.array([11.44363813, -9.40543896, 4.46367162, -0.06806179, 0.1024875, -0.01397152]),  # coil 5
    np.array([-9.00577939, -12.78905365, 5.98650851, 0.06473315, 0.10618968, -0.0151172]),  # coil 6
    np.array([0.92820081, 8.54965337, 8.72298349, -0.00381254, -0.08845466, -0.08874662]),   # coil 7
    np.array([8.7302819, -4.90773115, 7.00109937, -0.07977306, 0.04481733, -0.08536032]),   # coil 8
    np.array([-7.68962762, -6.83258326, 8.12112247, 0.07498008, 0.04542436, -0.08696975]),   # coil 9
    np.array([2.35614001, -1.11370036, 14.00304846, -0.00722183, 0.00029277, -0.12482979])   # coil 10
]

# Define symbolic dipole model
m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z = sp.symbols('m0 m1 m2 r0_0 r0_1 r0_2 X Y Z')
mu0 = 4 * sp.pi * 1e-7

dx = X - r0_0
dy = Y - r0_1
dz = Z - r0_2
# Use a larger epsilon for numerical stability (approx 1mm safety margin) inside the sqrt
# This prevents 1/r^5 from exploding when r -> 0
r_sq = dx**2 + dy**2 + dz**2
r_safe = sp.sqrt(r_sq + 1.0e-5) 

dot_product = m0 * dx + m1 * dy + m2 * dz

# Magnetic field components
model_Bx = (mu0 / (4 * sp.pi)) * (3 * dx * dot_product / r_safe**5 - m0 / r_safe**3)
model_By = (mu0 / (4 * sp.pi)) * (3 * dy * dot_product / r_safe**5 - m1 / r_safe**3)
model_Bz = (mu0 / (4 * sp.pi)) * (3 * dz * dot_product / r_safe**5 - m2 / r_safe**3)

# Gradient of magnetic field for force calculation
grad_Bx_x = sp.diff(model_Bx, X)
grad_Bx_y = sp.diff(model_Bx, Y)
grad_By_x = sp.diff(model_By, X)
grad_By_y = sp.diff(model_By, Y)

# Convert to numerical functions
dipole_model_Bx = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx, 'numpy')
dipole_model_By = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By, 'numpy')
dipole_model_Bz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bz, 'numpy')
dipole_grad_Bx_x = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), grad_Bx_x, 'numpy')
dipole_grad_Bx_y = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), grad_Bx_y, 'numpy')
dipole_grad_By_x = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), grad_By_x, 'numpy')
dipole_grad_By_y = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), grad_By_y, 'numpy')

# Load trajectory data
# Return x, y, theta, time arrays
def load_trajectory(filepath):
    """Load trajectory from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    trajectory = np.array(data['trajectory'])
    
    return {
        'x': trajectory[:, 0],
        'y': trajectory[:, 1],
        'theta': trajectory[:, 2],  # orientation
        'time': trajectory[:, 3],
        'start': data['start'],
        'end': data['end']
    }

# Calculate trajectory derivatives
# Returns velocity, acceleration, and moving direction
def calculate_trajectory_derivatives(trajectory_data):
    """Calculate velocity, acceleration, and moving direction from trajectory"""
    x = trajectory_data['x']
    y = trajectory_data['y']
    theta = trajectory_data['theta']
    t = trajectory_data['time']
    
    # Calculate time differences
    dt = np.diff(t)
    dt = np.append(dt, dt[-1])  # Extend for last point
    
    # Velocity (first derivative)
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    
    # Acceleration (second derivative)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    
    # Moving direction (normalized velocity direction)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    # Avoid division by zero
    direction_x = np.where(v_magnitude > 1e-10, vx / v_magnitude, np.cos(theta))
    direction_y = np.where(v_magnitude > 1e-10, vy / v_magnitude, np.sin(theta))
    
    return {
        'vx': vx,
        'vy': vy,
        'ax': ax,
        'ay': ay,
        'direction_x': direction_x,
        'direction_y': direction_y,
        'speed': v_magnitude
    }

# Pre-compute mapping from currents to magnetic field and force
# Returns mapping matrices
def precompute_field_and_force_mapping(magnet_state):
    """Pre-compute mapping from currents to magnetic field and force"""
    n_coils = 10
    
    # Mapping matrices
    Bx_vector = np.zeros(n_coils)
    By_vector = np.zeros(n_coils)
    Bz_vector = np.zeros(n_coils)
    Fx_vector = np.zeros(n_coils)
    Fy_vector = np.zeros(n_coils)
    
    x = magnet_state['X']
    y = magnet_state['Y']
    z = magnet_state['Z']
    m_mag = magnet_state['m']
    alpha = magnet_state['alpha']
    beta = magnet_state['beta']
    
    # Check for NaN inputs or extreme values (divergence protection)
    if np.isnan(x) or np.isnan(y) or np.abs(x) > 5.0 or np.abs(y) > 5.0:
        return {
            'Bx': Bx_vector, 'By': By_vector, 'Bz': Bz_vector, 'Fx': Fx_vector, 'Fy': Fy_vector
        }

    # Magnetic moment components
    mx = m_mag * np.sin(beta) * np.cos(alpha)
    my = m_mag * np.sin(beta) * np.sin(alpha)
    
    for coil_idx in range(n_coils):
        params = params_list[coil_idx]
        m0_val, m1_val, m2_val, r0_0_val, r0_1_val, r0_2_val = params
        
        # Magnetic field from unit current
        Bx_vector[coil_idx] = dipole_model_Bx(m0_val, m1_val, m2_val, r0_0_val, r0_1_val, r0_2_val, x, y, z)
        By_vector[coil_idx] = dipole_model_By(m0_val, m1_val, m2_val, r0_0_val, r0_1_val, r0_2_val, x, y, z)
        Bz_vector[coil_idx] = dipole_model_Bz(m0_val, m1_val, m2_val, r0_0_val, r0_1_val, r0_2_val, x, y, z)
        
        # Force from unit current
        dBx_dx = dipole_grad_Bx_x(m0_val, m1_val, m2_val, r0_0_val, r0_1_val, r0_2_val, x, y, z)
        dBx_dy = dipole_grad_Bx_y(m0_val, m1_val, m2_val, r0_0_val, r0_1_val, r0_2_val, x, y, z)
        dBy_dx = dipole_grad_By_x(m0_val, m1_val, m2_val, r0_0_val, r0_1_val, r0_2_val, x, y, z)
        dBy_dy = dipole_grad_By_y(m0_val, m1_val, m2_val, r0_0_val, r0_1_val, r0_2_val, x, y, z)
        
        Fx_vector[coil_idx] = mx * dBx_dx + my * dBy_dx
        Fy_vector[coil_idx] = mx * dBx_dy + my * dBy_dy
    
    return {
        'Bx': Bx_vector,
        'By': By_vector,
        'Bz': Bz_vector,
        'Fx': Fx_vector,
        'Fy': Fy_vector
    }

def unconstrained_objective(currents, mapping, target_Bx, target_By, target_Fx, target_Fy, 
                            penalty_weight=1e12):
    """Objective function with penalty method"""
    # Original objective: minimize power
    original_obj = np.sum(currents**2)
    
    # Magnetic field constraint penalties
    predicted_Bx = mapping['Bx'] @ currents
    predicted_By = mapping['By'] @ currents
    predicted_Bz = mapping['Bz'] @ currents
    
    # Weight field error higher to ensure B_magnitude is maintained (0.02T vs 0.1N)
    # Scaling factor to balance terms: (0.1/0.02)^2 = 25. Use 100 to prioritize field.
    field_weight = 100.0
    # Added Bz constraint (target Bz = 0)
    field_error = field_weight * ((predicted_Bx - target_Bx)**2 + (predicted_By - target_By)**2 + predicted_Bz**2)
    
    # Force constraint penalties
    predicted_Fx = mapping['Fx'] @ currents
    predicted_Fy = mapping['Fy'] @ currents
    force_error = (predicted_Fx - target_Fx)**2 + (predicted_Fy - target_Fy)**2
    
    # Current limit penalty
    current_violation = np.maximum(0, np.abs(currents) - current_limit)
    current_penalty = np.sum(current_violation**2)
    
    return original_obj + penalty_weight * (field_error + force_error + current_penalty)

def unconstrained_gradient(currents, mapping, target_Bx, target_By, target_Fx, target_Fy,
                          penalty_weight=1e12):
    """Analytical gradient"""
    # Gradient of original objective
    grad_original = 2 * currents
    
    field_weight = 100.0
    
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
    current_violation = np.maximum(0, np.abs(currents) - current_limit)
    grad_current = 2 * penalty_weight * current_violation * np.sign(currents)
    
    return grad_original + grad_field + grad_force + grad_current

def calculate_magnet_interaction_force(moving_magnet_state, fixed_magnet_state):
    """
    Calculate interaction force on moving magnet from fixed magnet
    Uses the dipole-dipole interaction model
    """
    mu0 = 4 * np.pi * 1e-7
    
    # Extract positions
    r_vec = np.array([
        moving_magnet_state['X'] - fixed_magnet_state['X'],
        moving_magnet_state['Y'] - fixed_magnet_state['Y'],
        moving_magnet_state['Z'] - fixed_magnet_state['Z']
    ])
    dist = np.linalg.norm(r_vec)
    
    # Check for invalid distance (NaN, Inf, or too large causing overflow)
    if np.isnan(dist) or np.isinf(dist) or dist > 10.0:
        return np.zeros(3)
    
    # Safety clamp: assume magnets cannot physically overlap closer than sum of radii
    # e.g., 5mm minimum distance to prevent singularity
    min_dist = 0.005 
    if dist < min_dist:
        dist = min_dist
        # Normalize r_vec to maintain direction but limit magnitude
        if np.linalg.norm(r_vec) > 1e-9:
            r_vec = r_vec / np.linalg.norm(r_vec) * dist
        else:
            r_vec = np.array([dist, 0, 0]) # Arbitrary direction if exactly on top
            
    r = dist
    r_hat = r_vec / r
    
    # Magnetic moment of moving magnet
    m_moving_mag = moving_magnet_state['m']
    alpha_m = moving_magnet_state['alpha']
    beta_m = moving_magnet_state['beta']
    m_moving = np.array([
        m_moving_mag * np.sin(beta_m) * np.cos(alpha_m),
        m_moving_mag * np.sin(beta_m) * np.sin(alpha_m),
        m_moving_mag * np.cos(beta_m)
    ])
    
    # Magnetic moment of fixed magnet
    m_fixed_mag = fixed_magnet_state['m']
    alpha_f = fixed_magnet_state['alpha']
    beta_f = fixed_magnet_state['beta']
    m_fixed = np.array([
        m_fixed_mag * np.sin(beta_f) * np.cos(alpha_f),
        m_fixed_mag * np.sin(beta_f) * np.sin(alpha_f),
        m_fixed_mag * np.cos(beta_f)
    ])
    
    # Calculate magnetic field gradient from fixed magnet at moving magnet location
    # B = (mu0/(4*pi)) * [3*(m_fixed·r_hat)*r_hat - m_fixed] / r^3
    m_fixed_dot_r_hat = np.dot(m_fixed, r_hat)
    
    # Force on moving magnet: F = ∇(m_moving · B)
    # Using dipole-dipole force formula:
    # F = (3*mu0/(4*pi*r^4)) * [(m_moving·r_hat)*m_fixed + (m_fixed·r_hat)*m_moving 
    #                            + (m_moving·m_fixed)*r_hat 
    #                            - 5*(m_moving·r_hat)*(m_fixed·r_hat)*r_hat]
    
    m_moving_dot_r_hat = np.dot(m_moving, r_hat)
    m_moving_dot_m_fixed = np.dot(m_moving, m_fixed)
    
    F_interaction = (3 * mu0 / (4 * np.pi * r**4)) * (
        m_moving_dot_r_hat * m_fixed +
        m_fixed_dot_r_hat * m_moving +
        m_moving_dot_m_fixed * r_hat -
        5 * m_moving_dot_r_hat * m_fixed_dot_r_hat * r_hat
    )
    
    # Clamp interaction force to prevent numerical explosion
    # Realistically, if force is > 10N, the model is probably invalid or magnets are touching
    max_interaction_force = 5.0 
    f_mag = np.linalg.norm(F_interaction)
    if f_mag > max_interaction_force:
        F_interaction = F_interaction / f_mag * max_interaction_force
    
    return F_interaction

# Feedback simulation parameters
feedback_config = {
    'position_noise_std': 0.0005,  # Standard deviation for position measurement noise (m)
    'orientation_noise_std': 0.02,  # Standard deviation for orientation noise (rad)
    'bias_position': np.array([0.0002, -0.0002]),  # Systematic bias in position [x, y] (m)
    'bias_orientation': 0.001,  # Systematic bias in orientation (rad)
    'delay_steps': 1,  # Measurement delay in control steps
    'dropout_probability': 0.002,  # Probability of measurement dropout
}

def simulate_feedback_measurement(true_state, previous_measurements, feedback_config):
    """
    Simulate realistic feedback measurements with noise, bias, delay, and dropout
    
    Args:
        true_state: dict with keys 'X', 'Y', 'theta' (true magnet state)
        previous_measurements: list of previous measurements for delay simulation
        feedback_config: dict with noise parameters
    
    Returns:
        measured_state: dict with noisy measurements
        measurement_valid: bool indicating if measurement is valid (not dropped out)
    """
    # Check for measurement dropout
    if np.random.random() < feedback_config['dropout_probability']:
        # Return last valid measurement if available
        if len(previous_measurements) > 0:
            return previous_measurements[-1], False
        else:
            return true_state.copy(), False
    
    # Add Gaussian noise to position
    position_noise = np.random.normal(0, feedback_config['position_noise_std'], size=2)
    measured_x = true_state['X'] + position_noise[0] + feedback_config['bias_position'][0]
    measured_y = true_state['Y'] + position_noise[1] + feedback_config['bias_position'][1]
    
    # Add Gaussian noise to orientation
    orientation_noise = np.random.normal(0, feedback_config['orientation_noise_std'])
    measured_theta = true_state['theta'] + orientation_noise + feedback_config['bias_orientation']
    
    # Wrap orientation to [-pi, pi]
    measured_theta = np.arctan2(np.sin(measured_theta), np.cos(measured_theta))
    
    measured_state = {
        'X': measured_x,
        'Y': measured_y,
        'theta': measured_theta
    }
    
    # Store measurement in history
    previous_measurements.append(measured_state.copy())
    
    # Apply delay: return delayed measurement
    delay_steps = feedback_config['delay_steps']
    if len(previous_measurements) > delay_steps:
        return previous_measurements[-delay_steps-1], True
    else:
        return measured_state, True

def apply_low_pass_filter(current_measurement, filtered_state, alpha=0.3):
    """
    Apply exponential moving average filter to reduce noise
    
    Args:
        current_measurement: dict with current noisy measurement
        filtered_state: dict with previous filtered state (or None for first measurement)
        alpha: filter coefficient (0 < alpha <= 1, smaller = more filtering)
    
    Returns:
        filtered_state: dict with filtered measurements
    """
    if filtered_state is None:
        return current_measurement.copy()
    
    filtered = {
        'X': alpha * current_measurement['X'] + (1 - alpha) * filtered_state['X'],
        'Y': alpha * current_measurement['Y'] + (1 - alpha) * filtered_state['Y'],
        'theta': alpha * current_measurement['theta'] + (1 - alpha) * filtered_state['theta']
    }
    
    return filtered

def estimate_velocity_acceleration(position_history, time_history, dt):
    """
    Estimate velocity and acceleration from position history using finite differences
    
    Args:
        position_history: list of dicts with 'X', 'Y', 'theta'
        time_history: list of timestamps
        dt: time step
    
    Returns:
        velocity: dict with estimated velocities
        acceleration: dict with estimated accelerations
    """
    if len(position_history) < 2:
        return {'vx': 0, 'vy': 0, 'vtheta': 0}, {'ax': 0, 'ay': 0, 'atheta': 0}
    
    # Velocity from last two positions (backward difference)
    if len(position_history) >= 2:
        vx = (position_history[-1]['X'] - position_history[-2]['X']) / dt
        vy = (position_history[-1]['Y'] - position_history[-2]['Y']) / dt
        vtheta = (position_history[-1]['theta'] - position_history[-2]['theta']) / dt
    else:
        vx = vy = vtheta = 0
    
    # Acceleration from last three positions (central difference if available)
    if len(position_history) >= 3:
        ax = (position_history[-1]['X'] - 2*position_history[-2]['X'] + position_history[-3]['X']) / (dt**2)
        ay = (position_history[-1]['Y'] - 2*position_history[-2]['Y'] + position_history[-3]['Y']) / (dt**2)
        atheta = (position_history[-1]['theta'] - 2*position_history[-2]['theta'] + position_history[-3]['theta']) / (dt**2)
    else:
        ax = ay = atheta = 0
    
    velocity = {'vx': vx, 'vy': vy, 'vtheta': vtheta}
    acceleration = {'ax': ax, 'ay': ay, 'atheta': atheta}
    
    return velocity, acceleration

def get_lookahead_point(x_traj, y_traj, x_robot, y_robot, lookahead_dist, last_index):
    """
    Find the point on the trajectory at distance 'lookahead_dist' from the robot.
    Starts searching from last_index to optimize performance.
    """
    # Search forward from the last known index
    for i in range(last_index, len(x_traj)):
        dx = x_traj[i] - x_robot
        dy = y_traj[i] - y_robot
        dist = np.sqrt(dx**2 + dy**2)
        
        # Found a point far enough ahead
        if dist >= lookahead_dist:
            return x_traj[i], y_traj[i], i
            
    # If we reach the end of the trajectory, return the last point
    return x_traj[-1], y_traj[-1], len(x_traj)-1

def run_trajectory_control(animate=True):
    """Main control loop for Case 6 with feedback simulation (Closed-Loop)"""
    # Load trajectory
    print("Loading trajectory data...")
    trajectory_data = load_trajectory("Case_6/trajectory_case6_3D_02.json")
    
    # Calculate derivatives
    print("Calculating trajectory derivatives...")
    derivatives = calculate_trajectory_derivatives(trajectory_data)
    
    times = trajectory_data['time']
    n_points = len(times)
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    print(f"Total trajectory points: {n_points}, Duration: {times[-1]:.3f} seconds, dt: {dt:.4f}s")
    
    # Setup animation
    if animate:
        plt.ion()
        fig, plot_ax = plt.subplots(figsize=(10, 10))  # Renamed ax to plot_ax
        plot_ax.set_aspect('equal')
        plot_ax.grid(True)
        plot_ax.set_xlabel('X (m)')
        plot_ax.set_ylabel('Y (m)')
        plot_ax.set_title('Closed-Loop Control Simulation (Pure Pursuit)')
        
        # Plot full trajectory background
        plot_ax.plot(trajectory_data['x'], trajectory_data['y'], 'k--', alpha=0.3, label='Reference Path')
        plot_ax.plot(fixed_magnet_config['X'], fixed_magnet_config['Y'], 'ms', markersize=10, label='Fixed Magnet')
        
        # Magnet visualization (True State)
        magnet_w, magnet_h = 0.004, 0.002  # Dimensions
        magnet_patch = patches.Rectangle((-magnet_w/2, -magnet_h/2), magnet_w, magnet_h, 
                                       fc='blue', ec='black', alpha=0.8, label='Actual')
        plot_ax.add_patch(magnet_patch)
        
        # Reference visualization (Ghost) - Now represents Lookahead Point
        ref_patch = patches.Circle((0, 0), 0.001, fc='green', alpha=0.5, label='Lookahead')
        plot_ax.add_patch(ref_patch)
        
        # Heading arrow
        quiver = plot_ax.quiver(0, 0, 1, 0, scale=20, color='red', width=0.005, label='Heading')
        
        # Magnetic Field arrow
        quiver_B = plot_ax.quiver(0, 0, 1, 0, scale=20, color='cyan', width=0.005, label='B-Field')
        
        # Text for status
        status_text = plot_ax.text(0.05, 0.95, '', transform=plot_ax.transAxes, verticalalignment='top')
        
        # Set limits with some padding
        x_min, x_max = np.min(trajectory_data['x']), np.max(trajectory_data['x'])
        y_min, y_max = np.min(trajectory_data['y']), np.max(trajectory_data['y'])
        margin = 0.01
        plot_ax.set_xlim(x_min - margin, x_max + margin)
        plot_ax.set_ylim(y_min - margin, y_max + margin)
        
        plt.legend(loc='lower left')

    # Storage for results
    results_data = {
        'time': [],
        'currents': [],
        'B_desired': [],
        'B_actual': [],
        'F_desired': [],
        'F_actual': [],
        'F_interaction': [],
        'F_coil': [],
        'F_body_frame': [],
        'optimization_time': [],
        'success': [],
        'B_error': [],
        'F_error': [],
        'max_current': [],
        'pos_error': [],
        'theta_error': [],
        'true_state': []
    }
    
    # Initial guess
    currents_guess = np.zeros(10)
    
    # Pure Pursuit State
    last_traj_index = 0
    
    # Initialize State (True Physics State)
    current_true_state = {
        'X': trajectory_data['x'][0],
        'Y': trajectory_data['y'][0],
        'Z': magnet_config['Z'],
        'theta': trajectory_data['theta'][0],
        'vx': derivatives['vx'][0],
        'vy': derivatives['vy'][0],
        'omega': 0.0
    }
    
    # Measurement history
    previous_measurements = []
    
    total_start_time = time.time()
    
    # Start timer for animation sync
    if animate:
        playback_start_time = time.time()
        start_sim_time = times[0] if n_points > 0 else 0
    
    step_count = 0
    t = times[0]
    
    # Run until lookahead reaches the end (with safety limit)
    while last_traj_index < n_points - 1 and step_count < n_points * 3:
        # Determine reference index (clamp to last point if we exceed trajectory duration)
        ref_idx = min(step_count, n_points - 1)
        
        # Update time
        if step_count < n_points:
            t = times[step_count]
        else:
            t += dt
        
        if step_count % 50 == 0 or step_count < 5:
            print(f"\nProcessing time {t:.3f}s (Step {step_count+1}, Lookahead Idx {last_traj_index}/{n_points-1})")
        
        # --- 0. SAFETY CHECK ---
        # If state is NaN or out of bounds, reset or stop
        if (np.isnan(current_true_state['X']) or np.isnan(current_true_state['Y']) or 
            np.abs(current_true_state['X']) > 2.0 or np.abs(current_true_state['Y']) > 2.0):
            print(f"CRITICAL: Instability detected at step {step_count}. Resetting to trajectory.")
            current_true_state['X'] = trajectory_data['x'][ref_idx]
            current_true_state['Y'] = trajectory_data['y'][ref_idx]
            current_true_state['vx'] = 0
            current_true_state['vy'] = 0
            
        # --- 1. SENSOR MEASUREMENT ---
        # Get noisy measurement of the current state
        measured_state_simple, _ = simulate_feedback_measurement(
            {'X': current_true_state['X'], 'Y': current_true_state['Y'], 'theta': current_true_state['theta']}, 
            previous_measurements, 
            feedback_config
        )
        
        # Estimate velocity from measurements (simple difference)
        if len(previous_measurements) >= 2:
            dt_meas = dt # Assuming constant rate
            meas_vx = (previous_measurements[-1]['X'] - previous_measurements[-2]['X']) / dt_meas
            meas_vy = (previous_measurements[-1]['Y'] - previous_measurements[-2]['Y']) / dt_meas
            
            # Angular velocity estimation
            d_theta = previous_measurements[-1]['theta'] - previous_measurements[-2]['theta']
            d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta)) # Wrap to [-pi, pi]
            meas_omega = d_theta / dt_meas
        else:
            meas_vx, meas_vy, meas_omega = 0, 0, 0

        # --- 2. CONTROLLER (Pure Pursuit) ---
        # 1. Find lookahead point
        lookahead_x, lookahead_y, last_traj_index = get_lookahead_point(
            trajectory_data['x'], trajectory_data['y'], 
            measured_state_simple['X'], measured_state_simple['Y'], 
            lookahead_distance, last_traj_index
        )
        
        # 2. Calculate vector to lookahead point
        dx_lookahead = lookahead_x - measured_state_simple['X']
        dy_lookahead = lookahead_y - measured_state_simple['Y']
        
        # 3. Calculate Control Force
        # F = Kp * (lookahead_pos - current_pos) - Kd * current_vel
        # This pulls the robot towards the lookahead point while damping velocity
        F_control_x = Kp_pos * dx_lookahead - Kd_pos * meas_vx
        F_control_y = Kp_pos * dy_lookahead - Kd_pos * meas_vy
        
        # Reference orientation (from trajectory at current time step)
        ref_theta = trajectory_data['theta'][ref_idx]
        
        # Orientation Error
        err_theta = ref_theta - measured_state_simple['theta']
        # Normalize angle error
        err_theta = np.arctan2(np.sin(err_theta), np.cos(err_theta))
        
        # Clamp control force to avoid huge requests
        F_control_mag = np.sqrt(F_control_x**2 + F_control_y**2)
        if F_control_mag > force_limit * 2: # Allow some overhead for control
            scale = (force_limit * 2) / F_control_mag
            F_control_x *= scale
            F_control_y *= scale

        # Interaction Force Compensation (Estimated using measured state)
        magnet_state_est = {
            'X': measured_state_simple['X'],
            'Y': measured_state_simple['Y'],
            'Z': magnet_config['Z'],
            'm': magnet_config['m'],
            'alpha': np.pi/2,
            'beta': measured_state_simple['theta']
        }
        F_interaction_est = calculate_magnet_interaction_force(magnet_state_est, fixed_magnet_config)
        
        # Required Coil Force (World Frame)
        target_Fx_coil = F_control_x - F_interaction_est[0]
        target_Fy_coil = F_control_y - F_interaction_est[1]
        
        # Orientation Control via Magnetic Field Direction
        # Desired heading is reference theta + correction (PD Control)
        # We add a derivative term to damp the rotation
        theta_correction = Kp_theta * err_theta - Kd_theta * meas_omega
        
        # Clamp correction to prevent field reversal relative to movement direction
        # This ensures the magnetic field always points roughly in the direction of motion (within +/- 85 deg)
        # and prevents the field from pointing backwards which causes instability
        max_angle_dev = 85 * np.pi / 180.0
        theta_correction = np.clip(theta_correction, -max_angle_dev, max_angle_dev)
        
        target_heading = ref_theta + theta_correction
        target_Bx = B_magnitude * np.cos(target_heading)
        target_By = B_magnitude * np.sin(target_heading)
        
        # Calculate actual tracking error for logging (closest point distance)
        # Simple approximation: distance to current trajectory point
        err_x = trajectory_data['x'][ref_idx] - measured_state_simple['X']
        err_y = trajectory_data['y'][ref_idx] - measured_state_simple['Y']

        if step_count < 5:
            print(f"  Lookahead: ({lookahead_x:.4f}, {lookahead_y:.4f}), Meas: ({measured_state_simple['X']:.4f}, {measured_state_simple['Y']:.4f})")
            print(f"  Error: ({err_x*1000:.2f}, {err_y*1000:.2f}) mm")
            print(f"  Control Force: ({F_control_x:.4f}, {F_control_y:.4f}) N")
        
        # --- 3. OPTIMIZATION ---
        # Compute mapping at MEASURED position
        mapping_est = precompute_field_and_force_mapping(magnet_state_est)
        
        opt_start_time = time.time()
        
        # Check if mapping is valid
        if np.any(np.isnan(mapping_est['Bx'])):
             print("Warning: NaN in mapping. Skipping optimization.")
             applied_currents = np.zeros(10)
             result = type('obj', (object,), {'success': False})
        else:
            result = minimize(
                lambda x: unconstrained_objective(x, mapping_est, target_Bx, target_By, 
                                                target_Fx_coil, target_Fy_coil),
                currents_guess,
                method='L-BFGS-B',
                jac=lambda x: unconstrained_gradient(x, mapping_est, target_Bx, target_By,
                                                    target_Fx_coil, target_Fy_coil),
                options={'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 500}
            )
            applied_currents = result.x
        
        opt_end_time = time.time()
        
        # --- 4. PHYSICS SIMULATION (The "Real World") ---
        # Calculate actual forces/fields acting on the TRUE magnet state
        # We use sub-stepping for stability
        n_substeps = 20
        dt_sub = dt / n_substeps
        
        # Coil forces are assumed constant during the control step (Zero-Order Hold)
        # But we need to calculate them at the start of the step
        true_magnet_state_dict = {
            'X': current_true_state['X'],
            'Y': current_true_state['Y'],
            'Z': magnet_config['Z'],
            'm': magnet_config['m'],
            'alpha': np.pi/2,
            'beta': current_true_state['theta']
        }
        
        # True mapping at start of step
        mapping_true = precompute_field_and_force_mapping(true_magnet_state_dict)
        
        # Actual Coil Force & Field (held constant for sub-steps)
        actual_Fx_coil = mapping_true['Fx'] @ applied_currents
        actual_Fy_coil = mapping_true['Fy'] @ applied_currents
        actual_Bx = mapping_true['Bx'] @ applied_currents
        actual_By = mapping_true['By'] @ applied_currents
        
        # Variables for logging (take the last value)
        log_total_Fx = 0
        log_total_Fy = 0
        log_interaction_Fx = 0
        log_interaction_Fy = 0
        
        for _ in range(n_substeps):
            # Update state dict for interaction calculation
            sub_state_dict = {
                'X': current_true_state['X'],
                'Y': current_true_state['Y'],
                'Z': magnet_config['Z'],
                'm': magnet_config['m'],
                'alpha': np.pi/2,
                'beta': current_true_state['theta']
            }
            
            # Actual Interaction Force (updates with position)
            F_interaction_true = calculate_magnet_interaction_force(sub_state_dict, fixed_magnet_config)
            
            # Total Force
            total_Fx = actual_Fx_coil + F_interaction_true[0]
            total_Fy = actual_Fy_coil + F_interaction_true[1]
            
            # Clamp total force to prevent physics explosion
            total_F_mag = np.sqrt(total_Fx**2 + total_Fy**2)
            max_physics_force = 10.0 # N (Reduced clamp for stability)
            if total_F_mag > max_physics_force:
                scale = max_physics_force / total_F_mag
                total_Fx *= scale
                total_Fy *= scale
                
            # Magnetic Torque: tau = m x B
            mx = magnet_config['m'] * np.cos(current_true_state['theta'])
            my = magnet_config['m'] * np.sin(current_true_state['theta'])
            torque_z = mx * actual_By - my * actual_Bx
            
            # Dynamics Update (Euler Integration)
            # Linear
            acc_x = (total_Fx - linear_damping * current_true_state['vx']) / magnet_mass
            acc_y = (total_Fy - linear_damping * current_true_state['vy']) / magnet_mass
            
            # Check for NaNs
            if np.isnan(acc_x) or np.isnan(acc_y):
                acc_x = 0
                acc_y = 0
                
            current_true_state['vx'] += acc_x * dt_sub
            current_true_state['vy'] += acc_y * dt_sub
            
            # Velocity Clamp (Safety)
            v_mag = np.sqrt(current_true_state['vx']**2 + current_true_state['vy']**2)
            max_v = 1.0 # m/s limit
            if v_mag > max_v:
                scale = max_v / v_mag
                current_true_state['vx'] *= scale
                current_true_state['vy'] *= scale

            current_true_state['X'] += current_true_state['vx'] * dt_sub
            current_true_state['Y'] += current_true_state['vy'] * dt_sub
            
            # Angular
            alpha_z = (torque_z - angular_damping * current_true_state['omega']) / rotational_inertia
            current_true_state['omega'] += alpha_z * dt_sub
            current_true_state['theta'] += current_true_state['omega'] * dt_sub
            
            # Capture last forces for logging
            log_total_Fx = total_Fx
            log_total_Fy = total_Fy
            log_interaction_Fx = F_interaction_true[0]
            log_interaction_Fy = F_interaction_true[1]
        
        # --- 5. LOGGING & VISUALIZATION ---
        
        # Transform total force to body frame (for logging consistency)
        cos_theta = np.cos(current_true_state['theta'])
        sin_theta = np.sin(current_true_state['theta'])
        F_body_x = log_total_Fx * cos_theta + log_total_Fy * sin_theta
        F_body_y = -log_total_Fx * sin_theta + log_total_Fy * cos_theta
        
        # Animation update
        if animate and step_count % 2 == 0:
            # Update magnet patch (True State)
            trans = Affine2D().rotate(current_true_state['theta']).translate(current_true_state['X'], current_true_state['Y']) + plot_ax.transData
            magnet_patch.set_transform(trans)
            
            # Update reference patch (Lookahead Point)
            trans_ref = Affine2D().translate(lookahead_x, lookahead_y) + plot_ax.transData
            ref_patch.set_transform(trans_ref)
            
            # Update arrow (Heading)
            quiver.set_offsets([current_true_state['X'], current_true_state['Y']])
            quiver.set_UVC(np.cos(current_true_state['theta']), np.sin(current_true_state['theta']))
            
            # Update arrow (B-Field)
            quiver_B.set_offsets([current_true_state['X'], current_true_state['Y']])
            B_mag_vis = np.sqrt(actual_Bx**2 + actual_By**2)
            if B_mag_vis > 1e-6:
                quiver_B.set_UVC(actual_Bx/B_mag_vis, actual_By/B_mag_vis)
            else:
                quiver_B.set_UVC(0, 0)
            
            # Update text
            status_text.set_text(f'Time: {t:.2f}s\n'
                               f'Err: {np.sqrt(err_x**2 + err_y**2)*1000:.1f}mm\n'
                               f'Max Current: {np.max(np.abs(applied_currents)):.2f}A')
            
            # Sync
            current_real_time = time.time()
            elapsed_real_time = current_real_time - playback_start_time
            target_sim_time = t - start_sim_time
            if target_sim_time > elapsed_real_time:
                time.sleep(target_sim_time - elapsed_real_time)
            
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Store results
        results_data['time'].append(t)
        results_data['currents'].append(applied_currents.copy())
        results_data['B_desired'].append([target_Bx, target_By])
        results_data['B_actual'].append([actual_Bx, actual_By])
        results_data['F_desired'].append([F_control_x, F_control_y])
        results_data['F_actual'].append([log_total_Fx, log_total_Fy])
        results_data['F_interaction'].append([log_interaction_Fx, log_interaction_Fy])
        results_data['F_coil'].append([actual_Fx_coil, actual_Fy_coil])
        results_data['F_body_frame'].append([F_body_x, F_body_y])
        results_data['optimization_time'].append(opt_end_time - opt_start_time)
        results_data['success'].append(result.success)
        results_data['B_error'].append(np.sqrt((actual_Bx - target_Bx)**2 + (actual_By - target_By)**2))
        results_data['F_error'].append(np.sqrt((log_total_Fx - F_control_x)**2 + (log_total_Fy - F_control_y)**2))
        results_data['max_current'].append(np.max(np.abs(applied_currents)))
        results_data['pos_error'].append(np.sqrt(err_x**2 + err_y**2))
        results_data['theta_error'].append(err_theta)
        results_data['true_state'].append(current_true_state.copy())
        
        # Warm start
        if result.success:
            currents_guess = applied_currents
            
        step_count += 1
    
    if animate:
        plt.ioff()
        plt.close()

    total_end_time = time.time()
    
    print(f"\n=== CLOSED-LOOP CONTROL COMPLETED ===")
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Avg Position Error: {np.mean(results_data['pos_error'])*1000:.2f} mm")
    print(f"Avg Theta Error: {np.mean(np.abs(results_data['theta_error']))*180/np.pi:.2f} deg")
    
    return results_data, trajectory_data, derivatives

def save_and_plot_results(results_data, trajectory_data, derivatives):
    """Save results and create plots"""
    times = np.array(results_data['time'])
    ref_times = trajectory_data['time'] # Use reference time for reference plots
    
    currents_array = np.array(results_data['currents'])
    B_desired = np.array(results_data['B_desired'])
    B_actual = np.array(results_data['B_actual'])
    F_desired = np.array(results_data['F_desired'])
    F_actual = np.array(results_data['F_actual'])
    F_interaction = np.array(results_data['F_interaction'])
    F_coil = np.array(results_data['F_coil'])
    F_body = np.array(results_data['F_body_frame'])
    
    # Save currents to CSV
    currents_df = pd.DataFrame()
    currents_df['Time(s)'] = times
    for i in range(10):
        currents_df[f'Current_Coil_{i+1}(A)'] = currents_array[:, i]
    
    output_path = "Case_6/coil_currents_case6_closed_loop.csv"
    currents_df.to_csv(output_path, index=False)
    print(f"\nCurrents saved to {output_path}")
    
    # Create comprehensive plots - now with 4x4 grid
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Trajectory in XY plane
    ax1 = plt.subplot(4, 4, 1)
    ax1.plot(trajectory_data['x'], trajectory_data['y'], 'k--', linewidth=1, label='Reference')
    
    # Extract actual path
    true_states = results_data['true_state']
    actual_x = [s['X'] for s in true_states]
    actual_y = [s['Y'] for s in true_states]
    
    ax1.plot(actual_x, actual_y, 'b-', linewidth=2, label='Actual')
    ax1.plot(trajectory_data['x'][0], trajectory_data['y'][0], 'go', markersize=10, label='Start')
    ax1.plot(trajectory_data['x'][-1], trajectory_data['y'][-1], 'ro', markersize=10, label='End')
    # Mark fixed magnet position
    ax1.plot(fixed_magnet_config['X'], fixed_magnet_config['Y'], 'ms', markersize=12, 
             label='Fixed Magnet', markeredgewidth=2, markerfacecolor='none')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory Tracking')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    # 2. Orientation vs time
    ax2 = plt.subplot(4, 4, 2)
    ax2.plot(ref_times, trajectory_data['theta'], 'k--', linewidth=1, label='Reference')
    actual_theta = [s['theta'] for s in true_states]
    ax2.plot(times, actual_theta, 'b-', linewidth=2, label='Actual')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Orientation (rad)')
    ax2.set_title('Orientation Tracking')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Coil currents
    ax3 = plt.subplot(4, 4, 3)
    for i in range(10):
        ax3.plot(times, currents_array[:, i], label=f'Coil {i+1}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Current (A)')
    ax3.set_title('Coil Currents')
    ax3.grid(True)
    # ax3.legend(ncol=2, fontsize=8)
    
    # 4. Max current vs limit
    ax4 = plt.subplot(4, 4, 4)
    ax4.plot(times, results_data['max_current'], 'b-', label='Max Current')
    ax4.axhline(y=current_limit, color='r', linestyle='--', label=f'Limit ({current_limit}A)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Current (A)')
    ax4.set_title('Maximum Current vs Limit')
    ax4.grid(True)
    ax4.legend()
    
    # 5. Magnetic field Bx
    ax5 = plt.subplot(4, 4, 5)
    ax5.plot(times, B_desired[:, 0], 'k-', label='Desired', linewidth=2)
    ax5.plot(times, B_actual[:, 0], 'r--', label='Actual', linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Bx (T)')
    ax5.set_title('Magnetic Field Bx')
    ax5.grid(True)
    ax5.legend()
    
    # 6. Magnetic field By
    ax6 = plt.subplot(4, 4, 6)
    ax6.plot(times, B_desired[:, 1], 'k-', label='Desired', linewidth=2)
    ax6.plot(times, B_actual[:, 1], 'r--', label='Actual', linewidth=1.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('By (T)')
    ax6.set_title('Magnetic Field By')
    ax6.grid(True)
    ax6.legend()
    
    # 7. Magnetic field magnitude
    ax7 = plt.subplot(4, 4, 7)
    B_mag_desired = np.sqrt(B_desired[:, 0]**2 + B_desired[:, 1]**2)
    B_mag_actual = np.sqrt(B_actual[:, 0]**2 + B_actual[:, 1]**2)
    ax7.plot(times, B_mag_desired, 'k-', label='Desired', linewidth=2)
    ax7.plot(times, B_mag_actual, 'r--', label='Actual', linewidth=1.5)
    ax7.axhline(y=B_magnitude, color='g', linestyle=':', label=f'Target ({B_magnitude}T)')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('|B| (T)')
    ax7.set_title('Magnetic Field Magnitude')
    ax7.grid(True)
    ax7.legend()
    
    # 8. Magnetic field error
    ax8 = plt.subplot(4, 4, 8)
    ax8.plot(times, results_data['B_error'], 'r-')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('B Error (T)')
    ax8.set_title('Magnetic Field Error')
    ax8.set_yscale('log')
    ax8.grid(True)
    
    # 9. Force Fx (world frame) - show breakdown
    ax9 = plt.subplot(4, 4, 9)
    ax9.plot(times, F_desired[:, 0], 'k-', label='Control Output', linewidth=2)
    ax9.plot(times, F_coil[:, 0], 'b--', label='Coil Force', linewidth=1.5)
    ax9.plot(times, F_interaction[:, 0], 'r--', label='Interaction', linewidth=1.5)
    ax9.plot(times, F_actual[:, 0], 'g:', label='Total Actual', linewidth=2)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Fx (N)')
    ax9.set_title('Force Fx Breakdown (World Frame)')
    ax9.grid(True)
    ax9.legend(fontsize=8)
    
    # 10. Force Fy (world frame) - show breakdown
    ax10 = plt.subplot(4, 4, 10)
    ax10.plot(times, F_desired[:, 1], 'k-', label='Control Output', linewidth=2)
    ax10.plot(times, F_coil[:, 1], 'b--', label='Coil Force', linewidth=1.5)
    ax10.plot(times, F_interaction[:, 1], 'r--', label='Interaction', linewidth=1.5)
    ax10.plot(times, F_actual[:, 1], 'g:', label='Total Actual', linewidth=2)
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('Fy (N)')
    ax10.set_title('Force Fy Breakdown (World Frame)')
    ax10.grid(True)
    
    # 11. Force magnitude (world frame)
    ax11 = plt.subplot(4, 4, 11)
    F_mag_desired = np.sqrt(F_desired[:, 0]**2 + F_desired[:, 1]**2)
    F_mag_actual = np.sqrt(F_actual[:, 0]**2 + F_actual[:, 1]**2)
    ax11.plot(times, F_mag_desired, 'k-', label='Desired', linewidth=2)
    ax11.plot(times, F_mag_actual, 'r--', label='Actual', linewidth=1.5)
    ax11.axhline(y=force_limit, color='r', linestyle='--', label=f'Limit ({force_limit}N)')
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('|F| (N)')
    ax11.set_title('Force Magnitude (World Frame)')
    ax11.grid(True)
    ax11.legend()
    
    # 12. Position Error
    ax12 = plt.subplot(4, 4, 12)
    ax12.plot(times, np.array(results_data['pos_error'])*1000, 'r-')
    ax12.set_xlabel('Time (s)')
    ax12.set_ylabel('Position Error (mm)')
    ax12.set_title('Tracking Error')
    ax12.grid(True)
    
    # 13. Force Fx (body frame)
    ax13 = plt.subplot(4, 4, 13)
    ax13.plot(times, F_body[:, 0], 'b-', linewidth=2, label='Fx (body)')
    ax13.set_xlabel('Time (s)')
    ax13.set_ylabel('Fx (N)')
    ax13.set_title('Force Fx (Body Frame)')
    ax13.grid(True)
    
    # 14. Force Fy (body frame)
    ax14 = plt.subplot(4, 4, 14)
    ax14.plot(times, F_body[:, 1], 'g-', linewidth=2, label='Fy (body)')
    ax14.set_xlabel('Time (s)')
    ax14.set_ylabel('Fy (N)')
    ax14.set_title('Force Fy (Body Frame)')
    ax14.grid(True)
    
    # 15. Force components (body frame) combined
    ax15 = plt.subplot(4, 4, 15)
    ax15.plot(times, F_body[:, 0], 'b-', linewidth=2, label='Fx (body)')
    ax15.plot(times, F_body[:, 1], 'g-', linewidth=2, label='Fy (body)')
    ax15.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax15.set_xlabel('Time (s)')
    ax15.set_ylabel('Force (N)')
    ax15.set_title('Force Components (Body Frame)')
    ax15.grid(True)
    
    # 16. Force magnitude (body frame)
    ax16 = plt.subplot(4, 4, 16)
    F_body_mag = np.sqrt(F_body[:, 0]**2 + F_body[:, 1]**2)
    ax16.plot(times, F_body_mag, 'm-', linewidth=2, label='|F| (body)')
    ax16.axhline(y=force_limit, color='r', linestyle='--', label=f'Limit ({force_limit}N)')
    ax16.set_xlabel('Time (s)')
    ax16.set_ylabel('|F| (N)')
    ax16.set_title('Force Magnitude (Body Frame)')
    ax16.grid(True)
    
    plt.tight_layout()
    plt.savefig('Case_6/control_results_closed_loop.png', dpi=300, bbox_inches='tight')
    print("Plots saved to Case_6/control_results_closed_loop.png")
    plt.show()

if __name__ == '__main__':
    results, trajectory, derivatives = run_trajectory_control(animate=True)
    save_and_plot_results(results, trajectory, derivatives)
