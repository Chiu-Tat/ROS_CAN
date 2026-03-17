import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json
import time
import matplotlib.pyplot as plt
from scipy.special import comb
import sys
import os

# Add parent directory to path to import WS_lib_new
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from WS_lib_new import calculate_Force_and_Torque

# Configuration
current_limit = 15  # Amperes
magnet_mass = 0.0015  # kg
control_frequency = 10  # Hz (10ms intervals for smooth control)

# Single magnet configuration (Magnet 2 - middle magnet)
magnet_config = {'X': 0.0, 'Z': -0.0015, 'alpha': np.pi/2, 'beta': np.pi/2, 'm': 0.145}

def load_trajectory_results(filepath):
    """Load trajectory from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    control_points_data = data["control_points"]
    trajectory_data = {
        "control_points": {
            # Only use Cy for magnet 2
            "Cy": np.array(control_points_data["Cy"]),
        },
        "time_segments": data["time_segments"],
        "config": data["config"],
        "num_segments": data["num_segments"],
    }
    return trajectory_data

def evaluate_bezier_segment(Cy, n_degree, t_normalized):
    """Evaluate Bezier curve and its derivatives at normalized time t"""
    if isinstance(t_normalized, (int, float)):
        t_normalized = np.array([t_normalized])
    
    n = n_degree
    # Position - Bernstein basis
    bernstein = np.zeros((n + 1, len(t_normalized)))
    for i in range(n + 1):
        bernstein[i, :] = comb(n, i) * (t_normalized**i) * ((1 - t_normalized)**(n - i))
    
    py = bernstein.T @ Cy
    
    # First derivative - velocity
    if n > 0:
        bernstein_dot = np.zeros((n, len(t_normalized)))
        for i in range(n):
            bernstein_dot[i, :] = comb(n-1, i) * (t_normalized**i) * ((1 - t_normalized)**(n - 1 - i))
        
        delta_Cy = n * np.diff(Cy)
        
        vy = bernstein_dot.T @ delta_Cy
    else:
        vy = np.zeros_like(py)
    
    # Second derivative - acceleration
    if n > 1:
        bernstein_ddot = np.zeros((n - 1, len(t_normalized)))
        for i in range(n - 1):
            bernstein_ddot[i, :] = comb(n-2, i) * (t_normalized**i) * ((1 - t_normalized)**(n - 2 - i))
        
        delta2_Cy = (n - 1) * np.diff(delta_Cy)
        
        ay = bernstein_ddot.T @ delta2_Cy
    else:
        ay = np.zeros_like(py)
    
    return py, vy, ay

def generate_trajectory_kinematics(trajectory_data, dt):
    """Generate position, velocity, and acceleration along trajectory for single magnet"""
    Cy = trajectory_data["control_points"]["Cy"]  # Y position for magnet 2
    time_segments = trajectory_data["time_segments"]
    n_degree = trajectory_data["config"]["N_DEGREE"]
    
    all_times = []
    all_positions = []
    all_velocities = []
    all_accelerations = []
    
    cumulative_time = 0
    for seg_idx in range(len(time_segments)):
        T = time_segments[seg_idx]
        num_points = max(2, int(np.ceil(T / dt)))
        t_normalized = np.linspace(0, 1, num_points)
        
        segment_Cy = Cy[:, seg_idx]
        
        pos, vel, acc = evaluate_bezier_segment(segment_Cy, n_degree, t_normalized)
        
        # Convert to absolute time and scale derivatives
        times = cumulative_time + t_normalized * T
        all_times.extend(times)
        
        # Position
        all_positions.extend(pos)
        
        # Velocity: dy/dt = (dy/ds) / T
        all_velocities.extend(vel / T)
        
        # Acceleration: d²y/dt² = (d²y/ds²) / T²
        all_accelerations.extend(acc / (T**2))
        
        cumulative_time += T
    
    return {
        'time': np.array(all_times),
        'position': np.array(all_positions),
        'velocity': np.array(all_velocities),
        'acceleration': np.array(all_accelerations)
    }

def precompute_force_mapping_single_magnet(magnet_state):
    """Pre-compute mapping matrix from currents to force for single magnet"""
    from WS_lib_new import calculate_Force_and_Torque
    import numpy as np
    
    n_coils = 10
    
    # Build mapping vector: fy for one magnet from each coil
    A_vector = np.zeros(n_coils)
    
    for coil_idx in range(n_coils):
        # Unit current in this coil
        currents = np.zeros(n_coils)
        currents[coil_idx] = 1.0
        
        # Calculate force from this coil
        Force, _ = calculate_Force_and_Torque(
            currents,
            magnet_state['X'], magnet_state['Y'], magnet_state['Z'],
            magnet_state['m'], magnet_state['alpha'], magnet_state['beta']
        )
        
        # Store fy component (index 1)
        A_vector[coil_idx] = Force[1]
    
    return A_vector

def unconstrained_objective(currents, A_vector, target_force, penalty_weight=1e12):
    """Objective function with penalty method for single magnet"""
    # Original objective: minimize power (sum of squares)
    original_obj = np.sum(currents**2)
    
    # Force constraint penalty
    predicted_force = A_vector @ currents
    force_error = (predicted_force - target_force)**2
    
    # Current limit penalty
    current_violation = np.maximum(0, np.abs(currents) - current_limit)
    current_penalty = np.sum(current_violation**2)
    
    return original_obj + penalty_weight * (force_error + current_penalty)

def unconstrained_gradient(currents, A_vector, target_force, penalty_weight=1e12):
    """Analytical gradient for optimization"""
    # Gradient of original objective
    grad_original = 2 * currents
    
    # Gradient of force constraint penalty
    predicted_force = A_vector @ currents
    force_residual = predicted_force - target_force
    grad_force = 2 * penalty_weight * force_residual * A_vector
    
    # Gradient of current limit penalty
    current_violation = np.maximum(0, np.abs(currents) - current_limit)
    grad_current = 2 * penalty_weight * current_violation * np.sign(currents)
    
    return grad_original + grad_force + grad_current

def run_trajectory_control():
    """Main control loop for single magnet along trajectory"""
    # Load trajectory
    print("Loading trajectory data...")
    trajectory_data = load_trajectory_results("src/Case_2/scripts/trajectory_case2_3D.json")
    
    # Generate kinematics at control frequency
    dt = 1.0 / control_frequency
    print(f"Generating trajectory kinematics at {control_frequency} Hz...")
    kinematics = generate_trajectory_kinematics(trajectory_data, dt)
    
    times = kinematics['time']
    print(f"Total time points: {len(times)}, Duration: {times[-1]:.3f} seconds")
    
    # Storage for results
    results_data = {
        'time': [],
        'currents': [],
        'force_desired': [],
        'force_actual': [],
        'optimization_time': [],
        'success': [],
        'force_error': [],
        'max_current': []
    }
    
    # Initial guess
    currents_guess = np.zeros(10)
    
    total_start_time = time.time()
    
    for i, t in enumerate(times):
        if i % 20 == 0 or i < 3:
            print(f"\nProcessing time {t:.3f}s ({i+1}/{len(times)})")
        
        # Get Y position for magnet 2
        y_pos = kinematics['position'][i]
        
        # Build magnet state with current position
        magnet_state = {
            'X': magnet_config['X'],
            'Y': y_pos,
            'Z': magnet_config['Z'],
            'm': magnet_config['m'],
            'alpha': magnet_config['alpha'],
            'beta': magnet_config['beta']
        }
        
        # Calculate desired force: F = ma
        ay = kinematics['acceleration'][i]
        desired_force = magnet_mass * ay
        
        if i < 3:
            print(f"  Desired acceleration: {ay:.6f} m/s²")
            print(f"  Desired force (F=ma): {desired_force:.6e} N")
        
        # Build mapping vector from currents to force (no magnet interactions)
        A_vector = precompute_force_mapping_single_magnet(magnet_state)
        
        # Optimize currents to achieve desired force
        opt_start_time = time.time()
        
        result = minimize(
            lambda x: unconstrained_objective(x, A_vector, desired_force),
            currents_guess,
            method='L-BFGS-B',
            jac=lambda x: unconstrained_gradient(x, A_vector, desired_force),
            options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1000}
        )
        
        opt_end_time = time.time()
        
        # Compute actual force
        actual_force = A_vector @ result.x
        force_error = actual_force - desired_force
        
        if i < 3:
            print(f"  Optimized currents: {result.x}")
            print(f"  Actual force: {actual_force:.6e} N")
            print(f"  Force error: {force_error:.6e} N")
        
        # Store results
        results_data['time'].append(t)
        results_data['currents'].append(result.x.copy())
        results_data['force_desired'].append(desired_force)
        results_data['force_actual'].append(actual_force)
        results_data['optimization_time'].append(opt_end_time - opt_start_time)
        results_data['success'].append(result.success)
        results_data['force_error'].append(abs(force_error))
        results_data['max_current'].append(np.max(np.abs(result.x)))
        
        # Warm start
        if result.success:
            currents_guess = result.x
    
    total_end_time = time.time()
    
    print(f"\n=== TRAJECTORY CONTROL COMPLETED ===")
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Average optimization time: {np.mean(results_data['optimization_time']):.4f} seconds")
    print(f"Success rate: {np.mean(results_data['success']):.1%}")
    print(f"Average force error: {np.mean(results_data['force_error']):.2e} N")
    print(f"Max force error: {np.max(results_data['force_error']):.2e} N")
    print(f"Max current: {np.max(results_data['max_current']):.2f} A")
    
    return results_data, kinematics

def save_and_plot_results(results_data, kinematics):
    """Save results to CSV and plot"""
    times = np.array(results_data['time'])
    currents_array = np.array(results_data['currents'])
    forces_actual = np.array(results_data['force_actual'])
    forces_desired = np.array(results_data['force_desired'])
    
    # Save currents to CSV
    # currents_df = pd.DataFrame()
    # currents_df['Time(s)'] = times
    # for i in range(10):
    #     currents_df[f'Current_Coil_{i+1}(A)'] = currents_array[:, i]
    
    # output_path = "Case_2/coil_currents_case2_single.csv"
    # currents_df.to_csv(output_path, index=False)
    # print(f"\nCurrents saved to {output_path}")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Currents
    axes[0, 0].plot(times, currents_array)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Current (A)')
    axes[0, 0].set_title('Coil Currents Over Time')
    axes[0, 0].grid(True)
    axes[0, 0].legend([f'Coil {i+1}' for i in range(10)], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Trajectory
    axes[0, 1].plot(kinematics['time'], kinematics['position'], 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Y Position (m)')
    axes[0, 1].set_title('Y Trajectory (Magnet 2)')
    axes[0, 1].grid(True)
    
    # Force comparison
    axes[0, 2].plot(times, forces_desired, 'k-', label='Desired', linewidth=2)
    axes[0, 2].plot(times, forces_actual, 'r--', label='Actual', linewidth=1.5)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Force (N)')
    axes[0, 2].set_title('Force Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Force error
    axes[1, 0].plot(times, results_data['force_error'], 'r-')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Force Error (N)')
    axes[1, 0].set_title('Force Tracking Error')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Current limits
    axes[1, 1].plot(times, results_data['max_current'], 'b-', label='Max Current')
    axes[1, 1].axhline(y=current_limit, color='r', linestyle='--', label=f'Limit ({current_limit}A)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Current (A)')
    axes[1, 1].set_title('Maximum Current vs Limit')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    # Acceleration profile
    axes[1, 2].plot(kinematics['time'], kinematics['acceleration'], 'g-')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Acceleration (m/s²)')
    axes[1, 2].set_title('Y Acceleration Profile')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    # plt.savefig('Case_2/control_results_case2_single.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    results, kinematics = run_trajectory_control()
    save_and_plot_results(results, kinematics)
