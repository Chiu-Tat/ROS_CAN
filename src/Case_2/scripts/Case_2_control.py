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
magnet_mass = 0.015  # kg
control_frequency = 10  # Hz (10ms intervals for smooth control)

# Magnet fixed positions
magnets_config = [
    {'X': -0.014, 'Z': -0.0015, 'alpha': np.pi/2, 'beta': np.pi/2, 'm': 0.145},  # Magnet 1 (Red)
    {'X': 0.014, 'Z': -0.0015, 'alpha': np.pi/2, 'beta': np.pi/2, 'm': 0.145}     # Magnet 3 (Blue)
]

def load_trajectory_results(filepath):
    """Load trajectory from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    control_points_data = data["control_points"]
    trajectory_data = {
        "control_points": {
            "Cx": np.array(control_points_data["Cx"]),
            "Cy": np.array(control_points_data["Cy"]),
        },
        "time_segments": data["time_segments"],
        "config": data["config"],
        "num_segments": data["num_segments"],
    }
    return trajectory_data

def evaluate_bezier_segment(Cx, Cy, Cz, n_degree, t_normalized):
    """Evaluate Bezier curve and its derivatives at normalized time t"""
    if isinstance(t_normalized, (int, float)):
        t_normalized = np.array([t_normalized])
    
    n = n_degree
    # Position - Bernstein basis
    bernstein = np.zeros((n + 1, len(t_normalized)))
    for i in range(n + 1):
        bernstein[i, :] = comb(n, i) * (t_normalized**i) * ((1 - t_normalized)**(n - i))
    
    px = bernstein.T @ Cx
    py = bernstein.T @ Cy
    pz = bernstein.T @ Cz if Cz is not None else None
    
    # First derivative - velocity
    if n > 0:
        bernstein_dot = np.zeros((n, len(t_normalized)))
        for i in range(n):
            bernstein_dot[i, :] = comb(n-1, i) * (t_normalized**i) * ((1 - t_normalized)**(n - 1 - i))
        
        delta_Cx = n * np.diff(Cx)
        delta_Cy = n * np.diff(Cy)
        delta_Cz = n * np.diff(Cz) if Cz is not None else None
        
        vx = bernstein_dot.T @ delta_Cx
        vy = bernstein_dot.T @ delta_Cy
        vz = bernstein_dot.T @ delta_Cz if delta_Cz is not None else None
    else:
        vx = vy = vz = np.zeros_like(px)
    
    # Second derivative - acceleration
    if n > 1:
        bernstein_ddot = np.zeros((n - 1, len(t_normalized)))
        for i in range(n - 1):
            bernstein_ddot[i, :] = comb(n-2, i) * (t_normalized**i) * ((1 - t_normalized)**(n - 2 - i))
        
        delta2_Cx = (n - 1) * np.diff(delta_Cx)
        delta2_Cy = (n - 1) * np.diff(delta_Cy)
        delta2_Cz = (n - 1) * np.diff(delta_Cz) if delta_Cz is not None else None
        
        ax = bernstein_ddot.T @ delta2_Cx
        ay = bernstein_ddot.T @ delta2_Cy
        az = bernstein_ddot.T @ delta2_Cz if delta2_Cz is not None else None
    else:
        ax = ay = az = np.zeros_like(px)
    
    return (px, py, pz), (vx, vy, vz), (ax, ay, az)

def generate_trajectory_kinematics(trajectory_data, dt):
    """Generate position, velocity, and acceleration along trajectory
    
    Note: For Case 2 (2 magnets), Cx represents Y positions for magnet 1 (Red)
    and Cy represents Y positions for magnet 3 (Blue)
    """
    Cx = trajectory_data["control_points"]["Cx"]  # Y position for magnet 1 (Red)
    Cy = trajectory_data["control_points"]["Cy"]  # Y position for magnet 3 (Blue)
    # Cz is not used
    time_segments = trajectory_data["time_segments"]
    n_degree = trajectory_data["config"]["N_DEGREE"]
    
    all_times = []
    all_positions = {'magnet1_y': [], 'magnet3_y': []}
    all_velocities = {'magnet1_y': [], 'magnet3_y': []}
    all_accelerations = {'magnet1_y': [], 'magnet3_y': []}
    
    cumulative_time = 0
    for seg_idx in range(len(time_segments)):
        T = time_segments[seg_idx]
        num_points = max(2, int(np.ceil(T / dt)))
        t_normalized = np.linspace(0, 1, num_points)
        
        segment_Cx = Cx[:, seg_idx]  # Magnet 1 Y trajectory
        segment_Cy = Cy[:, seg_idx]  # Magnet 3 Y trajectory
        
        pos, vel, acc = evaluate_bezier_segment(segment_Cx, segment_Cy, None, n_degree, t_normalized)
        
        # Convert to absolute time and scale derivatives
        times = cumulative_time + t_normalized * T
        all_times.extend(times)
        
        # Position (Cx -> magnet1_y, Cy -> magnet3_y)
        all_positions['magnet1_y'].extend(pos[0])
        all_positions['magnet3_y'].extend(pos[1])
        
        # Velocity: dy/dt = (dy/ds) / T
        all_velocities['magnet1_y'].extend(vel[0] / T)
        all_velocities['magnet3_y'].extend(vel[1] / T)
        
        # Acceleration: d²y/dt² = (d²y/ds²) / T²
        all_accelerations['magnet1_y'].extend(acc[0] / (T**2))
        all_accelerations['magnet3_y'].extend(acc[1] / (T**2))
        
        cumulative_time += T
    
    return {
        'time': np.array(all_times),
        'position': {k: np.array(v) for k, v in all_positions.items()},
        'velocity': {k: np.array(v) for k, v in all_velocities.items()},
        'acceleration': {k: np.array(v) for k, v in all_accelerations.items()}
    }

# def precompute_force_mapping(magnets_state):
#     """Pre-compute mapping matrix from currents to forces for all magnets"""
#     from WS_lib_new import Combined_Map_I2H
    
#     # Build target points list - only specify force requirements
#     target_points = []
#     for mag in magnets_state:
#         target_points.append({
#             'X': mag['X'], 'Y': mag['Y'], 'Z': mag['Z'],
#             'm': mag['m'], 'alpha': mag['alpha'], 'beta': mag['beta'],
#             'Bx': None, 'By': None, 'Bz': None,
#             'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None,
#             'By_dy': None, 'By_dz': None,
#             'fx': None, 'fy': True, 'fz': None,  # Only need fy component
#             'tx': None, 'ty': None, 'tz': None
#         })
    
#     A_matrix = Combined_Map_I2H(target_points)
#     return A_matrix

def precompute_force_mapping_coils_only(magnets_state):
    """Pre-compute mapping matrix from currents to forces (coils only, no magnet interactions)"""
    from WS_lib_new import calculate_Force_and_Torque
    import numpy as np
    
    n_coils = 10
    n_magnets = len(magnets_state)
    
    # Build mapping matrix: each row is fy for one magnet, each column is one coil
    A_matrix = np.zeros((n_magnets, n_coils))
    
    for mag_idx, mag in enumerate(magnets_state):
        for coil_idx in range(n_coils):
            # Unit current in this coil
            currents = np.zeros(n_coils)
            currents[coil_idx] = 1.0
            
            # Calculate force from this coil
            Force, _ = calculate_Force_and_Torque(
                currents,
                mag['X'], mag['Y'], mag['Z'],
                mag['m'], mag['alpha'], mag['beta']
            )
            
            # Store fy component (index 1)
            A_matrix[mag_idx, coil_idx] = Force[1]
    
    return A_matrix

def compute_magnet_interaction_forces(magnets_state):
    """Compute total magnet-magnet interaction forces for all magnets"""
    from WS_lib_new import calculate_Force_and_Torque_magnet
    import numpy as np
    
    n_magnets = len(magnets_state)
    interaction_forces = np.zeros(n_magnets)  # fy component only
    
    # # Debug: print magnet positions and orientations
    # print(f"\nDebug: Magnet states:")
    # for idx, mag in enumerate(magnets_state):
    #     print(f"  Magnet {idx+1}: X={mag['X']:.4f}, Y={mag['Y']:.4f}, Z={mag['Z']:.4f}, "
    #           f"alpha={mag['alpha']:.4f}, beta={mag['beta']:.4f}, m={mag['m']:.4f}")
    
    for i in range(n_magnets):
        for j in range(n_magnets):
            if i != j:
                # Force on magnet i due to magnet j
                Force_ij, _ = calculate_Force_and_Torque_magnet(magnets_state[i], magnets_state[j])
                
                # # Debug: print individual interaction
                # print(f"  Force on magnet {i+1} from magnet {j+1}: "
                #       f"fx={Force_ij[0]:.6e}, fy={Force_ij[1]:.6e}, fz={Force_ij[2]:.6e}")
                
                interaction_forces[i] += Force_ij[1]  # fy component
    
    # print(f"  Total interaction forces (fy): {interaction_forces}")
    return interaction_forces

def unconstrained_objective(currents, A_matrix, target_forces, penalty_weight=1e12):
    """Objective function with penalty method"""
    # Original objective: minimize power (sum of squares)
    original_obj = np.sum(currents**2)
    
    # Force constraint penalty
    predicted_forces = A_matrix @ currents
    force_error = np.sum((predicted_forces - target_forces)**2)
    
    # Current limit penalty
    current_violation = np.maximum(0, np.abs(currents) - current_limit)
    current_penalty = np.sum(current_violation**2)
    
    return original_obj + penalty_weight * (force_error + current_penalty)

def unconstrained_gradient(currents, A_matrix, target_forces, penalty_weight=1e12):
    """Analytical gradient for optimization"""
    # Gradient of original objective
    grad_original = 2 * currents
    
    # Gradient of force constraint penalty
    predicted_forces = A_matrix @ currents
    force_residual = predicted_forces - target_forces
    grad_force = 2 * penalty_weight * (A_matrix.T @ force_residual)
    
    # Gradient of current limit penalty
    current_violation = np.maximum(0, np.abs(currents) - current_limit)
    grad_current = 2 * penalty_weight * current_violation * np.sign(currents)
    
    return grad_original + grad_force + grad_current

def run_trajectory_control():
    """Main control loop for two magnets along trajectories"""
    # Load trajectory
    print("Loading trajectory data...")
    # Use absolute path relative to this script file
    trajectory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory_case2_2D.json")
    trajectory_data = load_trajectory_results(trajectory_path)
    
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
        'forces_desired': [],
        'forces_coil': [],
        'forces_interaction': [],
        'forces_actual': [],
        'optimization_time': [],
        'success': [],
        'max_force_error': [],
        'max_current': []
    }
    
    # Initial guess
    currents_guess = np.zeros(10)
    
    total_start_time = time.time()
    
    # Debug: Only process first few time steps to see interaction forces
    debug_steps = min(3, len(times))
    
    for i, t in enumerate(times[:debug_steps]):  # Only first few steps for debugging
        print(f"\n{'='*60}")
        print(f"Processing time {t:.3f}s ({i+1}/{debug_steps})")
        print(f"{'='*60}")
        
        # Get Y positions for each magnet (magnet1 and magnet3)
        y_pos_magnet1 = kinematics['position']['magnet1_y'][i]
        y_pos_magnet3 = kinematics['position']['magnet3_y'][i]
        
        # Build magnet states with current positions
        magnets_state = [
            {
                'X': magnets_config[0]['X'],
                'Y': y_pos_magnet1,
                'Z': magnets_config[0]['Z'],
                'm': magnets_config[0]['m'],
                'alpha': magnets_config[0]['alpha'],
                'beta': magnets_config[0]['beta']
            },
            {
                'X': magnets_config[1]['X'],
                'Y': y_pos_magnet3,
                'Z': magnets_config[1]['Z'],
                'm': magnets_config[1]['m'],
                'alpha': magnets_config[1]['alpha'],
                'beta': magnets_config[1]['beta']
            }
        ]
        
        # Step 1: Calculate total desired forces: F_total = ma (different for each magnet)
        ay_magnet1 = kinematics['acceleration']['magnet1_y'][i]
        ay_magnet3 = kinematics['acceleration']['magnet3_y'][i]
        
        total_desired_forces = np.array([
            magnet_mass * ay_magnet1,
            magnet_mass * ay_magnet3
        ])
        
        print(f"\nDesired accelerations: mag1={ay_magnet1:.6f}, mag3={ay_magnet3:.6f} m/s²")
        print(f"Total desired forces (F=ma): {total_desired_forces}")
        
        # Step 2: Compute magnet-magnet interaction forces
        interaction_forces = compute_magnet_interaction_forces(magnets_state)
        
        # Step 3: Calculate required force from coils: F_coil = F_total - F_interaction
        required_coil_forces = total_desired_forces - interaction_forces
        
        print(f"\nRequired coil forces: {required_coil_forces}")
        
        # Step 4: Build mapping matrix from currents to coil forces (without magnet interactions)
        A_matrix = precompute_force_mapping_coils_only(magnets_state)
        
        # Step 5: Optimize currents to achieve required coil forces
        opt_start_time = time.time()
        
        result = minimize(
            lambda x: unconstrained_objective(x, A_matrix, required_coil_forces),
            currents_guess,
            method='L-BFGS-B',
            jac=lambda x: unconstrained_gradient(x, A_matrix, required_coil_forces),
            options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1000}
        )
        
        opt_end_time = time.time()
        
        # Compute actual forces
        actual_coil_forces = A_matrix @ result.x
        actual_total_forces = actual_coil_forces + interaction_forces
        force_errors = actual_total_forces - total_desired_forces
        
        print(f"\nOptimized currents: {result.x}")
        print(f"Actual coil forces: {actual_coil_forces}")
        print(f"Actual total forces: {actual_total_forces}")
        print(f"Force errors: {force_errors}")
        
        # Store results
        results_data['time'].append(t)
        results_data['currents'].append(result.x.copy())
        results_data['forces_desired'].append(total_desired_forces.copy())
        results_data['forces_coil'].append(actual_coil_forces.copy())
        results_data['forces_interaction'].append(interaction_forces.copy())
        results_data['forces_actual'].append(actual_total_forces.copy())
        results_data['optimization_time'].append(opt_end_time - opt_start_time)
        results_data['success'].append(result.success)
        results_data['max_force_error'].append(np.max(np.abs(force_errors)))
        results_data['max_current'].append(np.max(np.abs(result.x)))
        
        # Warm start
        if result.success:
            currents_guess = result.x
    
    # Continue with remaining time steps without debug output
    for i, t in enumerate(times[debug_steps:], start=debug_steps):
        if i % 20 == 0:
            print(f"Processing time {t:.3f}s ({i+1}/{len(times)})")
        
        # Get Y positions for each magnet
        y_pos_magnet1 = kinematics['position']['magnet1_y'][i]
        y_pos_magnet3 = kinematics['position']['magnet3_y'][i]
        
        # Build magnet states
        magnets_state = [
            {
                'X': magnets_config[0]['X'], 'Y': y_pos_magnet1, 'Z': magnets_config[0]['Z'],
                'm': magnets_config[0]['m'], 'alpha': magnets_config[0]['alpha'], 'beta': magnets_config[0]['beta']
            },
            {
                'X': magnets_config[1]['X'], 'Y': y_pos_magnet3, 'Z': magnets_config[1]['Z'],
                'm': magnets_config[1]['m'], 'alpha': magnets_config[1]['alpha'], 'beta': magnets_config[1]['beta']
            }
        ]
        
        # Calculate desired forces
        ay_magnet1 = kinematics['acceleration']['magnet1_y'][i]
        ay_magnet3 = kinematics['acceleration']['magnet3_y'][i]
        
        total_desired_forces = np.array([
            magnet_mass * ay_magnet1,
            magnet_mass * ay_magnet3
        ])
        
        interaction_forces = compute_magnet_interaction_forces(magnets_state)
        required_coil_forces = total_desired_forces - interaction_forces
        A_matrix = precompute_force_mapping_coils_only(magnets_state)
        
        opt_start_time = time.time()
        result = minimize(
            lambda x: unconstrained_objective(x, A_matrix, required_coil_forces),
            currents_guess,
            method='L-BFGS-B',
            jac=lambda x: unconstrained_gradient(x, A_matrix, required_coil_forces),
            options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1000}
        )
        opt_end_time = time.time()
        
        actual_coil_forces = A_matrix @ result.x
        actual_total_forces = actual_coil_forces + interaction_forces
        force_errors = actual_total_forces - total_desired_forces
        
        results_data['time'].append(t)
        results_data['currents'].append(result.x.copy())
        results_data['forces_desired'].append(total_desired_forces.copy())
        results_data['forces_coil'].append(actual_coil_forces.copy())
        results_data['forces_interaction'].append(interaction_forces.copy())
        results_data['forces_actual'].append(actual_total_forces.copy())
        results_data['optimization_time'].append(opt_end_time - opt_start_time)
        results_data['success'].append(result.success)
        results_data['max_force_error'].append(np.max(np.abs(force_errors)))
        results_data['max_current'].append(np.max(np.abs(result.x)))
        
        if result.success:
            currents_guess = result.x
    
    total_end_time = time.time()
    
    print(f"\n=== TRAJECTORY CONTROL COMPLETED ===")
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Average optimization time: {np.mean(results_data['optimization_time']):.4f} seconds")
    print(f"Success rate: {np.mean(results_data['success']):.1%}")
    print(f"Average max force error: {np.mean(results_data['max_force_error']):.2e} N")
    print(f"Max current: {np.max(results_data['max_current']):.2f} A")
    
    return results_data, kinematics

def save_and_plot_results(results_data, kinematics):
    """Save results to CSV and plot"""
    times = np.array(results_data['time'])
    currents_array = np.array(results_data['currents'])
    forces_coil = np.array(results_data['forces_coil'])
    forces_interaction = np.array(results_data['forces_interaction'])
    forces_actual = np.array(results_data['forces_actual'])
    forces_desired = np.array(results_data['forces_desired'])
    
    # Save currents to CSV
    currents_df = pd.DataFrame()
    currents_df['Time(s)'] = times
    for i in range(10):
        currents_df[f'Current_Coil_{i+1}(A)'] = currents_array[:, i]
    
    output_path = "Case_2/coil_currents_case2.csv"
    currents_df.to_csv(output_path, index=False)
    print(f"\nCurrents saved to {output_path}")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Currents
    axes[0, 0].plot(times, currents_array)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Current (A)')
    axes[0, 0].set_title('Coil Currents Over Time')
    axes[0, 0].grid(True)
    axes[0, 0].legend([f'Coil {i+1}' for i in range(10)], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Trajectories for all two magnets
    axes[0, 1].plot(kinematics['time'], kinematics['position']['magnet1_y'], 'r-', label='Magnet 1 (Red)', linewidth=2)
    # axes[0, 1].plot(kinematics['time'], kinematics['position']['magnet2_y'], 'g-', label='Magnet 2', linewidth=2)
    axes[0, 1].plot(kinematics['time'], kinematics['position']['magnet3_y'], 'b-', label='Magnet 3 (Blue)', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Y Position (m)')
    axes[0, 1].set_title('Y Trajectories (2 Magnets)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Force breakdown for magnet 1
    axes[0, 2].plot(times, forces_desired[:, 0], 'k-', label='Total Desired', linewidth=2)
    axes[0, 2].plot(times, forces_coil[:, 0], 'b--', label='Coil Force')
    axes[0, 2].plot(times, forces_interaction[:, 0], 'r--', label='Interaction Force')
    axes[0, 2].plot(times, forces_actual[:, 0], 'g:', label='Actual Total', linewidth=2)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Force (N)')
    axes[0, 2].set_title('Force Breakdown - Magnet 1')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Force errors
    axes[1, 0].plot(times, results_data['max_force_error'], 'r-')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Max Force Error (N)')
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
    
    # Interaction forces for all magnets
    axes[1, 2].plot(times, forces_interaction[:, 0], label='Magnet 1')
    # axes[1, 2].plot(times, forces_interaction[:, 1], label='Magnet 2')
    axes[1, 2].plot(times, forces_interaction[:, 1], label='Magnet 3')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Interaction Force (N)')
    axes[1, 2].set_title('Magnet-Magnet Interaction Forces')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('Case_2/control_results_case2.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    results, kinematics = run_trajectory_control()
    save_and_plot_results(results, kinematics)
