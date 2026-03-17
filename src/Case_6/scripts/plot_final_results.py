import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Read the closed loop results CSV file (Case 6 format)
results_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'closed_loop_results.csv')
results_data = pd.read_csv(results_data_path)

# Convert to numeric (should already be numeric, but ensure it)
results_data['Time(s)'] = pd.to_numeric(results_data['Time(s)'], errors='coerce')
results_data['Desired_X(m)'] = pd.to_numeric(results_data['Desired_X(m)'], errors='coerce')
results_data['Desired_Y(m)'] = pd.to_numeric(results_data['Desired_Y(m)'], errors='coerce')
results_data['Desired_Theta(rad)'] = pd.to_numeric(results_data['Desired_Theta(rad)'], errors='coerce')
results_data['Actual_X(m)'] = pd.to_numeric(results_data['Actual_X(m)'], errors='coerce')
results_data['Actual_Y(m)'] = pd.to_numeric(results_data['Actual_Y(m)'], errors='coerce')
results_data['Actual_Theta(rad)'] = pd.to_numeric(results_data['Actual_Theta(rad)'], errors='coerce')

# Filter out rows where we have no data at all
valid_mask = (results_data['Desired_X(m)'].notna() | results_data['Actual_X(m)'].notna())
results_data = results_data[valid_mask].copy()

# Create aliases for easier access (matching the rest of the code)
sphere_data = results_data.copy()
sphere_data['Desired_X'] = sphere_data['Desired_X(m)']
sphere_data['Desired_Y'] = sphere_data['Desired_Y(m)']
sphere_data['Desired_Theta'] = sphere_data['Desired_Theta(rad)']
sphere_data['Green_X'] = sphere_data['Actual_X(m)']
sphere_data['Green_Y'] = sphere_data['Actual_Y(m)']
sphere_data['Green_Theta'] = sphere_data['Actual_Theta(rad)']
# Time column already exists as 'Time(s)'

# Optionally load trajectory JSON for reference
trajectory_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trajectory_case6_3D_02.json')
trajectory_data = None
if os.path.exists(trajectory_json_path):
    try:
        with open(trajectory_json_path, 'r') as f:
            traj_json = json.load(f)
            trajectory_data = np.array(traj_json['trajectory'])
            traj_times = trajectory_data[:, 3]
            traj_x = trajectory_data[:, 0]
            traj_y = trajectory_data[:, 1]
    except Exception as e:
        print(f"Warning: Could not load trajectory JSON: {e}")

# Create figure with multiple subplots (2 rows, 4 columns)
fig = plt.figure(figsize=(24, 12))

# Plot 1: X Position vs Time
ax1 = plt.subplot(2, 4, 1)
if sphere_data['Desired_X'].notna().any():
    ax1.plot(sphere_data['Time(s)'], sphere_data['Desired_X'], 'b-', linewidth=2, label='Desired', alpha=0.8)
if sphere_data['Green_X'].notna().any():
    ax1.plot(sphere_data['Time(s)'], sphere_data['Green_X'], 'r--', linewidth=1.5, label='Actual', alpha=0.8)
if trajectory_data is not None:
    ax1.plot(traj_times, traj_x, 'g:', linewidth=1, label='Trajectory (ref)', alpha=0.6)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('X Position (m)', fontsize=12)
ax1.set_title('X Position: Desired vs Actual', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Y Position vs Time
ax2 = plt.subplot(2, 4, 2)
if sphere_data['Desired_Y'].notna().any():
    ax2.plot(sphere_data['Time(s)'], sphere_data['Desired_Y'], 'b-', linewidth=2, label='Desired', alpha=0.8)
if sphere_data['Green_Y'].notna().any():
    ax2.plot(sphere_data['Time(s)'], sphere_data['Green_Y'], 'r--', linewidth=1.5, label='Actual', alpha=0.8)
if trajectory_data is not None:
    ax2.plot(traj_times, traj_y, 'g:', linewidth=1, label='Trajectory (ref)', alpha=0.6)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Y Position (m)', fontsize=12)
ax2.set_title('Y Position: Desired vs Actual', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Trajectory in XY Plane
ax3 = plt.subplot(2, 4, 3)
if sphere_data['Desired_X'].notna().any() and sphere_data['Desired_Y'].notna().any():
    desired_mask = sphere_data['Desired_X'].notna() & sphere_data['Desired_Y'].notna()
    ax3.plot(sphere_data.loc[desired_mask, 'Desired_X'], 
             sphere_data.loc[desired_mask, 'Desired_Y'], 
             'b-', linewidth=2, label='Desired', alpha=0.8, marker='o', markersize=2)
if sphere_data['Green_X'].notna().any() and sphere_data['Green_Y'].notna().any():
    actual_mask = sphere_data['Green_X'].notna() & sphere_data['Green_Y'].notna()
    ax3.plot(sphere_data.loc[actual_mask, 'Green_X'], 
             sphere_data.loc[actual_mask, 'Green_Y'], 
             'r--', linewidth=1.5, label='Actual', alpha=0.8, marker='s', markersize=2)
if trajectory_data is not None:
    ax3.plot(traj_x, traj_y, 'g:', linewidth=1, label='Trajectory (ref)', alpha=0.6)
# Mark start and end points
if sphere_data['Desired_X'].notna().any() and sphere_data['Desired_Y'].notna().any():
    first_valid = sphere_data[sphere_data['Desired_X'].notna() & sphere_data['Desired_Y'].notna()].iloc[0]
    last_valid = sphere_data[sphere_data['Desired_X'].notna() & sphere_data['Desired_Y'].notna()].iloc[-1]
    ax3.plot(first_valid['Desired_X'], first_valid['Desired_Y'], 'go', markersize=10, label='Start', zorder=5)
    ax3.plot(last_valid['Desired_X'], last_valid['Desired_Y'], 'ro', markersize=10, label='End', zorder=5)
ax3.set_xlabel('X Position (m)', fontsize=12)
ax3.set_ylabel('Y Position (m)', fontsize=12)
ax3.set_title('Trajectory in XY Plane: Desired vs Actual', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.axis('equal')

# Plot 4: X Position Error vs Time
ax4 = plt.subplot(2, 4, 4)
# Calculate error only where both desired and actual are available
error_mask = sphere_data['Desired_X'].notna() & sphere_data['Green_X'].notna()
if error_mask.any():
    x_error = sphere_data.loc[error_mask, 'Green_X'] - sphere_data.loc[error_mask, 'Desired_X']
    ax4.plot(sphere_data.loc[error_mask, 'Time(s)'], x_error * 1000, 'r-', linewidth=1.5, alpha=0.8)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Time (s)', fontsize=12)
ax4.set_ylabel('X Error (mm)', fontsize=12)
ax4.set_title('X Position Tracking Error', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Y Position Error vs Time
ax5 = plt.subplot(2, 4, 5)
# Calculate error only where both desired and actual are available
error_mask = sphere_data['Desired_Y'].notna() & sphere_data['Green_Y'].notna()
if error_mask.any():
    y_error = sphere_data.loc[error_mask, 'Green_Y'] - sphere_data.loc[error_mask, 'Desired_Y']
    ax5.plot(sphere_data.loc[error_mask, 'Time(s)'], y_error * 1000, 'r-', linewidth=1.5, alpha=0.8)
    ax5.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax5.set_xlabel('Time (s)', fontsize=12)
ax5.set_ylabel('Y Error (mm)', fontsize=12)
ax5.set_title('Y Position Tracking Error', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Position Error Magnitude vs Time
ax6 = plt.subplot(2, 4, 6)
error_mask = (sphere_data['Desired_X'].notna() & sphere_data['Green_X'].notna() & 
              sphere_data['Desired_Y'].notna() & sphere_data['Green_Y'].notna())
if error_mask.any():
    x_error = sphere_data.loc[error_mask, 'Green_X'] - sphere_data.loc[error_mask, 'Desired_X']
    y_error = sphere_data.loc[error_mask, 'Green_Y'] - sphere_data.loc[error_mask, 'Desired_Y']
    error_mag = np.sqrt(x_error**2 + y_error**2)
    ax6.plot(sphere_data.loc[error_mask, 'Time(s)'], error_mag * 1000, 'r-', linewidth=1.5, alpha=0.8)
    ax6.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax6.set_xlabel('Time (s)', fontsize=12)
ax6.set_ylabel('Position Error Magnitude (mm)', fontsize=12)
ax6.set_title('Position Error Magnitude', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot 7: Angle (Theta) vs Time
ax7 = plt.subplot(2, 4, 7)
if sphere_data['Desired_Theta'].notna().any():
    # Convert radians to degrees for display
    desired_theta_deg = np.degrees(sphere_data['Desired_Theta'])
    ax7.plot(sphere_data['Time(s)'], desired_theta_deg, 'b-', linewidth=2, label='Desired', alpha=0.8)
if sphere_data['Green_Theta'].notna().any():
    actual_theta_deg = np.degrees(sphere_data['Green_Theta'])
    ax7.plot(sphere_data['Time(s)'], actual_theta_deg, 'r--', linewidth=1.5, label='Actual', alpha=0.8)
ax7.set_xlabel('Time (s)', fontsize=12)
ax7.set_ylabel('Angle (degrees)', fontsize=12)
ax7.set_title('Angle (Theta): Desired vs Actual', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend()

# Plot 8: Angle Error vs Time
ax8 = plt.subplot(2, 4, 8)
error_mask_theta = sphere_data['Desired_Theta'].notna() & sphere_data['Green_Theta'].notna()
if error_mask_theta.any():
    theta_error = sphere_data.loc[error_mask_theta, 'Green_Theta'] - sphere_data.loc[error_mask_theta, 'Desired_Theta']
    # Normalize angle error to [-pi, pi]
    theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
    ax8.plot(sphere_data.loc[error_mask_theta, 'Time(s)'], np.degrees(theta_error), 'r-', linewidth=1.5, alpha=0.8)
    ax8.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax8.set_xlabel('Time (s)', fontsize=12)
ax8.set_ylabel('Angle Error (degrees)', fontsize=12)
ax8.set_title('Angle Tracking Error', fontsize=14, fontweight='bold')
ax8.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'case6_trajectory_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
plt.show()

# Calculate and print tracking errors
print("\n" + "="*60)
print("TRACKING PERFORMANCE ANALYSIS")
print("="*60)

if sphere_data['Time(s)'].notna().any():
    total_time = sphere_data['Time(s)'].max() - sphere_data['Time(s)'].min()
    print(f"\nTotal recording time: {total_time:.2f} seconds")
    print(f"Data points: {len(sphere_data)}")

# X Position Error
error_mask_x = sphere_data['Desired_X'].notna() & sphere_data['Green_X'].notna()
if error_mask_x.any():
    x_error = sphere_data.loc[error_mask_x, 'Green_X'] - sphere_data.loc[error_mask_x, 'Desired_X']
    x_error_abs = np.abs(x_error)
    print(f"\nX Position Tracking Error:")
    print(f"  Mean: {x_error_abs.mean()*1000:.4f} mm")
    print(f"  Max: {x_error_abs.max()*1000:.4f} mm")
    print(f"  RMS: {np.sqrt((x_error**2).mean())*1000:.4f} mm")
    print(f"  Std: {x_error.std()*1000:.4f} mm")
else:
    print("\nX Position Tracking Error: No valid data")

# Y Position Error
error_mask_y = sphere_data['Desired_Y'].notna() & sphere_data['Green_Y'].notna()
if error_mask_y.any():
    y_error = sphere_data.loc[error_mask_y, 'Green_Y'] - sphere_data.loc[error_mask_y, 'Desired_Y']
    y_error_abs = np.abs(y_error)
    print(f"\nY Position Tracking Error:")
    print(f"  Mean: {y_error_abs.mean()*1000:.4f} mm")
    print(f"  Max: {y_error_abs.max()*1000:.4f} mm")
    print(f"  RMS: {np.sqrt((y_error**2).mean())*1000:.4f} mm")
    print(f"  Std: {y_error.std()*1000:.4f} mm")
else:
    print("\nY Position Tracking Error: No valid data")

# Position Error Magnitude
error_mask_xy = (sphere_data['Desired_X'].notna() & sphere_data['Green_X'].notna() & 
                 sphere_data['Desired_Y'].notna() & sphere_data['Green_Y'].notna())
if error_mask_xy.any():
    x_error = sphere_data.loc[error_mask_xy, 'Green_X'] - sphere_data.loc[error_mask_xy, 'Desired_X']
    y_error = sphere_data.loc[error_mask_xy, 'Green_Y'] - sphere_data.loc[error_mask_xy, 'Desired_Y']
    error_mag = np.sqrt(x_error**2 + y_error**2)
    print(f"\nPosition Error Magnitude:")
    print(f"  Mean: {error_mag.mean()*1000:.4f} mm")
    print(f"  Max: {error_mag.max()*1000:.4f} mm")
    print(f"  RMS: {np.sqrt((error_mag**2).mean())*1000:.4f} mm")
else:
    print("\nPosition Error Magnitude: No valid data")

# Angle Error
error_mask_theta = sphere_data['Desired_Theta'].notna() & sphere_data['Green_Theta'].notna()
if error_mask_theta.any():
    theta_error = sphere_data.loc[error_mask_theta, 'Green_Theta'] - sphere_data.loc[error_mask_theta, 'Desired_Theta']
    # Normalize angle error to [-pi, pi]
    theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
    theta_error_abs = np.abs(theta_error)
    print(f"\nAngle (Theta) Tracking Error:")
    print(f"  Mean: {np.degrees(theta_error_abs.mean()):.4f} degrees")
    print(f"  Max: {np.degrees(theta_error_abs.max()):.4f} degrees")
    print(f"  RMS: {np.degrees(np.sqrt((theta_error**2).mean())):.4f} degrees")
    print(f"  Std: {np.degrees(theta_error.std()):.4f} degrees")
else:
    print("\nAngle (Theta) Tracking Error: No valid data")

# Trajectory statistics
print(f"\nTrajectory Statistics:")
if sphere_data['Desired_X'].notna().any():
    print(f"  Desired X range: [{sphere_data['Desired_X'].min():.6f}, {sphere_data['Desired_X'].max():.6f}] m")
if sphere_data['Desired_Y'].notna().any():
    print(f"  Desired Y range: [{sphere_data['Desired_Y'].min():.6f}, {sphere_data['Desired_Y'].max():.6f}] m")
if sphere_data['Green_X'].notna().any():
    print(f"  Actual X range: [{sphere_data['Green_X'].min():.6f}, {sphere_data['Green_X'].max():.6f}] m")
if sphere_data['Green_Y'].notna().any():
    print(f"  Actual Y range: [{sphere_data['Green_Y'].min():.6f}, {sphere_data['Green_Y'].max():.6f}] m")

if trajectory_data is not None:
    print(f"\nReference Trajectory Statistics:")
    print(f"  X range: [{traj_x.min():.6f}, {traj_x.max():.6f}] m")
    print(f"  Y range: [{traj_y.min():.6f}, {traj_y.max():.6f}] m")
    print(f"  Duration: {traj_times.max():.2f} seconds")

print("\n" + "="*60)
