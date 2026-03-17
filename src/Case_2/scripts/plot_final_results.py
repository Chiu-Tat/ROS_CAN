import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
# Read the magnet positions CSV file
magnet_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'magnet_positions_recorded.csv')
magnet_data = pd.read_csv(magnet_data_path)

# Read the trajectory CSV file
trajectory_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trajectory_case2_3D_sampled.csv')
trajectory_data = pd.read_csv(trajectory_data_path)

# Create figure with three subplots
fig = plt.figure(figsize=(18, 6))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)

# Plot 1: Magnet 1 (Desired vs Actual Y position)
ax1.plot(magnet_data['Time(s)'], magnet_data['Desired_Magnet1_Y(m)'], 'b-', linewidth=1, label='Desired', alpha=0.8)
ax1.plot(magnet_data['Time(s)'], magnet_data['Actual_Magnet1_Y(m)'], 'r--', linewidth=1, label='Actual', alpha=0.8)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Y Position (m)', fontsize=12)
ax1.set_title('Magnet 1: Desired vs Actual Y Position', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Magnet 3 (Desired vs Actual Y position)
ax2.plot(magnet_data['Time(s)'], magnet_data['Desired_Magnet3_Y(m)'], 'b-', linewidth=1.5, label='Desired', alpha=0.8)
ax2.plot(magnet_data['Time(s)'], magnet_data['Actual_Magnet3_Y(m)'], 'r--', linewidth=1.5, label='Actual', alpha=0.8)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Y Position (m)', fontsize=12)
ax2.set_title('Magnet 3: Desired vs Actual Y Position', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Magnet 1 Y vs Magnet 3 Y (Desired and Actual comparison)
ax3.plot(magnet_data['Desired_Magnet1_Y(m)'], magnet_data['Desired_Magnet3_Y(m)'], 'b-', linewidth=1.5, label='Desired', alpha=0.8)
ax3.plot(magnet_data['Actual_Magnet1_Y(m)'], magnet_data['Actual_Magnet3_Y(m)'], 'r--', linewidth=1.5, label='Actual', alpha=0.8)
ax3.set_xlabel('Magnet 1 Y Position (m)', fontsize=12)
ax3.set_ylabel('Magnet 3 Y Position (m)', fontsize=12)
ax3.set_title('Magnet 1 Y vs Magnet 3 Y: Desired vs Actual', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.axis('equal')

plt.tight_layout()
plt.savefig('magnet_and_trajectory_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate and print tracking errors
magnet1_error = np.abs(magnet_data['Actual_Magnet1_Y(m)'] - magnet_data['Desired_Magnet1_Y(m)'])
magnet3_error = np.abs(magnet_data['Actual_Magnet3_Y(m)'] - magnet_data['Desired_Magnet3_Y(m)'])

print(f"Total time: {magnet_data['Time(s)'].max():.2f} seconds")
print(f"\nMagnet 1 Tracking Error:")
print(f"  Mean: {magnet1_error.mean()*1000:.4f} mm")
print(f"  Max: {magnet1_error.max()*1000:.4f} mm")
print(f"  RMS: {np.sqrt((magnet1_error**2).mean())*1000:.4f} mm")
print(f"\nMagnet 3 Tracking Error:")
print(f"  Mean: {magnet3_error.mean()*1000:.4f} mm")
print(f"  Max: {magnet3_error.max()*1000:.4f} mm")
print(f"  RMS: {np.sqrt((magnet3_error**2).mean())*1000:.4f} mm")
print(f"\nTrajectory Statistics:")
print(f"X range: [{trajectory_data['x'].min():.4f}, {trajectory_data['x'].max():.4f}] m")
print(f"Y range: [{trajectory_data['y'].min():.4f}, {trajectory_data['y'].max():.4f}] m")
print(f"Z range: [{trajectory_data['z'].min():.4f}, {trajectory_data['z'].max():.4f}] m")
