import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

# Add current directory to path to allow importing Case_2_control
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from Case_2_control import load_trajectory_results, generate_trajectory_kinematics, control_frequency
except ImportError:
    print("Error: Could not import from Case_2_control.py. Make sure it is in the same directory.")
    sys.exit(1)

def plot_trajectory():
    """Load trajectory data and plot the paths for Red and Blue magnets"""
    
    # Path to the trajectory JSON file
    trajectory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory_case2_2D.json")
    
    if not os.path.exists(trajectory_path):
        print(f"Error: Trajectory file not found at {trajectory_path}")
        return

    print(f"Loading trajectory from {trajectory_path}...")
    
    # Load trajectory data
    trajectory_data = load_trajectory_results(trajectory_path)
    
    # Generate kinematics at control frequency
    dt = 1.0 / control_frequency
    print(f"Generating trajectory kinematics at {control_frequency} Hz...")
    kinematics = generate_trajectory_kinematics(trajectory_data, dt)
    
    times = kinematics['time']
    magnet1_y = kinematics['position']['magnet1_y'] # Red
    magnet3_y = kinematics['position']['magnet3_y'] # Blue
    
    print(f"Total points: {len(times)}")
    print(f"Duration: {times[-1]:.2f} seconds")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.plot(times, magnet1_y, 'r-', label='Magnet 1 (Red)', linewidth=2)
    plt.plot(times, magnet3_y, 'b-', label='Magnet 3 (Blue)', linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Planned Trajectories for Red and Blue Magnets')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory_plot_2D.png")
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_trajectory()

