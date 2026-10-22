import numpy as np
import math

def robot_arm_kinematics(alpha, beta):
    x1 = 0
    y1 = 0.0345
    # z1 = 0.04669 - 0.085
    z1 = 0.05-0.085
    z2 = z1 - 0.01947
    z3 = z1


    # Convert angles from degrees to radians
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)

    # Define the lengths of the robot arm segments
    L1 = 0.02  # Length of the first segment
    L2 = 0.02 # Length of the second segment

    # Calculate x2, y2
    # Initial pose is -90 degrees (negative y-axis).
    # Assuming alpha is the rotation from the initial pose.
    theta1 = alpha_rad - math.pi / 2
    x2 = x1 + L1 * math.cos(theta1)
    y2 = y1 + L1 * math.sin(theta1)

    # Calculate x3, y3
    # beta is the relative rotation from the first link.
    theta2 = theta1 + beta_rad
    x3 = x2 + L2 * math.cos(theta2)
    y3 = y2 + L2 * math.sin(theta2)

    return x1, y1, z1, x2, y2, z2, x3, y3, z3

if __name__ == "__main__":
    alpha = 0 # Example angle for the first joint
    beta =  0  # Example angle for the second joint
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha, beta)
    print(f"Joint 1: ({x1:.4f}, {y1:.4f}, {z1:.4f})")
    print(f"Joint 2: ({x2:.4f}, {y2:.4f}, {z2:.4f})")
    print(f"End Effector: ({x3:.4f}, {y3:.4f}, {z3:.4f})")