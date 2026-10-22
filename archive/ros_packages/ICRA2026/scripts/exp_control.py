"""
Experiment control for IROS paper

Robot description: Actuation method is the electromagnetic control. It's a 3-DOF robot arm with a gripper. The first two joints are revolute, and the third joint is the gripper. Each motor is composed of a magnet and a gearbox (1:50). We need to generate rotating magnetic fields to actuate three motors. Every motor rotating axis is parallel to the z-axis. The rotating direction of three motors' output shaft is the same to the input magnet. When rotating in clockwise direction, the gripper will open, and when rotating in counterclockwise direction, the gripper will close.

The kinematics function of the robot arm is x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha, beta), where alpha and beta are the angles of the two revolute joints. x1, y1, z1 are the coordinates of the first magnet, x2, y2, z2 are the coordinates of the second magnet, and x3, y3, z3 are the coordinates of the end magnet (gripper).

The control sequence is as follows:
Start pose (71,05, -33.16)
- Close the gripper; (rotate for a period of time, e.g., 5 seconds)
- Actuate motor 1 to the target angle; (71.05 -> 42.63)
- Actuate motor 2 to the target angle; (-33.16 -> 52.11)
- Actuate motor 1 to the target angle; (42.63 -> -90)
- Open the gripper; (rotate for a period of time, e.g., 5 seconds)
Final pose (-90, 52.11)

Magnetic field generation: the parameters of the coils is in params_list. It also contains how to generate the magnetic field. The control signal from the PID controller will be used to determine the current in the coils, which in turn generates the required magnetic field to actuate the motors.

Author: Da ZHAO
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Kinematics import robot_arm_kinematics
import sympy as sp
import pandas as pd

# ============================================================
# Control parameters
# ============================================================
field_amplitude = 0.04          # Tesla – target |B| at each motor
rotation_freq   = 1.0           # Hz – magnetic field rotation frequency
angular_velocity = 2 * np.pi * rotation_freq   # rad/s
gear_ratio      = 50            # gearbox ratio
motor_output_speed = 360.0 * rotation_freq / gear_ratio  # deg/s at output shaft
control_frequency = 20          # Hz
control_period    = 1.0 / control_frequency  # seconds

# Start / end joint angles (degrees)
alpha_start, beta_start = 71.05, -33.16

# Control sequence definition
# Each stage: (description, active_motor, alpha_start, alpha_end, beta_start, beta_end, duration)
#   active_motor: 1, 2, or 3 (gripper)
#   For gripper stages the joint angles stay constant; duration is fixed.
#   For joint stages the duration is computed from the angle change.
gripper_duration = 5.0  # seconds for gripper open/close

# Stage 0: Close gripper
s0_alpha, s0_beta = 71.05, -33.16

# Stage 1: Motor 1  71.05 -> 42.63
s1_alpha_end = 42.63
s1_dur = abs(s1_alpha_end - s0_alpha) / motor_output_speed

# Stage 2: Motor 2  -33.16 -> 52.11
s2_beta_end = 52.11
s2_dur = abs(s2_beta_end - s0_beta) / motor_output_speed

# Stage 3: Motor 1  42.63 -> -90
s3_alpha_end = -90.0
s3_dur = abs(s3_alpha_end - s1_alpha_end) / motor_output_speed

# Stage 4: Open gripper
s4_alpha, s4_beta = s3_alpha_end, s2_beta_end

control_stages = [
    # (name, active_motor, alpha_i, alpha_f, beta_i, beta_f, duration)
    ("Close gripper",  3, s0_alpha, s0_alpha,       s0_beta,   s0_beta,   gripper_duration),
    ("Motor1 71→43",   1, s0_alpha, s1_alpha_end,   s0_beta,   s0_beta,   s1_dur),
    ("Motor2 -33→52",  2, s1_alpha_end, s1_alpha_end, s0_beta, s2_beta_end, s2_dur),
    ("Motor1 43→-90",  1, s1_alpha_end, s3_alpha_end, s2_beta_end, s2_beta_end, s3_dur),
    ("Open gripper",   3, s4_alpha, s4_alpha,       s4_beta,   s4_beta,   gripper_duration),
]

# 10 coils data 3D
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

num_coils = len(params_list)

# Define the symbols
m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z = sp.symbols('m0 m1 m2 r0_0 r0_1 r0_2 X Y Z')

# Constants
mu0 = 4 * sp.pi * 1e-7

# Calculate displacement vector
dx = X - r0_0
dy = Y - r0_1
dz = Z - r0_2

# Calculate distance to the coordinate point
r = sp.sqrt(dx**2 + dy**2 + dz**2) + 1e-9  # Add a small constant to avoid division by zero

# Calculate dot product of displacement vector and magnetic dipole moment
dot_product = m0 * dx + m1 * dy + m2 * dz

# Calculate magnetic field components
model_Bx = (mu0 / (4 * sp.pi)) * (3 * dx * dot_product / r**5 - m0 / r**3)
model_By = (mu0 / (4 * sp.pi)) * (3 * dy * dot_product / r**5 - m1 / r**3)
model_Bz = (mu0 / (4 * sp.pi)) * (3 * dz * dot_product / r**5 - m2 / r**3)

# Convert the symbolic functions to numerical functions
dipole_model_Bx = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx, 'numpy')
dipole_model_By = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By, 'numpy')
dipole_model_Bz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bz, 'numpy')

# Calculate the partial derivatives
model_Bx_dx = sp.diff(model_Bx, X)
model_Bx_dy = sp.diff(model_Bx, Y)
model_Bx_dz = sp.diff(model_Bx, Z)
model_By_dy = sp.diff(model_By, Y)
model_By_dz = sp.diff(model_By, Z)
model_Bz_dz = sp.diff(model_Bz, Z)

# Convert the symbolic functions to numerical functions
dipole_model_Bx_dx = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx_dx, 'numpy')
dipole_model_Bx_dy = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx_dy, 'numpy')
dipole_model_Bx_dz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx_dz, 'numpy')
dipole_model_By_dy = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By_dy, 'numpy')
dipole_model_By_dz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By_dz, 'numpy')
dipole_model_Bz_dz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bz_dz, 'numpy')

# Calculate the magnetic field and its partial derivatives at given points
def calculate_B_and_derivatives(currents, X, Y, Z):
    Bx_total = 0
    By_total = 0
    Bz_total = 0
    Bx_dx_total = 0
    Bx_dy_total = 0
    Bx_dz_total = 0
    By_dy_total = 0
    By_dz_total = 0
    Bz_dz_total = 0

    # Loop over the coils
    for i in range(num_coils):
        # Get the coil parameters
        params = params_list[i]
        m0, m1, m2, r0_0, r0_1, r0_2 = params
        # Calculate the magnetic field produced by this coil
        Bx = dipole_model_Bx(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        By = dipole_model_By(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bz = dipole_model_Bz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]

        # Add to the total magnetic field
        Bx_total += Bx
        By_total += By
        Bz_total += Bz

        # Calculate the partial derivatives
        Bx_dx = dipole_model_Bx_dx(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bx_dy = dipole_model_Bx_dy(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bx_dz = dipole_model_Bx_dz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        By_dy = dipole_model_By_dy(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        By_dz = dipole_model_By_dz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bz_dz = dipole_model_Bz_dz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]

        # Add to the total partial derivatives
        Bx_dx_total += Bx_dx
        Bx_dy_total += Bx_dy
        Bx_dz_total += Bx_dz
        By_dy_total += By_dy
        By_dz_total += By_dz
        Bz_dz_total += Bz_dz

    return np.array([Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total])


# ============================================================
# Stage 1: Generate magnetic-field target sequence
# ============================================================
def generate_field_targets():
    """
    Walk through the control stages and produce a time-indexed table:
        time | stage | alpha | beta
              | motor1 (x,y,z, Bx, By)
              | motor2 (x,y,z, Bx, By)
              | motor3/gripper (x,y,z, Bx, By)
    
    Convention for rotation direction:
      - Angle decreasing  → clockwise field   (phase decreases)
      - Angle increasing  → counter-clockwise  (phase increases)
      - Close gripper     → clockwise
      - Open gripper      → counter-clockwise
    """
    records = []
    global_time = 0.0        # running clock (seconds)

    # Each motor has its own independent phase (rad).
    # Active motor: phase advances (rotating field).
    # Inactive motors: phase is frozen (field holds direction & amplitude).
    motor_phase = {1: 0.0, 2: 0.0, 3: 0.0}

    for stage_idx, (name, active_motor, a_i, a_f, b_i, b_f, dur) in enumerate(control_stages):
        n_steps = max(int(np.round(dur * control_frequency)), 1)
        dt = dur / n_steps

        # Determine rotation sign for the active motor
        if active_motor == 3:
            rot_sign = 1.0 if "Close" in name else -1.0    # close=CCW, open=CW
        elif active_motor == 1:
            rot_sign = -1.0 if a_f < a_i else 1.0
        else:  # motor 2
            rot_sign = -1.0 if b_f < b_i else 1.0

        for k in range(n_steps):
            frac = k / n_steps
            alpha_k = a_i + (a_f - a_i) * frac
            beta_k  = b_i + (b_f - b_i) * frac

            # Motor positions via forward kinematics (metres)
            x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha_k, beta_k)

            # Link orientation angles (match Kinematics.py)
            alpha_rad = np.radians(alpha_k)
            beta_rad  = np.radians(beta_k)
            theta1 = alpha_rad - np.pi / 2   # global angle of link 1
            theta2 = theta1 + beta_rad        # global angle of link 2

            # Magnetic field in global frame for each motor.
            # motor_phase[i] is the field angle in the motor's LOCAL frame.
            # Motor 1 is at the base (no link offset).
            # Motor 2 is mounted on link 1 → offset by theta1.
            # Motor 3 is mounted on link 2 → offset by theta2.
            global_angle_1 = motor_phase[1]              # base frame
            global_angle_2 = motor_phase[2] + theta1     # link-1 frame
            global_angle_3 = motor_phase[3] + theta2     # link-2 frame

            bx1 = field_amplitude * np.cos(global_angle_1)
            by1 = field_amplitude * np.sin(global_angle_1)
            bx2 = field_amplitude * np.cos(global_angle_2)
            by2 = field_amplitude * np.sin(global_angle_2)
            bx3 = field_amplitude * np.cos(global_angle_3)
            by3 = field_amplitude * np.sin(global_angle_3)

            records.append({
                'time': round(global_time, 6),
                'stage': stage_idx,
                'stage_name': name,
                'active_motor': active_motor,
                'alpha_deg': alpha_k,
                'beta_deg': beta_k,
                # Motor 1 position & target field
                'x1': x1, 'y1': y1, 'z1': z1, 'Bx1': bx1, 'By1': by1,
                # Motor 2 position & target field
                'x2': x2, 'y2': y2, 'z2': z2, 'Bx2': bx2, 'By2': by2,
                # Motor 3 (gripper) position & target field
                'x3': x3, 'y3': y3, 'z3': z3, 'Bx3': bx3, 'By3': by3,
            })

            # Only advance the phase of the *active* motor; others stay frozen
            motor_phase[active_motor] += rot_sign * angular_velocity * dt
            global_time += dt

    df = pd.DataFrame(records)
    return df


# ============================================================
# 3D Visualisation
# ============================================================
def visualize_targets_3d(df):
    """
    Visualise the magnetic field targets in 3D space.
    - Robot arm poses shown at many time-stamps with colour-coded stages
    - Large arrows show target B-field direction & amplitude at every motor
    """
    fig = plt.figure(figsize=(20, 12))

    # ---------- subplot 1: 3D robot arm + field arrows ----------
    ax = fig.add_subplot(121, projection='3d')

    # Sample every N-th row so we get ~60-80 arm poses (dense but readable)
    n_samples = 60
    step = max(1, len(df) // n_samples)
    sample_df = df.iloc[::step].copy()

    # Colour map: one colour per stage
    stage_ids = df['stage'].unique()
    cmap = plt.cm.Set1
    stage_colors = {s: cmap(i / max(len(stage_ids) - 1, 1)) for i, s in enumerate(stage_ids)}
    motor_colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}

    # Arrow scale: 0.03 T should produce a clearly visible arrow (~1 cm = 0.01 m)
    arrow_scale = 0.4   # 0.03 * 0.4 = 0.012 m arrow length

    for _, row in sample_df.iterrows():
        s = int(row['stage'])
        scolor = stage_colors[s]

        # Draw arm links (joint 1 → joint 2 → end-effector)
        xs = [row['x1'], row['x2'], row['x3']]
        ys = [row['y1'], row['y2'], row['y3']]
        zs = [row['z1'], row['z2'], row['z3']]
        ax.plot(xs, ys, zs, '-o', color=scolor, markersize=3, linewidth=1.0, alpha=0.35)

        # Draw B-field arrows at ALL three motors
        for motor_id in [1, 2, 3]:
            bx_val = row[f'Bx{motor_id}']
            by_val = row[f'By{motor_id}']
            px = row[f'x{motor_id}']
            py = row[f'y{motor_id}']
            pz = row[f'z{motor_id}']
            mc = motor_colors[motor_id]
            # Bold arrow for the active motor, thinner for inactive
            is_active = (int(row['active_motor']) == motor_id)
            lw = 1.8 if is_active else 0.8
            al = 0.9 if is_active else 0.45
            ax.quiver(px, py, pz,
                      bx_val * arrow_scale, by_val * arrow_scale, 0,
                      color=mc, arrow_length_ratio=0.18, linewidth=lw, alpha=al)

    # Draw full motor trajectories
    for motor_id in [1, 2, 3]:
        mc = motor_colors[motor_id]
        ax.plot(df[f'x{motor_id}'], df[f'y{motor_id}'], df[f'z{motor_id}'],
                '-', color=mc, linewidth=1.2, alpha=0.6, label=f'Motor {motor_id} path')

    # Mark start and end positions
    for motor_id in [1, 2, 3]:
        mc = motor_colors[motor_id]
        r0 = df.iloc[0]
        rf = df.iloc[-1]
        ax.scatter(r0[f'x{motor_id}'], r0[f'y{motor_id}'], r0[f'z{motor_id}'],
                   s=80, c=mc, marker='^', edgecolors='k', zorder=5, label=f'M{motor_id} start' if motor_id == 1 else '')
        ax.scatter(rf[f'x{motor_id}'], rf[f'y{motor_id}'], rf[f'z{motor_id}'],
                   s=80, c=mc, marker='s', edgecolors='k', zorder=5, label=f'M{motor_id} end' if motor_id == 1 else '')

    # Custom legend entries for stages
    from matplotlib.lines import Line2D
    handles = []
    for s in stage_ids:
        sname = df[df['stage'] == s].iloc[0]['stage_name']
        handles.append(Line2D([0], [0], color=stage_colors[s], linewidth=2, label=f'S{s}: {sname}'))
    for mid in [1, 2, 3]:
        handles.append(Line2D([0], [0], color=motor_colors[mid], linewidth=2, linestyle='--',
                               label=f'Motor {mid} path'))
    ax.legend(handles=handles, fontsize=7, loc='upper left', framealpha=0.8)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Robot Arm Poses & Target B-fields (arrows = B direction)')

    # ---------- subplot 2: Bx, By vs time ----------
    stage_boundaries = df.groupby('stage').first().index.tolist()
    ax2 = fig.add_subplot(222)
    t = df['time']
    for motor_id, c in [(1, 'tab:blue'), (2, 'tab:orange'), (3, 'tab:green')]:
        ax2.plot(t, df[f'Bx{motor_id}'] * 1e3, '-', color=c, linewidth=0.9,
                 label=f'Bx motor{motor_id}')
        ax2.plot(t, df[f'By{motor_id}'] * 1e3, '--', color=c, linewidth=0.9,
                 label=f'By motor{motor_id}')
    for s in stage_boundaries:
        t_start = df[df['stage'] == s].iloc[0]['time']
        ax2.axvline(t_start, color='grey', linestyle=':', linewidth=0.6)
        ax2.text(t_start + 0.2, ax2.get_ylim()[0] if ax2.get_ylim()[0] != 0 else -30,
                 df[df['stage'] == s].iloc[0]['stage_name'], fontsize=6, rotation=90, va='bottom')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('B (mT)')
    ax2.set_title('Target Bx / By at Each Motor')
    ax2.legend(fontsize=7, ncol=2, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ---------- subplot 3: Motor positions over time ----------
    ax3 = fig.add_subplot(224)
    for motor_id, c in [(1, 'tab:blue'), (2, 'tab:orange'), (3, 'tab:green')]:
        ax3.plot(t, df[f'x{motor_id}'] * 1e2, '-', color=c, linewidth=0.9,
                 label=f'x{motor_id} (cm)')
        ax3.plot(t, df[f'y{motor_id}'] * 1e2, '--', color=c, linewidth=0.9,
                 label=f'y{motor_id} (cm)')
    for s in stage_boundaries:
        t_start = df[df['stage'] == s].iloc[0]['time']
        ax3.axvline(t_start, color='grey', linestyle=':', linewidth=0.6)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (cm)')
    ax3.set_title('Motor XY Positions vs Time')
    ax3.legend(fontsize=7, ncol=2, loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("  Experiment Control – Stage 1: Generate Field Targets")
    print("="*60)
    print(f"  Field amplitude : {field_amplitude*1e3:.1f} mT")
    print(f"  Rotation freq   : {rotation_freq:.1f} Hz")
    print(f"  Gear ratio      : {gear_ratio}")
    print(f"  Motor speed     : {motor_output_speed:.2f} deg/s")
    print(f"  Control freq    : {control_frequency} Hz")
    print()

    # Print control sequence summary
    total_dur = 0
    for name, am, ai, af, bi, bf, dur in control_stages:
        print(f"  Stage: {name:20s}  active_motor={am}  "
              f"a {ai:7.2f}->{af:7.2f}  b {bi:7.2f}->{bf:7.2f}  "
              f"dur={dur:.2f}s")
        total_dur += dur
    print(f"  Total duration: {total_dur:.2f} s")
    print()

    # Generate targets
    df_targets = generate_field_targets()
    print(f"Generated {len(df_targets)} target samples")
    print()
    print("Sample data (first 5 rows):")
    print(df_targets[['time', 'stage_name', 'active_motor',
                       'x1', 'y1', 'z1', 'Bx1', 'By1',
                       'x2', 'y2', 'z2', 'Bx2', 'By2',
                       'x3', 'y3', 'z3', 'Bx3', 'By3']].head().to_string(index=False))
    print()
    print("Sample data (last 5 rows):")
    print(df_targets[['time', 'stage_name', 'active_motor',
                       'x1', 'y1', 'z1', 'Bx1', 'By1',
                       'x2', 'y2', 'z2', 'Bx2', 'By2',
                       'x3', 'y3', 'z3', 'Bx3', 'By3']].tail().to_string(index=False))

    # Save to CSV (same directory as this script)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'field_targets.csv')
    df_targets.to_csv(csv_path, index=False)
    print(f"\nTargets saved to {csv_path}")

    # Visualize
    visualize_targets_3d(df_targets)


