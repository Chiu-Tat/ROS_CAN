#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJECTORY_PATH = os.path.join(SCRIPT_DIR, "trajectory_case7_two_microspheres.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "target_trajectory_case7_two_microspheres.png")


def load_trajectory():
    with open(TRAJECTORY_PATH, "r") as f:
        return json.load(f)


def path_array(data, key):
    return np.array(data[key]["path"], dtype=float)


def raw_array(data, key):
    return np.array(data[key]["raw_path"], dtype=float)


def plot_time_markers(ax, path, dt_s, duration_s, color, y_offset, label_prefix):
    marker_times = np.arange(0.0, duration_s + 0.001, 5.0)
    for marker_time in marker_times:
        index = min(int(round(marker_time / dt_s)), len(path) - 1)
        x, y = path[index]
        ax.scatter([x], [y], s=72, color=color, edgecolor="black", linewidth=0.8, zorder=5)
        ax.text(
            x + 0.002,
            y + y_offset,
            "%s t=%.0fs" % (label_prefix, marker_time),
            color=color,
            fontsize=9,
            weight="bold",
        )


def main():
    data = load_trajectory()
    settings = data["settings"]
    dt_s = float(settings["dt_s"])
    workspace_radius = float(settings["workspace_radius_m"])

    m1 = path_array(data, "microsphere_1")
    m2 = path_array(data, "microsphere_2")
    raw_m1 = raw_array(data, "microsphere_1")
    raw_m2 = raw_array(data, "microsphere_2")
    point_count = max(len(m1), len(m2))
    duration_s = (point_count - 1) * dt_s

    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(
        workspace_radius * np.cos(theta),
        workspace_radius * np.sin(theta),
        "--",
        color="0.35",
        linewidth=2,
        label="Workspace radius %.2f m" % workspace_radius,
    )

    m1_color = "#c43c39"
    m1_waypoint_color = "#e07a72"
    m2_color = "#1f8f4d"
    m2_waypoint_color = "#76bf8a"

    ax.plot(m1[:, 0], m1[:, 1], color=m1_color, linewidth=2.8, label="M1 red sampled target")
    ax.plot(
        raw_m1[:, 0],
        raw_m1[:, 1],
        ":o",
        color=m1_waypoint_color,
        linewidth=1.8,
        markersize=4,
        label="M1 red raw waypoints",
    )
    ax.plot(m2[:, 0], m2[:, 1], color=m2_color, linewidth=2.8, label="M2 green sampled target")
    ax.plot(
        raw_m2[:, 0],
        raw_m2[:, 1],
        ":o",
        color=m2_waypoint_color,
        linewidth=1.8,
        markersize=4,
        label="M2 green raw waypoints",
    )

    plot_time_markers(ax, m1, dt_s, duration_s, m1_color, -0.004, "M1")
    plot_time_markers(ax, m2, dt_s, duration_s, m2_color, 0.003, "M2")

    ax.scatter(m1[0, 0], m1[0, 1], marker="*", s=220, color=m1_color, edgecolor="black", zorder=6)
    ax.scatter(m1[-1, 0], m1[-1, 1], marker="*", s=220, color=m1_color, edgecolor="black", zorder=6)
    ax.scatter(m2[0, 0], m2[0, 1], marker="*", s=220, color=m2_color, edgecolor="black", zorder=6)
    ax.scatter(m2[-1, 0], m2[-1, 1], marker="*", s=220, color=m2_color, edgecolor="black", zorder=6)

    ax.text(m1[0, 0] + 0.002, m1[0, 1] - 0.006, "M1 start / M2 end", color=m1_color, fontsize=10)
    ax.text(m1[-1, 0] + 0.002, m1[-1, 1] + 0.002, "M1 end / M2 start", color=m2_color, fontsize=10)

    info = (
        "total time = %.1f s\n"
        "dt = %.3f s, points = %d\n"
        "z = %.3f m\n"
        "I limit = +/-%.1f A\n"
        "Kp = %.3g, Kd = %.3g\n"
        "target radius = %.4f m"
    ) % (
        duration_s,
        dt_s,
        point_count,
        float(settings["z_plane_m"]),
        float(settings["current_limit_A"]),
        float(settings["kp_N_per_m"]),
        float(settings["kd_N_s_per_m"]),
        float(settings["target_radius_m"]),
    )
    ax.text(
        0.015,
        -0.066,
        info,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9),
    )

    ax.set_title("Case 7 Target Trajectories: Two Iron Microspheres (20 s)", fontsize=16, pad=10)
    ax.set_xlabel("x (m)", fontsize=13)
    ax.set_ylabel("y (m)", fontsize=13)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.075, 0.075)
    ax.set_ylim(-0.075, 0.075)
    ax.grid(True, linestyle=":", color="0.7")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print("Saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()
