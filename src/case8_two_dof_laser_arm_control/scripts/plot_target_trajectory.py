#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJECTORY_PATH = os.path.join(SCRIPT_DIR, "trajectory_case8_laser_spot.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "target_trajectory_case8_laser_spot.png")


def catmull_rom_point(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def resample_path(raw_path, samples_per_segment):
    raw_path = np.array(raw_path, dtype=float)
    if len(raw_path) < 2:
        return raw_path.copy()
    path = []
    for segment in range(len(raw_path) - 1):
        p0 = raw_path[max(segment - 1, 0)]
        p1 = raw_path[segment]
        p2 = raw_path[segment + 1]
        p3 = raw_path[min(segment + 2, len(raw_path) - 1)]
        for sample in range(samples_per_segment):
            path.append(catmull_rom_point(p0, p1, p2, p3, sample / float(samples_per_segment)))
    path.append(raw_path[-1])
    return np.array(path, dtype=float)


def path_length(path):
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def resample_path_by_target_speed(path, dt_s, target_speed_m_per_s):
    if target_speed_m_per_s <= 0.0 or len(path) < 2:
        return path
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    spacing = target_speed_m_per_s * dt_s
    if total_length <= 0.0 or spacing <= 0.0:
        return path
    point_count = max(2, int(round(total_length / spacing)) + 1)
    sample_distances = np.linspace(0.0, total_length, point_count)
    x = np.interp(sample_distances, cumulative, path[:, 0])
    y = np.interp(sample_distances, cumulative, path[:, 1])
    return np.column_stack((x, y))


def main():
    with open(TRAJECTORY_PATH, "r") as f:
        data = json.load(f)

    settings = data["settings"]
    dt_s = float(settings["dt_s"])
    samples_per_segment = int(settings.get("samples_per_segment", 36))
    target_speed_m_per_s = float(settings.get("target_speed_m_per_s", 0.0))
    workspace_radius = float(settings.get("workspace_radius_m", 0.12))
    raw_path = np.array(data["laser_spot"]["raw_path"], dtype=float)
    dense_path = resample_path(raw_path, samples_per_segment)
    path = resample_path_by_target_speed(dense_path, dt_s, target_speed_m_per_s)
    duration_s = (len(path) - 1) * dt_s
    nominal_speed = path_length(path) / duration_s if duration_s > 0.0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 9), dpi=160)
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    center = np.array(data["laser_spot"].get("center", [0.0, 0.0]), dtype=float)
    shape_radius = float(data["laser_spot"].get("radius_m", 0.0))

    ax.plot(
        workspace_radius * np.cos(theta),
        workspace_radius * np.sin(theta),
        "--",
        color="0.45",
        linewidth=1.8,
        label="workspace radius %.3f m" % workspace_radius,
    )
    if shape_radius > 0.0:
        ax.plot(
            center[0] + shape_radius * np.cos(theta),
            center[1] + shape_radius * np.sin(theta),
            ":",
            color="0.65",
            linewidth=1.2,
            label="source shape radius",
        )

    ax.plot(path[:, 0], path[:, 1], color="#c82333", linewidth=2.8, label="sampled laser target")
    ax.plot(raw_path[:, 0], raw_path[:, 1], "o:", color="#2f6f73", linewidth=1.8, markersize=5, label="raw waypoints")
    ax.scatter(path[0, 0], path[0, 1], marker="o", s=90, color="#c82333", edgecolor="black", zorder=5, label="start")
    ax.scatter(path[-1, 0], path[-1, 1], marker="*", s=160, color="#c82333", edgecolor="black", zorder=5, label="end")

    marker_step_s = 10.0 if duration_s > 30.0 else 1.0
    marker_times = np.arange(0.0, duration_s + 0.001, marker_step_s)
    for marker_time in marker_times:
        index = min(int(round(marker_time / dt_s)), len(path) - 1)
        x, y = path[index]
        ax.scatter([x], [y], s=42, color="#c82333", edgecolor="black", linewidth=0.6, zorder=4)
        ax.text(x + 0.002, y + 0.002, "t=%.0fs" % marker_time, fontsize=8, color="#8b1d1d")

    info = (
        "points = %d\n"
        "dt = %.3f s\n"
        "nominal time = %.2f s\n"
        "target speed = %.1f mm/s\n"
        "nominal speed = %.2f mm/s\n"
        "field = %.3g T\n"
        "I limit = +/-%.1f A\n"
        "target radius = %.4f m"
    ) % (
        len(path),
        dt_s,
        duration_s,
        target_speed_m_per_s * 1000.0,
        nominal_speed * 1000.0,
        float(settings["field_strength_T"]),
        float(settings["current_limit_A"]),
        float(settings["target_radius_m"]),
    )
    ax.text(
        0.02,
        -0.105,
        info,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.92),
    )

    ax.set_title("Case 8 Target Trajectory: Laser Spot", fontsize=15)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.13, 0.13)
    ax.set_ylim(-0.12, 0.13)
    ax.grid(True, linestyle=":", color="0.72")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print("Saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()
