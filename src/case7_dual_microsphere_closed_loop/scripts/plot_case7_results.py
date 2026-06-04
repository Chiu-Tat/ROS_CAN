#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import json
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TRAJECTORY_PATH = os.path.join(SCRIPT_DIR, "trajectory_case7_two_microspheres.json")


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def latest_position_csv():
    pattern = os.path.join(RESULTS_DIR, "sphere_positions_case7_*.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError("No sphere_positions_case7_*.csv file found in %s" % RESULTS_DIR)
    return files[-1]


def default_output_path(position_csv):
    base = os.path.splitext(os.path.basename(position_csv))[0]
    return os.path.join(RESULTS_DIR, base.replace("sphere_positions", "trajectory_comparison") + ".png")


def read_positions(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Position CSV is empty: %s" % path)

    def column(name):
        values = []
        for row in rows:
            text = row.get(name, "")
            if text == "":
                values.append(np.nan)
            else:
                values.append(float(text))
        return np.array(values, dtype=float)

    return {
        "time": column("Time(s)"),
        "m1_x": column("M1_Red_X(m)"),
        "m1_y": column("M1_Red_Y(m)"),
        "m2_x": column("M2_Green_X(m)"),
        "m2_y": column("M2_Green_Y(m)"),
        "m1_target_index": column("M1_Target_Index"),
        "m2_target_index": column("M2_Target_Index"),
    }


def read_reference(path):
    with open(path, "r") as f:
        data = json.load(f)

    settings = data["settings"]
    dt_s = float(settings["dt_s"])
    m1 = np.array(data["microsphere_1"]["path"], dtype=float)
    m2 = np.array(data["microsphere_2"]["path"], dtype=float)
    points = max(len(m1), len(m2))
    ref_time = np.arange(points, dtype=float) * dt_s
    return {
        "time": ref_time,
        "m1": m1,
        "m2": m2,
        "duration": (points - 1) * dt_s,
        "dt_s": dt_s,
        "workspace_radius": float(settings["workspace_radius_m"]),
        "time_based_reference": as_bool(settings.get("time_based_reference", False)),
    }


def reference_positions(ref, positions, sphere_key):
    path = ref[sphere_key]
    index_key = "%s_target_index" % sphere_key
    target_indices = positions.get(index_key)
    if not ref["time_based_reference"] and target_indices is not None:
        valid = np.isfinite(target_indices)
        if np.any(valid):
            x = np.full_like(positions["time"], np.nan, dtype=float)
            y = np.full_like(positions["time"], np.nan, dtype=float)
            indices = np.clip(target_indices[valid].astype(int), 0, len(path) - 1)
            x[valid] = path[indices, 0]
            y[valid] = path[indices, 1]
            return x, y

    ref_time = ref["time"]
    times = positions["time"]
    clipped = np.clip(times, ref_time[0], ref_time[-1])
    x = np.interp(clipped, ref_time, path[:, 0])
    y = np.interp(clipped, ref_time, path[:, 1])
    return x, y


def tracking_error_mm(actual_x, actual_y, ref_x, ref_y):
    valid = np.isfinite(actual_x) & np.isfinite(actual_y)
    error_m = np.full_like(actual_x, np.nan, dtype=float)
    error_m[valid] = np.sqrt((actual_x[valid] - ref_x[valid]) ** 2 + (actual_y[valid] - ref_y[valid]) ** 2)
    return error_m * 1000.0


def error_stats(error_mm):
    valid = error_mm[np.isfinite(error_mm)]
    if len(valid) == 0:
        return "mean=n/a, rms=n/a, max=n/a"
    rms = math.sqrt(float(np.mean(valid ** 2)))
    return "mean=%.2f mm, rms=%.2f mm, max=%.2f mm" % (
        float(np.mean(valid)),
        rms,
        float(np.max(valid)),
    )


def plot_results(positions, ref, output_path, title_suffix, show=False):
    time_s = positions["time"]
    m1_ref_x, m1_ref_y = reference_positions(ref, positions, "m1")
    m2_ref_x, m2_ref_y = reference_positions(ref, positions, "m2")
    m1_error = tracking_error_mm(positions["m1_x"], positions["m1_y"], m1_ref_x, m1_ref_y)
    m2_error = tracking_error_mm(positions["m2_x"], positions["m2_y"], m2_ref_x, m2_ref_y)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=160)
    ax_xy, ax_x, ax_y, ax_err = axes.ravel()

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    radius = ref["workspace_radius"]
    ax_xy.plot(radius * np.cos(theta), radius * np.sin(theta), "--", color="0.55", linewidth=1.5, label="workspace")
    ax_xy.plot(ref["m1"][:, 0], ref["m1"][:, 1], color="#8b1d1d", linewidth=2.2, label="M1 red reference")
    ax_xy.plot(ref["m2"][:, 0], ref["m2"][:, 1], color="#1f7a3a", linewidth=2.2, label="M2 green reference")
    ax_xy.plot(positions["m1_x"], positions["m1_y"], "--", color="#e65c55", linewidth=1.4, label="M1 red actual")
    ax_xy.plot(positions["m2_x"], positions["m2_y"], "--", color="#55b96f", linewidth=1.4, label="M2 green actual")
    ax_xy.scatter(ref["m1"][0, 0], ref["m1"][0, 1], marker="o", s=50, color="#8b1d1d", edgecolor="black", zorder=5)
    ax_xy.scatter(ref["m1"][-1, 0], ref["m1"][-1, 1], marker="*", s=130, color="#8b1d1d", edgecolor="black", zorder=5)
    ax_xy.scatter(ref["m2"][0, 0], ref["m2"][0, 1], marker="o", s=50, color="#1f7a3a", edgecolor="black", zorder=5)
    ax_xy.scatter(ref["m2"][-1, 0], ref["m2"][-1, 1], marker="*", s=130, color="#1f7a3a", edgecolor="black", zorder=5)
    ax_xy.set_title("XY trajectory")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.axis("equal")
    ax_xy.grid(True, linestyle=":", alpha=0.7)
    ax_xy.legend(fontsize=8)

    if ref["time_based_reference"]:
        ref_plot_time = ref["time"]
        m1_plot_x = ref["m1"][:, 0]
        m1_plot_y = ref["m1"][:, 1]
        m2_plot_x = ref["m2"][:, 0]
        m2_plot_y = ref["m2"][:, 1]
        ref_label_suffix = "reference"
    else:
        ref_plot_time = time_s
        m1_plot_x = m1_ref_x
        m1_plot_y = m1_ref_y
        m2_plot_x = m2_ref_x
        m2_plot_y = m2_ref_y
        ref_label_suffix = "active target"

    ax_x.plot(ref_plot_time, m1_plot_x, color="#8b1d1d", linewidth=2, label="M1 x %s" % ref_label_suffix)
    ax_x.plot(time_s, positions["m1_x"], "--", color="#e65c55", linewidth=1.2, label="M1 x actual")
    ax_x.plot(ref_plot_time, m2_plot_x, color="#1f7a3a", linewidth=2, label="M2 x %s" % ref_label_suffix)
    ax_x.plot(time_s, positions["m2_x"], "--", color="#55b96f", linewidth=1.2, label="M2 x actual")
    ax_x.set_title("X position vs time")
    ax_x.set_xlabel("time (s)")
    ax_x.set_ylabel("x (m)")
    ax_x.grid(True, linestyle=":", alpha=0.7)
    ax_x.legend(fontsize=8)

    ax_y.plot(ref_plot_time, m1_plot_y, color="#8b1d1d", linewidth=2, label="M1 y %s" % ref_label_suffix)
    ax_y.plot(time_s, positions["m1_y"], "--", color="#e65c55", linewidth=1.2, label="M1 y actual")
    ax_y.plot(ref_plot_time, m2_plot_y, color="#1f7a3a", linewidth=2, label="M2 y %s" % ref_label_suffix)
    ax_y.plot(time_s, positions["m2_y"], "--", color="#55b96f", linewidth=1.2, label="M2 y actual")
    ax_y.set_title("Y position vs time")
    ax_y.set_xlabel("time (s)")
    ax_y.set_ylabel("y (m)")
    ax_y.grid(True, linestyle=":", alpha=0.7)
    ax_y.legend(fontsize=8)

    ax_err.plot(time_s, m1_error, color="#8b1d1d", linewidth=1.4, label="M1 error")
    ax_err.plot(time_s, m2_error, color="#1f7a3a", linewidth=1.4, label="M2 error")
    ax_err.set_title("Position tracking error")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("error (mm)")
    ax_err.grid(True, linestyle=":", alpha=0.7)
    ax_err.legend(fontsize=8)

    summary = "M1 %s\nM2 %s" % (error_stats(m1_error), error_stats(m2_error))
    ax_err.text(
        0.02,
        0.98,
        summary,
        transform=ax_err.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    fig.suptitle("Case 7 actual vs reference trajectory%s" % title_suffix, fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    latest_path = os.path.join(os.path.dirname(output_path), "latest_trajectory_comparison_case7.png")
    if os.path.abspath(output_path) != os.path.abspath(latest_path):
        shutil.copyfile(output_path, latest_path)
    print("Saved", output_path)
    print("Saved", latest_path)
    print("M1", error_stats(m1_error))
    print("M2", error_stats(m2_error))
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Case 7 actual sphere trajectories against the reference trajectory.")
    parser.add_argument("--positions", default=None, help="Path to sphere_positions_case7_*.csv. Defaults to latest result.")
    parser.add_argument("--trajectory", default=TRAJECTORY_PATH, help="Path to trajectory_case7_two_microspheres.json.")
    parser.add_argument("--output", default=None, help="Output PNG path. Defaults to results/trajectory_comparison_*.png.")
    parser.add_argument("--show", action="store_true", help="Show the plot window after saving the PNG.")
    args = parser.parse_args()

    position_csv = args.positions or latest_position_csv()
    output_path = args.output or default_output_path(position_csv)
    positions = read_positions(position_csv)
    ref = read_reference(args.trajectory)
    if ref["time_based_reference"]:
        mode_text = "time reference %.1f s" % ref["duration"]
    else:
        mode_text = "progress-based reference"
    title_suffix = "\\n%s, %s" % (os.path.basename(position_csv), mode_text)
    plot_results(positions, ref, output_path, title_suffix, show=args.show)


if __name__ == "__main__":
    main()
