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
TRAJECTORY_PATH = os.path.join(SCRIPT_DIR, "trajectory_case8_laser_spot.json")


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


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


def latest_results_csv():
    pattern = os.path.join(RESULTS_DIR, "closed_loop_results_case8_*.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError("No closed_loop_results_case8_*.csv file found in %s" % RESULTS_DIR)
    return files[-1]


def default_output_path(results_csv):
    base = os.path.splitext(os.path.basename(results_csv))[0]
    return os.path.join(RESULTS_DIR, base.replace("closed_loop_results", "trajectory_comparison") + ".png")


def read_results(path):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Result CSV is empty: %s" % path)

    def column(name):
        values = []
        for row in rows:
            text = row.get(name, "")
            values.append(float(text) if text != "" else np.nan)
        return np.array(values, dtype=float)

    return {
        "time": column("Time(s)"),
        "target_x": column("Target_X(m)"),
        "target_y": column("Target_Y(m)"),
        "spot_x": column("Spot_X(m)"),
        "spot_y": column("Spot_Y(m)"),
        "yaw": column("Model_Yaw(rad)"),
        "pitch": column("Model_Pitch(rad)"),
        "fit_error": column("Field_Fit_Error(T)"),
        "target_index": column("Target_Index"),
    }


def read_reference(path):
    with open(path, "r") as f:
        data = json.load(f)
    settings = data["settings"]
    dt_s = float(settings["dt_s"])
    dense_path = resample_path(data["laser_spot"]["raw_path"], int(settings.get("samples_per_segment", 36)))
    path_points = resample_path_by_target_speed(
        dense_path,
        dt_s,
        float(settings.get("target_speed_m_per_s", 0.0)),
    )
    return {
        "path": path_points,
        "time": np.arange(len(path_points), dtype=float) * dt_s,
        "dt_s": dt_s,
        "duration": (len(path_points) - 1) * dt_s,
        "workspace_radius": float(settings["workspace_radius_m"]),
        "time_based_reference": as_bool(settings.get("time_based_reference", False)),
    }


def reference_positions(ref, results):
    if not ref["time_based_reference"] and "target_index" in results:
        valid = np.isfinite(results["target_index"])
        x = np.full_like(results["time"], np.nan, dtype=float)
        y = np.full_like(results["time"], np.nan, dtype=float)
        if np.any(valid):
            indices = np.clip(results["target_index"][valid].astype(int), 0, len(ref["path"]) - 1)
            x[valid] = ref["path"][indices, 0]
            y[valid] = ref["path"][indices, 1]
            return x, y
    clipped = np.clip(results["time"], ref["time"][0], ref["time"][-1])
    x = np.interp(clipped, ref["time"], ref["path"][:, 0])
    y = np.interp(clipped, ref["time"], ref["path"][:, 1])
    return x, y


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


def plot_results(results, ref, output_path, title_suffix, show=False):
    ref_x, ref_y = reference_positions(ref, results)
    valid = np.isfinite(results["spot_x"]) & np.isfinite(results["spot_y"])
    error_mm = np.full_like(results["time"], np.nan, dtype=float)
    error_mm[valid] = np.sqrt((results["spot_x"][valid] - ref_x[valid]) ** 2 +
                              (results["spot_y"][valid] - ref_y[valid]) ** 2) * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=160)
    ax_xy, ax_pos, ax_joint, ax_err = axes.ravel()
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    radius = ref["workspace_radius"]

    ax_xy.plot(radius * np.cos(theta), radius * np.sin(theta), "--", color="0.55", linewidth=1.4, label="workspace")
    ax_xy.plot(ref["path"][:, 0], ref["path"][:, 1], color="#c82333", linewidth=2.2, label="full reference")
    ax_xy.plot(ref_x, ref_y, color="#8b1d1d", linewidth=1.6, label="active target")
    ax_xy.plot(results["spot_x"], results["spot_y"], "--", color="#23777b", linewidth=1.5, label="measured spot")
    ax_xy.scatter(ref["path"][0, 0], ref["path"][0, 1], marker="o", s=60, color="#c82333", edgecolor="black", zorder=5)
    ax_xy.scatter(ref["path"][-1, 0], ref["path"][-1, 1], marker="*", s=130, color="#c82333", edgecolor="black", zorder=5)
    ax_xy.set_title("XY trajectory")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.axis("equal")
    ax_xy.grid(True, linestyle=":", alpha=0.7)
    ax_xy.legend(fontsize=8)

    ax_pos.plot(results["time"], ref_x, color="#8b1d1d", linewidth=2, label="target x")
    ax_pos.plot(results["time"], results["spot_x"], "--", color="#c82333", linewidth=1.2, label="spot x")
    ax_pos.plot(results["time"], ref_y, color="#135f67", linewidth=2, label="target y")
    ax_pos.plot(results["time"], results["spot_y"], "--", color="#2f9ca0", linewidth=1.2, label="spot y")
    ax_pos.set_title("Spot coordinates")
    ax_pos.set_xlabel("time (s)")
    ax_pos.set_ylabel("position (m)")
    ax_pos.grid(True, linestyle=":", alpha=0.7)
    ax_pos.legend(fontsize=8)

    ax_joint.plot(results["time"], np.rad2deg(results["yaw"]), color="#6f4da8", linewidth=1.5, label="model yaw")
    ax_joint.plot(results["time"], np.rad2deg(results["pitch"]), color="#b15c1b", linewidth=1.5, label="model pitch")
    ax_joint.set_title("Estimated joint pose")
    ax_joint.set_xlabel("time (s)")
    ax_joint.set_ylabel("angle (deg)")
    ax_joint.grid(True, linestyle=":", alpha=0.7)
    ax_joint.legend(fontsize=8)

    ax_err.plot(results["time"], error_mm, color="#c82333", linewidth=1.5, label="spot error")
    ax_err_twin = ax_err.twinx()
    ax_err_twin.plot(results["time"], results["fit_error"] * 1e3, color="#23777b", linewidth=1.1, alpha=0.75, label="field fit")
    ax_err.set_title("Tracking error and field fit")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("spot error (mm)")
    ax_err_twin.set_ylabel("field fit error (mT)")
    ax_err.grid(True, linestyle=":", alpha=0.7)
    ax_err.text(
        0.02,
        0.98,
        error_stats(error_mm),
        transform=ax_err.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    fig.suptitle("Case 8 laser actual vs reference trajectory%s" % title_suffix, fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    latest_path = os.path.join(os.path.dirname(output_path), "latest_trajectory_comparison_case8.png")
    if os.path.abspath(output_path) != os.path.abspath(latest_path):
        shutil.copyfile(output_path, latest_path)
    print("Saved", output_path)
    print("Saved", latest_path)
    print("Laser spot", error_stats(error_mm))
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Case 8 laser spot results against the reference trajectory.")
    parser.add_argument("--results", default=None, help="Path to closed_loop_results_case8_*.csv. Defaults to latest result.")
    parser.add_argument("--trajectory", default=TRAJECTORY_PATH, help="Path to trajectory_case8_laser_spot.json.")
    parser.add_argument("--output", default=None, help="Output PNG path. Defaults to results/trajectory_comparison_*.png.")
    parser.add_argument("--show", action="store_true", help="Show the plot window after saving the PNG.")
    args = parser.parse_args()

    results_csv = args.results or latest_results_csv()
    output_path = args.output or default_output_path(results_csv)
    results = read_results(results_csv)
    ref = read_reference(args.trajectory)
    mode_text = "time reference %.1f s" % ref["duration"] if ref["time_based_reference"] else "progress-based reference"
    title_suffix = "\n%s, %s" % (os.path.basename(results_csv), mode_text)
    plot_results(results, ref, output_path, title_suffix, show=args.show)


if __name__ == "__main__":
    main()
