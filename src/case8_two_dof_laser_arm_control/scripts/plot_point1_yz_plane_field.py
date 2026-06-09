#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot the Case 8 magnetic field on the y-z plane when the laser target is at
point 1 of the five-corner star.

This script does not open CAN and does not send any current command. It only
uses the same kinematics, coil model, and current solver as the controller so
the point-1 y-z plane field can be checked visually.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from closed_loop_control import COIL_PARAMS, ClosedLoopLaserSpotController, TwoDOFLaserModel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJECTORY_PATH = os.path.join(SCRIPT_DIR, "trajectory_case8_laser_spot.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "point1_yz_plane_magnetic_field.png")


def load_case8_settings():
    with open(TRAJECTORY_PATH, "r") as f:
        data = json.load(f)
    point1 = np.array(data["laser_spot"]["raw_path"][0], dtype=float)
    return data["settings"], point1


def solver_for_settings(settings, model):
    solver = object.__new__(ClosedLoopLaserSpotController)
    solver.model = model
    solver.settings = {
        "field_strength_T": float(settings["field_strength_T"]),
        "current_limit_A": float(settings["current_limit_A"]),
        "current_smoothing": float(settings["current_smoothing"]),
        "current_regularization": float(settings["current_regularization"]),
    }
    return solver


def solve_point1_yz_field(settings, point1):
    model = TwoDOFLaserModel()
    solver = solver_for_settings(settings, model)
    yaw, pitch = model.inverse_laser_target(point1[0], point1[1])
    kin = model.forward_kinematics(yaw, pitch)
    aligned_fields = ClosedLoopLaserSpotController._magnet_aligned_fields(solver, kin)
    target_fields = ClosedLoopLaserSpotController._project_fields_to_yz_plane(solver, aligned_fields)
    solution = ClosedLoopLaserSpotController._solve_currents_for_fields(
        solver,
        kin,
        target_fields,
        np.zeros(10),
    )
    currents = np.clip(solution["currents"], -settings["current_limit_A"], settings["current_limit_A"])
    actual_fields = ClosedLoopLaserSpotController._actual_fields(
        solver,
        currents,
        [
            {"position": kin["magnet1"], "field": target_fields[0]},
            {"position": kin["magnet2"], "field": target_fields[1]},
        ],
    )
    return {
        "model": model,
        "solver": solver,
        "yaw": yaw,
        "pitch": pitch,
        "kin": kin,
        "target_fields": target_fields,
        "actual_fields": actual_fields,
        "currents": currents,
        "fit_error": solution["fit_error"],
    }


def sample_yz_plane(solver, currents, y_values, z_values, x_plane=0.0):
    yy, zz = np.meshgrid(y_values, z_values)
    by = np.zeros_like(yy)
    bz = np.zeros_like(yy)
    bmag = np.zeros_like(yy)
    for row in range(yy.shape[0]):
        for col in range(yy.shape[1]):
            point = np.array([x_plane, yy[row, col], zz[row, col]], dtype=float)
            field = np.zeros(3)
            for params, current in zip(COIL_PARAMS, currents):
                field += ClosedLoopLaserSpotController._dipole_field(solver, params, point) * current
            by[row, col] = field[1]
            bz[row, col] = field[2]
            bmag[row, col] = np.linalg.norm(field)
    return yy, zz, by, bz, bmag


def plot_field(settings, point1, result):
    kin = result["kin"]
    solver = result["solver"]
    currents = result["currents"]
    y_values = np.linspace(0.0, 0.09, 29)
    z_values = np.linspace(-0.04, 0.08, 31)
    yy, zz, by, bz, bmag = sample_yz_plane(solver, currents, y_values, z_values)

    fig, ax = plt.subplots(figsize=(10.5, 8.2), dpi=160)
    contour = ax.contourf(
        yy,
        zz,
        bmag * 1000.0,
        levels=24,
        cmap="viridis",
        alpha=0.88,
    )
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("|B| (mT)")

    vector_norm = np.sqrt(by ** 2 + bz ** 2)
    scale = np.where(vector_norm > 1e-12, vector_norm, 1.0)
    ax.quiver(
        yy,
        zz,
        by / scale,
        bz / scale,
        color="white",
        pivot="mid",
        width=0.0025,
        scale=34,
        alpha=0.9,
    )

    ax.scatter([point1[1]], [result["model"].observation_plane_z], marker="*", s=180, color="#ff3333",
               edgecolor="black", label="laser point 1")
    ax.scatter([kin["magnet1"][1]], [kin["magnet1"][2]], marker="o", s=85, color="#f5b642",
               edgecolor="black", label="magnet 1")
    ax.scatter([kin["magnet2"][1]], [kin["magnet2"][2]], marker="s", s=85, color="#66d9ef",
               edgecolor="black", label="magnet 2")
    ax.axhline(result["model"].observation_plane_z, color="white", linestyle="--", linewidth=1.2,
               alpha=0.8, label="observation plane")

    text = (
        "Point 1 laser spot: x=%.3f mm, y=%.2f mm\n"
        "Pose: yaw=%.2f deg, pitch=%.2f deg\n"
        "Target B1 = [%.3f, %.3f, %.3f] mT\n"
        "Target B2 = [%.3f, %.3f, %.3f] mT\n"
        "Max current = %.2f A, fit error = %.3f mT"
    ) % (
        point1[0] * 1000.0,
        point1[1] * 1000.0,
        result["yaw"] * 180.0 / np.pi,
        result["pitch"] * 180.0 / np.pi,
        result["target_fields"][0][0] * 1000.0,
        result["target_fields"][0][1] * 1000.0,
        result["target_fields"][0][2] * 1000.0,
        result["target_fields"][1][0] * 1000.0,
        result["target_fields"][1][1] * 1000.0,
        result["target_fields"][1][2] * 1000.0,
        float(np.max(np.abs(currents))),
        result["fit_error"] * 1000.0,
    )
    ax.text(
        0.002,
        -0.036,
        text,
        color="black",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.55", alpha=0.92),
    )

    ax.set_title("Case 8 Point 1 Magnetic Field on y-z Plane (x = 0)")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(y_values[0], y_values[-1])
    ax.set_ylim(z_values[0], z_values[-1])
    ax.grid(True, linestyle=":", color="white", alpha=0.45)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print("Saved", OUTPUT_PATH)


def main():
    settings, point1 = load_case8_settings()
    result = solve_point1_yz_field(settings, point1)
    plot_field(settings, point1, result)


if __name__ == "__main__":
    main()
