#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case 8 data-driven step 3: evaluate the trained forward model.

Loads a model + dataset, reports prediction error (mm) and writes diagnostic
plots under models/. Use a *held-out* dataset (a file not used for training)
for an honest number; otherwise pass the same data with --val-frac to score the
same random validation split train_model.py used.

    python3 evaluate_model.py
    python3 evaluate_model.py --model models/forward_model_latest.npz \
        --data data/case8_dataset_HELDOUT.csv
"""

import argparse
import glob
import os

import numpy as np

from mlp_model import MLP
from train_model import load_datasets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")


def report(pred, truth, label):
    err = pred - truth
    dist_mm = np.linalg.norm(err, axis=1) * 1000.0
    print("\n=== %s (%d samples) ===" % (label, len(truth)))
    print("  RMS error   : %.3f mm" % np.sqrt(np.mean(dist_mm ** 2)))
    print("  mean error  : %.3f mm" % np.mean(dist_mm))
    print("  median error: %.3f mm" % np.median(dist_mm))
    print("  90th pct    : %.3f mm" % np.percentile(dist_mm, 90))
    print("  max error   : %.3f mm" % np.max(dist_mm))
    print("  per-axis RMS: x=%.3f mm  y=%.3f mm"
          % (np.sqrt(np.mean((err[:, 0] * 1000) ** 2)),
             np.sqrt(np.mean((err[:, 1] * 1000) ** 2))))
    return dist_mm


def main():
    ap = argparse.ArgumentParser(description="Evaluate Case 8 forward model")
    ap.add_argument("--model", default=os.path.join(MODEL_DIR, "forward_model_latest.npz"))
    ap.add_argument("--data", nargs="+",
                    default=[os.path.join(DATA_DIR, "case8_dataset_*.csv")])
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="if >0, score only a random held-out fraction (seed must match training)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = MLP.load(args.model)
    X, Y = load_datasets(args.data)

    if args.val_frac > 0.0:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(X))
        n_val = max(1, int(round(args.val_frac * len(X))))
        idx = perm[:n_val]
        X, Y = X[idx], Y[idx]
        label = "held-out split"
    else:
        label = "all provided data"

    pred = model.predict(X)
    dist_mm = report(pred, Y, label)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1) measured vs predicted spot, with error vectors
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(Y[:, 0] * 1000, Y[:, 1] * 1000, s=10, c="tab:blue", label="measured", alpha=0.6)
        ax.scatter(pred[:, 0] * 1000, pred[:, 1] * 1000, s=10, c="tab:red", label="predicted", alpha=0.6)
        for i in range(len(Y)):
            ax.plot([Y[i, 0] * 1000, pred[i, 0] * 1000],
                    [Y[i, 1] * 1000, pred[i, 1] * 1000], color="gray", lw=0.3, alpha=0.5)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal", "box")
        ax.set_title("Case 8 forward model: measured vs predicted spot")
        ax.legend()
        fig.tight_layout()
        p1 = os.path.join(MODEL_DIR, "eval_scatter.png")
        fig.savefig(p1, dpi=130)

        # 2) error histogram
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(dist_mm, bins=40, color="tab:purple", alpha=0.8)
        ax2.axvline(np.sqrt(np.mean(dist_mm ** 2)), color="k", ls="--",
                    label="RMS = %.2f mm" % np.sqrt(np.mean(dist_mm ** 2)))
        ax2.set_xlabel("Euclidean error (mm)")
        ax2.set_ylabel("count")
        ax2.set_title("Case 8 forward model error distribution")
        ax2.legend()
        fig2.tight_layout()
        p2 = os.path.join(MODEL_DIR, "eval_error_hist.png")
        fig2.savefig(p2, dpi=130)
        print("\nSaved plots:\n  %s\n  %s" % (p1, p2))
    except Exception as exc:
        print("Could not generate plots: %s" % exc)


if __name__ == "__main__":
    main()
