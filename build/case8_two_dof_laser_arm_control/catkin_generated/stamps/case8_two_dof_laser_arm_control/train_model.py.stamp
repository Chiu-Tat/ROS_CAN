#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case 8 data-driven step 2: train the forward model (currents -> laser spot).

Reads one or more dataset CSVs produced by collect_data.py, trains the NumPy
MLP from mlp_model.py with Adam + early stopping, and saves the model plus a
loss curve under models/.

This is a plain Python script (not a ROS node); run it anywhere with numpy:
    python3 train_model.py
    python3 train_model.py --data data/case8_dataset_*.csv --hidden 64 64 --epochs 4000
"""

import argparse
import glob
import os
import time

import numpy as np

from mlp_model import MLP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")


def load_datasets(patterns):
    """Load CSV(s), keep detected rows, return X (N,10) currents, Y (N,2) spot."""
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit("No dataset files matched: %s" % patterns)

    xs, ys, total = [], [], 0
    for path in files:
        rows = np.genfromtxt(path, delimiter=",", names=True)
        if rows.size == 0:
            continue
        rows = np.atleast_1d(rows)
        currents = np.column_stack([rows["I%d_A" % i] for i in range(10)])
        spot = np.column_stack([rows["spot_x_m"], rows["spot_y_m"]])
        detected = rows["detected"] > 0.5
        valid = detected & np.all(np.isfinite(currents), axis=1) & np.all(np.isfinite(spot), axis=1)
        xs.append(currents[valid])
        ys.append(spot[valid])
        total += int(np.sum(valid))
        print("  %s: %d/%d usable rows" % (os.path.basename(path), int(np.sum(valid)), len(rows)))
    X = np.vstack(xs)
    Y = np.vstack(ys)
    print("Loaded %d usable samples from %d file(s)" % (total, len(files)))
    return X, Y


def adam_train(model, Xtr, Ytr, Xva, Yva, epochs, batch_size, lr, weight_decay, patience):
    """Mini-batch Adam with early stopping on validation RMS error."""
    xtr = model._normalize_x(Xtr)
    ytr = (Ytr - model.y_mean) / model.y_std
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mW = [np.zeros_like(w) for w in model.W]
    vW = [np.zeros_like(w) for w in model.W]
    mb = [np.zeros_like(b) for b in model.b]
    vb = [np.zeros_like(b) for b in model.b]

    rng = np.random.default_rng(0)
    n = xtr.shape[0]
    t = 0
    history = {"train_loss": [], "val_rms_mm": [], "val_med_mm": []}
    best_val = np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss, dW, db = model.loss_and_grads(xtr[idx], ytr[idx])
            epoch_loss += loss * len(idx)
            t += 1
            for l in range(model.n_layers):
                if weight_decay > 0.0:
                    dW[l] = dW[l] + weight_decay * model.W[l]
                mW[l] = beta1 * mW[l] + (1 - beta1) * dW[l]
                vW[l] = beta2 * vW[l] + (1 - beta2) * (dW[l] ** 2)
                mb[l] = beta1 * mb[l] + (1 - beta1) * db[l]
                vb[l] = beta2 * vb[l] + (1 - beta2) * (db[l] ** 2)
                mW_hat = mW[l] / (1 - beta1 ** t)
                vW_hat = vW[l] / (1 - beta2 ** t)
                mb_hat = mb[l] / (1 - beta1 ** t)
                vb_hat = vb[l] / (1 - beta2 ** t)
                model.W[l] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                model.b[l] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        # validation error in mm (physical units): RMS for early stopping,
        # median as a robust companion that ignores hysteresis/glitch outliers.
        pred = model.predict(Xva)
        err_mm = np.sqrt(np.sum((pred - Yva) ** 2, axis=1)) * 1000.0
        val_rms_mm = float(np.sqrt(np.mean(err_mm ** 2)))
        val_med_mm = float(np.median(err_mm))
        history["train_loss"].append(epoch_loss / n)
        history["val_rms_mm"].append(val_rms_mm)
        history["val_med_mm"].append(val_med_mm)

        if val_rms_mm < best_val - 1e-6:
            best_val = val_rms_mm
            best_state = ([w.copy() for w in model.W], [b.copy() for b in model.b])
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 50 == 0 or epoch == epochs - 1:
            print("epoch %4d  train_loss=%.3e  val_rms=%.3f mm  val_median=%.3f mm  best_rms=%.3f mm"
                  % (epoch, epoch_loss / n, val_rms_mm, val_med_mm, best_val))
        if bad_epochs >= patience:
            print("Early stopping at epoch %d (no val improvement for %d epochs)" % (epoch, patience))
            break

    if best_state is not None:
        model.W, model.b = best_state
    return history, best_val


def error_metrics_mm(model, X, Y):
    """Euclidean spot error in mm: RMS (outlier-sensitive) + robust summaries."""
    err = np.linalg.norm(model.predict(X) - Y, axis=1) * 1000.0
    return {
        "rms": float(np.sqrt(np.mean(err ** 2))),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "max": float(err.max()),
        "n": int(len(err)),
    }


def fit(X, Y, args, label):
    """Split, normalize (from train only), build and train an MLP.

    Returns (model, history, best_val, (Xtr, Ytr, Xva, Yva)).
    """
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(round(args.val_frac * len(X))))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr, Xva, Yva = X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]
    print("[%s] Train/val split: %d / %d" % (label, len(Xtr), len(Xva)))

    # normalization from training set only (avoid val leakage)
    x_mean, x_std = Xtr.mean(axis=0), Xtr.std(axis=0)
    y_mean, y_std = Ytr.mean(axis=0), Ytr.std(axis=0)
    x_std[x_std < 1e-9] = 1.0
    y_std[y_std < 1e-9] = 1.0

    model = MLP([10] + list(args.hidden) + [2], x_mean, x_std, y_mean, y_std, seed=args.seed)
    history, best_val = adam_train(model, Xtr, Ytr, Xva, Yva,
                                   args.epochs, args.batch_size, args.lr,
                                   args.weight_decay, args.patience)
    return model, history, best_val, (Xtr, Ytr, Xva, Yva)


def main():
    ap = argparse.ArgumentParser(description="Train Case 8 forward model (currents -> spot)")
    ap.add_argument("--data", nargs="+",
                    default=[os.path.join(DATA_DIR, "case8_dataset_*.csv")],
                    help="dataset CSV path(s) or glob(s)")
    ap.add_argument("--hidden", nargs="+", type=int, default=[64, 64],
                    help="hidden layer sizes")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=300)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-clean", action="store_false", dest="clean",
                    help="disable robust two-pass outlier removal (on by default)")
    ap.set_defaults(clean=True)
    ap.add_argument("--clean-sigma", type=float, default=3.0,
                    help="drop rows with residual > median + sigma*1.4826*MAD")
    ap.add_argument("--clean-floor-mm", type=float, default=5.0,
                    help="never drop rows whose residual is below this (mm)")
    ap.add_argument("--out", default=None, help="output .npz path (default: models/forward_model_<ts>.npz)")
    args = ap.parse_args()

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    X, Y = load_datasets(args.data)
    if len(X) < 50:
        print("WARNING: only %d samples; collect more for a usable model." % len(X))

    t0 = time.time()
    model, history, best_val, (Xtr, Ytr, Xva, Yva) = fit(
        X, Y, args, "pass1" if args.clean else "fit")

    # Robust two-pass cleaning: the electromagnet has hysteresis and the tracker
    # occasionally glitches, so a minority of rows are physically inconsistent
    # (near-identical currents, very different spot). They dominate the squared
    # RMS. Use the pass-1 model's residuals + a median/MAD threshold to drop
    # them, then retrain from scratch on the cleaned set.
    if args.clean:
        err = np.linalg.norm(model.predict(X) - Y, axis=1) * 1000.0
        med = float(np.median(err))
        mad = float(np.median(np.abs(err - med)))
        thr = max(args.clean_floor_mm, med + args.clean_sigma * 1.4826 * mad)
        inlier = err < thr
        n_drop = int(np.sum(~inlier))
        print("Robust cleaning: residual median=%.2f mm, MAD=%.2f mm, threshold=%.2f mm"
              % (med, 1.4826 * mad, thr))
        print("Dropping %d / %d rows (%.1f%%) as outliers above threshold"
              % (n_drop, len(X), 100.0 * n_drop / len(X)))
        if n_drop == 0:
            print("No rows above threshold; keeping pass-1 model.")
        else:
            X_out, Y_out = X[~inlier], Y[~inlier]
            model, history, best_val, (Xtr, Ytr, Xva, Yva) = fit(
                X[inlier], Y[inlier], args, "clean")
            m_out = error_metrics_mm(model, X_out, Y_out)
            print("Final model on the %d dropped rows: median=%.1f mm (excluded from val)"
                  % (n_drop, m_out["median"]))

    print("Training finished in %.1f s. Best val RMS = %.3f mm" % (time.time() - t0, best_val))
    mv = error_metrics_mm(model, Xva, Yva)
    print("Final val metrics (mm): RMS=%.2f  median=%.2f  90pct=%.2f  max=%.2f  (n=%d)"
          % (mv["rms"], mv["median"], mv["p90"], mv["max"], mv["n"]))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(MODEL_DIR, "forward_model_%s.npz" % run_id)
    model.save(out_path)
    latest = os.path.join(MODEL_DIR, "forward_model_latest.npz")
    model.save(latest)
    print("Saved model to %s\nSaved latest pointer to %s" % (out_path, latest))

    # loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(history["train_loss"], label="train loss (norm)", color="tab:blue")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train loss (normalized)", color="tab:blue")
        ax1.set_yscale("log")
        ax2 = ax1.twinx()
        ax2.plot(history["val_rms_mm"], label="val RMS (mm)", color="tab:red")
        ax2.plot(history["val_med_mm"], label="val median (mm)", color="tab:orange")
        ax2.set_ylabel("val error (mm)", color="tab:red")
        ax2.legend(loc="upper right")
        ax1.set_title("Case 8 forward model training (post-cleaning)")
        fig.tight_layout()
        plot_path = os.path.join(MODEL_DIR, "training_loss_%s.png" % run_id)
        fig.savefig(plot_path, dpi=130)
        print("Saved loss curve to %s" % plot_path)
    except Exception as exc:
        print("Could not plot loss curve: %s" % exc)


if __name__ == "__main__":
    main()
