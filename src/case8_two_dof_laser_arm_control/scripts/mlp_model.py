#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-NumPy MLP for the Case 8 data-driven laser-spot model.

Maps the 10 coil currents (Amps) -> laser spot (x, y) in meters. This is the
*forward* model: well-posed (10 inputs, 2 outputs). The controller inverts it
online (see nn_closed_loop_control.py) using the analytic input Jacobian
exposed here, so we never have to learn the ill-posed inverse directly.

Why pure NumPy: this ROS/robot machine has numpy/scipy/matplotlib but no
torch/sklearn, and deployment must run with zero extra installs. For a 10->2
regression an MLP trained with Adam in NumPy is more than enough and fast.

Swap-in note: if you prefer PyTorch for training, keep the same .npz layout
(layer_sizes, x_mean/x_std, y_mean/y_std, W0..Wn, b0..bn) and deployment will
load it unchanged.
"""

import numpy as np


def _tanh(x):
    return np.tanh(x)


class MLP:
    """Fully-connected tanh network with linear output and built-in I/O scaling."""

    def __init__(self, layer_sizes, x_mean, x_std, y_mean, y_std,
                 weights=None, biases=None, seed=0):
        self.layer_sizes = [int(s) for s in layer_sizes]   # e.g. [10, 64, 64, 2]
        self.x_mean = np.asarray(x_mean, dtype=np.float64)
        self.x_std = np.asarray(x_std, dtype=np.float64)
        self.y_mean = np.asarray(y_mean, dtype=np.float64)
        self.y_std = np.asarray(y_std, dtype=np.float64)

        rng = np.random.default_rng(seed)
        if weights is None:
            self.W, self.b = [], []
            for i in range(len(self.layer_sizes) - 1):
                fan_in, fan_out = self.layer_sizes[i], self.layer_sizes[i + 1]
                limit = np.sqrt(6.0 / (fan_in + fan_out))   # Glorot for tanh
                self.W.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
                self.b.append(np.zeros(fan_out))
        else:
            self.W = [np.asarray(w, dtype=np.float64) for w in weights]
            self.b = [np.asarray(bb, dtype=np.float64) for bb in biases]

    @property
    def n_layers(self):
        return len(self.W)

    # --- scaling helpers -------------------------------------------------
    def _normalize_x(self, x):
        return (x - self.x_mean) / self.x_std

    def _denormalize_y(self, yn):
        return yn * self.y_std + self.y_mean

    # --- forward / inference --------------------------------------------
    def forward_norm(self, xn, cache=False):
        """Forward pass in normalized space. xn: (N, in) -> y_norm: (N, out)."""
        a = xn
        pres, acts = [], [a]
        last = self.n_layers - 1
        for l in range(self.n_layers):
            pre = a @ self.W[l] + self.b[l]
            pres.append(pre)
            a = pre if l == last else _tanh(pre)   # linear output layer
            acts.append(a)
        if cache:
            return a, pres, acts
        return a

    def predict(self, x):
        """Physical currents (N,10) or (10,) -> physical spot (N,2) or (2,)."""
        x = np.asarray(x, dtype=np.float64)
        single = (x.ndim == 1)
        y = self._denormalize_y(self.forward_norm(self._normalize_x(np.atleast_2d(x))))
        return y[0] if single else y

    def input_jacobian(self, x):
        """d(spot)/d(current) at a single point x (10,) -> (2,10) physical units.

        Forward-mode differentiation through the layers, then unwind the input
        and output scaling: J_phys = diag(y_std) @ (dy_norm/dxn) @ diag(1/x_std).
        """
        x = np.asarray(x, dtype=np.float64)
        xn = self._normalize_x(np.atleast_2d(x))
        _, pres, _ = self.forward_norm(xn, cache=True)
        last = self.n_layers - 1
        G = np.eye(self.layer_sizes[0])             # d xn / d xn
        for l in range(self.n_layers):
            dpre = self.W[l].T @ G                  # (out_l, in_dim)
            if l == last:
                G = dpre                            # linear output
            else:
                deriv = 1.0 - np.tanh(pres[l][0]) ** 2   # (out_l,)
                G = deriv[:, None] * dpre
        return (self.y_std[:, None] * G) / self.x_std[None, :]

    # --- training (used by train_model.py) ------------------------------
    def loss_and_grads(self, xn, yn):
        """Mean 0.5*||.||^2 loss in normalized space + parameter gradients."""
        n = xn.shape[0]
        y_pred, pres, acts = self.forward_norm(xn, cache=True)
        diff = y_pred - yn
        loss = 0.5 * np.mean(np.sum(diff ** 2, axis=1))

        dW = [None] * self.n_layers
        db = [None] * self.n_layers
        delta = diff / n                            # dL/d(output pre)
        for l in reversed(range(self.n_layers)):
            dW[l] = acts[l].T @ delta
            db[l] = np.sum(delta, axis=0)
            if l > 0:
                deriv = 1.0 - np.tanh(pres[l - 1]) ** 2
                delta = (delta @ self.W[l].T) * deriv
        return loss, dW, db

    # --- persistence -----------------------------------------------------
    def save(self, path):
        arrays = {
            "layer_sizes": np.array(self.layer_sizes),
            "n_layers": np.array([self.n_layers]),
            "x_mean": self.x_mean, "x_std": self.x_std,
            "y_mean": self.y_mean, "y_std": self.y_std,
        }
        for i in range(self.n_layers):
            arrays["W%d" % i] = self.W[i]
            arrays["b%d" % i] = self.b[i]
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        n = int(d["n_layers"][0])
        return cls(
            layer_sizes=list(d["layer_sizes"]),
            x_mean=d["x_mean"], x_std=d["x_std"],
            y_mean=d["y_mean"], y_std=d["y_std"],
            weights=[d["W%d" % i] for i in range(n)],
            biases=[d["b%d" % i] for i in range(n)],
        )


def numerical_input_jacobian(model, x, eps=1e-5):
    """Finite-difference Jacobian, for unit-testing input_jacobian()."""
    x = np.asarray(x, dtype=np.float64)
    base = model.predict(x)
    J = np.zeros((base.shape[0], x.shape[0]))
    for j in range(x.shape[0]):
        dx = x.copy()
        dx[j] += eps
        J[:, j] = (model.predict(dx) - base) / eps
    return J


if __name__ == "__main__":
    # Self-test: analytic Jacobian must match finite differences.
    rng = np.random.default_rng(1)
    m = MLP([10, 16, 8, 2],
            x_mean=np.zeros(10), x_std=np.ones(10),
            y_mean=np.zeros(2), y_std=np.ones(2), seed=3)
    x0 = rng.normal(size=10)
    err = np.max(np.abs(m.input_jacobian(x0) - numerical_input_jacobian(m, x0)))
    print("max |analytic - numerical| Jacobian error = %.2e" % err)
    assert err < 1e-6, "Jacobian mismatch"
    print("mlp_model self-test passed")
