"""
EML Tree Trainer v16_final

Goal:
- Keep v15 reproducibility and export format.
- Use a simpler, auditable training loop.
- Plot both optimized soft RMSE and hard-tau RMSE diagnostics.

Design:
1) SEARCH: fixed high tau, Adam, monitor plateau on soft loss.
2) HARDEN: tau annealing + entropy/binarity penalties.
3) OPTIONAL POLISH: short LBFGS at tau_hard.
4) SNAP + EVALUATE: hard 0/1 projection and exactness checks.
"""

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
import sys
from datetime import datetime
import numbers

import matplotlib
if "--skip-plot" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


DTYPE = torch.complex128
REAL_DTYPE = torch.float64
# Default: no practical clamping (1e300 is safely below float64 max ~1.8e308).
# Set to ~700 only if NaN restarts are excessive. Clamping at ~500 distorts EML.
_EML_CLAMP_DEFAULT = 1.0e300
_BYPASS_THR = 1.0 - torch.finfo(torch.float64).eps


# ---------------------------------------------------------------------------
# Target functions
# Examples generated from EML_expression_generation.nb
# ---------------------------------------------------------------------------
def _target_eml_depth2(x, y):
    """e - log(exp(y) - log(x))"""
    return np.exp(1) - np.log(np.exp(y) - np.log(x))


def _target_eml_depth2a(x, y):
    """e^e/y - log(exp(y) - log(x))"""
    return np.exp(np.exp(1)) / y - np.log(np.exp(y) - np.log(x))


def _target_eml_depth3(x, y):
    """e^e/(e^y - log(x))"""
    return np.exp(np.exp(1)) / (np.exp(y) - np.log(x))


def _target_eml_depth4(x, y):
    """log(e^x - log(y))"""
    return np.log(np.exp(x) - np.log(y))


def _target_eml_depth5(x, y):
    """log(e-log(e^x - log(y)))"""
    return np.log(np.exp(1) - np.log(np.exp(x) - np.log(y)))


def _target_eml_depth6(x, y):
    """e - y*exp(e - exp(x))"""
    return np.exp(1.0) - y * np.exp(np.exp(1.0) - np.exp(x))


def _target_multiply(x, y):
    """x * y depth=8"""
    return x * y


TARGET_FUNCTIONS = {
    # EML[1, EML[y, x]]
    "eml_depth2":  (_target_eml_depth2, "e - log(exp(y) - log(x))"),
    # EML depth-2 variant used in earlier sweeps
    "eml_depth2a": (_target_eml_depth2a, "e^e/y - log(exp(y) - log(x))"),
    # EML[EML[1, EML[y, x]], 1]
    "eml_depth3":  (_target_eml_depth3, "e^e/(e^y - log(x))"),
    # EML[1, EML[EML[1, EML[x, y]], 1]]
    "eml_depth4":  (_target_eml_depth4, "log(e^x - log(y))"),
    # EML[1, EML[EML[1, EML[1, EML[x, y]]], 1]]
    "eml_depth5":  (_target_eml_depth5, "log(e-log(e^x - log(y)))"),
    # EML[1, EML[EML[EML[1, EML[EML[x, y], 1]], 1], 1]]
    "eml_depth6":  (_target_eml_depth6, "e - y*exp(e - exp(x))"),
    # Exact EML construction:
    # times = EML[EML[1, EML[EML[EML[1, EML[EML[1, EML[1, x]], 1]], y], 1]], 1]
    # identity = EML[1, EML[EML[1, EML[x, 1]], 1]]
    # Bottom-layer x,y variant:
    # EML[EML[1, EML[EML[EML[1, EML[EML[1, EML[1, x]], 1]],
    # EML[1, EML[EML[1, EML[y, 1]], 1]]], 1]], 1]
    "multiply":    (_target_multiply, "x * y"),
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_target_fn(name):
    if name not in TARGET_FUNCTIONS:
        available = ", ".join(TARGET_FUNCTIONS.keys())
        raise ValueError(f"Unknown target '{name}'. Available: {available}")
    return TARGET_FUNCTIONS[name]


def snapshot(tree):
    """Detached copy of tree state dict."""
    return {k: v.detach().clone() for k, v in tree.state_dict().items()}


def _sanitize_for_json(obj):
    """Convert NaN/Inf to None and keep json.dump(..., allow_nan=False) valid."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, numbers.Real):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        if isinstance(obj, numbers.Integral):
            return int(obj)
        return v
    return obj


def eml_exact(x, y):
    """EML[x, y] = exp(x) - log(y), evaluated in complex plane."""
    return torch.exp(x) - torch.log(y)


# ===========================================================================
# EML Expression Parser
# ===========================================================================
def parse_eml_expr(s):
    s = s.strip()
    if s in ("1", "x", "y"):
        return s
    if s.startswith("EML[") and s.endswith("]"):
        inner = s[4:-1]
        depth = 0
        for i, c in enumerate(inner):
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            elif c == "," and depth == 0:
                left = parse_eml_expr(inner[:i])
                right = parse_eml_expr(inner[i + 1 :])
                return ("EML", left, right)
        raise ValueError(f"Malformed EML expression (no comma at depth 0): {s}")
    raise ValueError(f"Cannot parse EML expression: '{s}'.")


def expr_depth(node):
    if isinstance(node, str):
        return 0
    return 1 + max(expr_depth(node[1]), expr_depth(node[2]))


def validate_expr_for_tree(node, depth):
    errors = []

    def _check(n, d):
        if isinstance(n, str):
            if n in ("x", "y") and d != depth:
                errors.append(f"Terminal '{n}' at depth {d}, need {depth}")
            return
        _check(n[1], d + 1)
        _check(n[2], d + 1)

    _check(node, 0)
    return errors


def flat_node_idx(tree_depth, level_from_top, pos_in_level):
    return 2 ** tree_depth - 2 ** (level_from_top + 1) + pos_in_level


# ===========================================================================
# Manual Initialization
# ===========================================================================
def init_from_expr(tree, expr_str, k=32.0):
    parsed = parse_eml_expr(expr_str)
    ed = expr_depth(parsed)
    td = tree.depth
    if ed != td:
        raise ValueError(f"Expression depth ({ed}) != tree depth ({td}).")
    errors = validate_expr_for_tree(parsed, td)
    if errors:
        raise ValueError("Incompatible expression:\n  " + "\n  ".join(errors))

    with torch.no_grad():
        tree.leaf_logits.fill_(-k)
        tree.leaf_logits[:, 0] = k
        tree.blend_logits.fill_(k)

    choice_map = {"1": 0, "x": 1, "y": 2}

    def recurse(node, level, pos):
        if isinstance(node, str):
            if level == td:
                c = choice_map[node]
                with torch.no_grad():
                    tree.leaf_logits[pos] = torch.tensor([-k, -k, -k], dtype=REAL_DTYPE)
                    tree.leaf_logits[pos, c] = k
            return
        _, left, right = node
        nidx = flat_node_idx(td, level, pos)
        l1 = isinstance(left, str) and left == "1"
        r1 = isinstance(right, str) and right == "1"
        with torch.no_grad():
            tree.blend_logits[nidx, 0] = k if l1 else -k
            tree.blend_logits[nidx, 1] = k if r1 else -k
        if not l1:
            recurse(left, level + 1, 2 * pos)
        if not r1:
            recurse(right, level + 1, 2 * pos + 1)

    recurse(parsed, 0, 0)
    return parsed


def init_from_blend_leaves(tree, blend_str, leaf_str, k=32.0):
    blend_clean = blend_str.replace(" ", "")
    if "," in blend_clean:
        pairs = blend_clean.split(",")
        blend_bits = []
        for p in pairs:
            if len(p) != 2 or not all(c in "01" for c in p):
                raise ValueError(f"Bad blend pair: '{p}'")
            blend_bits.append((int(p[0]), int(p[1])))
    else:
        if len(blend_clean) % 2 != 0:
            raise ValueError(f"Blend bitstring length {len(blend_clean)} is odd.")
        if not all(c in "01" for c in blend_clean):
            raise ValueError("Blend bitstring must contain only '0' and '1'.")
        blend_bits = [
            (int(blend_clean[i]), int(blend_clean[i + 1]))
            for i in range(0, len(blend_clean), 2)
        ]
    if len(blend_bits) != tree.n_internal:
        raise ValueError(f"Blend gives {len(blend_bits)} nodes, tree has {tree.n_internal}.")

    leaf_clean = leaf_str.replace(" ", "")
    symbols = leaf_clean.split(",")
    if len(symbols) != tree.n_leaves:
        raise ValueError(f"Leaf gives {len(symbols)} leaves, tree has {tree.n_leaves}.")

    choice_map = {"1": 0, "x": 1, "y": 2}
    leaf_choices = []
    for i, s in enumerate(symbols):
        if s not in choice_map:
            raise ValueError(f"Leaf {i}: unknown '{s}'. Use 1, x, or y.")
        leaf_choices.append(choice_map[s])

    with torch.no_grad():
        for i, c in enumerate(leaf_choices):
            tree.leaf_logits[i] = torch.tensor([-k, -k, -k], dtype=REAL_DTYPE)
            tree.leaf_logits[i, c] = k
        for i, (bl, br) in enumerate(blend_bits):
            tree.blend_logits[i, 0] = k if bl else -k
            tree.blend_logits[i, 1] = k if br else -k


def add_init_noise(tree, noise_scale, seed=None):
    if noise_scale <= 0:
        return
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        tree.leaf_logits.add_(torch.randn_like(tree.leaf_logits) * noise_scale)
        tree.blend_logits.add_(torch.randn_like(tree.blend_logits) * noise_scale)


class EMLTree(nn.Module):
    """Full binary tree of depth `depth` with EML at every internal node."""

    def __init__(self, depth, init_scale=1.0, init_strategy="biased", eml_clamp=None):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1
        self.eml_clamp = eml_clamp if eml_clamp is not None else _EML_CLAMP_DEFAULT

        if init_strategy == "manual":
            leaf_init = torch.zeros(self.n_leaves, 3, dtype=REAL_DTYPE)
            gate_init = torch.zeros(self.n_internal, 2, dtype=REAL_DTYPE)
        elif init_strategy == "biased":
            leaf_init = torch.randn(self.n_leaves, 3, dtype=REAL_DTYPE) * init_scale
            leaf_init[:, 0] += 2.0
            gate_init = torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0
        elif init_strategy == "uniform":
            leaf_init = torch.randn(self.n_leaves, 3, dtype=REAL_DTYPE) * init_scale
            gate_init = torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0
        elif init_strategy == "xy_biased":
            leaf_init = torch.randn(self.n_leaves, 3, dtype=REAL_DTYPE) * init_scale
            leaf_init[:, 1] += 1.0
            leaf_init[:, 2] += 1.0
            gate_init = torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0
        elif init_strategy == "random_hot":
            leaf_init = torch.randn(self.n_leaves, 3, dtype=REAL_DTYPE) * init_scale
            hot_idx = torch.randint(0, 3, (self.n_leaves,))
            leaf_init[torch.arange(self.n_leaves), hot_idx] += 3.0
            gate_init = torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * init_scale + 3.0
            open_mask = torch.rand(self.n_internal, 2) < 0.25
            gate_init[open_mask] -= 6.0
        else:
            leaf_init = torch.randn(self.n_leaves, 3, dtype=REAL_DTYPE) * init_scale
            gate_init = torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0

        self.leaf_logits = nn.Parameter(leaf_init)
        self.blend_logits = nn.Parameter(gate_init)

    def forward(self, x, y, tau_leaf=1.0, tau_gate=1.0):
        x = x.to(DTYPE)
        y = y.to(DTYPE)
        batch_size = x.shape[0]

        leaf_probs = torch.softmax(self.leaf_logits / tau_leaf, dim=1)
        weights = leaf_probs.to(DTYPE)
        ones = torch.ones(batch_size, dtype=DTYPE)
        candidates = torch.stack([ones, x, y], dim=1)
        current_level = torch.matmul(candidates, weights.T)

        gate_probs_levels = []
        eml_outputs = []
        node_idx = 0
        while current_level.shape[1] > 1:
            n_pairs = current_level.shape[1] // 2
            left_children = current_level[:, 0::2]
            right_children = current_level[:, 1::2]
            s = torch.sigmoid(self.blend_logits[node_idx : node_idx + n_pairs] / tau_gate)
            gate_probs_levels.append(s)

            # s=1 -> constant 1, s=0 -> child value
            #
            # NaN prevention: complex multiplication (a+0j)*(c+0j)
            # computes imag = a*0 + 0*c; when either is Inf this
            # gives 0*Inf = NaN. By blending real & imag parts
            # separately with real-valued s we avoid the cross-term.
            # The _BYPASS_THR guard handles residual real-arithmetic
            # 0.0 * Inf = NaN when s rounds to exactly 1.
            s_left = s[:, 0].unsqueeze(0)
            s_right = s[:, 1].unsqueeze(0)
            bypass_left = s_left > _BYPASS_THR
            bypass_right = s_right > _BYPASS_THR
            oml = 1.0 - s_left
            omr = 1.0 - s_right

            lr = torch.where(bypass_left, 1.0, s_left + oml * left_children.real)
            li = torch.where(bypass_left, 0.0, oml * left_children.imag)
            rr = torch.where(bypass_right, 1.0, s_right + omr * right_children.real)
            ri = torch.where(bypass_right, 0.0, omr * right_children.imag)
            left_input = torch.complex(lr, li)
            right_input = torch.complex(rr, ri)

            current_level = eml_exact(left_input, right_input)
            # Clamp to prevent Inf cascades; scrub NaN from Inf-Inf in eml_exact.
            current_level = torch.complex(
                torch.nan_to_num(
                    current_level.real,
                    nan=0.0,
                    posinf=self.eml_clamp,
                    neginf=-self.eml_clamp,
                ).clamp(-self.eml_clamp, self.eml_clamp),
                torch.nan_to_num(
                    current_level.imag,
                    nan=0.0,
                    posinf=self.eml_clamp,
                    neginf=-self.eml_clamp,
                ).clamp(-self.eml_clamp, self.eml_clamp),
            )

            eml_outputs.append(current_level)
            node_idx += n_pairs

        gate_probs = torch.cat(gate_probs_levels, dim=0)
        return current_level.squeeze(1), leaf_probs, gate_probs, eml_outputs

    def _format_weights_mma(self, discretize=True, snap_threshold=0.0):
        """Format leafWeights / blendSigmoid for Mathematica export.

        If discretize=True and snap_threshold>0, only snaps confident entries.
        Returns (leaf_line, blend_line, uncertain_entries).
        """
        uncertain = []
        if discretize:
            leaf_probs = torch.softmax(self.leaf_logits, dim=1).detach().cpu().numpy()
            gate_probs = torch.sigmoid(self.blend_logits).detach().cpu().numpy()
            lw = np.zeros((self.n_leaves, 3))
            for i in range(self.n_leaves):
                max_p = leaf_probs[i].max()
                max_idx = leaf_probs[i].argmax()
                if snap_threshold > 0 and max_p < 1.0 - snap_threshold:
                    uncertain.append(f"leaf[{i}]: max_prob={max_p:.4f} {leaf_probs[i].tolist()}")
                    lw[i] = leaf_probs[i]
                else:
                    lw[i, max_idx] = 1.0
            bs = np.zeros((self.n_internal, 2))
            for i in range(self.n_internal):
                for j in range(2):
                    p = gate_probs[i, j]
                    if snap_threshold > 0 and snap_threshold < p < 1.0 - snap_threshold:
                        uncertain.append(f"gate[{i},{j}]: prob={p:.4f}")
                        bs[i, j] = p
                    else:
                        bs[i, j] = 1.0 if p >= 0.5 else 0.0

            def fmt_leaf(w):
                if all(v in (0.0, 1.0) for v in w):
                    return f"{{{int(w[0])}, {int(w[1])}, {int(w[2])}}}"
                return f"{{{w[0]:.12f}, {w[1]:.12f}, {w[2]:.12f}}}"

            def fmt_blend(b):
                if all(v in (0.0, 1.0) for v in b):
                    return f"{{{int(b[0])}, {int(b[1])}}}"
                return f"{{{b[0]:.12f}, {b[1]:.12f}}}"

            leaf_parts = [fmt_leaf(lw[i]) for i in range(self.n_leaves)]
            blend_parts = [fmt_blend(bs[i]) for i in range(self.n_internal)]
        else:
            lw = torch.softmax(self.leaf_logits, dim=1).detach().cpu().numpy()
            bs = torch.sigmoid(self.blend_logits).detach().cpu().numpy()
            leaf_parts = [f"{{{w[0]:.12f}, {w[1]:.12f}, {w[2]:.12f}}}" for w in lw]
            blend_parts = [f"{{{b[0]:.12f}, {b[1]:.12f}}}" for b in bs]

        leaf_line = "leafWeights = {" + ", ".join(leaf_parts) + "};"
        blend_line = "blendSigmoid = {" + ", ".join(blend_parts) + "};"
        return leaf_line, blend_line, uncertain

    def export_mathematica(self, filename, discretize=True, comment="", snap_threshold=0.0):
        leaf_line, blend_line, uncertain = self._format_weights_mma(discretize, snap_threshold)
        lines = [
            f"(* EML Tree exported from PyTorch v16_final *)",
            f"(* Depth: {self.depth}, Leaves: {self.n_leaves}, Internal nodes: {self.n_internal} *)",
            f"(* Discretized: {discretize} *)",
        ]
        if comment:
            lines.append(f"(* {comment} *)")
        if uncertain:
            lines.append(f"(* WARNING: {len(uncertain)} uncertain weights: *)")
            for u in uncertain:
                lines.append(f"(*   {u} *)")
        lines += [
            "",
            f"treeDepth = {self.depth};",
            f"nLeaves = {self.n_leaves};",
            f"nInternal = {self.n_internal};",
            "",
            "(* Leaf weights: {w1, wx, wy} for each leaf *)",
            leaf_line,
            "",
            "(* Blend sigmoid values: {left, right} for each internal node *)",
            blend_line,
            "",
            "{blendSigmoid, leafWeights} (* Direct output to Mma notebook *)",
        ]
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def _filter_real_domain(x, y, target_fn, imag_tol=1e-12, label="data"):
    """Evaluate in complex128 and keep only finite real-domain points.

    Rejected points include:
    - non-finite real part
    - non-negligible imaginary part
    """
    xc = x.astype(np.complex128)
    yc = y.astype(np.complex128)
    with np.errstate(all="ignore"):
        tc = target_fn(xc, yc)
    real_mask = (np.abs(tc.imag) < imag_tol) & np.isfinite(tc.real)
    n_orig = len(x)
    n_kept = int(real_mask.sum())
    n_rejected = n_orig - n_kept
    if n_rejected > 0:
        print(f"  {label}: {n_rejected}/{n_orig} points outside real domain (rejected)")
    return x[real_mask], y[real_mask], tc[real_mask].real


def make_grid_data(target_fn, lo=1.0, hi=3.0, step=0.1):
    xs = np.arange(lo, hi + step * 0.5, step)
    ys = np.arange(lo, hi + step * 0.5, step)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xx, yy = xx.ravel(), yy.ravel()
    xx, yy, tt = _filter_real_domain(xx, yy, target_fn, label="train grid")
    print(f"Training data: {len(xx)} valid points on [{lo}, {hi}]^2 step={step}")
    return (
        torch.tensor(xx, dtype=REAL_DTYPE),
        torch.tensor(yy, dtype=REAL_DTYPE),
        torch.tensor(tt, dtype=DTYPE),
    )


def make_generalization_data(target_fn, lo=0.5, hi=5.0, n=4000, seed=12345):
    rng = np.random.default_rng(seed)
    oversample = max(n * 4, n + 10000)
    x_all = rng.uniform(lo, hi, size=oversample)
    y_all = rng.uniform(lo, hi, size=oversample)
    x_ok, y_ok, t_ok = _filter_real_domain(x_all, y_all, target_fn, label="gen random")
    if len(x_ok) < n:
        print(f"  WARNING: only {len(x_ok)} valid generalization points (requested {n})")
    else:
        x_ok, y_ok, t_ok = x_ok[:n], y_ok[:n], t_ok[:n]
    print(f"Generalization data: {len(x_ok)} valid points on [{lo}, {hi}]^2")
    return (
        torch.tensor(x_ok, dtype=REAL_DTYPE),
        torch.tensor(y_ok, dtype=REAL_DTYPE),
        torch.tensor(t_ok, dtype=DTYPE),
    )


def compute_losses(
    pred,
    target,
    leaf_probs,
    gate_probs,
    eml_outputs,
    lam_ent,
    lam_bin,
    lam_inter,
    inter_threshold,
    lam_sparse=0.0,
    uncertainty_power=2.0,
):
    """Composite objective:
    - data MSE
    - uncertainty-weighted leaf entropy
    - uncertainty-weighted gate binarity penalty
    - intermediate-magnitude penalty
    """
    data_loss = torch.mean((pred - target).abs() ** 2).real
    eps = 1e-12

    leaf_max = leaf_probs.max(dim=1).values
    leaf_unc = torch.clamp((1.0 - leaf_max) / (2.0 / 3.0), 0.0, 1.0).pow(uncertainty_power)
    leaf_ent = -(leaf_probs * (leaf_probs + eps).log()).sum(dim=1)
    entropy = (leaf_ent * leaf_unc).mean()

    gate_unc = torch.clamp(1.0 - (2.0 * gate_probs - 1.0).abs(), 0.0, 1.0).pow(uncertainty_power)
    gate_bin = gate_probs * (1.0 - gate_probs)
    binarity = (gate_bin * gate_unc).mean()

    sparse = torch.mean(1.0 - gate_probs)
    inter_penalty = torch.tensor(0.0, dtype=REAL_DTYPE)
    if lam_inter > 0 and eml_outputs:
        for lo in eml_outputs:
            excess = torch.relu(lo.abs() - inter_threshold)
            inter_penalty = inter_penalty + excess.pow(2).mean()
        inter_penalty = inter_penalty / len(eml_outputs)

    total = data_loss + lam_ent * entropy + lam_bin * binarity + lam_inter * inter_penalty + lam_sparse * sparse
    ambiguity = torch.cat([leaf_unc, gate_unc.reshape(-1)]).mean()
    return total, data_loss, entropy, binarity, inter_penalty, sparse, ambiguity


def evaluate(tree, x_data, y_data, targets, tau=0.01):
    """Evaluate MSE and max real/imag errors at fixed tau."""
    with torch.no_grad():
        pred, _, _, _ = tree(x_data, y_data, tau_leaf=tau, tau_gate=tau)
        mse = torch.mean((pred - targets).abs() ** 2).real.item()
        max_real = torch.max((pred.real - targets.real).abs()).item()
        max_imag = torch.max(pred.imag.abs()).item()
    return mse, max_real, max_imag


def analyze_snap(tree, snap_threshold=0.01):
    """Check which weights cannot be cleanly snapped to 0/1.

    Returns dict with uncertain leaves/gates and total uncertain count.
    """
    with torch.no_grad():
        leaf_probs = torch.softmax(tree.leaf_logits, dim=1).cpu().numpy()
        gate_probs = torch.sigmoid(tree.blend_logits).cpu().numpy()

    names = {0: "1", 1: "x", 2: "y"}
    uncertain_leaves = []
    for i in range(tree.n_leaves):
        max_p = leaf_probs[i].max()
        max_idx = leaf_probs[i].argmax()
        if max_p < 1.0 - snap_threshold:
            uncertain_leaves.append(
                f"leaf[{i}]: best={names[max_idx]}({max_p:.4f}) "
                f"probs=[{leaf_probs[i][0]:.4f}, {leaf_probs[i][1]:.4f}, {leaf_probs[i][2]:.4f}]"
            )

    uncertain_gates = []
    for i in range(tree.n_internal):
        for j in range(2):
            p = gate_probs[i, j]
            if snap_threshold < p < 1.0 - snap_threshold:
                side = "left" if j == 0 else "right"
                uncertain_gates.append(f"gate[{i}].{side}: prob={p:.4f}")

    return {
        "uncertain_leaves": uncertain_leaves,
        "uncertain_gates": uncertain_gates,
        "n_uncertain": len(uncertain_leaves) + len(uncertain_gates),
    }


def hard_project_inplace(tree, k=24.0):
    """Snap all weights to nearest hard choice (argmax/sign)."""
    with torch.no_grad():
        lc = torch.argmax(tree.leaf_logits, dim=1)
        new_leaf = torch.full_like(tree.leaf_logits, -k)
        new_leaf[torch.arange(tree.n_leaves), lc] = k
        tree.leaf_logits.copy_(new_leaf)

        gc = (tree.blend_logits >= 0).to(tree.blend_logits.dtype)
        new_gate = torch.where(
            gc > 0.5,
            torch.full_like(tree.blend_logits, k),
            torch.full_like(tree.blend_logits, -k),
        )
        tree.blend_logits.copy_(new_gate)


INIT_STRATEGIES_ALL = ["biased", "uniform", "xy_biased", "random_hot"]


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _setup_loss_axis(ax, args, max_x, title):
    ax.set_title(title)
    ax.set_ylabel("RMSE")
    ax.set_yscale("log")
    ax.set_ylim(args.loss_y_min, args.loss_y_max)
    ax.set_xlim(0, max(10, max_x))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10)


def save_loss_plot(path, hist, title, args, hardening_iter):
    fig, (ax_loss, ax_aux) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )

    ax_loss.plot(hist["iter"], hist["soft_rmse"], color="#2ca02c", linewidth=1.6, label="soft RMSE")
    ax_loss.plot(hist["iter"], hist["best_soft_rmse"], color="#d62728", linestyle="--", linewidth=1.4, label="best soft")
    ax_loss.plot(hist["eval_iter"], hist["hard_rmse"], "o", color="#1f77b4", markersize=3.2, alpha=0.8, label="hard RMSE")
    if hist["snap_eval_iter"]:
        ax_loss.plot(hist["snap_eval_iter"], hist["snap_rmse"], marker="*", markersize=11, color="#111111", linestyle="None", label="snapped RMSE")
    if hardening_iter is not None:
        ax_loss.axvline(hardening_iter, linestyle="--", linewidth=1.0, color="#6c6c6c", alpha=0.8, label="harden start")

    _setup_loss_axis(ax_loss, args, (hist["iter"][-1] if hist["iter"] else 1) + 5, title)
    ax_loss.legend(loc="upper right", fontsize=8)

    ax_aux.plot(hist["iter"], hist["tau"], linewidth=1.2, label="tau")
    ax_aux.plot(hist["iter"], hist["entropy"], linewidth=1.2, label="H")
    ax_aux.plot(hist["iter"], hist["binarity"], linewidth=1.2, label="B")
    if hardening_iter is not None:
        ax_aux.axvline(hardening_iter, linestyle="--", linewidth=1.0, color="#6c6c6c", alpha=0.8)
    ax_aux.set_xlabel("Iteration")
    ax_aux.set_ylabel("Aux")
    ax_aux.grid(True, alpha=0.30)
    ax_aux.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_manual_init_fn(args):
    has_expr = bool(args.init_expr.strip())
    has_blend = bool(args.init_blend.strip()) and bool(args.init_leaves.strip())
    if not has_expr and not has_blend:
        return None

    if has_expr:
        if args.depth == 0:
            parsed = parse_eml_expr(args.init_expr)
            args.depth = expr_depth(parsed)
            print(f"Auto-detected depth={args.depth} from --init-expr")

        def _manual(tree):
            init_from_expr(tree, args.init_expr, k=args.init_k)

        return _manual

    def _manual(tree):
        init_from_blend_leaves(tree, args.init_blend, args.init_leaves, k=args.init_k)

    return _manual


def train_one_seed(seed, strategy, args, x_train, y_train, t_train, manual_init_fn=None):
    torch.manual_seed(seed)
    tree = EMLTree(depth=args.depth, init_scale=args.init_scale, init_strategy=strategy, eml_clamp=args.eml_clamp)
    if manual_init_fn is not None:
        manual_init_fn(tree)
        if args.init_noise > 0:
            add_init_noise(tree, args.init_noise, seed=seed)

    optimizer = torch.optim.Adam(tree.parameters(), lr=args.lr)

    hist = {
        "iter": [],
        "soft_rmse": [],
        "best_soft_rmse": [],
        "eval_iter": [],
        "hard_rmse": [],
        "tau": [],
        "entropy": [],
        "binarity": [],
        "snap_eval_iter": [],
        "snap_rmse": [],
    }

    # Optional live plot (disabled by --skip-plot).
    live_fig = None
    live_ax_loss = None
    live_ax_aux = None
    live_ln_soft = None
    live_ln_best = None
    live_ln_hard = None
    live_ln_tau = None
    live_ln_h = None
    live_ln_b = None
    live_hard_vline = None
    if not args.skip_plot:
        try:
            plt.ion()
            live_fig, (live_ax_loss, live_ax_aux) = plt.subplots(
                2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
            )
            live_ln_soft, = live_ax_loss.plot([], [], color="#2ca02c", linewidth=1.6, label="soft RMSE")
            live_ln_best, = live_ax_loss.plot([], [], color="#d62728", linestyle="--", linewidth=1.4, label="best soft")
            live_ln_hard, = live_ax_loss.plot([], [], "o", color="#1f77b4", markersize=3.0, alpha=0.8, label="hard RMSE")
            _setup_loss_axis(
                live_ax_loss,
                args,
                args.search_iters + args.hardening_iters + 5,
                f"EML v16_final | seed={seed} ({strategy})",
            )
            live_ax_loss.legend(loc="upper right", fontsize=8)

            live_ln_tau, = live_ax_aux.plot([], [], linewidth=1.2, label="tau")
            live_ln_h, = live_ax_aux.plot([], [], linewidth=1.2, label="H")
            live_ln_b, = live_ax_aux.plot([], [], linewidth=1.2, label="B")
            live_ax_aux.set_xlabel("Iteration")
            live_ax_aux.set_ylabel("Aux")
            live_ax_aux.grid(True, alpha=0.30)
            live_ax_aux.legend(loc="upper right", fontsize=8)
        except Exception as ex:
            print(f"seed={seed} live plot disabled: {ex}")
            live_fig = None

    phase = "search"
    hardening_iter = None
    hard_step = 0

    best_soft_loss = float("inf")
    best_soft_state = None
    best_hard_loss = float("inf")
    best_hard_state = None

    plateau_counter = 0
    hard_success_streak = 0
    hard_trigger_streak = 0
    nan_streak = 0
    nan_restarts = 0

    total_iters = args.search_iters + args.hardening_iters
    for it in range(1, total_iters + 1):
        if nan_restarts >= args.max_nan_restarts > 0:
            print(f"seed={seed} it={it:6d} abandoned: too many NaN restarts ({nan_restarts})")
            break

        if phase == "search":
            if it > args.search_iters or (
                plateau_counter >= args.patience and best_soft_loss < args.patience_threshold
            ):
                phase = "hardening"
                hardening_iter = it
                hard_step = 0
                if best_soft_state is not None:
                    tree.load_state_dict(best_soft_state)
                    optimizer = torch.optim.Adam(tree.parameters(), lr=args.lr)
                print(f"seed={seed} it={it:6d} HARDENING start")

        if phase == "search":
            tau = args.tau_search
            lam_ent = 0.0
            lam_bin = 0.0
            lr_mult = 1.0
        else:
            if hard_step >= args.hardening_iters:
                print(f"seed={seed} it={it:6d} HARDENING complete")
                break
            t = hard_step / max(1, args.hardening_iters)
            t_tau = t ** args.hardening_tau_power
            tau = args.tau_search * (args.tau_hard / args.tau_search) ** t_tau
            lam_ent = t * args.lam_ent_hard
            lam_bin = t * args.lam_bin_hard
            lr_mult = max(args.hardening_lr_floor, (1.0 - t) ** 2)
            hard_step += 1

        optimizer.param_groups[0]["lr"] = args.lr * lr_mult
        optimizer.zero_grad()

        pred, leaf_probs, gate_probs, eml_outputs = tree(x_train, y_train, tau_leaf=tau, tau_gate=tau)
        total, data_loss, entropy, binarity, inter_pen, _, _ = compute_losses(
            pred, t_train, leaf_probs, gate_probs, eml_outputs,
            lam_ent, lam_bin, args.lam_inter, args.inter_threshold,
            lam_sparse=0.0, uncertainty_power=1.0,
        )

        if not torch.isfinite(total):
            nan_streak += 1
            plateau_counter += 1
            if nan_streak >= args.nan_restart_patience:
                if best_soft_state is not None:
                    tree.load_state_dict(best_soft_state)
                    optimizer = torch.optim.Adam(tree.parameters(), lr=args.lr * lr_mult)
                nan_streak = 0
                nan_restarts += 1
            continue

        nan_streak = 0
        total.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        optimizer.step()

        soft_loss = float(data_loss.item())
        if np.isfinite(soft_loss) and soft_loss < best_soft_loss:
            rel_imp = (best_soft_loss - soft_loss) / max(best_soft_loss, 1e-15)
            best_soft_loss = soft_loss
            best_soft_state = snapshot(tree)
            plateau_counter = 0 if rel_imp > args.plateau_rtol else plateau_counter + 1
        else:
            plateau_counter += 1

        hist["iter"].append(it)
        hist["soft_rmse"].append(math.sqrt(max(soft_loss, 0.0)))
        hist["best_soft_rmse"].append(
            math.sqrt(max(best_soft_loss, 0.0)) if np.isfinite(best_soft_loss) else float("nan")
        )
        hist["tau"].append(float(tau))
        hist["entropy"].append(float(entropy.item()))
        hist["binarity"].append(float(binarity.item()))

        do_eval = (it % max(1, args.eval_every) == 0)
        if phase == "hardening" and args.tail_eval_every > 0 and tau <= args.tail_eval_tau:
            do_eval = do_eval or (it % max(1, args.tail_eval_every) == 0)
        if do_eval:
            hard_mse, _, _ = evaluate(tree, x_train, y_train, t_train, tau=args.tau_hard)
            hard_rmse = math.sqrt(max(hard_mse, 0.0)) if np.isfinite(hard_mse) else float("nan")
            hist["eval_iter"].append(it)
            hist["hard_rmse"].append(hard_rmse)

            if np.isfinite(hard_mse) and hard_mse < best_hard_loss:
                best_hard_loss = hard_mse
                best_hard_state = snapshot(tree)

            if phase == "hardening" and np.isfinite(hard_mse) and hard_mse < args.success_thr:
                hard_success_streak += 1
            else:
                hard_success_streak = 0

            if phase == "search" and np.isfinite(hard_mse) and hard_mse < args.hard_trigger_mse:
                hard_trigger_streak += 1
            elif phase == "search":
                hard_trigger_streak = 0

            phase_tag = " [HARD]" if phase == "hardening" else ""
            print(
                f"seed={seed} it={it:6d} soft_rmse={hist['soft_rmse'][-1]:.3e} "
                f"hard_rmse={hard_rmse:.3e} tau={tau:.3f} H={entropy.item():.4f} "
                f"B={binarity.item():.4f} inter={inter_pen.item():.3e} "
                f"nan_restarts={nan_restarts}{phase_tag}"
            )

            if hard_success_streak >= args.early_stop_count:
                print(f"seed={seed} it={it:6d} early stop: hard rmse below threshold")
                break

            if phase == "search" and hard_trigger_streak >= args.hard_trigger_count:
                phase = "hardening"
                hardening_iter = it + 1
                hard_step = 0
                if best_soft_state is not None:
                    tree.load_state_dict(best_soft_state)
                    optimizer = torch.optim.Adam(tree.parameters(), lr=args.lr)
                hard_trigger_streak = 0
                print(f"seed={seed} it={it:6d} HARDENING start (hard diagnostic already exact)")

            if live_fig is not None:
                live_ln_soft.set_data(hist["iter"], hist["soft_rmse"])
                live_ln_best.set_data(hist["iter"], hist["best_soft_rmse"])
                live_ln_hard.set_data(hist["eval_iter"], hist["hard_rmse"])
                live_ln_tau.set_data(hist["iter"], hist["tau"])
                live_ln_h.set_data(hist["iter"], hist["entropy"])
                live_ln_b.set_data(hist["iter"], hist["binarity"])
                if hardening_iter is not None and live_hard_vline is None:
                    live_hard_vline = live_ax_loss.axvline(
                        hardening_iter, linestyle="--", linewidth=1.0, color="#6c6c6c", alpha=0.8
                    )
                    live_ax_aux.axvline(
                        hardening_iter, linestyle="--", linewidth=1.0, color="#6c6c6c", alpha=0.8
                    )
                live_ax_aux.relim()
                live_ax_aux.autoscale_view()
                live_fig.canvas.draw()
                live_fig.canvas.flush_events()

    # Restore best hard model if available, otherwise best soft model.
    if best_hard_state is not None:
        tree.load_state_dict(best_hard_state)
    elif best_soft_state is not None:
        tree.load_state_dict(best_soft_state)

    if args.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            tree.parameters(),
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            p, lp, gp, eo = tree(x_train, y_train, tau_leaf=args.tau_hard, tau_gate=args.tau_hard)
            tot, _, _, _, _, _, _ = compute_losses(
                p, t_train, lp, gp, eo,
                args.lam_ent_hard, args.lam_bin_hard, args.lam_inter, args.inter_threshold,
                lam_sparse=0.0, uncertainty_power=1.0,
            )
            tot.backward()
            return tot

        try:
            lbfgs.step(closure)
        except Exception as ex:
            print(f"seed={seed} lbfgs skipped due to error: {ex}")

    snap_info = analyze_snap(tree, args.snap_threshold)
    snapped_tree = deepcopy(tree)
    hard_project_inplace(snapped_tree)
    snap_mse, snap_max_real, snap_max_imag = evaluate(snapped_tree, x_train, y_train, t_train, tau=0.01)
    snap_rmse = math.sqrt(max(snap_mse, 0.0)) if np.isfinite(snap_mse) else float("nan")
    snap_eval_x = (hist["iter"][-1] + 1) if hist["iter"] else 1
    hist["snap_eval_iter"].append(snap_eval_x)
    hist["snap_rmse"].append(snap_rmse)

    fit_success = bool(np.isfinite(snap_mse) and snap_mse < args.fit_success_thr)
    symbol_success = bool(np.isfinite(snap_mse) and snap_mse < args.success_thr)
    stable_symbol_success = bool(symbol_success and snap_info["n_uncertain"] <= args.max_uncertain_success)

    summary = {
        "snap_mse": snap_mse,
        "snap_rmse": snap_rmse,
        "snap_max_real": snap_max_real,
        "snap_max_imag": snap_max_imag,
        "fit_success": fit_success,
        "symbol_success": symbol_success,
        "stable_symbol_success": stable_symbol_success,
        "success": fit_success,
        "n_uncertain": snap_info["n_uncertain"],
        "hardening_iter": hardening_iter,
        "nan_restarts": nan_restarts,
    }
    if live_fig is not None:
        plt.ioff()
        plt.close(live_fig)
    return tree, snapped_tree, hist, summary


def parse_args():
    p = argparse.ArgumentParser(description="EML tree trainer (v16_final, simplified)")
    a = p.add_argument

    # Target / shape
    a("--target-fn", type=str, default="eml_depth3", choices=sorted(TARGET_FUNCTIONS.keys()))
    a("--depth", type=int, default=3, help="Tree depth; use 0 with --init-expr for auto-depth.")
    a("--init-scale", type=float, default=1.0)
    a("--init-strategy", type=str, default="all", help="biased/uniform/xy_biased/random_hot/manual/all")
    a("--init-expr", type=str, default="")
    a("--init-blend", type=str, default="")
    a("--init-leaves", type=str, default="")
    a("--init-k", type=float, default=32.0)
    a("--init-noise", type=float, default=0.0)

    # Seeds
    a("--seed0", type=int, default=137)
    a("--seeds", type=int, default=8)

    # Data
    a("--data-lo", type=float, default=1.0)
    a("--data-hi", type=float, default=3.0)
    a("--data-step", type=float, default=0.1)
    a("--gen-lo", type=float, default=0.5)
    a("--gen-hi", type=float, default=5.0)
    a("--generalization-points", type=int, default=4000)

    # Optimization
    a("--search-iters", type=int, default=6000)
    a("--hardening-iters", type=int, default=2000)
    a("--lr", type=float, default=0.01)
    a("--tau-search", type=float, default=2.5)
    a("--tau-hard", type=float, default=0.01)
    a("--hardening-tau-power", type=float, default=2.0)
    a("--hardening-lr-floor", type=float, default=0.01)
    a("--patience", type=int, default=4200)
    a("--patience-threshold", type=float, default=1e-2)
    a("--plateau-rtol", type=float, default=1e-3)
    a("--lam-ent-hard", type=float, default=2e-2)
    a("--lam-bin-hard", type=float, default=2e-2)
    a("--lam-inter", type=float, default=1e-4)
    a("--inter-threshold", type=float, default=50.0)
    a("--eml-clamp", type=float, default=1e300)

    # Diagnostics
    a("--eval-every", type=int, default=200)
    a("--tail-eval-every", type=int, default=50)
    a("--tail-eval-tau", type=float, default=0.2)
    a("--early-stop-count", type=int, default=10)
    a("--hard-trigger-mse", type=float, default=1e-20, help="If hard MSE stays below this during search, start hardening early.")
    a("--hard-trigger-count", type=int, default=3, help="Consecutive hard checks below --hard-trigger-mse to trigger early hardening.")
    a("--nan-restart-patience", type=int, default=50)
    a("--max-nan-restarts", type=int, default=100)

    # Success / snapping
    a("--fit-success-thr", type=float, default=1e-6)
    a("--success-thr", type=float, default=1e-20)
    a("--snap-threshold", type=float, default=0.01)
    a("--max-uncertain-success", type=int, default=0)

    # Polish
    a("--lbfgs-steps", type=int, default=0)
    a("--lbfgs-lr", type=float, default=0.6)

    # Output
    a("--save-prefix", type=str, default="v16_run")
    a("--export-m", type=str, default="eml_tree_v16_final.m")
    a("--skip-plot", action="store_true")
    a("--loss-y-min", type=float, default=1e-16)
    a("--loss-y-max", type=float, default=1e1)

    args = p.parse_args()

    if args.depth < 0:
        raise ValueError("--depth must be >= 0")
    if args.init_strategy not in INIT_STRATEGIES_ALL + ["manual", "all"]:
        raise ValueError(f"Unsupported --init-strategy: {args.init_strategy}")

    return args


def main():
    args = parse_args()
    target_fn, target_desc = get_target_fn(args.target_fn)

    output_dir = Path(args.save_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir = output_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    run_start = datetime.now()
    stamp = run_start.strftime("%Y%m%d-%H%M%S")
    run_tag = output_dir.name
    stdout_log = output_dir / f"{run_tag}_stdout_{stamp}.log"
    log_handle = open(stdout_log, "w", encoding="utf-8", buffering=1)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = TeeStream(orig_stdout, log_handle)
    sys.stderr = TeeStream(orig_stderr, log_handle)

    try:
        print(f"Command: {' '.join(sys.argv)}")
        print(f"Run start: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {output_dir.resolve()}")
        print(f"Logging to: {stdout_log}")

        x_train, y_train, t_train = make_grid_data(target_fn, lo=args.data_lo, hi=args.data_hi, step=args.data_step)
        x_gen, y_gen, t_gen = make_generalization_data(target_fn, lo=args.gen_lo, hi=args.gen_hi, n=args.generalization_points)
        manual_init_fn = make_manual_init_fn(args)
        if manual_init_fn is not None:
            strategies = ["manual"]
        elif args.init_strategy == "all":
            strategies = INIT_STRATEGIES_ALL
        elif args.init_strategy == "manual":
            raise ValueError("--init-strategy=manual requires --init-expr or --init-blend/--init-leaves")
        else:
            strategies = [args.init_strategy]

        if manual_init_fn is not None and args.init_noise == 0:
            seeds = [args.seed0]
        else:
            seeds = list(range(args.seed0, args.seed0 + args.seeds))
        run_plan = [(s, st) for s in seeds for st in strategies]

        print("\n=== EML Tree Training v16_final ===")
        print(f"target: {args.target_fn} = {target_desc}")
        print(f"depth={args.depth} leaves={2**args.depth} internal={2**args.depth - 1}")
        print(f"runs={len(run_plan)} seeds={len(seeds)} strategies={strategies}")
        print(f"search_iters={args.search_iters} hardening_iters={args.hardening_iters}")
        print(f"tau_search={args.tau_search} tau_hard={args.tau_hard}")
        print(f"fit_success_thr={args.fit_success_thr:.0e} success_thr={args.success_thr:.0e} snap_threshold={args.snap_threshold}")

        best = {"score": float("inf"), "seed": None, "strategy": None, "state": None}
        all_data = []
        n_success = 0
        n_fit_success = 0
        n_symbol_success = 0
        n_stable_symbol_success = 0

        for run_idx, (seed, strategy) in enumerate(run_plan, start=1):
            print(f"\n--- Run {run_idx}/{len(run_plan)}: seed={seed} strategy={strategy} ---")
            tree, snapped_tree, hist, summary = train_one_seed(
                seed, strategy, args, x_train, y_train, t_train, manual_init_fn=manual_init_fn if strategy == "manual" else None
            )

            gen_mse, gen_max_real, gen_max_imag = evaluate(snapped_tree, x_gen, y_gen, t_gen, tau=0.01)
            print(f"seed={seed} gen_rmse={math.sqrt(max(gen_mse, 0.0)):.3e} max_real={gen_max_real:.3e} max_imag={gen_max_imag:.3e}")

            seed_stem = f"{run_tag}_run{run_idx:02d}_seed{seed}_{strategy}_{stamp}"
            seed_base = output_dir / seed_stem
            torch.save(tree.state_dict(), f"{seed_base}.pt")
            snapped_tree.export_mathematica(f"{seed_base}.m", discretize=True, snap_threshold=args.snap_threshold)
            tree.export_mathematica(f"{seed_base}_continuous.m", discretize=False)

            plot_path = png_dir / f"{seed_stem}_loss.png"
            save_loss_plot(
                str(plot_path),
                hist,
                title=f"EML v16_final | run {run_idx}/{len(run_plan)} seed={seed} ({strategy})",
                args=args,
                hardening_iter=summary["hardening_iter"],
            )

            if summary["success"]:
                n_success += 1
            if summary["fit_success"]:
                n_fit_success += 1
            if summary["symbol_success"]:
                n_symbol_success += 1
            if summary["stable_symbol_success"]:
                n_stable_symbol_success += 1

            if np.isfinite(summary["snap_mse"]) and summary["snap_mse"] < best["score"]:
                best = {
                    "score": summary["snap_mse"],
                    "seed": seed,
                    "strategy": strategy,
                    "state": snapshot(snapped_tree),
                }

            all_data.append(
                {
                    "run_index": run_idx,
                    "seed": seed,
                    "strategy": strategy,
                    "summary": summary,
                    "gen_mse": gen_mse,
                }
            )

        print("\n" + "=" * 60)
        print(
            f"Results: success={n_success}/{len(run_plan)} "
            f"fit_success={n_fit_success}/{len(run_plan)} "
            f"symbol_success={n_symbol_success}/{len(run_plan)} "
            f"stable_symbol_success={n_stable_symbol_success}/{len(run_plan)}"
        )
        print(f"Best seed={best['seed']} strategy={best['strategy']} snapped_rmse={math.sqrt(max(best['score'], 0.0)):.3e}")

        if best["state"] is not None:
            best_tree = EMLTree(depth=args.depth, eml_clamp=args.eml_clamp)
            best_tree.load_state_dict(best["state"])
            export_path = output_dir / args.export_m
            best_tree.export_mathematica(str(export_path), discretize=True, snap_threshold=args.snap_threshold)
            train_mse, train_mr, train_mi = evaluate(best_tree, x_train, y_train, t_train, tau=0.01)
            gen_mse, gen_mr, gen_mi = evaluate(best_tree, x_gen, y_gen, t_gen, tau=0.01)
        else:
            train_mse, train_mr, train_mi = float("nan"), 0.0, 0.0
            gen_mse, gen_mr, gen_mi = float("nan"), 0.0, 0.0

        metrics = {
            "target_fn": args.target_fn,
            "target_desc": target_desc,
            "n_successes": n_success,
            "n_fit_successes": n_fit_success,
            "n_symbol_successes": n_symbol_success,
            "n_stable_symbol_successes": n_stable_symbol_success,
            "n_runs": len(run_plan),
            "best_seed": best["seed"],
            "best_strategy": best["strategy"],
            "best_snapped_mse": best["score"],
            "train": {"mse": train_mse, "max_real_err": train_mr, "max_imag": train_mi},
            "gen": {"mse": gen_mse, "max_real_err": gen_mr, "max_imag": gen_mi},
            "per_seed": [
                {
                    "run_index": d["run_index"],
                    "seed": d["seed"],
                    "strategy": d["strategy"],
                    "success": d["summary"]["success"],
                    "fit_success": d["summary"]["fit_success"],
                    "symbol_success": d["summary"]["symbol_success"],
                    "stable_symbol_success": d["summary"]["stable_symbol_success"],
                    "snap_mse": d["summary"]["snap_mse"],
                    "gen_mse": d["gen_mse"],
                    "n_uncertain": d["summary"]["n_uncertain"],
                    "hardening_iter": d["summary"]["hardening_iter"],
                }
                for d in all_data
            ],
            "args": vars(args),
            "command": " ".join(sys.argv),
            "stdout_log_file": str(stdout_log),
            "output_dir": str(output_dir.resolve()),
            "png_dir": str(png_dir.resolve()),
        }
        metrics_path = output_dir / f"{run_tag}_metrics_{stamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_sanitize_for_json(metrics), f, indent=2, allow_nan=False)
        print(f"Saved metrics: {metrics_path}")

    finally:
        run_end = datetime.now()
        print(f"Run end: {run_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed: {run_end - run_start}")
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
