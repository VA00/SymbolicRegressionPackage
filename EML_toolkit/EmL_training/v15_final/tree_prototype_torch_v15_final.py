"""
EML Tree Trainer v15_old

Clean rewrite of v12 with clear 5-step pipeline:
  1. INIT:      Random or manual + noise initialization
  2. SEARCH:    Adam at fixed high tau until plateau detected
  3. HARDENING: Gradually lower tau, add entropy/binarity penalties
  4. SNAP:      Project weights to 0/1 (with uncertainty threshold)
  5. EVALUATE:  Check snapped tree MSE; declare success or fail

Key changes from v12:
  - Single clear pipeline with named phases (no confusing dual eval)
  - Plotted loss = what optimizer actually sees (transitions naturally)
  - Discrete diagnostic plotted as separate line (blue dots)
  - Snap threshold: won't snap 0.51→1; reports uncertain weights
  - Command line logged in output for reproducibility
  - Default eml_clamp=1e300 (NaN prevention only, not EML distortion)
  - Grid lines every decade on loss plots
  - Seed progress counter on plots ("Seed 3 of 16")
  - Success/fail flag per seed with --success-thr
  - Domain-aware data generation: evaluates target in complex128,
    rejects points where Im!=0 or result is non-finite
  - NaN rate abandon: kills seeds with >95% NaN steps (--nan-rate-abandon)
    to catch sporadic-valid pathology where streak-based restarts never fire
  - Gentler hardening: --hardening-tau-power (default 1.5) keeps tau high
    longer; quadratic LR decay with --hardening-lr-floor (default 0.01)
  - Restored v12c NaN-safe blend: real-arithmetic component-wise blend
    avoids IEEE 754 complex cross-term 0*Inf=NaN; _BYPASS_THR guard
    for residual 0.0*Inf when s rounds to exactly 1.0
  - Post-EML nan_to_num: catches Inf-Inf=NaN from eml_exact itself
  - Event-driven hardening trigger from discrete plateau / ambiguity trends
  - Projection-consistency surrogate (soft vs near-discrete shadow outputs)
  - Uncertainty-weighted entropy/binarity and optional structure sparsity
  - Optional depth curriculum warm-start via lifted state dicts
  - Strict JSON metrics output (NaN/Inf sanitized)
"""

import argparse
import json
import math
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
# Default: no clamping (1e300 is safely below float64 max ~1.8e308).
# Set to ~700 only if NaN restarts are excessive.  Clamping at 500
# DISTORTS the EML function and should be avoided.
_EML_CLAMP_DEFAULT = 1.0e300
_BYPASS_THR = 1.0 - torch.finfo(torch.float64).eps  # s > this → pure bypass


# ---------------------------------------------------------------------------
# Target functions
# ---------------------------------------------------------------------------

def _target_eml_depth2(x, y):
    """e - log(exp(y) - log(x))  i.e.  EML(1, EML(y, x))"""
    return np.exp(1) - np.log(np.exp(y) - np.log(x))

def _target_eml_depth2a(x, y):
    """e^e/y - log(exp(y) - log(x))"""
    return np.exp(np.exp(1)) / y - np.log(np.exp(y) - np.log(x))

def _target_eml_depth3(x, y):
    """e^e/(e^y - log(x))  EML[EML[1, EML[y, x]], 1]"""
    return np.exp(np.exp(1)) / (np.exp(y) - np.log(x))

def _target_eml_depth4(x, y):
    """log(e^x - log(y))  EML[1,EML[EML[1, EML[x, y]], 1]] """
    return np.log(np.exp(x) - np.log(y))

def _target_eml_depth5(x, y):
    """log(e-log(e^x - log(y)))  EML[1,EML[EML[1,EML[1,EML[x,y]]],1]] """
    return np.log(np.exp(1) - np.log(np.exp(x) - np.log(y)))

def _target_multiply(x, y):
    """x * y depth=8

    Exact EML construction:
      times = EML[EML[1, EML[EML[EML[1, EML[EML[1, EML[1, x]], 1]], y], 1]], 1]
      identity    = EML[1, EML[EML[1, EML[x, 1]], 1]]
    Bottom-layer x,y variant:
      EML[EML[1, EML[EML[EML[1, EML[EML[1, EML[1, x]], 1]],
      EML[1, EML[EML[1, EML[y, 1]], 1]]], 1]], 1]
    """
    return x * y


TARGET_FUNCTIONS = {
    "eml_depth2":   (_target_eml_depth2, "e - log(exp(y) - log(x))"),
    "eml_depth2a":  (_target_eml_depth2a, "e^e/y - log(exp(y) - log(x))"),
    "eml_depth3":   (_target_eml_depth3, "e^e/(e^y - log(x))"),
    "eml_depth4":   (_target_eml_depth4, "log(e^x - log(y))"),
    "eml_depth5":   (_target_eml_depth5, "log(e-log(e^x - log(y)))"),
    "multiply":     (_target_multiply,   "x * y")
}



def get_target_fn(name):
    if name not in TARGET_FUNCTIONS:
        available = ", ".join(TARGET_FUNCTIONS.keys())
        raise ValueError(f"Unknown target '{name}'. Available: {available}")
    return TARGET_FUNCTIONS[name]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def snapshot(tree):
    """Detached copy of tree state dict."""
    return {k: v.detach().clone() for k, v in tree.state_dict().items()}


def _infer_depth_from_state(state):
    leaf = state.get("leaf_logits")
    if leaf is None or leaf.ndim != 2:
        return None
    n_leaves = int(leaf.shape[0])
    if n_leaves <= 0 or (n_leaves & (n_leaves - 1)) != 0:
        return None
    return int(round(math.log2(n_leaves)))


def load_warmstart_state(tree, state_path):
    """Lift a smaller/equal tree state into current depth by prefix copy."""
    loaded = torch.load(state_path, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise ValueError(f"Unsupported warmstart format: {type(loaded)}")

    if "leaf_logits" not in loaded or "blend_logits" not in loaded:
        raise ValueError("Warmstart state must contain leaf_logits and blend_logits.")

    with torch.no_grad():
        src_leaf = loaded["leaf_logits"].to(tree.leaf_logits.dtype)
        src_gate = loaded["blend_logits"].to(tree.blend_logits.dtype)
        n_leaf = min(src_leaf.shape[0], tree.leaf_logits.shape[0])
        n_gate = min(src_gate.shape[0], tree.blend_logits.shape[0])
        tree.leaf_logits[:n_leaf].copy_(src_leaf[:n_leaf])
        tree.blend_logits[:n_gate].copy_(src_gate[:n_gate])

    return _infer_depth_from_state(loaded), n_leaf, n_gate


def _sanitize_for_json(obj):
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


# ---------------------------------------------------------------------------
# EML operator
# ---------------------------------------------------------------------------

def eml_exact(x, y):
    """EML[x,y] = Exp[x] - Log[y], complex plane."""
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
                right = parse_eml_expr(inner[i + 1:])
                return ("EML", left, right)
        raise ValueError(f"Malformed EML expression (no comma at depth 0): {s}")
    raise ValueError(f"Cannot parse EML expression: '{s}'.")


def expr_depth(node):
    if isinstance(node, str):
        return 0
    return 1 + max(expr_depth(node[1]), expr_depth(node[2]))


def expr_to_string(node):
    if isinstance(node, str):
        return node
    return f"EML[{expr_to_string(node[1])}, {expr_to_string(node[2])}]"


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


# ===========================================================================
# Manual Initialization
# ===========================================================================

def flat_node_idx(tree_depth, level_from_top, pos_in_level):
    return 2 ** tree_depth - 2 ** (level_from_top + 1) + pos_in_level


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
        raise ValueError(
            f"Blend gives {len(blend_bits)} nodes, tree has {tree.n_internal}.")

    leaf_clean = leaf_str.replace(" ", "")
    symbols = leaf_clean.split(",")
    if len(symbols) != tree.n_leaves:
        raise ValueError(
            f"Leaf gives {len(symbols)} leaves, tree has {tree.n_leaves}.")

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


def validate_manual_init(tree, x_data, y_data, targets, label="INIT"):
    with torch.no_grad():
        pred, lp, gp, eo = tree(x_data, y_data, tau_leaf=0.01, tau_gate=0.01)
        mse = torch.mean((pred - targets).abs() ** 2).real.item()
        max_real = torch.max((pred.real - targets.real).abs()).item()
        max_imag = torch.max(pred.imag.abs()).item()
        any_bad = torch.isnan(pred).any() or torch.isinf(pred).any()
    status = "OK" if not any_bad else "*** NaN/Inf ***"
    print(f"{label}: mse={mse:.6e} max_err={max_real:.3e} max_imag={max_imag:.3e} [{status}]")
    return mse, max_real, max_imag


def describe_init(tree, show_numeric=False):
    with torch.no_grad():
        leaf_choices = torch.argmax(tree.leaf_logits, dim=1)
        gate_bypass = (tree.blend_logits >= 0).int()
    names = {0: "1", 1: "x", 2: "y"}
    leaf_desc = [names[c.item()] for c in leaf_choices]
    blend_desc = [
        f"{gate_bypass[i, 0].item()}{gate_bypass[i, 1].item()}"
        for i in range(tree.n_internal)
    ]
    print(f"  leaves:  [{', '.join(leaf_desc)}]")
    print(f"  blend:   [{', '.join(blend_desc)}]")
    if show_numeric:
        with torch.no_grad():
            for i in range(tree.n_leaves):
                if leaf_choices[i].item() != 0:
                    v = tree.leaf_logits[i].tolist()
                    print(f"  leaf[{i}] = [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}] -> {names[leaf_choices[i].item()]}")
            for i in range(tree.n_internal):
                if gate_bypass[i, 0].item() == 0 or gate_bypass[i, 1].item() == 0:
                    v = tree.blend_logits[i].tolist()
                    print(f"  gate[{i}] = [{v[0]:+.3f}, {v[1]:+.3f}]")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EMLTree(nn.Module):
    """Full binary tree of depth `depth` with EML at every internal node."""

    def __init__(self, depth, init_scale=1.0, init_strategy="biased",
                 eml_clamp=None):
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

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

            s = torch.sigmoid(
                self.blend_logits[node_idx:node_idx + n_pairs] / tau_gate
            )
            gate_probs_levels.append(s)

            # s=1 -> constant 1,  s=0 -> child value
            #
            # NaN prevention: complex multiplication (a+0j)*(c+0j)
            # computes imag = a*0 + 0*c; when either is Inf this
            # gives 0*Inf = NaN.  By blending real & imag parts
            # separately with real-valued s we avoid the cross-term.
            # The _BYPASS_THR guard handles the residual real-
            # arithmetic 0.0 * Inf = NaN when s rounds to exactly 1.
            s_left = s[:, 0].unsqueeze(0)            # [1, n_pairs] real
            s_right = s[:, 1].unsqueeze(0)
            bypass_left = s_left > _BYPASS_THR
            bypass_right = s_right > _BYPASS_THR
            oml = 1.0 - s_left                       # "one minus left"
            omr = 1.0 - s_right

            # real-arithmetic blend, component-wise
            lr = torch.where(bypass_left, 1.0, s_left + oml * left_children.real)
            li = torch.where(bypass_left, 0.0, oml * left_children.imag)
            rr = torch.where(bypass_right, 1.0, s_right + omr * right_children.real)
            ri = torch.where(bypass_right, 0.0, omr * right_children.imag)
            left_input = torch.complex(lr, li)
            right_input = torch.complex(rr, ri)

            current_level = eml_exact(left_input, right_input)

            # Clamp to prevent Inf cascade; scrub NaN from Inf-Inf in eml_exact.
            current_level = torch.complex(
                torch.nan_to_num(current_level.real, nan=0.0, posinf=self.eml_clamp, neginf=-self.eml_clamp).clamp(-self.eml_clamp, self.eml_clamp),
                torch.nan_to_num(current_level.imag, nan=0.0, posinf=self.eml_clamp, neginf=-self.eml_clamp).clamp(-self.eml_clamp, self.eml_clamp),
            )

            eml_outputs.append(current_level)
            node_idx += n_pairs

        gate_probs = torch.cat(gate_probs_levels, dim=0)
        return current_level.squeeze(1), leaf_probs, gate_probs, eml_outputs

    # --- Mathematica export -----------------------------------------------
    def _format_weights_mma(self, discretize=True, snap_threshold=0.0):
        """Format leafWeights / blendSigmoid for Mathematica.

        If discretize=True and snap_threshold>0, only snaps weights that
        exceed the threshold.  Returns (leaf_line, blend_line, uncertain).
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
                    lw[i] = leaf_probs[i]  # keep continuous
                else:
                    lw[i, max_idx] = 1.0
            bs = np.zeros((self.n_internal, 2))
            for i in range(self.n_internal):
                for j in range(2):
                    p = gate_probs[i, j]
                    if snap_threshold > 0 and snap_threshold < p < 1.0 - snap_threshold:
                        uncertain.append(f"gate[{i},{j}]: prob={p:.4f}")
                        bs[i, j] = p  # keep continuous
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

    def export_mathematica(self, filename, discretize=True, comment="",
                           snap_threshold=0.0):
        leaf_line, blend_line, uncertain = self._format_weights_mma(
            discretize, snap_threshold)

        lines = [
            f"(* EML Tree exported from PyTorch v15_old *)",
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

    def export_snapshot_mma(self, filename, iteration, mse=None):
        leaf_line, blend_line, _ = self._format_weights_mma(discretize=False)
        tag = f"it={iteration}"
        if mse is not None:
            tag += f" mse={mse:.3e}"
        lines = [
            f"(* {tag} *)",
            leaf_line,
            blend_line,
            "{blendSigmoid, leafWeights}",
        ]
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _filter_real_domain(x, y, target_fn, imag_tol=1e-12, label="data"):
    """Evaluate target_fn in complex128, keep only points where |Im(result)| < tol and result is finite."""
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
    return (torch.tensor(xx, dtype=REAL_DTYPE), torch.tensor(yy, dtype=REAL_DTYPE), torch.tensor(tt, dtype=DTYPE))


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
    return (torch.tensor(x_ok, dtype=REAL_DTYPE), torch.tensor(y_ok, dtype=REAL_DTYPE), torch.tensor(t_ok, dtype=DTYPE))


# ---------------------------------------------------------------------------
# Losses & evaluation
# ---------------------------------------------------------------------------

def compute_losses(pred, target, leaf_probs, gate_probs, eml_outputs,
                   lam_ent, lam_bin, lam_inter, inter_threshold,
                   lam_sparse=0.0, uncertainty_power=2.0):
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

    total = (data_loss + lam_ent * entropy + lam_bin * binarity
             + lam_inter * inter_penalty + lam_sparse * sparse)
    ambiguity = torch.cat([leaf_unc, gate_unc.reshape(-1)]).mean()
    return total, data_loss, entropy, binarity, inter_penalty, sparse, ambiguity


def evaluate(tree, x_data, y_data, targets, tau=0.01):
    with torch.no_grad():
        pred, _, _, _ = tree(x_data, y_data, tau_leaf=tau, tau_gate=tau)
        mse = torch.mean((pred - targets).abs() ** 2).real.item()
        max_real = torch.max((pred.real - targets.real).abs()).item()
        max_imag = torch.max(pred.imag.abs()).item()
    return mse, max_real, max_imag


# ---------------------------------------------------------------------------
# Snap analysis  (for interpreting converged weights)
# ---------------------------------------------------------------------------

def analyze_snap(tree, snap_threshold=0.01):
    """Check which weights can be cleanly snapped to 0/1.

    Returns dict with 'uncertain_leaves', 'uncertain_gates', 'n_uncertain'.
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
            uncertain_leaves.append(f"leaf[{i}]: best={names[max_idx]}({max_p:.4f}) probs=[{leaf_probs[i][0]:.4f}, {leaf_probs[i][1]:.4f}, {leaf_probs[i][2]:.4f}]")

    uncertain_gates = []
    for i in range(tree.n_internal):
        for j in range(2):
            p = gate_probs[i, j]
            if snap_threshold < p < 1.0 - snap_threshold:
                side = "left" if j == 0 else "right"
                uncertain_gates.append(
                    f"gate[{i}].{side}: prob={p:.4f}")

    return {
        "uncertain_leaves": uncertain_leaves,
        "uncertain_gates": uncertain_gates,
        "n_uncertain": len(uncertain_leaves) + len(uncertain_gates),
    }


def hard_project_inplace(tree, k=24.0):
    """Snap all weights to nearest integer (argmax for leaves, sign for gates)."""
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_loss_axis(ax, args, n_iters, title=""):
    """Configure a loss axis: RMSE, grid every decade, labels every 3."""
    ax.set_ylabel("RMSE")
    ax.set_ylim(args.loss_y_min, args.loss_y_max)
    ax.set_xlim(0, max(10, n_iters))
    if title:
        ax.set_title(title)
    # Major ticks every 3 decades (labeled): 1e-15, 1e-12, ..., 1e3
    y_lo_exp = int(math.floor(math.log10(max(args.loss_y_min, 1e-300))))
    y_hi_exp = int(math.ceil(math.log10(max(args.loss_y_max, 1e-300))))
    major_ticks = [10.0**e for e in range(y_lo_exp, y_hi_exp + 1, 3)]
    ax.set_yticks(major_ticks)
    # Minor ticks every decade (grid lines, no labels)
    minor_ticks = [10.0**e for e in range(y_lo_exp, y_hi_exp + 1)]
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True, which="major", alpha=0.4, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.2, linewidth=0.5)


def add_transition_line(ax, x_pos, label=None):
    ax.axvline(x_pos, color="0.35", linestyle="--", linewidth=1.2, alpha=0.7)
    if label:
        ax.text(x_pos, ax.get_ylim()[1], f" {label}", va="top", ha="left",
                fontsize=7, color="0.35")


def save_loss_plot(path, histories, title, n_iters, hardening_iter, args, focus_last_iters=0):
    fig, (ax_loss, ax_aux) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    plot_max_iter = max(n_iters, max(histories["iter"]) if histories["iter"] else 0)
    if histories.get("post_iter"):
        plot_max_iter = max(plot_max_iter, max(histories["post_iter"]) + 1)

    ax_loss.semilogy(histories["iter"], histories["loss"],
                     color="#2ca02c", linewidth=1.8, alpha=0.75,
                     label="optimizer RMSE")
    ax_loss.semilogy(histories["iter"], histories["best"],
                     color="#d62728", linestyle="-.", linewidth=1.4,
                     label="best search")
    if histories["disc_iter"]:
        ax_loss.semilogy(histories["disc_iter"], histories["disc"],
                         "o", color="#1f77b4", markersize=3, alpha=0.7,
                         label="discrete RMSE", zorder=5)
    ax_loss.semilogy(histories["iter"], histories["inter"],
                     color="#ff7f0e", linestyle=":", linewidth=1.2,
                     alpha=0.6, label="inter penalty")
    if hardening_iter is not None:
        add_transition_line(ax_loss, hardening_iter, "HARDEN")
    if histories.get("post_iter"):
        ax_loss.semilogy(histories["post_iter"], histories["post_soft"],
                         "s-", color="#9467bd", markersize=4, linewidth=1.1,
                         alpha=0.9, label="post soft")
        ax_loss.semilogy(histories["post_iter"], histories["post_snap"],
                         "D-", color="#111111", markersize=4, linewidth=1.1,
                         alpha=0.9, label="post snap")
        add_transition_line(ax_loss, histories["post_iter"][0], "POST")
        for x, y, label in zip(histories["post_iter"], histories["post_snap"], histories["post_label"]):
            if label.startswith("lbfgs_"):
                continue
            if np.isfinite(y):
                ax_loss.text(x, y, f" {label}", fontsize=7, color="#111111", rotation=25, va="bottom")
    _setup_loss_axis(ax_loss, args, plot_max_iter, title)
    ax_loss.legend(loc="upper right", fontsize=8)

    ax_aux.plot(histories["iter"], histories["tau"],
                linewidth=1.3, label="\u03c4")
    ax_aux.plot(histories["iter"], histories["H"],
                linewidth=1.3, label="H")
    ax_aux.plot(histories["iter"], histories["B"],
                linewidth=1.3, label="B")
    if hardening_iter is not None:
        add_transition_line(ax_aux, hardening_iter)
    ax_aux.set_xlabel("Iteration")
    ax_aux.set_ylabel("Aux")
    if histories.get("post_iter"):
        add_transition_line(ax_aux, histories["post_iter"][0])
    ax_aux.set_xlim(0, max(10, plot_max_iter))
    ax_aux.grid(True, alpha=0.3)
    ax_aux.legend(loc="upper right", ncol=4, fontsize=8)

    if focus_last_iters and focus_last_iters > 0:
        x_hi = max(histories["iter"]) if histories["iter"] else max(10, n_iters)
        x_lo = max(0, x_hi - int(focus_last_iters))
        ax_loss.set_xlim(x_lo, x_hi)
        ax_aux.set_xlim(x_lo, x_hi)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_seed_grid_plot_single(path, all_data, args, target_desc="", page_label=""):
    n = len(all_data)
    if n == 0:
        return
    ncols = max(1, args.grid_max_cols)
    nrows = max(1, int(np.ceil(max(1, args.grid_max_panels) / ncols)))
    fig = plt.figure(figsize=(6.5 * ncols, 4.8 * nrows), constrained_layout=False)
    outer = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.25)

    for idx, item in enumerate(all_data):
        r, c = idx // ncols, idx % ncols
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.04)
        ax_loss = fig.add_subplot(inner[0, 0])
        ax_aux = fig.add_subplot(inner[1, 0], sharex=ax_loss)
        h = item["histories"]

        ax_loss.semilogy(h["iter"], h["loss"],
                         color="#2ca02c", linewidth=1.5, alpha=0.75,
                         label="RMSE")
        ax_loss.semilogy(h["iter"], h["best"],
                         color="#d62728", linestyle="-.", linewidth=1.2,
                         label="best")
        if h["disc_iter"]:
            ax_loss.semilogy(h["disc_iter"], h["disc"],
                             "o", color="#1f77b4", markersize=2, alpha=0.7,
                             label="discrete")
        ax_loss.semilogy(h["iter"], h["inter"],
                         color="#ff7f0e", linestyle=":", linewidth=1.0,
                         alpha=0.6, label="inter")

        n_iters = item["n_iters"]
        if h.get("post_iter"):
            n_iters = max(n_iters, max(h["post_iter"]) + 1)
        flag = "\u2713" if item.get("success") else "\u2717"
        seed_label = f"seed={item['seed']} ({item['strategy']}) [{idx+1}/{n}] {flag}"
        _setup_loss_axis(ax_loss, args, n_iters, seed_label)
        ax_loss.legend(loc="upper right", fontsize=6)
        ax_loss.tick_params(axis="x", labelbottom=False)

        if item.get("hardening_iter") is not None:
            add_transition_line(ax_loss, item["hardening_iter"])
            add_transition_line(ax_aux, item["hardening_iter"])
        if h.get("post_iter"):
            ax_loss.semilogy(h["post_iter"], h["post_soft"], "s-", color="#9467bd", markersize=2, linewidth=0.8, alpha=0.85)
            ax_loss.semilogy(h["post_iter"], h["post_snap"], "D-", color="#111111", markersize=2, linewidth=0.8, alpha=0.85)
            add_transition_line(ax_loss, h["post_iter"][0])
            add_transition_line(ax_aux, h["post_iter"][0])

        ax_aux.plot(h["iter"], h["tau"], linewidth=1.2, label="\u03c4")
        ax_aux.plot(h["iter"], h["H"], linewidth=1.2, label="H")
        ax_aux.plot(h["iter"], h["B"], linewidth=1.2, label="B")
        ax_aux.set_xlim(0, max(10, n_iters))
        ax_aux.grid(True, alpha=0.25)
        ax_aux.legend(loc="upper right", ncol=4, fontsize=7)

    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        fig.add_subplot(outer[r, c]).axis("off")

    suptitle = "EML Tree Training v15_old \u2014 All Seeds"
    if target_desc:
        suptitle += f"  |  target: {target_desc}"
    if page_label:
        suptitle += f"  |  {page_label}"
    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(path, dpi=args.grid_dpi)
    plt.close(fig)


def save_seed_grid_plots(path, all_data, args, target_desc=""):
    n = len(all_data)
    if n == 0:
        return []
    max_panels = max(1, args.grid_max_panels)
    n_pages = int(np.ceil(n / max_panels))
    base = Path(path)
    out_paths = []
    for page_idx, start in enumerate(range(0, n, max_panels), start=1):
        chunk = all_data[start:start + max_panels]
        if n_pages == 1:
            out_path = base
            page_label = ""
        else:
            out_path = base.with_name(f"{base.stem}_p{page_idx:02d}of{n_pages:02d}{base.suffix}")
            page_label = f"page {page_idx}/{n_pages}"
        _save_seed_grid_plot_single(str(out_path), chunk, args, target_desc=target_desc, page_label=page_label)
        out_paths.append(str(out_path))
    return out_paths


# ---------------------------------------------------------------------------
# Training one seed
# ---------------------------------------------------------------------------

def train_one_seed(seed, args, x_train, y_train, t_train,
                   init_strategy="biased", manual_init_fn=None,
                   seed_idx=0, total_seeds=1):
    """Train one seed through: SEARCH -> HARDENING -> SNAP -> EVALUATE.

    Returns (tree, snapped_tree, histories, summary).
    """
    torch.manual_seed(seed)
    tree = EMLTree(depth=args.depth, init_scale=args.init_scale, init_strategy=init_strategy, eml_clamp=args.eml_clamp)
    if args.warmstart_state:
        src_depth, n_leaf, n_gate = load_warmstart_state(tree, args.warmstart_state)
        print(f"seed={seed} warmstart loaded from={args.warmstart_state} src_depth={src_depth} copied leaf={n_leaf} gate={n_gate}")
        if args.warmstart_noise > 0:
            add_init_noise(tree, args.warmstart_noise, seed=seed)
            print(f"seed={seed} warmstart_noise={args.warmstart_noise:.3f} added")

    # --- Manual init ---------------------------------------------------------
    output_dir = Path(args.save_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    if manual_init_fn is not None:
        manual_init_fn(tree)
        print(f"seed={seed} MANUAL INIT:")
        describe_init(tree)
        validate_manual_init(tree, x_train, y_train, t_train, label=f"seed={seed} EXACT INIT")
        init_path = output_dir / f"{args.save_prefix}_seed{seed}_init_exact.m"
        tree.export_mathematica(str(init_path), discretize=False, comment=f"seed={seed} exact init")
        if args.init_noise > 0:
            add_init_noise(tree, args.init_noise, seed=seed)
            print(f"seed={seed} noise={args.init_noise:.3f} added")
            describe_init(tree, show_numeric=True)
            validate_manual_init(tree, x_train, y_train, t_train, label=f"seed={seed} NOISY INIT")
    else:
        print(f"seed={seed} init_strategy={init_strategy} init_scale={args.init_scale}")
        init_path = output_dir / f"{args.save_prefix}_seed{seed}_init.m"
        tree.export_mathematica(str(init_path), discretize=False, comment=f"seed={seed} {init_strategy} init")

    optimizer = torch.optim.Adam(tree.parameters(), lr=args.lr)

    # Snapshot dir
    snap_dir = None
    if args.snapshot_every > 0:
        snap_dir = output_dir / f"snapshots_seed{seed}"
        snap_dir.mkdir(parents=True, exist_ok=True)

    # --- State ---------------------------------------------------------------
    phase = "search"
    hardening_start = None

    best_search_loss = float("inf")
    best_search_state = None
    plateau_counter = 0

    best_discrete_mse = float("inf")
    best_discrete_state = None
    early_stop_streak = 0
    discrete_plateau_checks = 0
    best_logged_disc = float("inf")
    best_ambiguity = float("inf")

    nan_steps = 0
    nan_streak = 0
    total_restarts = 0
    valid_steps = 0

    # Histories
    hist = {
        "iter": [], "loss": [], "best": [],
        "disc_iter": [], "disc": [],
        "tau": [], "H": [], "B": [], "inter": [],
        "sparse": [], "proj": [], "ambig": [],
        "post_iter": [], "post_soft": [], "post_snap": [], "post_label": [],
    }

    max_iter = args.search_iters + args.hardening_iters

    # --- Live plot -----------------------------------------------------------
    live_fig = None
    if not args.skip_plot:
        plt.ion()
        live_fig, (live_ax_loss, live_ax_aux) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
        live_ln_loss, = live_ax_loss.semilogy([], [], color="#2ca02c", linewidth=1.8, alpha=0.75, label="RMSE")
        live_ln_best, = live_ax_loss.semilogy([], [], color="#d62728", linestyle="-.", linewidth=1.4, label="best")
        live_ln_disc, = live_ax_loss.semilogy([], [], "o", color="#1f77b4", markersize=3, alpha=0.7, label="discrete")
        live_ln_inter, = live_ax_loss.semilogy([], [], color="#ff7f0e", linestyle=":", linewidth=1.2, alpha=0.6, label="inter")
        seed_title = f"EML v15_old | Run {seed_idx+1}/{total_seeds} seed={seed} ({init_strategy})"
        _setup_loss_axis(live_ax_loss, args, max_iter, seed_title)
        live_ax_loss.legend(loc="upper right")

        live_ln_tau, = live_ax_aux.plot([], [], label="\u03c4", linewidth=1.3)
        live_ln_h, = live_ax_aux.plot([], [], label="H", linewidth=1.3)
        live_ln_b, = live_ax_aux.plot([], [], label="B", linewidth=1.3)
        live_ln_ambig, = live_ax_aux.plot([], [], label="U", linewidth=1.1)
        live_ax_aux.set_xlabel("Iteration")
        live_ax_aux.set_ylabel("Aux")
        live_ax_aux.set_xlim(0, max(10, max_iter))
        live_ax_aux.grid(True, alpha=0.3)
        live_ax_aux.legend(loc="upper right", ncol=4, fontsize=8)

    # === MAIN LOOP ===========================================================

    for it in range(1, max_iter + 1):

        # --- Abandon if too many NaN restarts --------------------------------
        if args.max_nan_restarts > 0 and total_restarts >= args.max_nan_restarts:
            print(f"seed={seed} it={it:7d} *** ABANDONED *** ({total_restarts} NaN restarts)")
            break

        # --- Abandon if NaN rate too high (catches sporadic-valid pathology) --
        if it >= args.nan_rate_window and nan_steps > 0:
            nan_rate = nan_steps / it
            if nan_rate > args.nan_rate_abandon:
                print(f"seed={seed} it={it:7d} *** ABANDONED *** NaN rate {nan_rate:.1%} > {args.nan_rate_abandon:.0%} (nan={nan_steps} valid={valid_steps})")
                break

        # --- Phase transition: SEARCH -> HARDENING ---------------------------
        if phase == "search":
            trigger = None
            if (plateau_counter >= args.patience
                    and best_search_loss < args.patience_threshold):
                trigger = f"plateau ({plateau_counter} iters, best={best_search_loss:.3e})"
            elif (discrete_plateau_checks >= args.hardening_trigger_checks
                  and best_logged_disc < args.hardening_trigger_disc):
                trigger = (f"discrete plateau ({discrete_plateau_checks} checks, "
                           f"best_disc={best_logged_disc:.3e})")
            elif (it >= args.hardening_trigger_min_iter
                  and best_ambiguity < args.hardening_trigger_ambiguity):
                trigger = f"ambiguity low ({best_ambiguity:.3e})"
            elif it > args.search_iters:
                trigger = f"max search iters ({args.search_iters})"

            if trigger:
                phase = "hardening"
                hardening_start = it
                # Restore best search state before hardening
                if best_search_state is not None:
                    tree.load_state_dict(best_search_state)
                    optimizer = torch.optim.Adam(tree.parameters(), lr=args.lr)
                print(f"seed={seed} it={it:7d} *** HARDENING *** reason: {trigger}")

        # Check if hardening is done
        if phase == "hardening":
            h_elapsed = it - hardening_start
            if h_elapsed >= args.hardening_iters:
                print(f"seed={seed} it={it:7d} *** HARDENING COMPLETE ***")
                break

        # --- Schedule --------------------------------------------------------
        if phase == "search":
            tau = args.tau_search
            lam_ent = 0.0
            lam_bin = 0.0
            lr_mult = 1.0
        else:
            t = min(1.0, h_elapsed / max(1, args.hardening_iters))
            t_tau = t ** args.hardening_tau_power
            tau = args.tau_search * (args.tau_hard / args.tau_search) ** t_tau
            lam_ent = t * args.lam_ent_hard
            lam_bin = t * args.lam_bin_hard
            lr_mult = max(args.hardening_lr_floor, (1.0 - t) ** 2)

        optimizer.param_groups[0]["lr"] = args.lr * lr_mult

        # --- Forward ---------------------------------------------------------
        optimizer.zero_grad()
        pred, leaf_probs, gate_probs, eml_outputs = tree(x_train, y_train, tau_leaf=tau, tau_gate=tau)

        # NaN check on output
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            nan_steps += 1
            nan_streak += 1
            plateau_counter += 1
            hist["iter"].append(it)
            hist["loss"].append(float("nan"))
            hist["best"].append(
                np.sqrt(best_search_loss) if np.isfinite(best_search_loss)
                else float("nan"))
            hist["tau"].append(tau)
            hist["H"].append(float("nan"))
            hist["B"].append(float("nan"))
            hist["inter"].append(float("nan"))
            hist["sparse"].append(float("nan"))
            hist["proj"].append(float("nan"))
            hist["ambig"].append(float("nan"))

            if nan_streak >= args.nan_restart_patience:
                state = best_search_state
                if state is not None:
                    tree.load_state_dict(state)
                optimizer = torch.optim.Adam(
                    tree.parameters(), lr=args.lr * lr_mult)
                nan_streak = 0
                total_restarts += 1
                if it % args.log_every == 0:
                    print(f"seed={seed} it={it:7d} NaN RESTART #{total_restarts} (nan_steps={nan_steps})")
            continue

        # --- Compute losses --------------------------------------------------
        nan_streak = 0
        lam_sparse = args.lam_sparse_search if phase == "search" else args.lam_sparse_hard
        total, data_loss, entropy, binarity, inter_pen, sparse_pen, ambiguity = compute_losses(
            pred, t_train, leaf_probs, gate_probs, eml_outputs,
            lam_ent, lam_bin, args.lam_inter, args.inter_threshold,
            lam_sparse=lam_sparse, uncertainty_power=args.uncertainty_power,
        )
        proj_pen = torch.tensor(0.0, dtype=REAL_DTYPE)
        if args.proj_consistency_weight > 0:
            with torch.no_grad():
                shadow_pred, _, _, _ = tree(
                    x_train, y_train, tau_leaf=args.proj_tau, tau_gate=args.proj_tau)
            proj_pen = torch.mean((pred - shadow_pred).abs() ** 2).real
            w_proj = args.proj_consistency_weight
            if phase == "hardening":
                w_proj *= args.proj_hardening_boost
            total = total + w_proj * proj_pen

        if torch.isnan(total) or torch.isinf(total):
            nan_steps += 1
            nan_streak += 1
            plateau_counter += 1
            hist["iter"].append(it)
            hist["loss"].append(float("nan"))
            hist["best"].append(
                np.sqrt(best_search_loss) if np.isfinite(best_search_loss)
                else float("nan"))
            hist["tau"].append(tau)
            hist["H"].append(float("nan"))
            hist["B"].append(float("nan"))
            hist["inter"].append(float("nan"))
            hist["sparse"].append(float("nan"))
            hist["proj"].append(float("nan"))
            hist["ambig"].append(float("nan"))
            if nan_streak >= args.nan_restart_patience:
                if best_search_state is not None:
                    tree.load_state_dict(best_search_state)
                optimizer = torch.optim.Adam(
                    tree.parameters(), lr=args.lr * lr_mult)
                nan_streak = 0
                total_restarts += 1
            continue

        # --- Backward + step -------------------------------------------------
        total.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        optimizer.step()
        valid_steps += 1

        data_val = data_loss.item()

        # --- Track best (search phase: for plateau detection) ----------------
        if phase == "search":
            if np.isfinite(data_val) and data_val < best_search_loss:
                rel_imp = ((best_search_loss - data_val)
                           / max(best_search_loss, 1e-15))
                best_search_loss = data_val
                best_search_state = snapshot(tree)
                if rel_imp > args.plateau_rtol:
                    plateau_counter = 0
                else:
                    plateau_counter += 1
            else:
                plateau_counter += 1

        # --- Record history (all as RMSE for plotting) --------------------
        hist["iter"].append(it)
        hist["loss"].append(np.sqrt(data_val) if np.isfinite(data_val)
                            else float("nan"))
        hist["best"].append(np.sqrt(best_search_loss)
                            if np.isfinite(best_search_loss)
                            else float("nan"))
        hist["tau"].append(tau)
        hist["H"].append(entropy.item())
        hist["B"].append(binarity.item())
        hist["inter"].append(
            inter_pen.item() if args.lam_inter > 0 else 0.0)
        hist["sparse"].append(sparse_pen.item())
        hist["proj"].append(proj_pen.item())
        hist["ambig"].append(ambiguity.item())
        if np.isfinite(ambiguity.item()):
            best_ambiguity = min(best_ambiguity, ambiguity.item())

        # --- Periodic logging + discrete eval --------------------------------
        log_interval = max(1, args.log_every)
        if phase == "hardening" and args.tail_log_every > 0 and tau <= args.tail_log_tau:
            log_interval = min(log_interval, max(1, args.tail_log_every))

        should_log = (it % log_interval == 0)
        if should_log:
            with torch.no_grad():
                pred_d, _, _, _ = tree(x_train, y_train, tau_leaf=0.01, tau_gate=0.01)
                disc_mse = torch.mean((pred_d - t_train).abs() ** 2).real.item()

            hist["disc_iter"].append(it)
            hist["disc"].append(np.sqrt(disc_mse) if np.isfinite(disc_mse)
                                else float("nan"))

            if np.isfinite(disc_mse) and disc_mse < best_discrete_mse:
                best_discrete_mse = disc_mse
                best_discrete_state = snapshot(tree)
            if np.isfinite(disc_mse):
                if best_logged_disc == float("inf") or disc_mse < best_logged_disc * (1.0 - args.disc_plateau_rtol):
                    best_logged_disc = disc_mse
                    discrete_plateau_checks = 0
                else:
                    discrete_plateau_checks += 1

            # Early stop: during hardening, discrete already exact
            if phase == "hardening" and disc_mse < args.success_thr:
                early_stop_streak += 1
                if early_stop_streak >= args.early_stop_count:
                    print(f"seed={seed} it={it:7d} *** EARLY STOP *** discrete_rmse={np.sqrt(disc_mse):.3e} (mse={disc_mse:.3e} < {args.success_thr:.0e}) for {early_stop_streak} checks")
                    break
            else:
                early_stop_streak = 0

            rmse = (np.sqrt(data_val) if np.isfinite(data_val)
                    else float("nan"))
            disc_rmse = (np.sqrt(disc_mse) if np.isfinite(disc_mse)
                         else float("nan"))
            phase_tag = f" [{phase.upper()}]" if phase == "hardening" else ""
            print(
                f"seed={seed} it={it:7d} rmse={rmse:.3e} discrete={disc_rmse:.3e} "
                f"tau={tau:.3f} H={entropy.item():.4f} B={binarity.item():.4f} "
                f"S={sparse_pen.item():.4f} U={ambiguity.item():.4f} "
                f"proj={proj_pen.item():.3e} inter={inter_pen.item():.3e} "
                f"nan={nan_steps} restarts={total_restarts}{phase_tag}"
            )

        # --- Periodic snapshot -----------------------------------------------
        if snap_dir is not None and it % args.snapshot_every == 0:
            sp = snap_dir / f"it{it:07d}.m"
            tree.export_snapshot_mma(str(sp), it, mse=data_val)

        # --- Live plot update ------------------------------------------------
        if live_fig is not None and should_log:
            live_ln_loss.set_data(hist["iter"], hist["loss"])
            live_ln_best.set_data(hist["iter"], hist["best"])
            live_ln_disc.set_data(hist["disc_iter"], hist["disc"])
            live_ln_inter.set_data(hist["iter"], hist["inter"])
            live_ln_tau.set_data(hist["iter"], hist["tau"])
            live_ln_h.set_data(hist["iter"], hist["H"])
            live_ln_b.set_data(hist["iter"], hist["B"])
            live_ln_ambig.set_data(hist["iter"], hist["ambig"])
            live_ax_aux.relim()
            live_ax_aux.autoscale_view()
            live_fig.canvas.draw()
            live_fig.canvas.flush_events()

    # === POST-TRAINING =======================================================

    if live_fig is not None:
        plt.ioff()
        plt.close(live_fig)

    post_base_iter = hist["iter"][-1] if hist["iter"] else 0

    def log_post_stage(stage, model_for_soft, snap_probe_model=None):
        soft_mse, soft_max_real, _ = evaluate(model_for_soft, x_train, y_train, t_train, tau=0.01)
        if snap_probe_model is None:
            snap_probe_model = EMLTree(depth=args.depth, eml_clamp=args.eml_clamp)
            snap_probe_model.load_state_dict(snapshot(model_for_soft))
            hard_project_inplace(snap_probe_model)
        snap_mse, snap_max_real, _ = evaluate(snap_probe_model, x_train, y_train, t_train, tau=0.01)
        p_it = post_base_iter + (len(hist["post_iter"]) + 1) * max(1, args.post_plot_stride)
        hist["post_iter"].append(p_it)
        hist["post_soft"].append(np.sqrt(soft_mse) if np.isfinite(soft_mse) else float("nan"))
        hist["post_snap"].append(np.sqrt(snap_mse) if np.isfinite(snap_mse) else float("nan"))
        hist["post_label"].append(stage)
        print(
            f"seed={seed} POST {stage}: soft_rmse={np.sqrt(soft_mse):.3e} "
            f"soft_max={soft_max_real:.3e} snap_probe_rmse={np.sqrt(snap_mse):.3e} "
            f"snap_probe_max={snap_max_real:.3e}"
        )
        return soft_mse, snap_mse

    # Post trace before restoration
    log_post_stage("loop_end", tree)

    # Restore best state (prefer best-by-discrete, then best-by-search)
    restore_source = None
    if best_discrete_state is not None:
        tree.load_state_dict(best_discrete_state)
        restore_source = "best_discrete_state"
    elif best_search_state is not None:
        tree.load_state_dict(best_search_state)
        restore_source = "best_search_state"
    else:
        restore_source = "current_state"
    print(f"seed={seed} POST restore: source={restore_source}")
    log_post_stage("after_restore", tree)

    # L-BFGS polish at low tau
    if args.lbfgs_steps > 0:
        print(f"seed={seed} POST lbfgs: start steps={args.lbfgs_steps} lr={args.lbfgs_lr} trace_every={args.lbfgs_trace_every}")
        lbfgs = torch.optim.LBFGS(tree.parameters(), lr=args.lbfgs_lr, max_iter=1, history_size=50, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            p, lp, gp, eo = tree(x_train, y_train, tau_leaf=args.tau_hard, tau_gate=args.tau_hard)
            t, _, _, _, _, _, _ = compute_losses(
                p, t_train, lp, gp, eo,
                args.lam_ent_hard, args.lam_bin_hard,
                args.lam_inter, args.inter_threshold,
                lam_sparse=args.lam_sparse_hard,
                uncertainty_power=args.uncertainty_power,
            )
            t.backward()
            return t
        lbfgs_ok = True
        for lb_step in range(1, args.lbfgs_steps + 1):
            try:
                lbfgs.step(closure)
            except Exception as e:
                print(f"seed={seed} L-BFGS polish failed at step {lb_step}: {e}")
                lbfgs_ok = False
                break
            trace_every = max(1, args.lbfgs_trace_every)
            should_trace = (lb_step % trace_every == 0) or (lb_step == 1) or (lb_step == args.lbfgs_steps)
            if should_trace:
                log_post_stage(f"lbfgs_{lb_step:03d}", tree)
        if lbfgs_ok:
            print(f"seed={seed} POST lbfgs: complete")
        log_post_stage("after_lbfgs", tree)
    else:
        print(f"seed={seed} POST lbfgs: skipped (lbfgs_steps=0)")

    # --- Snap analysis -------------------------------------------------------
    snap_info = analyze_snap(tree, args.snap_threshold)
    if snap_info["n_uncertain"] > 0:
        print(f"seed={seed} WARNING: {snap_info['n_uncertain']} uncertain weights (snap_thr={args.snap_threshold}):")
        for u in snap_info["uncertain_leaves"]:
            print(f"  {u}")
        for u in snap_info["uncertain_gates"]:
            print(f"  {u}")

    # --- Create snapped tree and evaluate ------------------------------------
    snapped_tree = EMLTree(depth=args.depth, eml_clamp=args.eml_clamp)
    snapped_tree.load_state_dict(snapshot(tree))
    hard_project_inplace(snapped_tree)

    snap_mse, snap_max_real, snap_max_imag = evaluate(
        snapped_tree, x_train, y_train, t_train, tau=0.01)
    p_it = post_base_iter + (len(hist["post_iter"]) + 1) * max(1, args.post_plot_stride)
    hist["post_iter"].append(p_it)
    hist["post_soft"].append(float("nan"))
    hist["post_snap"].append(np.sqrt(snap_mse) if np.isfinite(snap_mse) else float("nan"))
    hist["post_label"].append("hard_project_eval")
    print(
        f"seed={seed} POST hard_project_eval: snap_rmse={np.sqrt(snap_mse):.3e} "
        f"snap_max={snap_max_real:.3e}"
    )

    fit_success = snap_mse < args.fit_success_thr
    symbol_success = snap_mse < args.success_thr
    stable_symbol_success = (symbol_success
                             and snap_info["n_uncertain"] <= args.max_uncertain_success)
    success = fit_success

    flag = "SUCCESS" if success else "FAIL"
    if success and not symbol_success:
        flag = "FIT-SUCCESS"
    extra = ""
    if snap_info["n_uncertain"] > 0:
        extra = f" (note: {snap_info['n_uncertain']} weights not fully converged)"
    print(
        f"seed={seed} *** {flag} *** snapped_rmse={np.sqrt(snap_mse):.3e} "
        f"max_err={snap_max_real:.3e} fit_success={fit_success} "
        f"symbol_success={symbol_success} "
        f"stable_symbol_success={stable_symbol_success}{extra}"
    )

    summary = {
        "phase_reached": phase,
        "valid_steps": valid_steps,
        "nan_steps": nan_steps,
        "total_restarts": total_restarts,
        "best_search_loss": (best_search_loss if np.isfinite(best_search_loss) else None),
        "best_discrete_mse": (best_discrete_mse if np.isfinite(best_discrete_mse) else None),
        "best_ambiguity": (best_ambiguity if np.isfinite(best_ambiguity) else None),
        "snap_mse": snap_mse,
        "snap_max_real": snap_max_real,
        "snap_max_imag": snap_max_imag,
        "success": success,
        "fit_success": fit_success,
        "symbol_success": symbol_success,
        "stable_symbol_success": stable_symbol_success,
        "n_uncertain": snap_info["n_uncertain"],
        "hardening_triggered": hardening_start is not None,
        "hardening_iter": hardening_start,
        "post_stages": [
            {"iter": i, "label": lbl, "soft_rmse": s, "snap_rmse": d}
            for i, lbl, s, d in zip(hist["post_iter"], hist["post_label"], hist["post_soft"], hist["post_snap"])
        ],
    }

    return tree, snapped_tree, hist, hardening_start, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

INIT_STRATEGIES_ALL = ["biased", "uniform", "xy_biased", "random_hot"]


def parse_args():
    p = argparse.ArgumentParser(description="EML tree trainer (v15_old)")
    a = p.add_argument
    # Target & structure
    a("--target-fn",            type=str,   default="eml_depth3")
    a("--depth",                type=int,   default=3,            help="Tree depth (0 = auto from expr)")
    a("--init-scale",           type=float, default=1.0)
    a("--init-strategy",        type=str,   default="all",        help="biased/uniform/xy_biased/random_hot/manual/all")
    # Manual init
    a("--init-expr",            type=str,   default="")
    a("--init-blend",           type=str,   default="")
    a("--init-leaves",          type=str,   default="")
    a("--init-noise",           type=float, default=0.0)
    a("--init-k",               type=float, default=32.0)
    a("--warmstart-state",      type=str,   default="",           help="Optional .pt/.pth state to lift into current depth")
    a("--warmstart-noise",      type=float, default=0.0,          help="Noise added after warmstart load")
    # Seed sweep
    a("--seeds",                type=int,   default=4)
    a("--seed0",                type=int,   default=7777)
    # Training
    a("--search-iters",         type=int,   default=30000,        help="Max iterations for search phase")
    a("--hardening-iters",      type=int,   default=10000,        help="Iterations for hardening phase")
    a("--lr",                   type=float, default=0.01)
    a("--log-every",            type=int,   default=200)
    a("--tail-log-every",       type=int,   default=25,           help="More frequent logging in late hardening (set 0 to disable)")
    a("--tail-log-tau",         type=float, default=0.2,          help="Enable tail logging when tau <= this value")
    # Data
    a("--data-lo",              type=float, default=1.0)
    a("--data-hi",              type=float, default=3.0)
    a("--data-step",            type=float, default=0.1)
    a("--gen-lo",               type=float, default=0.5,          help="Generalization data lower bound")
    a("--gen-hi",               type=float, default=5.0,          help="Generalization data upper bound")
    # Intermediate penalty
    a("--lam-inter",            type=float, default=1e-4)
    a("--inter-threshold",      type=float, default=50.0)
    a("--eml-clamp",            type=float, default=1e300,        help="Clamp EML outputs (NaN prevention ONLY; default=1e300=off)")
    # Search phase
    a("--tau-search",           type=float, default=2.5)
    a("--patience",             type=int,   default=10000,        help="Iters without improvement -> hardening")
    a("--patience-threshold",   type=float, default=1e-2,         help="Only trigger hardening if best < this")
    a("--plateau-rtol",         type=float, default=1e-3)
    # Hardening phase
    a("--tau-hard",             type=float, default=0.01,         help="Final tau (near-discrete)")
    a("--hardening-tau-power",  type=float, default=1.5,          help="Tau schedule power: t^power (>1 = gentler early hardening)")
    a("--hardening-lr-floor",   type=float, default=0.01,         help="Min LR multiplier during hardening (quadratic decay)")
    a("--lam-ent-hard",         type=float, default=2e-2)
    a("--lam-bin-hard",         type=float, default=2e-2)
    a("--lam-sparse-search",    type=float, default=0.0,          help="Structure sparsity in search phase")
    a("--lam-sparse-hard",      type=float, default=5e-4,         help="Structure sparsity in hardening phase")
    a("--uncertainty-power",    type=float, default=2.0,          help="Power for uncertainty-weighted penalties")
    a("--proj-consistency-weight", type=float, default=0.0,       help="Weight for soft-vs-shadow projection consistency")
    a("--proj-hardening-boost", type=float, default=1.5,          help="Multiplier for proj consistency during hardening")
    a("--proj-tau",             type=float, default=0.02,         help="Tau for near-discrete shadow predictions")
    a("--hardening-trigger-checks", type=int, default=6,          help="Discrete plateau checks before hardening trigger")
    a("--hardening-trigger-disc", type=float, default=1.0,        help="Enable discrete-plateau trigger when best discrete mse below this")
    a("--hardening-trigger-min-iter", type=int, default=4000,     help="Minimum iteration before ambiguity trigger is active")
    a("--hardening-trigger-ambiguity", type=float, default=0.35,  help="Ambiguity threshold trigger for hardening")
    a("--disc-plateau-rtol",    type=float, default=1e-3,         help="Relative improvement required to reset discrete plateau counter")
    # Success / snapping
    a("--fit-success-thr",      type=float, default=1e-6,         help="Fit success threshold (for reporting only)")
    a("--success-thr",          type=float, default=1e-20,        help="Snapped MSE below this = success")
    a("--max-uncertain-success", type=int, default=0,             help="Require <= this many uncertain weights for stable_symbol_success")
    a("--snap-threshold",       type=float, default=0.01,         help="Don't snap weights with prob in (thr, 1-thr)")
    a("--early-stop-count",     type=int,   default=10,           help="Discrete checks below success_thr -> early stop")
    # NaN handling
    a("--nan-restart-patience", type=int,   default=50)
    a("--max-nan-restarts",     type=int,   default=100)
    a("--nan-rate-abandon",     type=float, default=0.95,         help="Abandon seed if nan_steps/total_steps > this after warmup")
    a("--nan-rate-window",      type=int,   default=1000,         help="Min iterations before nan-rate-abandon check activates")
    # Polish
    a("--lbfgs-steps",          type=int,   default=80)
    a("--lbfgs-lr",             type=float, default=0.6)
    a("--lbfgs-trace-every",    type=int,   default=1,           help="Trace every N LBFGS steps in log/plot")
    # Output
    a("--generalization-points", type=int,  default=4000)
    a("--save-prefix",          type=str,   default="eml_v15_old")
    a("--skip-plot",            action="store_true")
    a("--export-m",             type=str,   default="eml_tree_v15_old.m")
    a("--snapshot-every",       type=int,   default=0)
    a("--loss-y-min",           type=float, default=1e-16)
    a("--loss-y-max",           type=float, default=1e5)
    a("--tail-plot-window",     type=int,   default=4000,         help="Tail loss zoom window (iterations, 0 disables)")
    a("--post-plot-stride",     type=int,   default=25,           help="Horizontal spacing for post/LBFGS trace points on plots")
    a("--grid-max-panels",      type=int,   default=16,           help="Maximum plots per collage page")
    a("--grid-max-cols",        type=int,   default=4,            help="Maximum subplot columns on collage pages")
    a("--grid-dpi",             type=int,   default=150,          help="DPI for collage pages")

    args = p.parse_args()
    args.warmstart_state = args.warmstart_state.strip()
    if args.warmstart_state and not Path(args.warmstart_state).exists():
        raise FileNotFoundError(f"--warmstart-state not found: {args.warmstart_state}")

    # Auto-detect depth from expression
    if args.init_expr and args.depth == 0:
        parsed = parse_eml_expr(args.init_expr)
        args.depth = expr_depth(parsed)
        print(f"Auto-detected depth={args.depth} from expression")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    target_fn, target_desc = get_target_fn(args.target_fn)

    # Output directory
    output_dir = Path(args.save_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_start = datetime.now()
    stamp = run_start.strftime("%Y%m%d-%H%M%S")

    # Set up logging
    stdout_log_path = output_dir / f"{args.save_prefix}_stdout_{stamp}.log"
    log_handle = open(stdout_log_path, "w", encoding="utf-8", buffering=1)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = TeeStream(orig_stdout, log_handle)
    sys.stderr = TeeStream(orig_stderr, log_handle)

    try:
        # Log command line for reproducibility
        print(f"Command: {' '.join(sys.argv)}")
        print(f"Run start: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {output_dir.resolve()}")
        print(f"Logging to: {stdout_log_path}")
        png_dir = output_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=True)

        # Data
        x_train, y_train, t_train = make_grid_data(target_fn, args.data_lo, args.data_hi, args.data_step)
        x_gen, y_gen, t_gen = make_generalization_data(target_fn, lo=args.gen_lo, hi=args.gen_hi, n=args.generalization_points)

        # Determine init strategies
        has_expr = bool(args.init_expr)
        has_blend = bool(args.init_blend) and bool(args.init_leaves)
        manual_mode = "expr" if has_expr else ("blend" if has_blend else None)

        if manual_mode:
            strategy_set = ["manual"]
        elif args.init_strategy == "all":
            strategy_set = INIT_STRATEGIES_ALL
        elif args.init_strategy in INIT_STRATEGIES_ALL:
            strategy_set = [args.init_strategy]
        else:
            strategy_set = INIT_STRATEGIES_ALL

        if manual_mode and args.init_noise == 0.0 and args.warmstart_noise == 0.0:
            # Manual deterministic init: seed does not change the initialized model.
            # Run once by default to avoid redundant repeated runs.
            seed_values = [args.seed0]
        else:
            seed_values = list(range(args.seed0, args.seed0 + args.seeds))
        run_plan = [(seed, strategy) for seed in seed_values for strategy in strategy_set]
        n_runs = len(run_plan)

        # Manual init function
        manual_init_fn = None
        if manual_mode == "expr":
            def manual_init_fn(tree):
                init_from_expr(tree, args.init_expr, k=args.init_k)
        elif manual_mode == "blend":
            def manual_init_fn(tree):
                init_from_blend_leaves(
                    tree, args.init_blend, args.init_leaves, k=args.init_k)

        # Print config
        print(f"\n=== EML Tree Training v15_old ===")
        print(f"target: {args.target_fn}  =  {target_desc}")
        print(f"depth={args.depth} leaves={2**args.depth} internal={2**args.depth - 1}")
        print(f"params per run={2**args.depth * 3 + (2**args.depth - 1) * 2}")
        print(f"train samples={len(x_train)} seeds={len(seed_values)} (requested={args.seeds}) strategies={len(strategy_set)} runs={n_runs}")
        print(f"search_iters={args.search_iters} hardening_iters={args.hardening_iters} (max total={args.search_iters + args.hardening_iters})")
        print(f"tau_search={args.tau_search} tau_hard={args.tau_hard}")
        print(f"patience={args.patience} patience_threshold={args.patience_threshold} plateau_rtol={args.plateau_rtol}")
        print(f"fit_success_thr={args.fit_success_thr:.0e} success_thr={args.success_thr:.0e} snap_threshold={args.snap_threshold}")
        print(f"eml_clamp={args.eml_clamp:.0e}")
        print(f"lam_inter={args.lam_inter} inter_threshold={args.inter_threshold}")
        print(f"lam_sparse_search={args.lam_sparse_search} lam_sparse_hard={args.lam_sparse_hard} uncertainty_power={args.uncertainty_power}")
        print(f"proj_consistency_weight={args.proj_consistency_weight} proj_hardening_boost={args.proj_hardening_boost} proj_tau={args.proj_tau}")
        print(f"nan_restart_patience={args.nan_restart_patience} max_nan_restarts={args.max_nan_restarts}")
        print(f"nan_rate_abandon={args.nan_rate_abandon} nan_rate_window={args.nan_rate_window}")
        print(f"hardening_tau_power={args.hardening_tau_power} hardening_lr_floor={args.hardening_lr_floor}")
        print(f"hardening_trigger_checks={args.hardening_trigger_checks} hardening_trigger_disc={args.hardening_trigger_disc}")
        print(f"hardening_trigger_min_iter={args.hardening_trigger_min_iter} hardening_trigger_ambiguity={args.hardening_trigger_ambiguity}")
        print(f"gen range=[{args.gen_lo}, {args.gen_hi}]")
        print(f"seed range=[{args.seed0}, {args.seed0 + args.seeds - 1}]")
        if manual_mode and len(seed_values) == 1 and args.seeds > 1:
            print("manual init deterministic mode: using single seed run (set init/warmstart noise > 0 to sweep seeds)")
        if args.warmstart_state:
            print(f"warmstart_state={args.warmstart_state} warmstart_noise={args.warmstart_noise}")
        print(f"init strategies: {strategy_set}")
        if manual_mode:
            print(f"manual init: {manual_mode}")
            if has_expr:
                print(f"  expression: {args.init_expr}")
            if has_blend:
                print(f"  blend: {args.init_blend}")
                print(f"  leaves: {args.init_leaves}")
            print(f"  noise: {args.init_noise}  k: {args.init_k}")

        best_overall = {
            "seed": None, "strategy": None,
            "score": float("inf"), "state": None,
            "success": False,
        }
        all_seed_data = []
        n_successes = 0
        n_fit_successes = 0
        n_symbol_successes = 0
        n_stable_symbol_successes = 0

        for seed_idx, (seed, strategy) in enumerate(run_plan):
            cur_manual = manual_init_fn if strategy == "manual" else None

            print(f"\n--- Run {seed_idx+1}/{n_runs}: seed={seed} strategy={strategy} ---")

            tree, snapped_tree, hist, hardening_iter, summary = train_one_seed(seed, args, x_train, y_train, t_train, init_strategy=strategy, manual_init_fn=cur_manual, seed_idx=seed_idx, total_seeds=n_runs)

            # Evaluate on generalization set
            gen_mse, gen_max_real, gen_max_imag = evaluate(snapped_tree, x_gen, y_gen, t_gen, tau=0.01)
            print(f"seed={seed} gen: rmse={np.sqrt(gen_mse):.6e} max_real={gen_max_real:.3e} max_imag={gen_max_imag:.3e}")

            if summary["success"]:
                n_successes += 1
            if summary["fit_success"]:
                n_fit_successes += 1
            if summary["symbol_success"]:
                n_symbol_successes += 1
            if summary["stable_symbol_success"]:
                n_stable_symbol_successes += 1

            # Save per-seed outputs
            seed_stem = f"{args.save_prefix}_run{seed_idx+1:02d}_seed{seed}_{strategy}_{stamp}"
            seed_base = output_dir / seed_stem
            torch.save(tree.state_dict(), f"{seed_base}.pt")
            snapped_tree.export_mathematica(f"{seed_base}.m", discretize=True, snap_threshold=args.snap_threshold)
            tree.export_mathematica(f"{seed_base}_continuous.m", discretize=False)
            print(f"seed={seed} saved: {seed_base}.m")

            loss_png = png_dir / f"{seed_stem}_loss.png"
            save_loss_plot(str(loss_png), hist, title=f"EML v15_old | Run {seed_idx+1}/{n_runs} seed={seed} ({strategy})", n_iters=args.search_iters + args.hardening_iters, hardening_iter=hardening_iter, args=args)
            if args.tail_plot_window > 0:
                tail_png = png_dir / f"{seed_stem}_loss_tail.png"
                save_loss_plot(
                    str(tail_png),
                    hist,
                    title=f"EML v15_old | Run {seed_idx+1}/{n_runs} seed={seed} ({strategy}) | tail",
                    n_iters=args.search_iters + args.hardening_iters,
                    hardening_iter=hardening_iter,
                    args=args,
                    focus_last_iters=args.tail_plot_window,
                )

            # Track best overall (by snapped MSE)
            snap_mse = summary["snap_mse"]
            if np.isfinite(snap_mse) and snap_mse < best_overall["score"]:
                best_overall["seed"] = seed
                best_overall["strategy"] = strategy
                best_overall["score"] = snap_mse
                best_overall["state"] = snapshot(snapped_tree)
                best_overall["success"] = summary["success"]

            all_seed_data.append({
                "run_index": seed_idx + 1,
                "seed": seed,
                "strategy": strategy,
                "n_iters": args.search_iters + args.hardening_iters,
                "histories": hist,
                "hardening_iter": hardening_iter,
                "success": summary["success"],
                "summary": summary,
                "gen_mse": gen_mse,
            })

        # ==== Best overall ====================================================
        print(f"\n{'='*60}")
        print(
            f"=== Results: success={n_successes}/{n_runs} "
            f"fit_success={n_fit_successes}/{n_runs} "
            f"symbol_success={n_symbol_successes}/{n_runs} "
            f"stable_symbol_success={n_stable_symbol_successes}/{n_runs} ==="
        )
        print(f"Best seed={best_overall['seed']} strategy={best_overall['strategy']} snapped_rmse={np.sqrt(best_overall['score']):.6e}")

        # Export best model
        if best_overall["state"] is not None:
            best_tree = EMLTree(depth=args.depth, eml_clamp=args.eml_clamp)
            best_tree.load_state_dict(best_overall["state"])

            export_path = output_dir / args.export_m
            best_tree.export_mathematica(str(export_path), discretize=True, snap_threshold=args.snap_threshold)
            print(f"Exported best model: {export_path}")

            # Final evaluation
            train_mse, train_mr, train_mi = evaluate(best_tree, x_train, y_train, t_train, tau=0.01)
            gen_mse, gen_mr, gen_mi = evaluate(best_tree, x_gen, y_gen, t_gen, tau=0.01)
            print(f"train: rmse={np.sqrt(train_mse):.6e} max_err={train_mr:.3e} max_imag={train_mi:.3e}")
            print(f"gen  : rmse={np.sqrt(gen_mse):.6e} max_err={gen_mr:.3e} max_imag={gen_mi:.3e}")
        else:
            train_mse, train_mr, train_mi = float("nan"), 0, 0
            gen_mse, gen_mr, gen_mi = float("nan"), 0, 0

        # Grid plot
        grid_path = png_dir / f"{args.save_prefix}_all_seeds_grid_{stamp}.png"
        grid_paths = save_seed_grid_plots(str(grid_path), all_seed_data, args, target_desc=target_desc)
        if len(grid_paths) == 1:
            print(f"Saved grid plot: {grid_paths[0]}")
        else:
            print("Saved grid plots:")
            for gp in grid_paths:
                print(f"  - {gp}")

        # Metrics JSON
        metrics = {
            "target_fn": args.target_fn,
            "target_desc": target_desc,
            "n_successes": n_successes,
            "n_fit_successes": n_fit_successes,
            "n_symbol_successes": n_symbol_successes,
            "n_stable_symbol_successes": n_stable_symbol_successes,
            "n_seeds": len(seed_values),
            "n_requested_seeds": args.seeds,
            "n_strategies": len(strategy_set),
            "n_runs": n_runs,
            "best_seed": best_overall["seed"],
            "best_strategy": best_overall["strategy"],
            "best_snapped_mse": best_overall["score"],
            "best_success": best_overall["success"],
            "train": {"mse": train_mse, "max_real_err": train_mr,
                      "max_imag": train_mi},
            "gen": {"mse": gen_mse, "max_real_err": gen_mr,
                    "max_imag": gen_mi},
            "per_seed": [
                {
                    "seed": d["seed"],
                    "run_index": d["run_index"],
                    "strategy": d["strategy"],
                    "success": d["summary"]["success"],
                    "fit_success": d["summary"]["fit_success"],
                    "symbol_success": d["summary"]["symbol_success"],
                    "stable_symbol_success": d["summary"]["stable_symbol_success"],
                    "snap_mse": d["summary"]["snap_mse"],
                    "gen_mse": d["gen_mse"],
                    "n_uncertain": d["summary"]["n_uncertain"],
                    "hardening_triggered": d["summary"]["hardening_triggered"],
                }
                for d in all_seed_data
            ],
            "args": vars(args),
            "command": " ".join(sys.argv),
            "stdout_log_file": str(stdout_log_path),
            "output_dir": str(output_dir.resolve()),
            "png_dir": str(png_dir.resolve()),
            "grid_plot_files": grid_paths,
        }
        metrics_path = output_dir / f"{args.save_prefix}_metrics_{stamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(_sanitize_for_json(metrics), f, indent=2, default=str, allow_nan=False)
        print(f"Saved metrics: {metrics_path}")

    finally:
        run_end = datetime.now()
        print(f"\nRun end:   {run_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed:   {run_end - run_start}")
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_handle.close()


if __name__ == "__main__":
    main()

