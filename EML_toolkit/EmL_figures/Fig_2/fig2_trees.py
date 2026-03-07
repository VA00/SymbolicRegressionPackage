from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from scipy.optimize import LinearConstraint, minimize


@dataclass
class Node:
    label: str
    left: "Node | None" = None
    right: "Node | None" = None
    depth: int = 0
    x: float = 0.0
    y: float = 0.0
    name: str = field(default="", repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class Parser:
    def __init__(self, text: str):
        self.text = "".join(text.split())
        self.pos = 0

    def parse(self) -> Node:
        node = self._parse_node()
        if self.pos != len(self.text):
            raise ValueError(f"Unexpected trailing text at position {self.pos}: {self.text[self.pos:]}")
        return node

    def _peek(self) -> str:
        return self.text[self.pos : self.pos + 1]

    def _consume(self, token: str) -> None:
        if not self.text.startswith(token, self.pos):
            got = self.text[self.pos : self.pos + len(token)]
            raise ValueError(f"Expected '{token}' at position {self.pos}, got '{got}'")
        self.pos += len(token)

    def _parse_atom(self) -> str:
        for atom in ("EML", "1", "x", "y"):
            if self.text.startswith(atom, self.pos):
                self.pos += len(atom)
                return atom
        raise ValueError(f"Unexpected token at position {self.pos}: {self.text[self.pos:self.pos+12]}")

    def _parse_node(self) -> Node:
        atom = self._parse_atom()
        if atom != "EML":
            return Node(atom)
        self._consume("[")
        left = self._parse_node()
        self._consume(",")
        right = self._parse_node()
        self._consume("]")
        return Node("EML", left=left, right=right)


def parse_eml(text: str) -> Node:
    return Parser(text).parse()


def max_depth(node: Node) -> int:
    if node.is_leaf:
        return 0
    return 1 + max(max_depth(node.left), max_depth(node.right))


def assign_layout(node: Node, global_depth: int, dx: float, dy: float) -> tuple[float, float, float]:
    leaf_index = 0

    def walk(cur: Node, depth: int) -> None:
        nonlocal leaf_index
        cur.depth = depth
        if cur.is_leaf:
            cur.y = leaf_index * dy
            leaf_index += 1
        else:
            walk(cur.left, depth + 1)
            walk(cur.right, depth + 1)
            cur.y = (cur.left.y + cur.right.y) / 2.0
        cur.x = (global_depth - depth) * dx

    walk(node, 0)
    baseline = node.y
    for cur in collect_nodes(node):
        cur.y -= baseline
    min_y = min(cur.y for cur in collect_nodes(node))
    max_y = max(cur.y for cur in collect_nodes(node))
    return global_depth * dx, min_y, max_y


def collect_nodes(node: Node) -> list[Node]:
    nodes = [node]
    if node.left is not None:
        nodes.extend(collect_nodes(node.left))
    if node.right is not None:
        nodes.extend(collect_nodes(node.right))
    return nodes


def node_halfheight(node: Node) -> float:
    return 0.12 if node.is_leaf else 0.17


def optimize_layout(root: Node, rms_cap: float = 0.10) -> None:
    nodes = collect_nodes(root)
    index = {id(node): idx for idx, node in enumerate(nodes)}
    root_idx = index[id(root)]
    y0 = np.array([node.y for node in nodes], dtype=float)
    panel_height = max(float(np.max(y0) - np.min(y0)), 1e-9)

    variable_indices = [idx for idx in range(len(nodes)) if idx != root_idx]
    inverse = {node_idx: var_idx for var_idx, node_idx in enumerate(variable_indices)}

    groups: dict[int, list[int]] = {}
    for idx, node in enumerate(nodes):
        groups.setdefault(node.depth, []).append(idx)
    for depth in groups:
        groups[depth].sort(key=lambda idx: y0[idx])

    edges: list[tuple[int, int]] = []
    centers: list[tuple[int, int, int]] = []
    for idx, node in enumerate(nodes):
        if node.left is not None:
            edges.append((idx, index[id(node.left)]))
        if node.right is not None:
            edges.append((idx, index[id(node.right)]))
        if node.left is not None and node.right is not None:
            centers.append((idx, index[id(node.left)], index[id(node.right)]))

    def unpack(z: np.ndarray) -> np.ndarray:
        y = y0.copy()
        y[root_idx] = y0[root_idx]
        for node_idx, var_idx in inverse.items():
            y[node_idx] = z[var_idx]
        return y

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for depth, ordered in groups.items():
        for prev_idx, cur_idx in zip(ordered, ordered[1:]):
            row = np.zeros(len(variable_indices))
            if prev_idx != root_idx:
                row[inverse[prev_idx]] = -1.0
            if cur_idx != root_idx:
                row[inverse[cur_idx]] = 1.0
            gap = node_halfheight(nodes[prev_idx]) + node_halfheight(nodes[cur_idx]) - 0.06
            rows.append(row)
            lower.append(gap)
            upper.append(np.inf)
    linear_constraint = LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))

    def loss(z: np.ndarray) -> float:
        y = unpack(z)

        edge_deltas = np.array([abs(y[parent] - y[child]) for parent, child in edges], dtype=float)
        target_delta = float(np.mean(edge_deltas))
        angle_loss = float(np.mean((edge_deltas - target_delta) ** 2))

        spans: list[float] = []
        spacing_residuals: list[float] = []
        for ordered in groups.values():
            if len(ordered) <= 1:
                continue
            values = y[ordered]
            gaps = np.diff(values)
            spans.append(float(np.sum(gaps)))
        common_span = float(np.mean(spans)) if spans else 0.0
        for ordered in groups.values():
            if len(ordered) <= 1:
                continue
            values = y[ordered]
            gaps = np.diff(values)
            target_gap = common_span / (len(ordered) - 1)
            spacing_residuals.extend((gaps - target_gap).tolist())
        spacing_loss = float(np.mean(np.square(spacing_residuals))) if spacing_residuals else 0.0

        center_residuals = [
            2.0 * y[parent] - y[left] - y[right]
            for parent, left, right in centers
        ]
        center_loss = float(np.mean(np.square(center_residuals))) if center_residuals else 0.0

        drift = (y - y0) / panel_height
        drift_loss = float(np.mean(drift * drift))
        rms = float(np.sqrt(drift_loss))
        rms_penalty = 0.0 if rms <= rms_cap else ((rms - rms_cap) / rms_cap) ** 2

        return (
            2.6 * angle_loss
            + 2.4 * spacing_loss
            + 5.5 * center_loss
            + 0.15 * drift_loss
            + 90.0 * rms_penalty
        )

    z0 = y0[variable_indices]
    result = minimize(
        loss,
        z0,
        method="SLSQP",
        constraints=[linear_constraint],
        options={"maxiter": 600, "ftol": 1e-9},
    )

    y = unpack(result.x if result.success else z0)
    y -= y[root_idx]
    for idx, node in enumerate(nodes):
        node.y = float(y[idx])


def math_label(label: str) -> str:
    if label == "1":
        return "$1$"
    return f"${label}$"


def render_panel(
    panel: dict[str, str],
    row_idx: int,
    global_depth: int,
    dx: float,
    dy: float,
    expr_gap: float,
) -> tuple[str, float, float]:
    root = parse_eml(panel["expr"])
    width, min_y, max_y = assign_layout(root, global_depth, dx, dy)
    optimize_layout(root, rms_cap=0.10)
    min_y = min(node.y for node in collect_nodes(root))
    max_y = max(node.y for node in collect_nodes(root))
    for idx, node in enumerate(collect_nodes(root)):
        node.name = f"row{row_idx}_n{idx}"
    lines: list[str] = []

    for node in collect_nodes(root):
        y = -node.y
        if node.is_leaf:
            style = "termnode"
            label = math_label(node.label)
        else:
            style = "emlnode"
            label = r"$\eml$"
        lines.append(
            f"\\node[{style}] ({node.name}) at ({node.x:.3f}cm,{y:.3f}cm) {{{label}}};"
        )

    for node in collect_nodes(root):
        if node.left is not None:
            lines.append(
                f"\\draw[branch] ({node.left.name}.east) -- ([yshift=1.15mm]{node.name}.west);"
            )
        if node.right is not None:
            lines.append(
                f"\\draw[branch] ({node.right.name}.east) -- ([yshift=-1.15mm]{node.name}.west);"
            )

    lines.append(
        f"\\node[exprlabel, anchor=west] at ({width + expr_gap:.3f}cm,0cm) {{$\\mkern-4mu\\raisebox{{-0.18ex}}{{\\ensuremath{{\\to}}}}\\;{panel['title_expr']}$}};"
    )

    return "\n".join(lines), -min_y, max_y


def build_document() -> str:
    panels = [
        {
            "title_expr": r"\ln x",
            "expr": "EML[1,EML[EML[1,x],1]]",
        },
        {
            "title_expr": r"x",
            "expr": "EML[1,EML[EML[1,EML[x,1]],1]]",
        },
        {
            "title_expr": r"-x",
            "expr": "EML[EML[1,EML[1,EML[1,EML[EML[1,1],1]]]],EML[x,1]]",
        },
        {
            "title_expr": r"x^{-1}",
            "expr": "EML[EML[EML[1,EML[1,EML[1,EML[EML[1,1],1]]]],x],1]",
        },
        {
            "title_expr": r"xy",
            "expr": (
                "EML[EML[1,EML[EML[EML[1,EML[EML[1,EML[1,x]],1]],"
                "EML[1,EML[EML[1,EML[y,1]],1]]],1]],1]"
            ),
        },
    ]

    global_depth = max(max_depth(parse_eml(panel["expr"])) for panel in panels)
    dx = 1.40
    dy = 0.32
    row_gap = 0.82
    expr_gap = 0.40

    rendered = [
        render_panel(panel, idx, global_depth, dx, dy, expr_gap)
        for idx, panel in enumerate(panels)
    ]

    scopes = []
    y_cursor = 0.0
    for idx, (body, top_extent, bottom_extent) in enumerate(rendered):
        if idx == 0:
            y_shift = -top_extent
        else:
            prev_bottom = rendered[idx - 1][2]
            y_shift = y_cursor - prev_bottom - row_gap - top_extent
        y_cursor = y_shift
        scopes.append(
            "\n".join(
                [
                    f"\\begin{{scope}}[shift={{(0cm,{y_shift:.3f}cm)}}]",
                    body,
                    "\\end{scope}",
                ]
            )
        )

    return "\n".join(
        [
            r"\documentclass[tikz,border=6pt]{standalone}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{newtxtext,newtxmath}",
            r"\DeclareMathOperator{\eml}{eml}",
            r"\definecolor{emlStroke}{RGB}{56,63,71}",
            r"\definecolor{emlFill}{RGB}{247,243,235}",
            r"\tikzset{",
            r"  emlnode/.style={",
            r"    draw=emlStroke,",
            r"    fill=emlFill,",
            r"    rounded corners=1.8pt,",
            r"    line width=0.5pt,",
            r"    minimum width=7.4mm,",
            r"    minimum height=4.0mm,",
            r"    inner sep=0.20pt,",
            r"    font=\fontsize{10.4}{11.2}\selectfont",
            r"  },",
            r"  termnode/.style={",
            r"    text=black!78,",
            r"    font=\fontsize{14.0}{14.8}\selectfont,",
            r"    inner sep=0.3pt,",
            r"    minimum width=3.2mm,",
            r"    minimum height=3.2mm",
            r"  },",
            r"  exprlabel/.style={text=black, font=\fontsize{15.0}{16.2}\selectfont},",
            r"  branch/.style={draw=black!68, line width=0.42pt, line cap=round, line join=round}",
            r"}",
            r"\begin{document}",
            r"\begin{tikzpicture}",
            r"\path[use as bounding box] (-0.30cm,1.80cm) rectangle (13.60cm,-11.80cm);",
            *scopes,
            r"\end{tikzpicture}",
            r"\end{document}",
        ]
    )


def main() -> None:
    default_output = Path(__file__).resolve().with_name("Fig2_trees.tex")
    parser = argparse.ArgumentParser(description="Generate a camera-ready Fig. 2 EML tree figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to the generated standalone LaTeX file.",
    )
    args = parser.parse_args()

    args.output.write_text(build_document(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
