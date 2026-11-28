#!/usr/bin/env python3
"""
Figure 1: Spiral dependency graph of elementary functions from EML.

Self-contained script. Reads rust_verify.log, generates:
  - graph_spiral.png   (raster, 250 dpi, matplotlib/Bezier edges)
  - graph_spiral.pdf   (vector, matplotlib/Bezier edges)
  - graph_spiral.tex   (standalone TikZ, circular arc edges)

Requirements: Python 3.10+, numpy, matplotlib, networkx

Usage:
    python gen_spiral_graph.py                          # default paths
    python gen_spiral_graph.py rust_verify.log output/  # custom paths

TikZ compilation:
    pdflatex graph_spiral.tex
"""
import re
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx


# ======================== LOG PARSER ========================
# Test constants used as placeholders for x, y — NOT real dependencies
TEST_CONSTANTS = {"EulerGamma", "Glaisher"}

# Display names for nodes (Unicode math notation, used in matplotlib output)
DISPLAY_NAMES = {
    'EML': 'EML',
    '1': '1',
    'E': 'e',
    'Exp': 'exp',
    'Log': 'ln',
    'Subtract': '\u2212',       # −
    '-1': '\u22121',            # −1
    '2': '2',
    'Minus': '\u2212x',         # −x
    'Plus': '+',
    'Inv': '1/x',
    'Times': '\u00d7',          # ×
    'Sqr': 'x\u00b2',          # x²
    'Divide': '\u00f7',         # ÷
    'Half': 'x/2',
    'Avg': 'avg',
    'Sqrt': '\u221a',           # √
    'Power': 'x^y',
    'Pi': '\u03c0',             # π
    'Hypot': 'hypot',
    'LogisticSigmoid': '\u03c3',  # σ
    'Cosh': 'cosh',
    'Sinh': 'sinh',
    'Tanh': 'tanh',
    'Cos': 'cos',
    'Sin': 'sin',
    'Tan': 'tan',
    'ArcSinh': 'asinh',
    'ArcCosh': 'acosh',
    'ArcCos': 'acos',
    'ArcTanh': 'atanh',
    'ArcSin': 'asin',
    'ArcTan': 'atan',
}


def parse_log(filepath):
    """Parse rust_verify.log and extract discoveries in order.

    Returns (primitives, discoveries) where:
        primitives = list of primitive names
        discoveries = list of dicts {name, type, k, witness, deps, order}
    """
    text = Path(filepath).read_text()

    primitives = set()
    m = re.search(r'--constants\s+(\S+)', text)
    if m:
        primitives.update(m.group(1).split(','))
    m = re.search(r'--operations\s+(\S+)', text)
    if m:
        primitives.update(m.group(1).split(','))

    pattern = re.compile(
        r'Found\s+(constant|unary function|binary operation|ternary operation):\s+(\S+)\s*\n'
        r'\s*witness\[k=(\d+)\]:\s*(.+)',
        re.MULTILINE
    )

    known = set(primitives)
    discoveries = []

    for match in pattern.finditer(text):
        dtype = match.group(1)
        name = match.group(2)
        k = int(match.group(3))
        witness = match.group(4).strip()

        tokens = set(re.findall(r'[A-Za-z]\w*', witness))
        nums = set(re.findall(r'-?\d+', witness))

        deps = set()
        for t in tokens:
            if t in TEST_CONSTANTS:
                continue
            if t in known:
                deps.add(t)
        for n in nums:
            if n in known:
                deps.add(n)

        discoveries.append({
            'name': name,
            'type': dtype,
            'k': k,
            'witness': witness,
            'deps': deps,
            'order': len(discoveries) + 1,
        })
        known.add(name)

    return list(primitives), discoveries


def find_primary_dep(d, discoveries, primitives):
    """Find the primary dependency (most recently discovered ingredient)."""
    if not d['deps']:
        return None
    name_to_order = {p: 0 for p in primitives}
    for disc in discoveries:
        name_to_order[disc['name']] = disc['order']
    best_dep = None
    best_order = -1
    for dep in d['deps']:
        order = name_to_order.get(dep, -1)
        if order > best_order:
            best_order = order
            best_dep = dep
    return best_dep


# ======================== STYLING ========================
TRIG = {'Cos', 'Sin', 'Tan', 'ArcCos', 'ArcSin', 'ArcTan'}
HYP = {'Cosh', 'Sinh', 'Tanh', 'ArcCosh', 'ArcSinh', 'ArcTanh'}


def get_fill(name, dtype, is_primitive=False):
    if is_primitive and name == 'EML':
        return '#ffcdd2'        # red — EML primitive
    if is_primitive or dtype == 'constant':
        return '#c8e6c9'        # green — constants
    if name in TRIG:
        return '#bbdefb'        # blue — trigonometric
    if name in HYP:
        return '#f8bbd0'        # pink — hyperbolic
    if dtype == 'unary function':
        return '#f5f5f5'        # light gray — unary
    return '#eeeeee'            # gray — binary


def get_edge_color(name, dtype, is_primitive=False):
    if is_primitive and name == 'EML':
        return '#c62828'
    if is_primitive or dtype == 'constant':
        return '#2e7d32'
    if name in TRIG:
        return '#1565c0'
    if name in HYP:
        return '#ad1457'
    return '#424242'


# ======================== GRAPH CONSTRUCTION ========================
def build_graph(primitives, discoveries):
    """Build networkx DiGraph with styling metadata."""
    G = nx.DiGraph()
    for p in primitives:
        G.add_node(p, display=DISPLAY_NAMES.get(p, p), dtype='primitive',
                   fill=get_fill(p, 'primitive', True),
                   edge_color=get_edge_color(p, 'primitive', True))

    name_seen = {}
    for d in discoveries:
        name = d['name']
        if name in name_seen:
            name_seen[name] += 1
            display = 'log_b' if name == 'Log' else f'{DISPLAY_NAMES.get(name, name)}_{name_seen[name]}'
        else:
            name_seen[name] = 1
            display = DISPLAY_NAMES.get(name, name)

        node_id = name if name_seen[name] == 1 else f'{name}_{name_seen[name]}'
        G.add_node(node_id, display=display, dtype=d['type'],
                   fill=get_fill(name, d['type']),
                   edge_color=get_edge_color(name, d['type']))
        d['node_id'] = node_id

        primary_dep = find_primary_dep(d, discoveries, primitives)
        for dep_name in d['deps']:
            is_primary = (dep_name == primary_dep)
            G.add_edge(dep_name, node_id, primary=is_primary)

    return G


# ======================== SPIRAL LAYOUT ========================
def archimedean_spiral_uniform(n_points, r_start=2.5, r_end=14.0, turns=2.2):
    """Place n_points on Archimedean spiral with UNIFORM arc-length spacing.

    Archimedean spiral: r(theta) = a + b*theta
    Arc length element: ds = sqrt(r^2 + b^2) d(theta)
    We numerically integrate and invert to find theta for equal spacing.
    """
    theta_max = 2 * np.pi * turns
    a = r_start
    b = (r_end - r_start) / theta_max

    N_fine = 10000
    theta_fine = np.linspace(0, theta_max, N_fine)
    r_fine = a + b * theta_fine
    ds = np.sqrt(r_fine**2 + b**2) * (theta_fine[1] - theta_fine[0])
    s_cumulative = np.concatenate([[0], np.cumsum(ds)])
    total_length = s_cumulative[-1]

    target_s = np.linspace(0, total_length, n_points)
    theta_uniform = np.interp(target_s, s_cumulative, np.concatenate([[0], theta_fine]))

    r_uniform = a + b * theta_uniform
    x = r_uniform * np.cos(theta_uniform)
    y = r_uniform * np.sin(theta_uniform)

    return x, y, theta_uniform, r_uniform


# ======================== MATPLOTLIB RENDERING ========================
def draw_arrow_bezier(ax, x1, y1, x2, y2, cx, cy, r_node, **style):
    """Draw a quadratic Bezier curve with arrowhead, clipped at node boundaries."""
    t = np.linspace(0, 1, 80)
    bx = (1 - t)**2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2
    by = (1 - t)**2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2

    # Clip: find entry/exit points at node boundaries
    dist_to_target = np.sqrt((bx - x2)**2 + (by - y2)**2)
    outside = np.where(dist_to_target > r_node)[0]
    t_end = outside[-1] if len(outside) > 0 else len(t) - 2

    dist_to_source = np.sqrt((bx - x1)**2 + (by - y1)**2)
    inside_src = np.where(dist_to_source < r_node)[0]
    t_start = inside_src[-1] + 1 if len(inside_src) > 0 else 0

    if t_start >= t_end:
        return

    ax.plot(bx[t_start:t_end+1], by[t_start:t_end+1], **style)

    # Arrowhead triangle
    arrow_len = r_node * 0.6
    dx = bx[t_end] - bx[max(t_end - 3, t_start)]
    dy = by[t_end] - by[max(t_end - 3, t_start)]
    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0.001:
        dx, dy = dx / norm, dy / norm
        tip_x, tip_y = bx[t_end], by[t_end]
        perp_x, perp_y = -dy, dx
        base1_x = tip_x - arrow_len * dx + arrow_len * 0.35 * perp_x
        base1_y = tip_y - arrow_len * dy + arrow_len * 0.35 * perp_y
        base2_x = tip_x - arrow_len * dx - arrow_len * 0.35 * perp_x
        base2_y = tip_y - arrow_len * dy - arrow_len * 0.35 * perp_y
        triangle = plt.Polygon(
            [[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]],
            closed=True, facecolor=style.get('color', 'black'),
            edgecolor='none', alpha=style.get('alpha', 1.0),
            zorder=style.get('zorder', 2))
        ax.add_patch(triangle)


def render_matplotlib(G, ordered_nodes, positions, r_circle, outdir, ts):
    """Render PNG + PDF via matplotlib (Bezier edges, outward-convex)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 18))

    # Edges: EML edges black, others gray
    for u, v, data in G.edges(data=True):
        if u not in positions or v not in positions:
            continue
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        mid_r = np.sqrt(mx**2 + my**2)
        if mid_r > 0.01:
            out_x, out_y = mx / mid_r, my / mid_r
        else:
            out_x, out_y = 0, 1
        bulge = dist * 0.3
        cx, cy = mx + bulge * out_x, my + bulge * out_y

        if u == 'EML':
            style = dict(color='black', lw=1.2, alpha=0.85, zorder=2)
        else:
            style = dict(color='#888888', lw=0.6, alpha=0.45, zorder=1)

        draw_arrow_bezier(ax, x1, y1, x2, y2, cx, cy, r_circle, **style)

    # Nodes on top
    for node in ordered_nodes:
        x, y = positions[node]
        data = G.nodes[node]
        fill = data.get('fill', '#eeeeee')
        ec = data.get('edge_color', '#333333')
        display = data.get('display', node)
        circle = plt.Circle((x, y), r_circle, facecolor=fill, edgecolor=ec,
                             linewidth=2.0, zorder=5)
        ax.add_patch(circle)
        fontsize = 14 if len(display) <= 4 else 11 if len(display) <= 5 else 9
        ax.text(x, y, display, ha='center', va='center', fontsize=fontsize,
                fontfamily='serif', fontweight='bold', zorder=6)

    ax.set_xlim(-18, 18)
    ax.set_ylim(-18, 18)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    # Timestamped + canonical
    for suffix, path in [
        ('png', outdir / f'graph_spiral_{ts}.png'),
        ('pdf', outdir / f'graph_spiral_{ts}.pdf'),
    ]:
        fig.savefig(str(path), dpi=250 if suffix == 'png' else None, bbox_inches='tight')
        print(f"Saved: {path}")

    fig.savefig(str(outdir / 'graph_spiral.png'), dpi=250, bbox_inches='tight')
    try:
        fig.savefig(str(outdir / 'graph_spiral.pdf'), bbox_inches='tight')
        print(f"Saved: {outdir / 'graph_spiral.pdf'}")
    except PermissionError:
        print(f"SKIPPED (locked): graph_spiral.pdf — use timestamped version")

    plt.close()


# ======================== TIKZ RENDERING ========================
def render_tikz(G, ordered_nodes, positions, r_circle, outdir, ts):
    """Generate standalone TikZ file (circular arc edges, layered rendering)."""

    color_defs = {
        '#ffcdd2': 'fillEML',
        '#c8e6c9': 'fillConst',
        '#bbdefb': 'fillTrig',
        '#f8bbd0': 'fillHyp',
        '#f5f5f5': 'fillUnary',
        '#eeeeee': 'fillBinary',
    }
    edge_color_defs = {
        '#c62828': 'edgeEML',
        '#2e7d32': 'edgeConst',
        '#1565c0': 'edgeTrig',
        '#ad1457': 'edgeHyp',
        '#424242': 'edgeDefault',
    }

    # TeX display names (ISO 80000-2 for inverse functions)
    TEX_NAMES = {
        'EML':    r'\textsf{EML}',
        '1':      r'$1$',
        'e':      r'$e$',
        '−1':     r'$-1$',
        '2':      r'$2$',
        'π':      r'$\pi$',
        'exp':    r'$\exp$',
        'ln':     r'$\ln$',
        '−x':     r'$-x$',
        '1/x':    r'$1/x$',
        'x²':     r'$x^2$',
        'x/2':    r'$x/2$',
        '√':      r'$\sqrt{x}$',
        'σ':      r'$\sigma$',
        '−':      r'$-$',
        '+':      r'$+$',
        '×':      r'$\times$',
        '÷':      r'$\div$',
        'avg':    r'$\frac{x+y}{2}$',
        'x^y':    r'$x^y$',
        'log_b':  r'$\log_x y$',
        'hypot':  r'$\sqrt{x^2\!+\!y^2}$',
        'cosh':   r'$\cosh$',
        'sinh':   r'$\sinh$',
        'tanh':   r'$\tanh$',
        'cos':    r'$\cos$',
        'sin':    r'$\sin$',
        'tan':    r'$\tan$',
        'asin':   r'$\arcsin$',
        'acos':   r'$\arccos$',
        'atan':   r'$\arctan$',
        'asinh':  r'$\mathrm{arsinh}$',
        'acosh':  r'$\mathrm{arcosh}$',
        'atanh':  r'$\mathrm{artanh}$',
    }

    def tex_escape(display):
        if display in TEX_NAMES:
            return TEX_NAMES[display]
        s = display
        for ch, repl in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#')]:
            s = s.replace(ch, repl)
        return s

    lines = []
    lines.append(r'\documentclass[border=5mm]{standalone}')
    lines.append(r'\usepackage{tikz}')
    lines.append(r'\usetikzlibrary{arrows.meta}')
    lines.append(r'\pgfdeclarelayer{background}')
    lines.append(r'\pgfsetlayers{background,main}')
    lines.append(r'\begin{document}')
    lines.append(r'\begin{tikzpicture}[')
    lines.append(r'  scale=0.8, every node/.style={transform shape},')
    lines.append(r'  every node/.append style={font=\sffamily\bfseries},')
    lines.append(r'  >=Stealth,')
    lines.append(r']')
    lines.append('')

    for hex_col, name in color_defs.items():
        r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
        lines.append(f'\\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}')
    for hex_col, name in edge_color_defs.items():
        r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
        lines.append(f'\\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}')
    lines.append('')

    node_diameter = 1.3
    lines.append(f'\\tikzset{{mynode/.style={{circle, line width=1.0pt, '
                 f'minimum size={node_diameter:.2f}cm, inner sep=0pt}}}}')
    lines.append('')

    # Nodes (main layer, on top)
    lines.append('% --- Nodes ---')
    node_tikz_id = {}
    for idx, node in enumerate(ordered_nodes):
        x, y = positions[node]
        data = G.nodes[node]
        fill_hex = data.get('fill', '#eeeeee')
        ec_hex = data.get('edge_color', '#424242')
        display = data.get('display', node)

        fill_name = color_defs.get(fill_hex, 'fillBinary')
        ec_name = edge_color_defs.get(ec_hex, 'edgeDefault')
        tex_display = tex_escape(display)

        tid = f'n{idx}'
        node_tikz_id[node] = tid

        if len(display) <= 2:
            fontsize = r'\Large'
        elif len(display) <= 4:
            fontsize = r'\large'
        elif len(display) <= 5:
            fontsize = r'\normalsize'
        else:
            fontsize = r'\small'

        lines.append(f'\\node[mynode, fill={fill_name}, draw={ec_name}] '
                     f'({tid}) at ({x:.3f},{y:.3f}) {{{fontsize} {tex_display}}};')

    lines.append('')

    # Edges (background layer, behind nodes) — circular arcs
    lines.append(r'\begin{pgfonlayer}{background}')
    lines.append('% --- Edges (circular arcs, bending outward from center) ---')
    bend_angle = 25
    for u, v, data in G.edges(data=True):
        if u not in positions or v not in positions:
            continue
        if u not in node_tikz_id or v not in node_tikz_id:
            continue
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        mid_r = np.sqrt(mx**2 + my**2)
        if mid_r > 0.01:
            out_x, out_y = mx / mid_r, my / mid_r
        else:
            out_x, out_y = 0, 1
        cross = (x2 - x1) * out_y - (y2 - y1) * out_x
        bend_dir = 'bend left' if cross > 0 else 'bend right'

        src = node_tikz_id[u]
        dst = node_tikz_id[v]

        if u == 'EML':
            style = f'black, line width=0.6pt, opacity=0.85, ->, {bend_dir}={bend_angle}'
        else:
            style = f'black!50, line width=0.3pt, opacity=0.45, ->, {bend_dir}={bend_angle}'

        lines.append(f'\\draw[{style}] ({src}) to ({dst});')

    lines.append(r'\end{pgfonlayer}')
    lines.append('')
    lines.append(r'\end{tikzpicture}')
    lines.append(r'\end{document}')

    tikz_text = '\n'.join(lines)

    tikz_ts = outdir / f'graph_spiral_{ts}.tex'
    Path(tikz_ts).write_text(tikz_text, encoding='utf-8')
    print(f"TikZ saved: {tikz_ts}")

    tikz_canonical = outdir / 'graph_spiral.tex'
    Path(tikz_canonical).write_text(tikz_text, encoding='utf-8')
    print(f"TikZ saved: {tikz_canonical}")


# ======================== MAIN ========================
def main():
    logfile = sys.argv[1] if len(sys.argv) > 1 else 'rust_verify.log'
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')

    primitives, discoveries = parse_log(logfile)
    G = build_graph(primitives, discoveries)

    # EML at center of spiral, then constant 1, then discoveries in order
    ordered_nodes = ['EML', '1'] + [d['node_id'] for d in discoveries]
    n = len(ordered_nodes)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Spiral layout with uniform arc-length spacing
    xs, ys, thetas, radii = archimedean_spiral_uniform(n)
    positions = {node: (xs[i], ys[i]) for i, node in enumerate(ordered_nodes)}

    r_circle = 0.65  # node radius in matplotlib coordinates

    render_matplotlib(G, ordered_nodes, positions, r_circle, outdir, ts)
    render_tikz(G, ordered_nodes, positions, r_circle, outdir, ts)

    print(f"\nAll outputs in: {outdir}")


if __name__ == '__main__':
    main()
