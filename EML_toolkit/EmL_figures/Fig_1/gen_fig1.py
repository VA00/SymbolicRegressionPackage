#!/usr/bin/env python3
"""
Figure 1: EML dependency graph — spiral embedding + adjacency matrix.

Self-contained script. Reads rust_verify.log, generates:
  graph_spiral.{png,pdf}       Spiral graph (matplotlib, Bezier edges)
  graph_spiral.tex             Spiral graph (standalone TikZ)
  adjacency_matrix.{png,pdf}   Adjacency matrix (matplotlib)

Both figures share the same log, color scheme, labels, and node ordering.

Requirements: Python 3.10+, numpy, matplotlib, networkx
TikZ compilation: pdflatex graph_spiral.tex

Usage:
    python gen_fig1.py                                   # default paths
    python gen_fig1.py rust_verify.log output_dir/       # custom paths
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
TEST_CONSTANTS = {"EulerGamma", "Glaisher"}


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


# ======================== SHARED STYLING ========================
TRIG = {'Cos', 'Sin', 'Tan', 'ArcCos', 'ArcSin', 'ArcTan'}
HYP  = {'Cosh', 'Sinh', 'Tanh', 'ArcCosh', 'ArcSinh', 'ArcTanh'}

# Display names — Unicode, used in matplotlib and as keys for TikZ lookup.
# Inverse hyperbolic: ISO 80000-2 notation (arsinh, arcosh, artanh).
DISPLAY_NAMES = {
    'EML': 'EML',  '1': '1',  'E': 'e',
    'Exp': 'exp',  'Log': 'ln',
    'Subtract': '\u2212',  '-1': '\u22121',  '2': '2',
    'Minus': '\u2212x',  'Plus': '+',  'Inv': '1/x',
    'Times': '\u00d7',  'Sqr': 'x\u00b2',  'Divide': '\u00f7',
    'Half': 'x/2',  'Avg': 'avg',  'Sqrt': '\u221ax',
    'Power': 'x\u02b8',  'Pi': '\u03c0',  'Hypot': 'hypot',
    'LogisticSigmoid': '\u03c3',
    'Cosh': 'cosh',  'Sinh': 'sinh',  'Tanh': 'tanh',
    'Cos': 'cos',  'Sin': 'sin',  'Tan': 'tan',
    'ArcSinh': 'arsinh',  'ArcCosh': 'arcosh',  'ArcCos': 'arccos',
    'ArcTanh': 'artanh',  'ArcSin': 'arcsin',  'ArcTan': 'arctan',
}

# Category-based color scheme — IDENTICAL colors in spiral and adjacency matrix.
#   fill = node fill (spiral) and square color (adjacency matrix)
#   edge = darker border for spiral nodes
# Original colors from the spiral graph in EML.tex (Fig1.pdf).
#   fill = pastel node fill (spiral nodes)
#   edge = saturated border (spiral node borders AND adjacency matrix squares)
CATEGORIES = {
    'eml':      {'fill': '#ef9a9a', 'edge': '#c62828'},
    'constant': {'fill': '#c8e6c9', 'edge': '#2e7d32'},
    'unary':    {'fill': '#cccccc', 'edge': '#424242'},
    'binary':   {'fill': '#c0c0c0', 'edge': '#424242'},
    'trig':     {'fill': '#bbdefb', 'edge': '#1565c0'},
    'hyp':      {'fill': '#f8bbd0', 'edge': '#ad1457'},
}


def get_category(name, dtype, is_primitive=False):
    """Return visual category for a function."""
    if is_primitive and name == 'EML':
        return 'eml'
    if is_primitive or dtype == 'constant':
        return 'constant'
    if name in TRIG:
        return 'trig'
    if name in HYP:
        return 'hyp'
    if dtype == 'unary function':
        return 'unary'
    return 'binary'


def get_fill(name, dtype, is_primitive=False):
    return CATEGORIES[get_category(name, dtype, is_primitive)]['fill']


def get_edge_color(name, dtype, is_primitive=False):
    return CATEGORIES[get_category(name, dtype, is_primitive)]['edge']


def get_bold_color(name, dtype, is_primitive=False):
    """Saturated border color — used for adjacency matrix squares."""
    return CATEGORIES[get_category(name, dtype, is_primitive)]['edge']


def get_text_color(name, dtype, is_primitive=False):
    """Dark text on light pastel fills."""
    return 'black'


# ======================== GRAPH CONSTRUCTION ========================
def build_graph(primitives, discoveries):
    """Build networkx DiGraph with styling metadata."""
    G = nx.DiGraph()
    for p in primitives:
        cat = get_category(p, 'primitive', True)
        G.add_node(p, display=DISPLAY_NAMES.get(p, p), dtype='primitive',
                   category=cat,
                   fill=CATEGORIES[cat]['fill'],
                   edge_color=CATEGORIES[cat]['edge'],
                   text_color='black')

    name_seen = {}
    for d in discoveries:
        name = d['name']
        if name in name_seen:
            name_seen[name] += 1
            display = 'log_b' if name == 'Log' else \
                f'{DISPLAY_NAMES.get(name, name)}_{name_seen[name]}'
        else:
            name_seen[name] = 1
            display = DISPLAY_NAMES.get(name, name)

        node_id = name if name_seen[name] == 1 else f'{name}_{name_seen[name]}'
        cat = get_category(name, d['type'])
        G.add_node(node_id, display=display, dtype=d['type'],
                   category=cat,
                   fill=CATEGORIES[cat]['fill'],
                   edge_color=CATEGORIES[cat]['edge'],
                   text_color='black')
        d['node_id'] = node_id

        primary_dep = find_primary_dep(d, discoveries, primitives)
        for dep_name in d['deps']:
            is_primary = (dep_name == primary_dep)
            G.add_edge(dep_name, node_id, primary=is_primary)

    return G


# ======================== SPIRAL LAYOUT ========================
def archimedean_spiral_uniform(n_points, r_start=2.5, r_end=14.0, turns=2.2):
    """Place n_points on Archimedean spiral with uniform arc-length spacing."""
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
    theta_uniform = np.interp(target_s, s_cumulative,
                              np.concatenate([[0], theta_fine]))

    r_uniform = a + b * theta_uniform
    x = r_uniform * np.cos(theta_uniform)
    y = r_uniform * np.sin(theta_uniform)

    return x, y, theta_uniform, r_uniform


# ======================== MATPLOTLIB SPIRAL ========================
def draw_arrow_bezier(ax, x1, y1, x2, y2, cx, cy, r_node, **style):
    """Draw a quadratic Bezier curve with solid arrowhead, clipped at nodes."""
    t = np.linspace(0, 1, 80)
    bx = (1 - t)**2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2
    by = (1 - t)**2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2

    dist_to_target = np.sqrt((bx - x2)**2 + (by - y2)**2)
    outside = np.where(dist_to_target > r_node)[0]
    t_end = outside[-1] if len(outside) > 0 else len(t) - 2

    dist_to_source = np.sqrt((bx - x1)**2 + (by - y1)**2)
    inside_src = np.where(dist_to_source < r_node)[0]
    t_start = inside_src[-1] + 1 if len(inside_src) > 0 else 0

    if t_start >= t_end:
        return

    ax.plot(bx[t_start:t_end+1], by[t_start:t_end+1], **style)

    # Solid arrowhead triangle
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


def render_spiral_mpl(G, ordered_nodes, positions, r_circle, outdir, ts):
    """Render spiral graph via matplotlib (Bezier edges)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 18))

    # Edges: EML solid, "1" dashed (same weight), others thin gray
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
        elif u == '1':
            style = dict(color='black', lw=1.2, alpha=0.85, zorder=2,
                         linestyle='dashed')
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
        tc = data.get('text_color', 'white')
        fontsize = 14 if len(display) <= 4 else 11 if len(display) <= 5 else 9
        ax.text(x, y, display, ha='center', va='center', fontsize=fontsize,
                fontfamily='serif', fontweight='bold', color=tc, zorder=6)

    ax.set_xlim(-18, 18)
    ax.set_ylim(-18, 18)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    for suffix in ('png', 'pdf'):
        path = outdir / f'graph_spiral_{ts}.{suffix}'
        fig.savefig(str(path),
                    dpi=250 if suffix == 'png' else None, bbox_inches='tight')
        print(f"Saved: {path}")

    fig.savefig(str(outdir / 'graph_spiral.png'), dpi=250, bbox_inches='tight')
    try:
        fig.savefig(str(outdir / 'graph_spiral.pdf'), bbox_inches='tight')
        print(f"Saved: {outdir / 'graph_spiral.pdf'}")
    except PermissionError:
        print(f"SKIPPED (locked): graph_spiral.pdf")
    plt.close()


# ======================== TIKZ SPIRAL ========================
def render_spiral_tikz(G, ordered_nodes, positions, r_circle, outdir, ts):
    """Generate standalone TikZ file (circular arc edges, layered rendering)."""

    color_defs = {
        '#ef9a9a': 'fillEML',
        '#c8e6c9': 'fillConst',
        '#bbdefb': 'fillTrig',
        '#f8bbd0': 'fillHyp',
        '#cccccc': 'fillUnary',
        '#c0c0c0': 'fillBinary',
    }
    edge_color_defs = {
        '#c62828': 'edgeEML',
        '#2e7d32': 'edgeConst',
        '#1565c0': 'edgeTrig',
        '#ad1457': 'edgeHyp',
        '#424242': 'edgeArith',
    }

    # TeX display names (ISO 80000-2 for inverse functions)
    # Compact forms using \smash, \! and scriptsize stacking for long labels
    TEX_NAMES = {
        'EML':     r'$\operatorname{eml}$',
        '1':       r'$1$',
        'e':       r'$e$',
        '\u22121': r'$\!-\!1$',
        '2':       r'$2$',
        '\u03c0':  r'$\pi$',
        'exp':     r'$\exp$',
        'ln':      r'$\ln$',
        '\u2212x': r'$-x$',
        '1/x':     r'$1/x$',
        'x\u00b2': r'$x^2$',
        'x/2':     r'$x/2$',
        '\u221ax': r'$\!\sqrt{x}$',
        '\u03c3':  r'$\sigma$',
        '\u2212':  r'$-$',
        '+':       r'$+$',
        '\u00d7':  r'$\times$',
        '\u00f7':  r'$\div$',
        'avg':     r'$\smash{\frac{x\!+\!y}{2}}$',
        'x\u02b8': r'$x^y$',
        'log_b':   r'$\!\log_{\!x}\!y$',
        'hypot':   r'$\smash{\sqrt{\!x^2\!\!+\!y^2}}$',
        'cosh':    r'$\cosh$',
        'sinh':    r'$\sinh$',
        'tanh':    r'$\tanh$',
        'cos':     r'$\cos$',
        'sin':     r'$\sin$',
        'tan':     r'$\tan$',
        'arcsin':  r'$\arcsin$',
        'arccos':  r'$\arccos$',
        'arctan':  r'$\arctan$',
        'arsinh':  r'$\mathrm{arsinh}$',
        'arcosh':  r'$\mathrm{arcosh}$',
        'artanh':  r'$\mathrm{artanh}$',
    }

    def tex_escape(display):
        if display in TEX_NAMES:
            return TEX_NAMES[display]
        s = display
        for ch, repl in [('_', r'\_'), ('&', r'\&'), ('%', r'\%'),
                         ('#', r'\#')]:
            s = s.replace(ch, repl)
        return s

    lines = []
    lines.append(r'\documentclass[border=5mm]{standalone}')
    lines.append(r'\usepackage{amsmath}')
    lines.append(r'\usepackage{tikz}')
    lines.append(r'\usetikzlibrary{arrows.meta}')
    lines.append(r'\pgfdeclarelayer{background}')
    lines.append(r'\pgfsetlayers{background,main}')
    lines.append(r'\newsavebox{\hypotbox}')
    lines.append(r'\begin{document}')
    lines.append(r'\begin{tikzpicture}[')
    lines.append(r'  scale=0.8, every node/.style={transform shape},')
    lines.append(r'  every node/.append style={font=\sffamily\bfseries},')
    lines.append(r'  >={Stealth[length=5pt,width=4pt]},')
    lines.append(r']')
    lines.append('')

    for hex_col, name in color_defs.items():
        r, g, b = (int(hex_col[1:3], 16), int(hex_col[3:5], 16),
                   int(hex_col[5:7], 16))
        lines.append(f'\\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}')
    for hex_col, name in edge_color_defs.items():
        r, g, b = (int(hex_col[1:3], 16), int(hex_col[3:5], 16),
                   int(hex_col[5:7], 16))
        lines.append(f'\\definecolor{{{name}}}{{RGB}}{{{r},{g},{b}}}')
    lines.append('')

    # Uniform node size: measure widest label (hypot), scale by 1.15
    lines.append(r'\savebox{\hypotbox}{\fontsize{14}{18}\selectfont '
                 r'$\smash{\sqrt{\!x^2\!\!+\!y^2}}$}')
    lines.append(r'\pgfmathsetlengthmacro{\nodesize}'
                 r'{1.15 * max(1.3cm, \wd\hypotbox)}')
    lines.append(r'\tikzset{mynode/.style={circle, line width=1.0pt, '
                 r'minimum size=\nodesize, inner sep=0pt}}')
    lines.append('')

    # Nodes (main layer, on top)
    lines.append('% --- Nodes ---')
    node_tikz_id = {}
    for idx, node in enumerate(ordered_nodes):
        x, y = positions[node]
        data = G.nodes[node]
        fill_hex = data.get('fill', '#f5f5f5')
        ec_hex = data.get('edge_color', '#424242')
        display = data.get('display', node)

        fill_name = color_defs.get(fill_hex, 'fillUnary')
        ec_name = edge_color_defs.get(ec_hex, 'edgeArith')
        tex_display = tex_escape(display)

        tid = f'n{idx}'
        node_tikz_id[node] = tid

        if len(display) <= 2:
            fontsize_pt = 29
        elif len(display) <= 4:
            fontsize_pt = 23
        elif len(display) <= 6:
            fontsize_pt = 18
        else:
            fontsize_pt = 14

        # Center on math axis: axis_height = 0.25 * fontsize (CM fonts).
        # text_height = 2 * axis_height so bounding box center = axis.
        th = fontsize_pt * 0.5
        leading = fontsize_pt + 4
        fontcmd = (f'\\fontsize{{{fontsize_pt}}}{{{leading}}}'
                   r'\selectfont')

        lines.append(f'\\node[mynode, fill={fill_name}, draw={ec_name}, '
                     f'text height={th:.1f}pt, text depth=0pt] '
                     f'({tid}) at ({x:.3f},{y:.3f}) '
                     f'{{{fontcmd} {tex_display}}};')

    lines.append('')

    # Edges (background layer) — circular arcs
    # EML: solid black, "1": dashed black (same weight), others: thin gray
    lines.append(r'\begin{pgfonlayer}{background}')
    lines.append('% --- Edges ---')
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
            style = (f'black, line width=0.9pt, '
                     f'->, {bend_dir}={bend_angle}')
        elif u == '1':
            style = (f'black, line width=0.9pt, dashed, '
                     f'->, {bend_dir}={bend_angle}')
        else:
            style = (f'black!50, line width=0.5pt, opacity=0.45, '
                     f'->, {bend_dir}={bend_angle}')

        lines.append(f'\\draw[{style}] ({src}) to ({dst});')

    lines.append(r'\end{pgfonlayer}')
    lines.append('')
    lines.append(r'\end{tikzpicture}')
    lines.append(r'\end{document}')

    tikz_text = '\n'.join(lines)

    tikz_ts = outdir / f'graph_spiral_{ts}.tex'
    Path(tikz_ts).write_text(tikz_text, encoding='utf-8')
    print(f"TikZ saved: {tikz_ts}")

    tikz_canonical = outdir / 'Fig1_graph_spiral.tex'
    Path(tikz_canonical).write_text(tikz_text, encoding='utf-8')
    print(f"TikZ saved: {tikz_canonical}")


# ======================== ADJACENCY MATRIX ========================
# Matplotlib mathtext overrides for axis labels
ADJ_LABEL_OVERRIDES = {
    'x\u00b2':  r'$x^2$',
    'x\u02b8':  r'$x^y$',
    'log_b':    r'$\log_x y$',
    '\u03c0':   r'$\pi$',
    '\u03c3':   r'$\sigma$',
    '\u221ax':  r'$\sqrt{x}$',
}


def adj_label(display_name):
    """Get matplotlib-rendered label for adjacency matrix axis."""
    return ADJ_LABEL_OVERRIDES.get(display_name, display_name)


def render_adjacency_matrix(primitives, discoveries, outdir, ts):
    """Render adjacency matrix with colors matching the spiral graph."""

    # Build ordered node list: EML first, then 1, then discoveries
    prim_ordered = ['EML', '1'] if 'EML' in primitives else sorted(primitives)
    nodes = list(prim_ordered)
    display_names = [DISPLAY_NAMES.get(p, p) for p in prim_ordered]
    col_fills = [get_fill(p, 'primitive', True) for p in prim_ordered]
    col_edges = [get_edge_color(p, 'primitive', True) for p in prim_ordered]

    name_seen = {}
    for d in discoveries:
        name = d['name']
        if name in name_seen:
            name_seen[name] += 1
            display = 'log_b' if name == 'Log' else \
                f'{DISPLAY_NAMES.get(name, name)}_{name_seen[name]}'
        else:
            name_seen[name] = 1
            display = DISPLAY_NAMES.get(name, name)
        nodes.append(name)
        display_names.append(display)
        col_fills.append(get_fill(name, d['type']))
        col_edges.append(get_edge_color(name, d['type']))

    display_labels = [adj_label(dn) for dn in display_names]

    n = len(nodes)
    name_to_idx = {}
    for i, name in enumerate(nodes):
        if name not in name_to_idx:
            name_to_idx[name] = i

    # Adjacency: M[i,j] = 1 means node i depends on ingredient j
    M = np.zeros((n, n), dtype=int)
    for d_idx, d in enumerate(discoveries):
        i = len(primitives) + d_idx
        for dep_name in d['deps']:
            j = name_to_idx.get(dep_name)
            if j is not None:
                M[i, j] = 1

    # Remove primitive rows (they have no dependencies)
    n_prim = len(primitives)
    M_show = M[n_prim:, :]
    row_labels = display_labels[n_prim:]
    all_col_labels = display_labels
    n_rows, n_cols = M_show.shape

    col_used = M_show.sum(axis=0) > 0

    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
    })

    fig, ax = plt.subplots(1, 1, figsize=(14, 13))
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    # Grey out empty columns (ingredients never used)
    for j in range(n_cols):
        if not col_used[j]:
            ax.add_patch(plt.Rectangle(
                (j - 0.5, -0.5), 1.0, n_rows,
                facecolor='#f0f0f0', edgecolor='none', zorder=0))

    # Colored squares — pastel fills matching spiral node fills, no frame
    for i in range(n_rows):
        for j in range(n_cols):
            if M_show[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor=col_fills[j], edgecolor='none',
                    zorder=2))

    # Grid
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.3, zorder=3)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color='black', linewidth=0.3, zorder=3)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color('black')

    # Legend — identical to spiral node colors (pastel fill + saturated border)
    legend_items = [
        (CATEGORIES['eml']['fill'],      CATEGORIES['eml']['edge'],      'EML'),
        (CATEGORIES['constant']['fill'], CATEGORIES['constant']['edge'], 'constant'),
        (CATEGORIES['unary']['fill'],    CATEGORIES['unary']['edge'],    'arithmetic'),
        (CATEGORIES['trig']['fill'],     CATEGORIES['trig']['edge'],     'trigonometric'),
        (CATEGORIES['hyp']['fill'],      CATEGORIES['hyp']['edge'],      'hyperbolic'),
        ('#f0f0f0',                      '#cccccc',                      'unused'),
    ]
    lx, ly = n_cols - 10, 4
    for k, (fill, ec, label) in enumerate(legend_items):
        y = ly + k
        ax.add_patch(plt.Rectangle(
            (lx - 0.5, y - 0.5), 1.0, 1.0,
            facecolor=fill, edgecolor='none', zorder=4))
        ax.text(lx + 1, y, label, fontsize=28, fontfamily='serif',
                va='center', ha='left', zorder=4)

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(all_col_labels, rotation=90, fontsize=28,
                       fontfamily='serif', color='black')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=28, fontfamily='serif',
                       color='black')

    ax.set_xlabel('Ingredients', fontsize=28, fontfamily='serif', labelpad=12)
    ax.set_ylabel('Constructed function', fontsize=28, fontfamily='serif',
                  labelpad=12)

    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    ax.tick_params(axis='both', which='both', length=0)
    ax.tick_params(axis='y', pad=10)

    plt.tight_layout()

    for suffix in ('png', 'pdf'):
        path = outdir / f'adjacency_matrix_{ts}.{suffix}'
        fig.savefig(str(path),
                    dpi=300 if suffix == 'png' else None, bbox_inches='tight')
        print(f"Saved: {path}")

    fig.savefig(str(outdir / 'SI_adjacency_matrix.png'), dpi=300,
                bbox_inches='tight')
    try:
        fig.savefig(str(outdir / 'SI_adjacency_matrix.pdf'), bbox_inches='tight')
        print(f"Saved: {outdir / 'SI_adjacency_matrix.pdf'}")
    except PermissionError:
        print(f"SKIPPED (locked): SI_adjacency_matrix.pdf")
    plt.close()


# ======================== MAIN ========================
def main():
    logfile = sys.argv[1] if len(sys.argv) > 1 else 'rust_verify.log'
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')

    primitives, discoveries = parse_log(logfile)
    G = build_graph(primitives, discoveries)

    # Node ordering: EML first, then 1, then discoveries in bootstrapping order
    ordered_nodes = ['EML', '1'] + [d['node_id'] for d in discoveries]
    n = len(ordered_nodes)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Spiral layout with uniform arc-length spacing
    xs, ys, thetas, radii = archimedean_spiral_uniform(n)
    positions = {node: (xs[i], ys[i]) for i, node in enumerate(ordered_nodes)}
    r_circle = 0.75

    # Generate all outputs
    render_spiral_mpl(G, ordered_nodes, positions, r_circle, outdir, ts)
    render_spiral_tikz(G, ordered_nodes, positions, r_circle, outdir, ts)
    render_adjacency_matrix(primitives, discoveries, outdir, ts)

    print(f"\nAll outputs in: {outdir}")


if __name__ == '__main__':
    main()
