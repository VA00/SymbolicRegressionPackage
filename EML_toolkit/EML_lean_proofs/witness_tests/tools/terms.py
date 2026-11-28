"""Terms of the EML proofs as trees, read from three kinds of source, and compared.

A term is a binary tree with leaves 1, x, y and inner nodes eml.  Every tree is stored once
(hash-consing), so two terms are the same tree exactly when they have the same number here.
Sources: Lean definitions (`def W (s t : Term n) : Term n := ...`), Wolfram Language definitions
(`W[x_, y_] := ...;`), and Python scripts, whose term definitions are run with eml replaced by
the builder of trees.
"""
import re, sys

class Store:
    def __init__(self):
        self.key = {}                 # (left, right) -> id
        self.leaves = [1, 1, 1, 1]    # ids 0..3: the leaves 1, x, y, z
        self.names = ["1", "x", "y", "z"]
    ONE, X, Y, Z = 0, 1, 2, 3
    def eml(self, a, b):
        k = (a, b)
        i = self.key.get(k)
        if i is None:
            i = len(self.leaves)
            self.key[k] = i
            self.leaves.append(self.leaves[a] + self.leaves[b])
        return i
    def K(self, i):                   # leaves + eml nodes = 2 * leaves - 1
        return 2 * self.leaves[i] - 1

class Defs:
    """name -> (list of parameters, body); body is ('atom', name) or ('app', head, [bodies])."""
    def __init__(self, store, primitive="eml", one=("one", "1")):
        self.s, self.d, self.memo = store, {}, {}
        self.primitive, self.one = primitive, set(one)
    def add(self, name, params, body):
        self.d[name] = (params, body)
    def value(self, name, args=()):
        k = (name, tuple(args))
        if k not in self.memo:
            params, body = self.d[name]
            if len(params) != len(args):
                raise ValueError("%s takes %d arguments, got %d" % (name, len(params), len(args)))
            self.memo[k] = self.ev(body, dict(zip(params, args)))
        return self.memo[k]
    def ev(self, b, loc):
        if b[0] == "atom":
            n = b[1]
            if n in loc: return loc[n]
            if n in self.one: return Store.ONE
            if n in self.d: return self.value(n)
            raise KeyError("unknown name: " + n)
        head, args = b[1], [self.ev(a, loc) for a in b[2]]
        if head == self.primitive:
            if len(args) != 2: raise ValueError("eml with %d arguments" % len(args))
            return self.s.eml(args[0], args[1])
        return self.value(head, args)

# ---------------------------------------------------------------- Lean
def lean_defs(text):
    """Definitions with result type `Term n`."""
    text = re.sub(r"/-.*?-/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    text = re.sub(r"--[^\n]*", "", text)
    out = []
    pat = re.compile(r"^(?:noncomputable\s+)?def\s+([\w']+)\s*((?:\([^()]*\)\s*)*):\s*Term\s+\w+\s*:=", re.M)
    stop = re.compile(r"^\s*$|^(?:theorem|lemma|def|noncomputable|@\[|end\b|namespace|section|open|private|instance|abbrev|example|variable)", re.M)
    for m in pat.finditer(text):
        params = []
        for g in re.findall(r"\(([^()]*)\)", m.group(2)):
            names, typ = g.split(":")
            if typ.strip().startswith("Term"):
                params += names.split()
            else:
                params = None; break
        if params is None: continue
        rest = text[m.end():]
        e = stop.search(rest, 1 if rest.startswith("\n") and rest[1:2] not in ("\n",) else 0)
        # the body may start on the next line; cut at the first blank line or next declaration
        lines, body = rest.split("\n"), []
        for j, ln in enumerate(lines):
            if j > 0 and (ln.strip() == "" or stop.match(ln) and ln.strip() != ""):
                break
            body.append(ln)
        out.append((m.group(1), params, lean_expr(" ".join(body))))
    return out

def lean_expr(src):
    toks = re.findall(r"[\w'.]+|[()]", src)
    pos = [0]
    def atom():
        t = toks[pos[0]]; pos[0] += 1
        if t == "(":
            e = seq(); assert toks[pos[0]] == ")", src; pos[0] += 1
            return e
        return ("atom", t)
    def seq():
        items = []
        while pos[0] < len(toks) and toks[pos[0]] != ")":
            items.append(atom())
        if len(items) == 1: return items[0]
        assert items and items[0][0] == "atom", src
        return ("app", items[0][1], items[1:])
    e = seq()
    assert pos[0] == len(toks), src
    return e

# ---------------------------------------------------------------- Wolfram Language
WL_NAMES = {r"\[CapitalEpsilon]": "EPS", r"\[CapitalNu]": "NU", r"\[CapitalDelta]": "DELTA",
            r"\[Tau]": "TAU", r"\[Sigma]": "SIGMA"}
def wl_defs(text, until=None):
    if until and until in text: text = text[:text.index(until)]
    text = re.sub(r"\(\*.*?\*\)", " ", text, flags=re.S)
    for k, v in WL_NAMES.items(): text = text.replace(k, v)
    out = []
    for m in re.finditer(r"^([A-Za-z]\w*)(?:\[([^\]]*)\])?\s*(:=|=)\s*(.*?);", text, flags=re.M | re.S):
        name, params, body = m.group(1), m.group(2), m.group(4)
        params = [p.strip().rstrip("_") for p in params.split(",")] if params else []
        try:
            out.append((name, params, wl_expr(body)))
        except Exception:
            pass                      # not a term (a message, a number, the operator itself)
    return out

def wl_expr(src):
    toks = re.findall(r"[A-Za-z]\w*|\d+|[\[\],]", src)
    if "".join(toks) != re.sub(r"\s+", "", src): raise ValueError(src)
    pos = [0]
    def expr():
        t = toks[pos[0]]; pos[0] += 1
        if pos[0] < len(toks) and toks[pos[0]] == "[":
            pos[0] += 1; args = [expr()]
            while toks[pos[0]] == ",":
                pos[0] += 1; args.append(expr())
            assert toks[pos[0]] == "]"; pos[0] += 1
            return ("app", t, args)
        return ("atom", t)
    e = expr()
    if pos[0] != len(toks): raise ValueError(src)
    return e

def load(defs, items, skip=()):
    for name, params, body in items:
        if name not in skip: defs.add(name, params, body)

def compare(store, A, B, pairs, label):
    """pairs: (name in A, name in B, number of arguments).  Prints one line per pair."""
    bad = 0
    for na, nb, k in pairs:
        args = [Store.X, Store.Y, Store.Z][:k]
        try:
            a, b = A.value(na, args), B.value(nb, args)
        except Exception as ex:
            print("  ERROR  %-16s %-18s %s" % (na, nb, ex)); bad += 1; continue
        same = a == b
        bad += not same
        print("  %-5s  %-16s %-18s leaves %s%s" % ("same" if same else "DIFF", na, nb, store.leaves[a],
              "" if same else " / %s" % store.leaves[b]))
    print("%s: %d pairs, %d differ" % (label, len(pairs), bad))
    return bad

# ---------------------------------------------------------------- Python
def py_terms(source, store, start, stop, extra=None):
    """Runs the lines of a Python script between the markers `start` and `stop` (the term definitions)
    with eml replaced by the tree builder, so that every lambda returns a tree.  Returns the names."""
    a = source.index(start); b = source.index(stop, a)
    ns = {"eml": store.eml, "mpc": lambda *v: Store.ONE, "mpf": lambda *v: Store.ONE, "isinstance": lambda *v: True,
          "abs": lambda v: 0, "pi": 0}
    if extra: ns.update(extra)
    exec(compile(source[a:b], "<terms>", "exec"), ns)
    return ns
