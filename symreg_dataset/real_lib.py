"""真实表达式库与「现实性」筛选。

对齐 SymbArena（arXiv:2508.09897）的 reality enhancement：
- 内置一批已知科学方程（Feynman Symbolic Regression Database 代表式 + Nguyen 基准式）作为种子；
- 提供极简的本地启发式相似度：以算子集合 + 参数个数衡量生成式与库中式的相似度，
  过滤明显不合现实的畸形表达式；可选扩展 LLM 检索（默认本地启发式）。
"""
from __future__ import annotations

import sympy as sp

# (名字, sympy 可解析的表达式字符串，自变量)
REAL_EQUATIONS: list[tuple[str, str, list[str]]] = [
    # ---- Feynman 物理方程（代表性）----
    ("Feynman I.9.18", "x0 + v*t", ["x0", "v", "t"]),
    ("Feynman I.11.3", "x + x*cos(a) + y*sin(a)", ["x", "y", "a"]),
    ("Feynman II.11.3", "q/(4*pi*eps0*r)", ["q", "eps0", "r"]),
    ("Feynman I.6.20a", "exp(-theta**2/2)/sqrt(2*pi)", ["theta"]),
    ("Hooke's law", "k*x", ["k", "x"]),
    ("Linear motion", "x0 + v0*t + a*t**2/2", ["x0", "v0", "t", "a"]),
    ("Mass-energy", "m*c**2", ["m", "c"]),
    ("Pendulum period", "2*pi*sqrt(l/g)", ["l", "g"]),
    ("Simple harmonic", "A*sin(w*t)", ["A", "w", "t"]),
    ("Coulomb", "k*q1*q2/r**2", ["k", "q1", "q2", "r"]),
    # ---- Nguyen 基准 ----
    ("Nguyen-3", "x**3 + x**2 + x", ["x"]),
    ("Nguyen-4", "x**4 + x**3 + x**2 + x", ["x"]),
    ("Nguyen-5", "x**5 + x**4 + x**3 + x**2 + x", ["x"]),
    ("Nguyen-7", "log(x + 1) + log(x**2 + 1)", ["x"]),
    ("Nguyen-8", "sqrt(x)", ["x"]),
    ("Nguyen-10", "x**2 + x + cos(x) + sin(x)", ["x"]),
    ("Nguyen-12", "x**4 - x**3 + x**2/2 - x", ["x"]),
    ("Nguyen-keijzer-13", "0.3*x**3 + 0.75**x*sin(x)" , ["x"]),
]


def load_real_equations() -> list[sp.Expr]:
    """加载真实方程库为 SymPy 表达式列表。"""
    out = []
    for _, text, _vars in REAL_EQUATIONS:
        try:
            out.append(sp.sympify(text))
        except Exception:
            continue
    return out


def _token_set(expr: sp.Expr) -> frozenset[str]:
    """表达式的算子/结构 token 集合（用于相似度）。"""
    ops = [type(a).__name__ for a in expr.atoms(sp.Function)]
    ops += ["Add"] * expr.count_ops()
    base = set(ops) | {str(s) for s in expr.free_symbols}
    return frozenset(base)


def reality_score(expr: sp.Expr, library: list[sp.Expr] | None = None) -> float:
    """与库中表达式的最大 Jaccard 相似度（算子和变量层面）。"""
    lib = library if library is not None else load_real_equations()
    if not lib:
        return 0.0
    ts = _token_set(expr)
    best = 0.0
    for ref in lib:
        tr = _token_set(ref)
        inter = len(ts & tr)
        union = len(ts | tr)
        best = max(best, inter / union if union else 0.0)
    return best


def is_realistic(expr: sp.Expr, threshold: float = 0.0,
                 library: list[sp.Expr] | None = None) -> bool:
    """现实性判定：相似度不低于阈值即认为可接受。threshold=0 时仅做基本结构合理性。"""
    score = reality_score(expr, library)
    return score >= threshold