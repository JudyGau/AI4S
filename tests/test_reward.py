import sympy as sp

from symreg_dataset import reward
from symreg_dataset.symbols import make_symbols


def test_compute_reward_perfect_and_wrong():
    var_symbols = make_symbols(1)
    ref_expr = sp.sympify("sin(x)")
    pts = [0.0, 0.5, 1.0, 1.5, 2.0]
    var_mat = [pts]
    ref_ys = [float(ref_expr.subs(var_symbols[0], p)) for p in pts]

    good = reward.compute_reward("sin(x)", ref_expr, var_mat, ref_ys, var_symbols)
    bad = reward.compute_reward("cos(x)", ref_expr, var_mat, ref_ys, var_symbols)
    assert good["total"] > bad["total"]
    assert good["rmse"] < 1e-6


def test_form_similarity():
    ref = sp.sympify("2*sin(x) + 3")
    assert reward.form_similarity("2*sin(x) + 3", ref) > 0.99
    assert reward.form_similarity("cos(x) - 7", ref) < 0.6