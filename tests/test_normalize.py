import sympy as sp

from symreg_dataset.normalize import canonical, serialize, parse_expression_text


def test_serialize_rounds_constants():
    e = sp.sympify("2.0*x + 3.14159")
    assert "3.14" in serialize(e, precision=2)


def test_parse_drops_prefix():
    assert parse_expression_text("f(x) = x**2 + 1") == sp.sympify("x**2 + 1")
    assert parse_expression_text("y = sin(x)") == sp.sympify("sin(x)")


def test_canonical_stable():
    e1 = sp.sympify("2*sin(x) + 3")
    e2 = sp.sympify("3 + 2*sin(x)")
    assert canonical(e1) == canonical(e2)