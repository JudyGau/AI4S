"""奖励函数：数值（R² / RMSE / Acc_τ）+ 形式一致性。

对齐 SymbArena / Symbolic-R1（arXiv:2508.09897）：
- 数值奖励度量拟合保真度；
- 形式奖励基于「系数抽象后的规范形式」串相似度，度量结构级一致性。
用于 GRPO（Form-GRPO）与 DPO 偏好校验。
"""
from __future__ import annotations

import math
from difflib import SequenceMatcher

import numpy as np
import sympy as sp

from .normalize import canonical, parse_expression_text
from .symbols import make_symbols


def _isfinite(v: float) -> bool:
    return math.isfinite(v)


def _eval_points(expr: sp.Expr, vars_matrix: list[list[float]], var_symbols: list):
    fn = sp.lambdify(var_symbols, expr, "numpy")
    ys = []
    for i in range(len(vars_matrix[0])):
        point = [v[i] for v in vars_matrix]
        with np.errstate(all="ignore"):
            ys.append(float(np_re(fn(*point))))
    return ys


def np_re(x):
    """提取 numpy/scalar 数值的实部；非数值返回 NaN。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        try:
            return float(np.real(x))
        except Exception:
            return float("nan")


def rmse(pred_str: str, ref_ys: list[float], vars_matrix: list[list[float]],
         var_symbols: list) -> float:
    """候选表达式在给定数据上的均方根误差。无法解析/取值失败返回 inf。"""
    try:
        expr = parse_expression_text(pred_str)
        ys = _eval_points(expr, vars_matrix, var_symbols)
    except Exception:
        return float("inf")
    if any(not _isfinite(y) for y in ys) or any(not _isfinite(y) for y in ref_ys):
        return float("inf")
    # 按量级缩放后计算，再乘回比例，避免超大值平方溢出（RMSE 恢复绝对量纲）
    scale = max(abs(v) for v in ys + ref_ys) or 1.0
    errs = [(a / scale - b / scale) for a, b in zip(ys, ref_ys)]
    return scale * math.sqrt(sum(e * e for e in errs) / len(ys))


def r2(pred_str: str, ref_ys: list[float], vars_matrix: list[list[float]],
       var_symbols: list) -> float:
    """决定系数（尺度无关，缩放后计算避免溢出）。ref_ys 方差为 0 时退化处理。"""
    try:
        expr = parse_expression_text(pred_str)
        ys = _eval_points(expr, vars_matrix, var_symbols)
    except Exception:
        return float("-inf")
    if any(not _isfinite(v) for v in ys) or any(not _isfinite(v) for v in ref_ys):
        return float("-inf")
    scale = max(abs(v) for v in ys + ref_ys) or 1.0
    ys = [v / scale for v in ys]
    ref_ys = [v / scale for v in ref_ys]
    mean_ref = sum(ref_ys) / len(ref_ys)
    ss_tot = sum((y - mean_ref) ** 2 for y in ref_ys)
    if ss_tot == 0:
        return 1.0 if all(abs(a - b) < 1e-9 for a, b in zip(ys, ref_ys)) else float("-inf")
    ss_res = sum((a - b) ** 2 for a, b in zip(ys, ref_ys))
    return 1.0 - ss_res / ss_tot


def acc_tau(pred_str: str, ref_ys: list[float], vars_matrix: list[list[float]],
            var_symbols: list, tau: float = 0.05) -> float:
    """容差精度：|pred - ref| <= max(tau, tau*|ref|) 的比例。"""
    try:
        expr = parse_expression_text(pred_str)
        ys = _eval_points(expr, vars_matrix, var_symbols)
    except Exception:
        return 0.0
    good = 0
    for a, b in zip(ys, ref_ys):
        if _isfinite(a) and abs(a - b) <= max(tau, tau * abs(b)):
            good += 1
    return good / len(ref_ys)


def form_similarity(pred_str: str, ref_expr: sp.Expr) -> float:
    """形式一致性：对系数抽象后的规范形式计算字符串相似度（0~1）。"""
    try:
        p = parse_expression_text(pred_str)
    except Exception:
        return 0.0
    return SequenceMatcher(None, canonical(p), canonical(ref_expr)).ratio()


def compute_reward(pred_str: str, ref_expr: sp.Expr, vars_matrix: list[list[float]],
                   ref_ys: list[float], var_symbols: list,
                   w_num: float = 0.6, w_form: float = 0.4,
                   tau: float = 0.05) -> dict:
    """组合奖励字典。数值项与形式项都在 [0,1] 量级。"""
    r2v = r2(pred_str, ref_ys, vars_matrix, var_symbols)
    acc = acc_tau(pred_str, ref_ys, vars_matrix, var_symbols, tau)
    rms = rmse(pred_str, ref_ys, vars_matrix, var_symbols)
    form = form_similarity(pred_str, ref_expr)
    # 数值项用 R² 截断 + 容差精度
    num = max(r2v, 0.0) * 0.5 + acc * 0.5
    total = w_num * num + w_form * form
    return {
        "rmse": rms,
        "r2": r2v,
        "acc_tau": acc,
        "form": form,
        "num": num,
        "total": total,
    }