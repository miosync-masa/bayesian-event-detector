#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lambda3 Structural Benchmark Suite
==================================

A benchmark harness for *structural coupling analysis* rather than forecasting.
This version is intentionally designed to run from a *single file*.

How to use Lambda3 in this file
-------------------------------
1. Make sure your Lambda3 implementation is importable in the same Python session,
   OR edit `_try_import_lambda3_symbols()` to import from your own module.
2. Set `enable_lambda3=True` in `main()` (or call `main(enable_lambda3=True)`).
3. The function `example_lambda3_runner_stub()` is already a real bridge function.
   It will call your Lambda3 functions if they are available.

Expected Lambda3 symbols
------------------------
- L3Config
- calc_lambda3_features_v2
- fit_l3_bayesian_regression_asymmetric
- calculate_sync_profile

Design goals
------------
1. Align the benchmark with the paper's revised introduction:
   - detect structural coupling changes
   - detect asymmetric interactions
   - detect lag structure
   - detect regime-switch reorganization
2. Avoid unfairly assigning 0.0 to methods that do not natively output
   direction / lag / regime boundaries. Such metrics are marked as NaN (N/A).
3. Keep the suite modular while still allowing fully single-file execution.
"""

from __future__ import annotations

import os
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


from lambda3_benchmark.methods.lambda3_detectors import create_lambda3_detector

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt

try:
    import arviz as az
    _HAS_ARVIZ = True
except Exception:
    az = None
    _HAS_ARVIZ = False

try:
    from statsmodels.tsa.api import VAR
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

try:
    from pyinform.transferentropy import transfer_entropy as _pyinform_transfer_entropy
    _HAS_PYINFORM = True
except Exception:
    _pyinform_transfer_entropy = None
    _HAS_PYINFORM = False

try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    _HAS_TIGRAMITE = True
except Exception:
    pp = None
    PCMCI = None
    ParCorr = None
    _HAS_TIGRAMITE = False


# ---------------------------------------------------------------------
# Optional Lambda3 symbol resolver
# ---------------------------------------------------------------------

def _try_import_lambda3_symbols() -> Dict[str, Any]:
    """
    Resolve Lambda3 symbols.

    Priority:
    1. Already present in globals() (e.g. notebook session)
    2. Optional module imports (edit this list to match your environment)

    Returns
    -------
    dict containing any resolved symbols.
    """
    needed = [
        "L3Config",
        "calc_lambda3_features_v2",
        "fit_l3_bayesian_regression_asymmetric",
        "calculate_sync_profile",
    ]

    resolved: Dict[str, Any] = {}

    # 1) from current globals
    g = globals()
    for name in needed:
        if name in g:
            resolved[name] = g[name]

    if len(resolved) == len(needed):
        return resolved

    # 2) optional module imports
    candidate_modules = [
        "lambda3_core",
        "lambda3_main",
        "lambda3",
        "lambda3_analytics",
        "lambda3_financial",
    ]

    for module_name in candidate_modules:
        try:
            mod = __import__(module_name, fromlist=needed)
            for name in needed:
                if name not in resolved and hasattr(mod, name):
                    resolved[name] = getattr(mod, name)
        except Exception:
            pass

        if len(resolved) == len(needed):
            break

    return resolved


def _require_lambda3_symbols() -> Dict[str, Any]:
    resolved = _try_import_lambda3_symbols()
    needed = [
        "L3Config",
        "calc_lambda3_features_v2",
        "fit_l3_bayesian_regression_asymmetric",
        "calculate_sync_profile",
    ]
    missing = [name for name in needed if name not in resolved]

    if missing:
        raise RuntimeError(
            "Lambda3 symbols could not be resolved. Missing: "
            + ", ".join(missing)
            + "\nEither run this file in the same session as your Lambda3 code, "
              "or edit `_try_import_lambda3_symbols()` to import from your module."
        )
    if not _HAS_ARVIZ:
        raise RuntimeError("arviz is required for the Lambda3 adapter path.")
    return resolved


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """Configuration shared by scenario generators."""
    T: int = 400
    burn_in: int = 100
    n_series: int = 3
    max_lag: int = 8
    noise_scale: float = 0.35
    regime_split: int = 200
    event_percentile: float = 97.0
    zero_inflation_prob: float = 0.88
    seed: int = 42


@dataclass
class GroundTruth:
    """
    Structural ground truth for one synthetic sample.
    """
    names: List[str]
    adjacency: np.ndarray                  # directed binary N x N
    lag_matrix: Optional[np.ndarray] = None
    sign_matrix: Optional[np.ndarray] = None  # -1, 0, +1
    notes: str = ""

    # Optional structural / scenario annotations
    regime_boundaries: Optional[List[int]] = None
    adjacency_by_regime: Optional[List[np.ndarray]] = None
    lag_by_regime: Optional[List[np.ndarray]] = None
    sign_by_regime: Optional[List[np.ndarray]] = None
    low_corr_edges: Optional[List[Tuple[int, int]]] = None
    forbidden_edges: Optional[List[Tuple[int, int]]] = None

    def undirected_adjacency(self) -> np.ndarray:
        return ((self.adjacency + self.adjacency.T) > 0).astype(int)


@dataclass
class MethodOutput:
    """
    Standardized output from any method adapter.
    """
    method_name: str
    names: List[str]
    adjacency_scores: Optional[np.ndarray] = None
    adjacency_bin: Optional[np.ndarray] = None

    directed_support: bool = True
    lag_support: bool = False
    sign_support: bool = False
    regime_support: bool = False

    lag_matrix: Optional[np.ndarray] = None
    sign_matrix: Optional[np.ndarray] = None

    regime_boundaries: Optional[List[int]] = None
    adjacency_by_regime: Optional[List[np.ndarray]] = None
    lag_by_regime: Optional[List[np.ndarray]] = None
    sign_by_regime: Optional[List[np.ndarray]] = None

    meta: Dict[str, Any] = field(default_factory=dict)

    def undirected_bin(self) -> Optional[np.ndarray]:
        if self.adjacency_bin is None:
            return None
        return ((self.adjacency_bin + self.adjacency_bin.T) > 0).astype(int)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x


def _diff_events(x: np.ndarray, percentile: float = 97.0) -> np.ndarray:
    """
    Event indicator from absolute first differences.
    """
    d = np.diff(x, prepend=x[0])
    thr = np.percentile(np.abs(d), percentile)
    return (np.abs(d) > thr).astype(int)


def _pairwise_corr(x: np.ndarray) -> np.ndarray:
    return np.corrcoef(x.T)


def _shifted_overlap(a: np.ndarray, b: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positive lag means b lags a, i.e., compare a[t] with b[t+lag].
    """
    if lag > 0:
        return a[:-lag], b[lag:]
    if lag < 0:
        return a[-lag:], b[:lag]
    return a, b


def _quantile_discretize(x: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """
    Quantile discretization robust to ties.
    """
    x = np.asarray(x)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    # make strictly increasing if ties appear
    edges = np.unique(edges)
    if len(edges) <= 2:
        # fallback to uniform bins if quantiles collapse
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmin == xmax:
            return np.zeros_like(x, dtype=int)
        edges = np.linspace(xmin, xmax, n_bins + 1)
    bins = np.digitize(x, edges[1:-1], right=False)
    return bins.astype(int)


def _te_discrete(source: np.ndarray, target: np.ndarray, lag: int = 1, n_bins: int = 4) -> float:
    """
    Simple discrete transfer entropy estimator:
        TE_{X->Y}(lag) = I(X_{t-lag}; Y_t | Y_{t-1})
    using empirical frequencies.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")

    x = _quantile_discretize(source, n_bins=n_bins)
    y = _quantile_discretize(target, n_bins=n_bins)

    t0 = max(lag, 1)
    y_t = y[t0:]
    y_prev = y[t0 - 1:-1]
    x_lag = x[t0 - lag: len(x) - lag]

    if not (len(y_t) == len(y_prev) == len(x_lag)):
        n = min(len(y_t), len(y_prev), len(x_lag))
        y_t = y_t[:n]
        y_prev = y_prev[:n]
        x_lag = x_lag[:n]

    n = len(y_t)
    if n <= 5:
        return 0.0

    from collections import Counter
    c_xyz = Counter(zip(x_lag.tolist(), y_prev.tolist(), y_t.tolist()))
    c_xy = Counter(zip(x_lag.tolist(), y_prev.tolist()))
    c_yz = Counter(zip(y_prev.tolist(), y_t.tolist()))
    c_y = Counter(y_prev.tolist())

    te = 0.0
    for (xs, yp, yt), c in c_xyz.items():
        p_xyz = c / n
        p_yt_given_xyp = c / c_xy[(xs, yp)]
        p_yt_given_yp = c_yz[(yp, yt)] / c_y[yp]
        if p_yt_given_xyp > 0 and p_yt_given_yp > 0:
            te += p_xyz * math.log(p_yt_given_xyp / p_yt_given_yp)
    return float(te)


def _te_pyinform(source: np.ndarray, target: np.ndarray, lag: int = 1, n_bins: int = 4, k: int = 1) -> float:
    """
    Transfer entropy via PyInform on lag-aligned, discretized series.
    Positive lag means target lags source.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if not _HAS_PYINFORM:
        raise RuntimeError("pyinform is not installed")

    xs, ys = _shifted_overlap(np.asarray(source), np.asarray(target), lag)
    xs = _quantile_discretize(xs, n_bins=n_bins).astype(int).tolist()
    ys = _quantile_discretize(ys, n_bins=n_bins).astype(int).tolist()
    if len(xs) <= max(k + 2, 8) or len(ys) <= max(k + 2, 8):
        return 0.0
    try:
        te = _pyinform_transfer_entropy(xs, ys, k=k)
        if np.isnan(te) or np.isinf(te):
            return 0.0
        return float(te)
    except Exception:
        return 0.0


def _event_sync_score(a_events: np.ndarray, b_events: np.ndarray, lag: int) -> float:
    """
    Event overlap score for ordered pairs.
    Positive lag means b lags a, so this evaluates a -> b.
    """
    aa, bb = _shifted_overlap(a_events, b_events, lag)
    if len(aa) == 0:
        return 0.0
    return float(np.mean(aa * bb))


def _binarize_adjacency_from_scores(
    scores: np.ndarray,
    threshold: Optional[float] = None,
    percentile: Optional[float] = None,
    symmetric: bool = False,
) -> np.ndarray:
    s = np.asarray(scores).copy()
    np.fill_diagonal(s, 0.0)
    vals = s[~np.eye(s.shape[0], dtype=bool)]
    if threshold is None:
        if percentile is None:
            percentile = 75.0
        threshold = float(np.percentile(vals, percentile)) if len(vals) else 0.0
    out = (s >= threshold).astype(int)
    np.fill_diagonal(out, 0)
    if symmetric:
        out = ((out + out.T) > 0).astype(int)
        np.fill_diagonal(out, 0)
    return out


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _boundary_f1(true_boundaries: Sequence[int], pred_boundaries: Sequence[int], tolerance: int = 5) -> float:
    """
    A tolerant boundary F1 score.
    """
    true_boundaries = list(true_boundaries or [])
    pred_boundaries = list(pred_boundaries or [])

    if not true_boundaries and not pred_boundaries:
        return 1.0
    if not true_boundaries or not pred_boundaries:
        return 0.0

    matched_true = set()
    matched_pred = set()

    for i, tb in enumerate(true_boundaries):
        for j, pb in enumerate(pred_boundaries):
            if abs(tb - pb) <= tolerance and j not in matched_pred:
                matched_true.add(i)
                matched_pred.add(j)
                break

    tp = len(matched_true)
    fp = len(pred_boundaries) - tp
    fn = len(true_boundaries) - tp

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------
# Synthetic structural scenarios
# ---------------------------------------------------------------------

def generate_null_independent(cfg: ScenarioConfig, seed: int) -> Tuple[pd.DataFrame, GroundTruth]:
    rng = _rng(seed)
    n = cfg.n_series
    T = cfg.T + cfg.burn_in
    x = np.zeros((T, n))
    for t in range(1, T):
        x[t] = 0.65 * x[t - 1] + rng.normal(scale=cfg.noise_scale, size=n)
    x = x[cfg.burn_in:]
    names = [f"X{i+1}" for i in range(n)]
    gt = GroundTruth(
        names=names,
        adjacency=np.zeros((n, n), dtype=int),
        lag_matrix=np.zeros((n, n), dtype=int),
        sign_matrix=np.zeros((n, n), dtype=int),
        notes="Null independent AR(1) system.",
    )
    return pd.DataFrame(x, columns=names), gt


def generate_delayed_directional(cfg: ScenarioConfig, seed: int) -> Tuple[pd.DataFrame, GroundTruth]:
    """
    S1: One delayed directional edge A -> B.
    """
    rng = _rng(seed)
    T = cfg.T + cfg.burn_in
    lag = 3
    x = np.zeros((T, 3))
    for t in range(1, T):
        x[t, 0] = 0.75 * x[t - 1, 0] + rng.normal(scale=cfg.noise_scale)
        x[t, 2] = 0.55 * x[t - 1, 2] + rng.normal(scale=cfg.noise_scale)
        effect = 0.0
        if t - lag >= 0:
            effect = 1.15 * x[t - lag, 0]
        x[t, 1] = 0.60 * x[t - 1, 1] + effect + 0.15 * x[t - 1, 2] + rng.normal(scale=cfg.noise_scale)
    x = x[cfg.burn_in:]
    names = ["A", "B", "C"]
    adj = np.zeros((3, 3), dtype=int)
    lagm = np.zeros((3, 3), dtype=int)
    signm = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    lagm[0, 1] = lag
    signm[0, 1] = 1
    gt = GroundTruth(names=names, adjacency=adj, lag_matrix=lagm, sign_matrix=signm,
                     notes="Delayed directional coupling A -> B.")
    return pd.DataFrame(x, columns=names), gt


def generate_asymmetric_coupling(cfg: ScenarioConfig, seed: int) -> Tuple[pd.DataFrame, GroundTruth]:
    """
    S2: A strongly drives B, reverse effect negligible.
    """
    rng = _rng(seed)
    T = cfg.T + cfg.burn_in
    x = np.zeros((T, 3))
    for t in range(1, T):
        x[t, 0] = 0.70 * x[t - 1, 0] + rng.normal(scale=cfg.noise_scale)
        strong = 1.35 * x[t - 1, 0]
        weak_reverse = 0.08 * x[t - 1, 1]
        x[t, 1] = 0.45 * x[t - 1, 1] + strong + rng.normal(scale=cfg.noise_scale)
        x[t, 0] += weak_reverse
        x[t, 2] = 0.60 * x[t - 1, 2] + rng.normal(scale=cfg.noise_scale)
    x = x[cfg.burn_in:]
    names = ["A", "B", "Noise"]
    adj = np.zeros((3, 3), dtype=int)
    lagm = np.zeros((3, 3), dtype=int)
    signm = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    lagm[0, 1] = 1
    signm[0, 1] = 1
    gt = GroundTruth(names=names, adjacency=adj, lag_matrix=lagm, sign_matrix=signm,
                     notes="Asymmetric coupling A -> B only (strong); reverse negligible.")
    return pd.DataFrame(x, columns=names), gt


def generate_hidden_lowcorr(cfg: ScenarioConfig, seed: int) -> Tuple[pd.DataFrame, GroundTruth]:
    """
    S3: A drives rare events in B, but global Pearson correlation stays low.
    """
    rng = _rng(seed)
    T = cfg.T + cfg.burn_in
    lag = 2
    x = np.zeros((T, 3))
    base_phase = rng.uniform(0, 2 * np.pi)
    for t in range(1, T):
        x[t, 0] = 0.72 * x[t - 1, 0] + rng.normal(scale=cfg.noise_scale)
        if rng.random() < 0.04:
            x[t, 0] += rng.normal(loc=2.8, scale=0.4)
        oscillatory = 1.1 * np.sin(0.14 * t + base_phase)
        event_effect = 0.0
        if t - lag >= 0:
            jump = abs(x[t - lag, 0] - x[t - lag - 1, 0]) if t - lag - 1 >= 0 else 0.0
            event_effect = 2.2 if jump > np.percentile(np.abs(np.diff(x[max(0, t-50):t+1, 0], prepend=x[max(0, t-50), 0])), 90) else 0.0
        x[t, 1] = 0.20 * x[t - 1, 1] + oscillatory + event_effect + rng.normal(scale=0.55)
        x[t, 2] = 0.60 * x[t - 1, 2] + rng.normal(scale=cfg.noise_scale)
    x = x[cfg.burn_in:]
    names = ["A", "B", "C"]
    adj = np.zeros((3, 3), dtype=int)
    lagm = np.zeros((3, 3), dtype=int)
    signm = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    lagm[0, 1] = lag
    signm[0, 1] = 1
    gt = GroundTruth(
        names=names,
        adjacency=adj,
        lag_matrix=lagm,
        sign_matrix=signm,
        low_corr_edges=[(0, 1)],
        notes="Hidden coupling: A drives rare structural events in B with low global correlation.",
    )
    return pd.DataFrame(x, columns=names), gt


def generate_regime_switch(cfg: ScenarioConfig, seed: int) -> Tuple[pd.DataFrame, GroundTruth]:
    """
    S4: Same observable target B, but the driving structure changes by regime.
    Regime 1: A -> B (positive, lag 2)
    Regime 2: C -> B (negative, lag 1)
    """
    rng = _rng(seed)
    T = cfg.T + cfg.burn_in
    split = cfg.regime_split + cfg.burn_in
    x = np.zeros((T, 3))
    for t in range(1, T):
        x[t, 0] = 0.70 * x[t - 1, 0] + rng.normal(scale=cfg.noise_scale)
        x[t, 2] = 0.70 * x[t - 1, 2] + rng.normal(scale=cfg.noise_scale)

        if t < split:
            effect = 1.10 * x[t - 2, 0] if t - 2 >= 0 else 0.0
        else:
            effect = -1.15 * x[t - 1, 2] if t - 1 >= 0 else 0.0

        x[t, 1] = 0.55 * x[t - 1, 1] + effect + rng.normal(scale=cfg.noise_scale)

    x = x[cfg.burn_in:]
    names = ["A", "B", "C"]

    adj_global = np.zeros((3, 3), dtype=int)
    adj_global[0, 1] = 1
    adj_global[2, 1] = 1
    lag_global = np.zeros((3, 3), dtype=int)
    lag_global[0, 1] = 2
    lag_global[2, 1] = 1
    sign_global = np.zeros((3, 3), dtype=int)
    sign_global[0, 1] = 1
    sign_global[2, 1] = -1

    adj_r1 = np.zeros((3, 3), dtype=int)
    adj_r1[0, 1] = 1
    lag_r1 = np.zeros((3, 3), dtype=int)
    lag_r1[0, 1] = 2
    sign_r1 = np.zeros((3, 3), dtype=int)
    sign_r1[0, 1] = 1

    adj_r2 = np.zeros((3, 3), dtype=int)
    adj_r2[2, 1] = 1
    lag_r2 = np.zeros((3, 3), dtype=int)
    lag_r2[2, 1] = 1
    sign_r2 = np.zeros((3, 3), dtype=int)
    sign_r2[2, 1] = -1

    gt = GroundTruth(
        names=names,
        adjacency=adj_global,
        lag_matrix=lag_global,
        sign_matrix=sign_global,
        regime_boundaries=[cfg.regime_split],
        adjacency_by_regime=[adj_r1, adj_r2],
        lag_by_regime=[lag_r1, lag_r2],
        sign_by_regime=[sign_r1, sign_r2],
        notes="Regime switch: A->B in regime 1, C->B in regime 2.",
    )
    return pd.DataFrame(x, columns=names), gt


def generate_confounder(cfg: ScenarioConfig, seed: int) -> Tuple[pd.DataFrame, GroundTruth]:
    """
    S5: Common driver C drives both A and B, but no direct A<->B edge.
    """
    rng = _rng(seed)
    T = cfg.T + cfg.burn_in
    x = np.zeros((T, 3))
    for t in range(1, T):
        x[t, 2] = 0.80 * x[t - 1, 2] + rng.normal(scale=cfg.noise_scale)  # driver C
        x[t, 0] = 0.50 * x[t - 1, 0] + 1.20 * x[t - 1, 2] + rng.normal(scale=cfg.noise_scale)
        x[t, 1] = 0.55 * x[t - 1, 1] - 1.05 * x[t - 2, 2] if t - 2 >= 0 else 0.0
        x[t, 1] += rng.normal(scale=cfg.noise_scale)
    x = x[cfg.burn_in:]
    names = ["A", "B", "C"]
    adj = np.zeros((3, 3), dtype=int)
    lagm = np.zeros((3, 3), dtype=int)
    signm = np.zeros((3, 3), dtype=int)
    adj[2, 0] = 1
    adj[2, 1] = 1
    lagm[2, 0] = 1
    lagm[2, 1] = 2
    signm[2, 0] = 1
    signm[2, 1] = -1
    gt = GroundTruth(
        names=names,
        adjacency=adj,
        lag_matrix=lagm,
        sign_matrix=signm,
        forbidden_edges=[(0, 1), (1, 0)],
        notes="Common-driver confounding: C -> A and C -> B, but no direct A-B edge.",
    )
    return pd.DataFrame(x, columns=names), gt


def generate_sparse_event(cfg: ScenarioConfig, seed: int) -> Tuple[pd.DataFrame, GroundTruth]:
    """
    S6: Sparse / zero-inflated event-like variable B.
    A drives the latent state of B, but B is mostly zero.
    """
    rng = _rng(seed)
    T = cfg.T + cfg.burn_in
    x = np.zeros((T, 3))
    latent = np.zeros(T)
    for t in range(1, T):
        x[t, 0] = 0.72 * x[t - 1, 0] + rng.normal(scale=cfg.noise_scale)
        x[t, 2] = 0.60 * x[t - 1, 2] + rng.normal(scale=cfg.noise_scale)
        latent[t] = 0.45 * latent[t - 1] + 1.25 * x[t - 1, 0] + rng.normal(scale=0.45)
        p_nonzero = 1 / (1 + np.exp(-(latent[t] - 1.4)))
        if rng.random() < (1 - cfg.zero_inflation_prob) * p_nonzero:
            x[t, 1] = max(0.0, latent[t] + rng.normal(scale=0.25))
        else:
            x[t, 1] = 0.0
    x = x[cfg.burn_in:]
    names = ["Driver", "SparseEvent", "Noise"]
    adj = np.zeros((3, 3), dtype=int)
    lagm = np.zeros((3, 3), dtype=int)
    signm = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    lagm[0, 1] = 1
    signm[0, 1] = 1
    gt = GroundTruth(
        names=names,
        adjacency=adj,
        lag_matrix=lagm,
        sign_matrix=signm,
        notes="Sparse zero-inflated event variable driven by A.",
    )
    return pd.DataFrame(x, columns=names), gt


SCENARIOS: Dict[str, Callable[[ScenarioConfig, int], Tuple[pd.DataFrame, GroundTruth]]] = {
    "S0_null_independent": generate_null_independent,
    "S1_delayed_directional": generate_delayed_directional,
    "S2_asymmetric": generate_asymmetric_coupling,
    "S3_hidden_lowcorr": generate_hidden_lowcorr,
    "S4_regime_switch": generate_regime_switch,
    "S5_common_driver": generate_confounder,
    "S6_sparse_event": generate_sparse_event,
}


# ---------------------------------------------------------------------
# Method adapters
# ---------------------------------------------------------------------

class BaseAdapter:
    method_name: str = "Base"

    def fit(self, df: pd.DataFrame, cfg: ScenarioConfig) -> MethodOutput:
        raise NotImplementedError


class VARGrangerAdapter(BaseAdapter):
    method_name = "VAR_Granger"

    def __init__(self, maxlags: int = 8, alpha: float = 0.05, ic: str = "aic"):
        self.maxlags = maxlags
        self.alpha = alpha
        self.ic = ic

    def fit(self, df: pd.DataFrame, cfg: ScenarioConfig) -> MethodOutput:
        if not _HAS_STATSMODELS:
            raise RuntimeError("statsmodels is required for VARGrangerAdapter.")

        names = list(df.columns)
        n = len(names)
        x = df.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(x)
            results = model.fit(maxlags=min(self.maxlags, max(1, len(df) // 10)), ic=self.ic)

        pmat = np.ones((n, n))
        scores = np.zeros((n, n))
        adjacency = np.zeros((n, n), dtype=int)
        lagm = np.zeros((n, n), dtype=int)
        signm = np.zeros((n, n), dtype=int)

        params = results.params.copy()

        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                if i == j:
                    continue
                try:
                    test = results.test_causality(caused=dst, causing=[src], kind='f')
                    pval = float(test.pvalue)
                except Exception:
                    pval = 1.0

                pmat[i, j] = pval
                scores[i, j] = -math.log10(max(pval, 1e-12))
                adjacency[i, j] = int(pval < self.alpha)

                coef_vals = []
                coef_lags = []
                for lag in range(1, results.k_ar + 1):
                    row_name = f"L{lag}.{src}"
                    if row_name in params.index and dst in params.columns:
                        v = float(params.loc[row_name, dst])
                        coef_vals.append(v)
                        coef_lags.append(lag)
                if coef_vals:
                    idx = int(np.argmax(np.abs(coef_vals)))
                    lagm[i, j] = coef_lags[idx]
                    signm[i, j] = int(np.sign(coef_vals[idx]))

        np.fill_diagonal(scores, 0.0)
        np.fill_diagonal(adjacency, 0)
        np.fill_diagonal(lagm, 0)
        np.fill_diagonal(signm, 0)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=adjacency,
            directed_support=True,
            lag_support=True,
            sign_support=True,
            regime_support=False,
            lag_matrix=lagm,
            sign_matrix=signm,
            meta={"selected_order": results.k_ar, "pvalues": pmat},
        )


class TransferEntropyAdapter(BaseAdapter):
    method_name = "TransferEntropy"

    def __init__(
        self,
        max_lag: int = 8,
        alpha: float = 0.05,
        n_bins: int = 4,
        n_perm: int = 30,
        backend: str = "pyinform",
        k_history: int = 1,
    ):
        self.max_lag = max_lag
        self.alpha = alpha
        self.n_bins = n_bins
        self.n_perm = n_perm
        self.backend = backend
        self.k_history = k_history

    def _score(self, source: np.ndarray, target: np.ndarray, lag: int) -> float:
        if self.backend == "pyinform":
            if _HAS_PYINFORM:
                return _te_pyinform(source, target, lag=lag, n_bins=self.n_bins, k=self.k_history)
            warnings.warn("pyinform not installed; falling back to internal discrete TE estimator.")
            return _te_discrete(source, target, lag=lag, n_bins=self.n_bins)
        elif self.backend == "discrete":
            return _te_discrete(source, target, lag=lag, n_bins=self.n_bins)
        else:
            raise ValueError(f"Unknown TE backend: {self.backend}")

    def fit(self, df: pd.DataFrame, cfg: ScenarioConfig) -> MethodOutput:
        names = list(df.columns)
        n = len(names)
        x = df.to_numpy()
        rng = _rng(12345)

        scores = np.zeros((n, n))
        adjacency = np.zeros((n, n), dtype=int)
        lagm = np.zeros((n, n), dtype=int)
        pvals = np.ones((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                best_te = -np.inf
                best_lag = 1
                for lag in range(1, min(self.max_lag, len(df) - 2) + 1):
                    te = self._score(x[:, i], x[:, j], lag=lag)
                    if te > best_te:
                        best_te = te
                        best_lag = lag

                null_scores = []
                for _ in range(self.n_perm):
                    perm_src = rng.permutation(x[:, i])
                    null_scores.append(self._score(perm_src, x[:, j], lag=best_lag))
                null_scores = np.asarray(null_scores)
                pval = (1 + np.sum(null_scores >= best_te)) / (self.n_perm + 1)

                scores[i, j] = max(best_te, 0.0)
                lagm[i, j] = best_lag
                pvals[i, j] = pval
                adjacency[i, j] = int(pval < self.alpha)

        np.fill_diagonal(scores, 0.0)
        np.fill_diagonal(adjacency, 0)
        np.fill_diagonal(lagm, 0)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=adjacency,
            directed_support=True,
            lag_support=True,
            sign_support=False,
            regime_support=False,
            lag_matrix=lagm,
            sign_matrix=None,
            meta={
                "pvalues": pvals,
                "n_bins": self.n_bins,
                "n_perm": self.n_perm,
                "backend": self.backend,
                "k_history": self.k_history,
                "pyinform_available": _HAS_PYINFORM,
            },
        )


class PCMCIPlusAdapter(BaseAdapter):
    method_name = "PCMCIPlus"

    def __init__(self, tau_max: int = 8, pc_alpha: float = 0.05, verbosity: int = 0):
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.verbosity = verbosity

    def fit(self, df: pd.DataFrame, cfg: ScenarioConfig) -> MethodOutput:
        if not _HAS_TIGRAMITE:
            raise RuntimeError(
                "tigramite is not installed. Install it with `pip install tigramite`."
            )

        names = list(df.columns)
        x = df.to_numpy(dtype=float)
        n = len(names)

        tg_df = pp.DataFrame(x, var_names=names)
        pcmci = PCMCI(dataframe=tg_df, cond_ind_test=ParCorr(significance='analytic'), verbosity=self.verbosity)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=self.tau_max, pc_alpha=self.pc_alpha)

        graph = results.get("graph")
        val_matrix = results.get("val_matrix")
        p_matrix = results.get("p_matrix")

        adjacency = np.zeros((n, n), dtype=int)
        scores = np.zeros((n, n), dtype=float)
        lagm = np.zeros((n, n), dtype=int)
        signm = np.zeros((n, n), dtype=int)

        if graph is not None:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    best_abs = 0.0
                    best_tau = 0
                    best_sign = 0
                    found = False
                    # ignore tau=0 for this structural-lag benchmark
                    for tau in range(1, min(self.tau_max, graph.shape[2] - 1) + 1):
                        g = graph[i, j, tau]
                        if isinstance(g, bytes):
                            g = g.decode()
                        if g is None:
                            g = ""
                        if str(g).strip() != "":
                            found = True
                            val = 0.0
                            if val_matrix is not None:
                                try:
                                    val = float(val_matrix[i, j, tau])
                                except Exception:
                                    val = 0.0
                            if abs(val) >= best_abs:
                                best_abs = abs(val)
                                best_tau = tau
                                best_sign = int(np.sign(val)) if val != 0 else 0
                    adjacency[i, j] = int(found)
                    scores[i, j] = best_abs
                    lagm[i, j] = best_tau
                    signm[i, j] = best_sign

        np.fill_diagonal(adjacency, 0)
        np.fill_diagonal(scores, 0.0)
        np.fill_diagonal(lagm, 0)
        np.fill_diagonal(signm, 0)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=adjacency,
            directed_support=True,
            lag_support=True,
            sign_support=True,
            regime_support=False,
            lag_matrix=lagm,
            sign_matrix=signm,
            meta={
                "pc_alpha": self.pc_alpha,
                "tau_max": self.tau_max,
                "p_matrix": p_matrix,
                "backend": "tigramite_pcmciplus",
            },
        )


class GraphicalLassoAdapter(BaseAdapter):
    method_name = "GraphicalLasso"

    def __init__(self, edge_threshold: float = 0.03):
        self.edge_threshold = edge_threshold

    def fit(self, df: pd.DataFrame, cfg: ScenarioConfig) -> MethodOutput:
        names = list(df.columns)
        x = StandardScaler().fit_transform(df.to_numpy())
        model = GraphicalLassoCV()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x)

        precision = np.abs(model.precision_)
        np.fill_diagonal(precision, 0.0)
        adjacency = (precision >= self.edge_threshold).astype(int)
        adjacency = ((adjacency + adjacency.T) > 0).astype(int)
        np.fill_diagonal(adjacency, 0)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=precision,
            adjacency_bin=adjacency,
            directed_support=False,
            lag_support=False,
            sign_support=False,
            regime_support=False,
            lag_matrix=None,
            sign_matrix=None,
            meta={"alpha_": getattr(model, "alpha_", None)},
        )


class EventCrossCorrelationAdapter(BaseAdapter):
    method_name = "EventXCorr"

    def __init__(self, max_lag: int = 8, event_percentile: float = 97.0, alpha: float = 0.05, n_perm: int = 30):
        self.max_lag = max_lag
        self.event_percentile = event_percentile
        self.alpha = alpha
        self.n_perm = n_perm

    def fit(self, df: pd.DataFrame, cfg: ScenarioConfig) -> MethodOutput:
        names = list(df.columns)
        n = len(names)
        x = df.to_numpy()
        rng = _rng(54321)

        events = np.column_stack([_diff_events(x[:, i], percentile=self.event_percentile) for i in range(n)])
        scores = np.zeros((n, n))
        adjacency = np.zeros((n, n), dtype=int)
        lagm = np.zeros((n, n), dtype=int)
        pvals = np.ones((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                best_score = -np.inf
                best_lag = 1
                for lag in range(1, min(self.max_lag, len(df) - 2) + 1):
                    s = _event_sync_score(events[:, i], events[:, j], lag=lag)
                    if s > best_score:
                        best_score = s
                        best_lag = lag
                null_scores = []
                for _ in range(self.n_perm):
                    perm_src = rng.permutation(events[:, i])
                    null_scores.append(_event_sync_score(perm_src, events[:, j], lag=best_lag))
                null_scores = np.asarray(null_scores)
                pval = (1 + np.sum(null_scores >= best_score)) / (self.n_perm + 1)

                scores[i, j] = max(best_score, 0.0)
                lagm[i, j] = best_lag
                pvals[i, j] = pval
                adjacency[i, j] = int(pval < self.alpha)

        np.fill_diagonal(scores, 0.0)
        np.fill_diagonal(adjacency, 0)
        np.fill_diagonal(lagm, 0)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=adjacency,
            directed_support=True,
            lag_support=True,
            sign_support=False,
            regime_support=False,
            lag_matrix=lagm,
            sign_matrix=None,
            meta={"event_percentile": self.event_percentile, "pvalues": pvals},
        )

def make_lambda3_adapter(
    runner: Callable[[pd.DataFrame, ScenarioConfig], Dict[str, Any]],
    method_name: str = "Lambda3",
) -> BaseAdapter:
    """
    Create an adapter from the user's Lambda3 runner.
    Parallelized version for A100 / high-core environments.
    """
    class _Lambda3Adapter(BaseAdapter):
        def __init__(self):
            self.method_name = method_name

        def fit(self, df: pd.DataFrame, cfg: ScenarioConfig) -> MethodOutput:
            """
            Lambda3 adapter fit (parallel version)

            Key features
            -------------
            1) unordered pair only: (i, j) with i < j
            2) one detector call per pair
            3) ThreadPoolExecutor for parallel MCMC across pairs
            4) forward/backward written separately to adjacency
            5) dominant-direction rule avoids spurious bidirectional edges
            """
            from concurrent.futures import ThreadPoolExecutor, as_completed

            names = list(df.columns)
            n = len(names)

            scores = np.zeros((n, n), dtype=float)
            adjacency = np.zeros((n, n), dtype=int)
            lagm = np.zeros((n, n), dtype=int)
            signm = np.zeros((n, n), dtype=int)

            # tune these if needed
            edge_threshold = getattr(self, "edge_threshold", 0.25)
            asymmetry_ratio_threshold = getattr(self, "asymmetry_ratio_threshold", 1.5)
            detector_mode = getattr(self, "detector_mode", "bidirectional")
            verbose = getattr(self, "verbose", False)

            def _extract_directional(result):
                """
                Normalize result schema from Lambda3 detectors.
                """
                # Bidirectional detector
                if "forward" in result and "backward" in result:
                    fwd = result["forward"]
                    bwd = result["backward"]
                    fwd_beta = float(fwd.get("beta_lambda", fwd.get("beta", 0.0)))
                    bwd_beta = float(bwd.get("beta_lambda", bwd.get("beta", 0.0)))
                    fwd_lag = int(fwd.get("lag", result.get("lag", 0)))
                    bwd_lag = int(bwd.get("lag", result.get("lag", 0)))
                    return fwd_beta, bwd_beta, fwd_lag, bwd_lag

                # Hierarchical detector
                if "pairwise_results" in result and result["pairwise_results"] is not None:
                    pr = result["pairwise_results"]
                    fwd = pr.get("forward", {})
                    bwd = pr.get("backward", {})
                    fwd_beta = float(fwd.get("beta_lambda", fwd.get("beta", 0.0)))
                    bwd_beta = float(bwd.get("beta_lambda", bwd.get("beta", 0.0)))
                    main_lag = int(result.get("lag", 0))
                    fwd_lag = int(fwd.get("lag", main_lag))
                    bwd_lag = int(bwd.get("lag", main_lag))
                    return fwd_beta, bwd_beta, fwd_lag, bwd_lag

                return 0.0, 0.0, 0, 0

            # ---------------------------------------------------------
            # Build pair list
            # ---------------------------------------------------------
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

            def _run_pair(pair_tuple):
                i, j = pair_tuple
                pair_data = df.iloc[:, [i, j]].to_numpy()
                local_detector = create_lambda3_detector(
                    mode=detector_mode,
                    verbose=verbose
                )
                try:
                    result = local_detector.detect(pair_data)

                    # ★★★ デバッグ：returnの前！★★★
                    print(f"=== DEBUG: pair ({names[i]}, {names[j]}) ===")
                    print(f"result keys: {list(result.keys())}")
                    if "forward" in result:
                        print(f"  forward beta: {result['forward'].get('beta', 'N/A')}")
                        print(f"  forward beta_lambda: {result['forward'].get('beta_lambda', 'N/A')}")
                        print(f"  backward beta: {result['backward'].get('beta', 'N/A')}")
                    if "primary_beta" in result:
                        print(f"  primary_beta: {result['primary_beta']}")

                    fwd_beta, bwd_beta, fwd_lag, bwd_lag = _extract_directional(result)
                    print(f"  extracted: fwd={fwd_beta:.4f}, bwd={bwd_beta:.4f}, edge_threshold={edge_threshold}")
                    
                    return (i, j, fwd_beta, bwd_beta, fwd_lag, bwd_lag, True)
                except Exception as e:
                    if verbose:
                        print(f"[{self.method_name}] pair ({names[i]}, {names[j]}) failed: {e}")
                    return (i, j, 0.0, 0.0, 0, 0, False) 

            # ---------------------------------------------------------
            # Parallel execution
            # ---------------------------------------------------------
            max_workers = min(len(pairs), 6)  # A100: safe up to 6
            pair_results = []

            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_run_pair, p): p for p in pairs}
                    for future in as_completed(futures):
                        pair_results.append(future.result())
            else:
                # Single pair: no need for thread overhead
                for p in pairs:
                    pair_results.append(_run_pair(p))

            # ---------------------------------------------------------
            # Write results to adjacency matrix
            # ---------------------------------------------------------
            for (i, j, beta_ij, beta_ji, lag_ij, lag_ji, success) in pair_results:
                if not success:
                    continue

                a_ij = abs(beta_ij)
                a_ji = abs(beta_ji)

                if a_ij < edge_threshold and a_ji < edge_threshold:
                    continue

                # strong asymmetry => keep only dominant direction
                if a_ij >= edge_threshold and a_ij >= asymmetry_ratio_threshold * max(a_ji, 1e-12):
                    adjacency[i, j] = 1
                    scores[i, j] = a_ij
                    lagm[i, j] = lag_ij
                    signm[i, j] = int(np.sign(beta_ij))

                elif a_ji >= edge_threshold and a_ji >= asymmetry_ratio_threshold * max(a_ij, 1e-12):
                    adjacency[j, i] = 1
                    scores[j, i] = a_ji
                    lagm[j, i] = lag_ji
                    signm[j, i] = int(np.sign(beta_ji))

                else:
                    # near-symmetric: keep both if above threshold
                    if a_ij >= edge_threshold:
                        adjacency[i, j] = 1
                        scores[i, j] = a_ij
                        lagm[i, j] = lag_ij
                        signm[i, j] = int(np.sign(beta_ij))

                    if a_ji >= edge_threshold:
                        adjacency[j, i] = 1
                        scores[j, i] = a_ji
                        lagm[j, i] = lag_ji
                        signm[j, i] = int(np.sign(beta_ji))

            np.fill_diagonal(scores, 0.0)
            np.fill_diagonal(adjacency, 0)
            np.fill_diagonal(lagm, 0)
            np.fill_diagonal(signm, 0)

            return MethodOutput(
                method_name=self.method_name,
                names=names,
                adjacency_scores=scores,
                adjacency_bin=adjacency,
                directed_support=True,
                lag_support=True,
                sign_support=False,
                regime_support=False,
                lag_matrix=lagm,
                sign_matrix=None,
                meta={
                    "edge_threshold": edge_threshold,
                    "asymmetry_ratio_threshold": asymmetry_ratio_threshold,
                    "detector_mode": detector_mode,
                    "max_workers": max_workers,
                },
            )

    return _Lambda3Adapter()

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_shared_tasks(output: MethodOutput, gt: GroundTruth) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if output.adjacency_bin is None:
        metrics["edge_f1_undirected"] = np.nan
        metrics["edge_precision_undirected"] = np.nan
        metrics["edge_recall_undirected"] = np.nan
    else:
        pred_u = output.undirected_bin()
        true_u = gt.undirected_adjacency()
        mask = ~np.eye(len(gt.names), dtype=bool)
        prf = _precision_recall_f1(true_u[mask], pred_u[mask])
        metrics["edge_precision_undirected"] = prf["precision"]
        metrics["edge_recall_undirected"] = prf["recall"]
        metrics["edge_f1_undirected"] = prf["f1"]

    if output.directed_support and output.adjacency_bin is not None:
        mask = ~np.eye(len(gt.names), dtype=bool)
        prf = _precision_recall_f1(gt.adjacency[mask], output.adjacency_bin[mask])
        metrics["edge_precision_directed"] = prf["precision"]
        metrics["edge_recall_directed"] = prf["recall"]
        metrics["edge_f1_directed"] = prf["f1"]

        true_edges = np.argwhere(gt.adjacency == 1)
        if len(true_edges):
            direction_hits = []
            for src, dst in true_edges:
                direction_hits.append(int(output.adjacency_bin[src, dst] == 1))
            metrics["direction_recall_on_true_edges"] = float(np.mean(direction_hits))
        else:
            metrics["direction_recall_on_true_edges"] = np.nan
    else:
        metrics["edge_precision_directed"] = np.nan
        metrics["edge_recall_directed"] = np.nan
        metrics["edge_f1_directed"] = np.nan
        metrics["direction_recall_on_true_edges"] = np.nan

    if output.lag_support and output.lag_matrix is not None and gt.lag_matrix is not None:
        true_edges = np.argwhere(gt.adjacency == 1)
        errs = []
        for src, dst in true_edges:
            if output.adjacency_bin is not None and output.adjacency_bin[src, dst] == 1:
                errs.append(abs(int(output.lag_matrix[src, dst]) - int(gt.lag_matrix[src, dst])))
        metrics["lag_mae_on_detected_true_edges"] = float(np.mean(errs)) if errs else np.nan
    else:
        metrics["lag_mae_on_detected_true_edges"] = np.nan

    if output.sign_support and output.sign_matrix is not None and gt.sign_matrix is not None:
        true_edges = np.argwhere(gt.adjacency == 1)
        hits = []
        for src, dst in true_edges:
            if output.adjacency_bin is not None and output.adjacency_bin[src, dst] == 1:
                hits.append(int(np.sign(output.sign_matrix[src, dst]) == np.sign(gt.sign_matrix[src, dst])))
        metrics["sign_acc_on_detected_true_edges"] = float(np.mean(hits)) if hits else np.nan
    else:
        metrics["sign_acc_on_detected_true_edges"] = np.nan

    if gt.regime_boundaries is not None:
        if output.regime_support and output.regime_boundaries is not None:
            metrics["regime_boundary_f1"] = _boundary_f1(gt.regime_boundaries, output.regime_boundaries, tolerance=5)
        else:
            metrics["regime_boundary_f1"] = np.nan
    else:
        metrics["regime_boundary_f1"] = np.nan

    if gt.adjacency_by_regime is not None and output.adjacency_by_regime is not None:
        vals = []
        for true_adj, pred_adj in zip(gt.adjacency_by_regime, output.adjacency_by_regime):
            mask = ~np.eye(len(gt.names), dtype=bool)
            prf = _precision_recall_f1(true_adj[mask], pred_adj[mask])
            vals.append(prf["f1"])
        metrics["regime_specific_edge_f1_directed"] = float(np.mean(vals)) if vals else np.nan
    else:
        metrics["regime_specific_edge_f1_directed"] = np.nan

    if gt.low_corr_edges:
        if output.adjacency_bin is not None:
            hits = [int(output.adjacency_bin[src, dst] == 1) for src, dst in gt.low_corr_edges]
            metrics["lowcorr_edge_recall"] = float(np.mean(hits))
        else:
            metrics["lowcorr_edge_recall"] = np.nan
    else:
        metrics["lowcorr_edge_recall"] = np.nan

    if gt.forbidden_edges:
        if output.adjacency_bin is not None:
            fp = [int(output.adjacency_bin[src, dst] == 1) for src, dst in gt.forbidden_edges]
            metrics["spurious_forbidden_edge_rate"] = float(np.mean(fp))
        else:
            metrics["spurious_forbidden_edge_rate"] = np.nan
    else:
        metrics["spurious_forbidden_edge_rate"] = np.nan

    return metrics


def aggregate_metrics(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("scenario", "method", "repeat")]
    agg_rows = []
    for (scenario, method), g in df.groupby(["scenario", "method"], dropna=False):
        row = {"scenario": scenario, "method": method}
        for col in metric_cols:
            row[col] = float(np.nanmean(g[col].to_numpy(dtype=float))) if g[col].notna().any() else np.nan
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def shared_task_score(df_agg: pd.DataFrame) -> pd.DataFrame:
    metric_weights = {
        "edge_f1_undirected": 1.0,
        "edge_f1_directed": 1.2,
        "direction_recall_on_true_edges": 1.0,
        "lag_mae_on_detected_true_edges": 0.8,
        "regime_boundary_f1": 1.2,
        "regime_specific_edge_f1_directed": 1.2,
        "lowcorr_edge_recall": 1.0,
        "spurious_forbidden_edge_rate": 0.8,
    }
    rows = []
    for (method,), g in df_agg.groupby(["method"]):
        vals = []
        weights = []
        for _, row in g.iterrows():
            for metric, w in metric_weights.items():
                v = row.get(metric, np.nan)
                if np.isnan(v):
                    continue
                if metric == "lag_mae_on_detected_true_edges":
                    v = 1.0 / (1.0 + v)
                elif metric == "spurious_forbidden_edge_rate":
                    v = 1.0 - v
                vals.append(v * w)
                weights.append(w)
        score = float(np.sum(vals) / np.sum(weights)) if weights else np.nan
        rows.append({"method": method, "shared_structural_score": score})
    return pd.DataFrame(rows).sort_values("shared_structural_score", ascending=False)


# ---------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------

def run_benchmark(
    adapters: Sequence[BaseAdapter],
    scenarios: Optional[Dict[str, Callable[[ScenarioConfig, int], Tuple[pd.DataFrame, GroundTruth]]]] = None,
    cfg: Optional[ScenarioConfig] = None,
    repeats: int = 10,
    base_seed: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if scenarios is None:
        scenarios = SCENARIOS
    if cfg is None:
        cfg = ScenarioConfig()

    raw_rows: List[Dict[str, Any]] = []

    for s_idx, (scenario_name, scenario_fn) in enumerate(scenarios.items()):
        if verbose:
            print("=" * 88)
            print(f"Scenario: {scenario_name}")
            print("=" * 88)
        for rep in range(repeats):
            seed = base_seed + 1000 * s_idx + rep
            df, gt = scenario_fn(cfg, seed)

            if verbose:
                print(f"  repeat {rep+1:02d}/{repeats}: n={len(df)}, variables={list(df.columns)}")

            for adapter in adapters:
                try:
                    out = adapter.fit(df, cfg)
                    metrics = evaluate_shared_tasks(out, gt)
                    row = {"scenario": scenario_name, "repeat": rep, "method": out.method_name}
                    row.update(metrics)
                    raw_rows.append(row)
                    if verbose:
                        key_bits = []
                        if not np.isnan(metrics["edge_f1_undirected"]):
                            key_bits.append(f"undirF1={metrics['edge_f1_undirected']:.2f}")
                        if not np.isnan(metrics["edge_f1_directed"]):
                            key_bits.append(f"dirF1={metrics['edge_f1_directed']:.2f}")
                        if not np.isnan(metrics["regime_boundary_f1"]):
                            key_bits.append(f"regF1={metrics['regime_boundary_f1']:.2f}")
                        print(f"    - {out.method_name:18s}  " + ", ".join(key_bits))
                except Exception as e:
                    if verbose:
                        print(f"    - {adapter.method_name:18s}  FAILED: {e}")
                    row = {"scenario": scenario_name, "repeat": rep, "method": adapter.method_name}
                    raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    agg_df = aggregate_metrics(raw_rows)
    return raw_df, agg_df


# ---------------------------------------------------------------------
# Reporting / plotting
# ---------------------------------------------------------------------

def print_summary_tables(agg_df: pd.DataFrame) -> None:
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 50)

    cols = [
        "scenario",
        "method",
        "edge_f1_undirected",
        "edge_f1_directed",
        "direction_recall_on_true_edges",
        "lag_mae_on_detected_true_edges",
        "regime_boundary_f1",
        "regime_specific_edge_f1_directed",
        "lowcorr_edge_recall",
        "spurious_forbidden_edge_rate",
    ]
    cols = [c for c in cols if c in agg_df.columns]
    print("\nAggregated benchmark metrics")
    print("-" * 88)
    print(agg_df[cols].sort_values(["scenario", "method"]).to_string(index=False))

    overall = shared_task_score(agg_df)
    print("\nShared structural score (N/A-aware)")
    print("-" * 88)
    print(overall.to_string(index=False))


def plot_shared_scores(agg_df: pd.DataFrame, outfile: Optional[str] = None) -> None:
    overall = shared_task_score(agg_df)
    methods = overall["method"].tolist()
    scores = overall["shared_structural_score"].to_numpy()

    plt.figure(figsize=(8, 4.5))
    y = np.arange(len(methods))
    bars = plt.barh(y, scores)
    plt.yticks(y, methods)
    plt.xlim(0, 1.05)
    plt.xlabel("Shared structural score")
    plt.title("Benchmark summary (N/A-aware structural score)")
    for bar, score in zip(bars, scores):
        plt.text(min(score + 0.02, 1.02), bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------
# Single-file Lambda3 runner
# ---------------------------------------------------------------------

def example_lambda3_runner_stub(df: pd.DataFrame, cfg: ScenarioConfig) -> Dict[str, Any]:
    """
    Real single-file bridge: benchmark -> Lambda3.

    This function will work as-is if the following Lambda3 symbols are available
    in the current Python session or importable by `_try_import_lambda3_symbols()`:
      - L3Config
      - calc_lambda3_features_v2
      - fit_l3_bayesian_regression_asymmetric
      - calculate_sync_profile

    Returns
    -------
    dict with keys suitable for `make_lambda3_adapter(...)`.
    """
    syms = _require_lambda3_symbols()
    L3Config = syms["L3Config"]
    calc_lambda3_features_v2 = syms["calc_lambda3_features_v2"]
    fit_l3_bayesian_regression_asymmetric = syms["fit_l3_bayesian_regression_asymmetric"]
    calculate_sync_profile = syms["calculate_sync_profile"]

    names = list(df.columns)
    n = len(names)

    l3cfg = L3Config(
        T=len(df),
        draws=1000,
        tune=1000,
        target_accept=0.90,
    )

    features: Dict[str, Dict[str, Any]] = {}
    for col in names:
        x = df[col].to_numpy(dtype=float)
        delta_pos, delta_neg, rho_t, time_trend, local_jump = calc_lambda3_features_v2(x, l3cfg)
        features[col] = {
            "data": x,
            "delta_LambdaC_pos": np.asarray(delta_pos, dtype=np.int32),
            "delta_LambdaC_neg": np.asarray(delta_neg, dtype=np.int32),
            "rho_T": np.asarray(rho_t, dtype=float),
            "time_trend": np.asarray(time_trend, dtype=float),
            "local_jump": np.asarray(local_jump, dtype=np.int32),
        }

    adjacency_bin = np.zeros((n, n), dtype=int)
    adjacency_scores = np.zeros((n, n), dtype=float)
    lag_matrix = np.zeros((n, n), dtype=int)
    sign_matrix = np.zeros((n, n), dtype=int)

    def _get_hdi_cols(summary_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        hdi_cols = [c for c in summary_df.columns if str(c).startswith("hdi_")]
        if len(hdi_cols) >= 2:
            def _key(c: str) -> float:
                m = re.search(r"hdi_([0-9.]+)", str(c))
                return float(m.group(1)) if m else 0.0
            hdi_cols = sorted(hdi_cols, key=_key)
            return str(hdi_cols[0]), str(hdi_cols[-1])
        return None, None

    for src_idx, src in enumerate(names):
        for dst_idx, dst in enumerate(names):
            if src == dst:
                continue

            f_src = features[src]
            f_dst = features[dst]

            trace = fit_l3_bayesian_regression_asymmetric(
                data=f_dst["data"],
                features_dict={
                    "delta_LambdaC_pos": f_dst["delta_LambdaC_pos"],
                    "delta_LambdaC_neg": f_dst["delta_LambdaC_neg"],
                    "rho_T": f_dst["rho_T"],
                    "time_trend": f_dst["time_trend"],
                },
                config=l3cfg,
                interaction_pos=f_src["delta_LambdaC_pos"],
                interaction_neg=f_src["delta_LambdaC_neg"],
                interaction_rhoT=f_src["rho_T"],
            )

            summary = az.summary(
                trace,
                var_names=["beta_interact_pos", "beta_interact_neg", "beta_interact_stress"],
                hdi_prob=getattr(l3cfg, "hdi_prob", 0.94),
            )

            lo_col, hi_col = _get_hdi_cols(summary)
            coeffs: Dict[str, Dict[str, Any]] = {}
            for var in ["beta_interact_pos", "beta_interact_neg", "beta_interact_stress"]:
                if var in summary.index:
                    mean_val = float(summary.loc[var, "mean"])
                    if lo_col is not None and hi_col is not None:
                        lo = float(summary.loc[var, lo_col])
                        hi = float(summary.loc[var, hi_col])
                        supported = (lo > 0.0) or (hi < 0.0)
                    else:
                        supported = False
                    coeffs[var] = {"mean": mean_val, "supported": supported}
                else:
                    coeffs[var] = {"mean": 0.0, "supported": False}

            supported_vars = [k for k, v in coeffs.items() if v["supported"]]
            edge_exists = len(supported_vars) > 0
            adjacency_bin[src_idx, dst_idx] = int(edge_exists)

            if edge_exists:
                dominant_var = max(supported_vars, key=lambda k: abs(coeffs[k]["mean"]))
            else:
                dominant_var = max(coeffs.keys(), key=lambda k: abs(coeffs[k]["mean"]))

            dominant_beta = float(coeffs[dominant_var]["mean"])
            adjacency_scores[src_idx, dst_idx] = abs(dominant_beta)
            sign_matrix[src_idx, dst_idx] = int(np.sign(dominant_beta))

            try:
                _, max_sync, optimal_lag = calculate_sync_profile(
                    f_src["delta_LambdaC_pos"].astype(np.float64),
                    f_dst["delta_LambdaC_pos"].astype(np.float64),
                    lag_window=min(getattr(cfg, "max_lag", 10), 10),
                )
                lag_matrix[src_idx, dst_idx] = int(optimal_lag)
            except Exception:
                lag_matrix[src_idx, dst_idx] = 0

    return {
        "adjacency_scores": adjacency_scores,
        "adjacency_bin": adjacency_bin,
        "lag_matrix": lag_matrix,
        "sign_matrix": sign_matrix,
        "regime_boundaries": None,
        "adjacency_by_regime": None,
        "lag_by_regime": None,
        "sign_by_regime": None,
        "meta": {
            "note": "Single-file Lambda3 bridge",
            "names": names,
            "draws": getattr(l3cfg, "draws", None),
            "tune": getattr(l3cfg, "tune", None),
        },
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(enable_lambda3: bool = False, repeats: int = 5) -> None:
    cfg = ScenarioConfig(T=350, burn_in=80, regime_split=175, max_lag=8, noise_scale=0.35)
    adapters: List[BaseAdapter] = [
        VARGrangerAdapter(maxlags=8, alpha=0.05),
        TransferEntropyAdapter(max_lag=8, alpha=0.05, n_bins=4, n_perm=20, backend="pyinform", k_history=1),
        # Optional additional causal-discovery baseline (uncomment if tigramite is installed)
        PCMCIPlusAdapter(tau_max=8, pc_alpha=0.05, verbosity=0),
        GraphicalLassoAdapter(edge_threshold=0.03),
        EventCrossCorrelationAdapter(max_lag=8, event_percentile=97.0, alpha=0.05, n_perm=20),
    ]

    if enable_lambda3:
        adapters.append(make_lambda3_adapter(example_lambda3_runner_stub, method_name="Lambda3"))

    raw_df, agg_df = run_benchmark(adapters, cfg=cfg, repeats=repeats, base_seed=42, verbose=True)
    print_summary_tables(agg_df)
    plot_shared_scores(agg_df, outfile="lambda3_benchmark_shared_scores.png")

    raw_df.to_csv("lambda3_benchmark_raw_results.csv", index=False)
    agg_df.to_csv("lambda3_benchmark_aggregated_results.csv", index=False)
    print("\nSaved:")
    print("  - lambda3_benchmark_raw_results.csv")
    print("  - lambda3_benchmark_aggregated_results.csv")
    print("  - lambda3_benchmark_shared_scores.png")

    if enable_lambda3:
        print("  - Lambda3 path: ENABLED (single-file bridge)")
    else:
        print("  - Lambda3 path: DISABLED (baselines only)")

if __name__ == "__main__":
    # Set enable_lambda3=True once your Lambda3 functions are available.
    main(enable_lambda3=True, repeats=1)
