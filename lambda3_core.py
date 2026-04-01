# ==========================================================
# Λ³ CORE: Universal Structural Tensor Field Engine
# ==========================================================
# Domain-independent structural tensor dynamics analyzer.
# Zero-shot recognition of universal patterns through
# ΔΛC pulsations in semantic tensor space.
#
# Pure structural analysis — no domain-specific dependencies.
# Domain extensions (financial, medical, IoT, NLP, etc.)
# should inherit from this module's base classes.
#
# Author: Masamichi Iizumi (Miosync, Inc.)
# Core Theory: Lambda³ (Λ³) Structural Tensor Dynamics
# ==========================================================

# ===============================
# Standard Library
# ===============================
import pickle
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any, Union, Callable

# ===============================
# Scientific Computing
# ===============================
import numpy as np
import pandas as pd

# ===============================
# Probabilistic Programming
# ===============================
import pymc as pm
import arviz as az

# ===============================
# Machine Learning & Clustering
# ===============================
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# ===============================
# Performance Optimization
# ===============================
from numba import njit, prange

# ===============================
# Visualization
# ===============================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns

# ===============================
# Network Analysis
# ===============================
import networkx as nx


# ===================================================================
#  SECTION 0: GLOBAL CONSTANTS & CONFIGURATION
# ===================================================================

DELTA_PERCENTILE = 97.0
LOCAL_JUMP_PERCENTILE = 95.0
WINDOW_SIZE = 10
LOCAL_WINDOW_SIZE = 5
LAG_WINDOW_DEFAULT = 10
SYNC_THRESHOLD_DEFAULT = 0.3


@dataclass
class L3Config:
    """Configuration for Lambda³ structural tensor analysis.

    Controls feature extraction, Bayesian estimation, and
    hierarchical structural change detection parameters.
    """
    T: int = 150
    window: int = WINDOW_SIZE
    local_window: int = LOCAL_WINDOW_SIZE
    global_window: int = 30
    delta_percentile: float = DELTA_PERCENTILE
    local_jump_percentile: float = LOCAL_JUMP_PERCENTILE
    local_threshold_percentile: float = 85.0
    global_threshold_percentile: float = 92.5
    draws: int = 8000
    tune: int = 8000
    target_accept: float = 0.95
    hdi_prob: float = 0.94
    hierarchical: bool = True


# ===================================================================
#  SECTION 1: DATA LOADING (Domain-agnostic)
# ===================================================================

def load_csv_data(
    filepath: str,
    time_column: Optional[str] = None,
    value_columns: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load time-series data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file.
    time_column : str, optional
        Column name used for temporal ordering (sorted, not included
        in the returned dict).
    value_columns : list of str, optional
        Columns to analyse. ``None`` → all numeric columns.

    Returns
    -------
    dict
        ``{column_name: np.ndarray}`` for each selected column.
    """
    df = pd.read_csv(filepath, parse_dates=True)
    print(f"Loaded CSV — shape: {df.shape}, columns: {list(df.columns)}")

    if time_column and time_column in df.columns:
        df = df.sort_values(by=time_column)

    if value_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_column and time_column in numeric_cols:
            numeric_cols.remove(time_column)
        value_columns = numeric_cols

    series_dict: Dict[str, np.ndarray] = {}
    for col in value_columns:
        if col in df.columns:
            data = df[col].values
            if pd.isna(data).any():
                data = pd.Series(data).ffill().bfill().values
            series_dict[col] = data.astype(np.float64)
    return series_dict


# ===================================================================
#  SECTION 2: JIT-COMPILED CORE FUNCTIONS
# ===================================================================

@njit
def _diff_and_threshold(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    n = data.shape[0]
    diff = np.empty(n)
    diff[0] = 0.0
    for i in range(1, n):
        diff[i] = data[i] - data[i - 1]
    threshold = np.percentile(np.abs(diff), percentile)
    return diff, threshold


@njit
def _detect_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    n = diff.shape[0]
    pos = np.zeros(n, dtype=np.int32)
    neg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if diff[i] > threshold:
            pos[i] = 1
        elif diff[i] < -threshold:
            neg[i] = 1
    return pos, neg


@njit
def _local_std(data: np.ndarray, window: int) -> np.ndarray:
    n = data.shape[0]
    out = np.empty(n)
    for i in range(n):
        s = max(0, i - window)
        e = min(n, i + window + 1)
        sub = data[s:e]
        m = np.mean(sub)
        out[i] = np.sqrt(np.sum((sub - m) ** 2) / (e - s))
    return out


@njit
def _rho_t(data: np.ndarray, window: int) -> np.ndarray:
    """Tension scalar ρT — local structural volatility."""
    n = data.shape[0]
    out = np.empty(n)
    for i in range(n):
        s = max(0, i - window)
        e = i + 1
        sub = data[s:e]
        if (e - s) > 1:
            m = np.mean(sub)
            out[i] = np.sqrt(np.sum((sub - m) ** 2) / (e - s))
        else:
            out[i] = 0.0
    return out


@njit
def _sync_rate_at_lag(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    la, lb = a.shape[0], b.shape[0]
    if lag < 0:
        return np.mean(a[-lag:] * b[:lag]) if -lag < la else 0.0
    elif lag > 0:
        return np.mean(a[:-lag] * b[lag:]) if lag < lb else 0.0
    return np.mean(a * b)


@njit(parallel=True)
def _sync_profile(a: np.ndarray, b: np.ndarray, lag_window: int
                  ) -> Tuple[np.ndarray, np.ndarray, float, int]:
    n_lags = 2 * lag_window + 1
    lags = np.arange(-lag_window, lag_window + 1)
    vals = np.empty(n_lags)
    for i in prange(n_lags):
        vals[i] = _sync_rate_at_lag(a, b, lags[i])
    mx, opt = 0.0, 0
    for i in range(n_lags):
        if vals[i] > mx:
            mx = vals[i]
            opt = lags[i]
    return lags, vals, mx, opt


@njit
def _hierarchical_jumps(
    data: np.ndarray,
    local_window: int,
    global_window: int,
    local_pct: float,
    global_pct: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect structural changes at two hierarchical scales."""
    n = data.shape[0]
    diff = np.empty(n)
    diff[0] = 0.0
    for i in range(1, n):
        diff[i] = data[i] - data[i - 1]

    l_pos = np.zeros(n, dtype=np.int32)
    l_neg = np.zeros(n, dtype=np.int32)
    g_pos = np.zeros(n, dtype=np.int32)
    g_neg = np.zeros(n, dtype=np.int32)

    # Local criteria (adaptive threshold per neighbourhood)
    for i in range(n):
        s = max(0, i - local_window)
        e = min(n, i + local_window + 1)
        thr = np.percentile(np.abs(diff[s:e]), local_pct)
        if diff[i] > thr:
            l_pos[i] = 1
        elif diff[i] < -thr:
            l_neg[i] = 1

    # Global criteria (single threshold across entire series)
    g_thr = np.percentile(np.abs(diff), global_pct)
    for i in range(n):
        if diff[i] > g_thr:
            g_pos[i] = 1
        elif diff[i] < -g_thr:
            g_neg[i] = 1

    return l_pos, l_neg, g_pos, g_neg


# ===================================================================
#  SECTION 3: UNIFIED FEATURE EXTRACTION
# ===================================================================

def calc_lambda3_features(data: np.ndarray, config: L3Config) -> Dict[str, np.ndarray]:
    """Extract Lambda³ structural tensor features from a time series.

    Returns a dictionary containing ΔΛC±, ρT, hierarchical
    decomposition (if ``config.hierarchical``), and auxiliary arrays.
    """
    if config.hierarchical:
        lp, ln, gp, gn = _hierarchical_jumps(
            data, config.local_window, config.global_window,
            config.local_threshold_percentile,
            config.global_threshold_percentile,
        )
        combined_pos = np.maximum(lp.astype(np.float64), gp.astype(np.float64))
        combined_neg = np.maximum(ln.astype(np.float64), gn.astype(np.float64))

        n = data.shape[0]
        pure_lp = np.zeros(n, dtype=np.float64)
        pure_ln = np.zeros(n, dtype=np.float64)
        pure_gp = np.zeros(n, dtype=np.float64)
        pure_gn = np.zeros(n, dtype=np.float64)
        mix_p = np.zeros(n, dtype=np.float64)
        mix_n = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if lp[i] and gp[i]:
                mix_p[i] = 1.0
            elif lp[i]:
                pure_lp[i] = 1.0
            elif gp[i]:
                pure_gp[i] = 1.0
            if ln[i] and gn[i]:
                mix_n[i] = 1.0
            elif ln[i]:
                pure_ln[i] = 1.0
            elif gn[i]:
                pure_gn[i] = 1.0

        rho = _rho_t(data, config.window)
        diff, _ = _diff_and_threshold(data, config.delta_percentile)
        lstd = _local_std(data, config.local_window)
        score = np.abs(diff) / (lstd + 1e-8)
        lthr = np.percentile(score, config.local_jump_percentile)

        return {
            "data": data,
            "delta_LambdaC_pos": combined_pos,
            "delta_LambdaC_neg": combined_neg,
            "rho_T": rho,
            "time_trend": np.arange(n, dtype=np.float64),
            "local_jump_detect": (score > lthr).astype(int),
            "local_pos": lp.astype(np.float64),
            "local_neg": ln.astype(np.float64),
            "global_pos": gp.astype(np.float64),
            "global_neg": gn.astype(np.float64),
            "pure_local_pos": pure_lp,
            "pure_local_neg": pure_ln,
            "pure_global_pos": pure_gp,
            "pure_global_neg": pure_gn,
            "mixed_pos": mix_p,
            "mixed_neg": mix_n,
        }
    else:
        diff, thr = _diff_and_threshold(data, config.delta_percentile)
        dp, dn = _detect_jumps(diff, thr)
        lstd = _local_std(data, config.local_window)
        score = np.abs(diff) / (lstd + 1e-8)
        lthr = np.percentile(score, config.local_jump_percentile)
        rho = _rho_t(data, config.window)

        return {
            "data": data,
            "delta_LambdaC_pos": dp.astype(np.float64),
            "delta_LambdaC_neg": dn.astype(np.float64),
            "rho_T": rho,
            "time_trend": np.arange(len(data), dtype=np.float64),
            "local_jump_detect": (score > lthr).astype(int),
        }


# ===================================================================
#  SECTION 4: BAYESIAN HDI LOGGER
# ===================================================================

class BayesianHDILogger:
    """Structured logging for Bayesian HDI estimates.

    Records posterior summaries from ``arviz`` traces,
    classifies parameters into structural categories (ΔΛC, ρT,
    interactions, hierarchical effects, etc.), and generates
    summary reports.
    """

    def __init__(self, hdi_prob: float = 0.94):
        self.hdi_prob = hdi_prob
        self.all_results: Dict[str, Any] = {}

    # ----- public API -----

    def log_trace(
        self,
        trace,
        model_id: str,
        model_type: str = "general",
        series_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        summary = az.summary(trace, hdi_prob=self.hdi_prob)
        hdi = self._extract_hdi(summary)
        classified = self._classify(hdi, model_type, series_names)
        self.all_results[model_id] = {
            "raw_hdi": hdi,
            "structured": classified,
            "model_type": model_type,
            "series": series_names,
            "timestamp": pd.Timestamp.now(),
        }
        if verbose:
            self._display(model_id, classified, series_names)
        return classified

    def generate_summary_report(self) -> pd.DataFrame:
        rows = []
        for mid, r in self.all_results.items():
            st = r["structured"]
            n_p = sum(len(g) for g in st.values())
            n_s = sum(
                sum(1 for p in g.values() if p["excludes_zero"])
                for g in st.values()
            )
            mx_e, mx_p = 0.0, ""
            for g in st.values():
                for pn, d in g.items():
                    if abs(d["mean"]) > mx_e:
                        mx_e, mx_p = abs(d["mean"]), pn
            rows.append({
                "model_id": mid,
                "model_type": r["model_type"],
                "series": " ⇄ ".join(r["series"]) if r["series"] else "N/A",
                "n_parameters": n_p,
                "n_significant": n_s,
                "signif_rate": n_s / n_p if n_p else 0,
                "max_effect_param": mx_p,
                "max_effect_size": mx_e,
            })
        return pd.DataFrame(rows)

    # ----- internals -----

    def _extract_hdi(self, summary: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        lo = f"hdi_{(1 - self.hdi_prob) / 2 * 100:.0f}%"
        hi = f"hdi_{(1 + self.hdi_prob) / 2 * 100:.0f}%"
        out: Dict[str, Dict[str, float]] = {}
        for p in summary.index:
            out[p] = {
                "mean": summary.loc[p, "mean"],
                "sd": summary.loc[p, "sd"],
                "hdi_low": summary.loc[p, lo],
                "hdi_high": summary.loc[p, hi],
                "width": summary.loc[p, hi] - summary.loc[p, lo],
                "excludes_zero": not (summary.loc[p, lo] < 0 < summary.loc[p, hi]),
            }
        return out

    def _classify(self, hdi, model_type, series_names):
        cats = {
            "structural_changes": {},
            "tension_scalars": {},
            "interactions": {},
            "hierarchical": {},
            "temporal": {},
            "observation": {},
        }
        for p, d in hdi.items():
            if any(x in p for x in ("dLC", "delta_Lambda", "_pos", "_neg")):
                cats["structural_changes"][p] = d
            elif any(x in p for x in ("rhoT", "rho_T", "tension", "stress")):
                cats["tension_scalars"][p] = d
            elif any(x in p for x in ("interact", "cross", "_ab_", "_ba_")):
                cats["interactions"][p] = d
            elif any(x in p for x in ("alpha_", "escalation", "deescalation",
                                       "local_", "global_", "hierarchical")):
                cats["hierarchical"][p] = d
            elif any(x in p for x in ("time", "lag", "temporal")):
                cats["temporal"][p] = d
            elif any(x in p for x in ("sigma", "rho_ab", "correlation")):
                cats["observation"][p] = d
        return cats

    def _display(self, model_id, structured, series_names):
        print(f"\n{'=' * 70}")
        print(f"Bayesian HDI — {model_id}")
        if series_names:
            print(f"Series: {' ⇄ '.join(series_names)}")
        print(f"HDI prob: {self.hdi_prob * 100:.0f}%")
        print(f"{'=' * 70}")
        for cat_name, params in structured.items():
            if not params:
                continue
            print(f"\n[{cat_name}]")
            print(f"{'Param':<30} {'Mean':>8} {'HDI_lo':>8} {'HDI_hi':>8} {'Sig':>5}")
            print("-" * 65)
            for p, d in params.items():
                sig = "YES" if d["excludes_zero"] else "no"
                print(f"{p:<30} {d['mean']:>8.3f} {d['hdi_low']:>8.3f} "
                      f"{d['hdi_high']:>8.3f} {sig:>5}")


# ===================================================================
#  SECTION 5: BAYESIAN MODELS
# ===================================================================

def fit_asymmetric_regression(
    data: np.ndarray,
    features: Dict[str, np.ndarray],
    config: L3Config,
    interact_pos: Optional[np.ndarray] = None,
    interact_neg: Optional[np.ndarray] = None,
    interact_rhoT: Optional[np.ndarray] = None,
    logger: Optional[BayesianHDILogger] = None,
    model_name: Optional[str] = None,
):
    """Bayesian regression with asymmetric cross-series interactions."""
    with pm.Model():
        b0 = pm.Normal("beta_0", mu=0, sigma=2)
        bt = pm.Normal("beta_time", mu=0, sigma=1)
        bp = pm.Normal("beta_dLC_pos", mu=0, sigma=5)
        bn = pm.Normal("beta_dLC_neg", mu=0, sigma=5)
        br = pm.Normal("beta_rhoT", mu=0, sigma=3)

        mu = (b0 + bt * features["time_trend"]
              + bp * features["delta_LambdaC_pos"]
              + bn * features["delta_LambdaC_neg"]
              + br * features["rho_T"])

        if interact_pos is not None:
            bip = pm.Normal("beta_interact_pos", mu=0, sigma=3)
            mu = mu + bip * interact_pos
        if interact_neg is not None:
            bin_ = pm.Normal("beta_interact_neg", mu=0, sigma=3)
            mu = mu + bin_ * interact_neg
        if interact_rhoT is not None:
            bis = pm.Normal("beta_interact_stress", mu=0, sigma=2)
            mu = mu + bis * interact_rhoT

        sig = pm.HalfNormal("sigma_obs", sigma=1)
        pm.Normal("y_obs", mu=mu, sigma=sig, observed=data)

        trace = pm.sample(draws=config.draws, tune=config.tune,
                          target_accept=config.target_accept,
                          return_inferencedata=True, cores=4, chains=4)

    if logger is not None:
        logger.log_trace(trace, model_name or "asymmetric_regression",
                         "asymmetric_regression", verbose=False)
    return trace


def fit_pairwise_system(
    data_dict: Dict[str, np.ndarray],
    features_dict: Dict[str, Dict[str, np.ndarray]],
    config: L3Config,
    series_pair: Optional[Tuple[str, str]] = None,
    logger: Optional[BayesianHDILogger] = None,
):
    """Pairwise bidirectional structural tensor model.

    Models A→B and B→A interactions simultaneously in a joint
    multivariate-normal observation model with correlation (ρ_ab).
    """
    names = list(series_pair) if series_pair else list(data_dict.keys())[:2]
    na, nb = names
    da, db = data_dict[na], data_dict[nb]
    fa, fb = features_dict[na], features_dict[nb]

    with pm.Model() as model:
        # --- Series A self terms ---
        b0a = pm.Normal("beta_0_a", mu=0, sigma=2)
        bta = pm.Normal("beta_time_a", mu=0, sigma=1)
        bpa = pm.Normal("beta_dLC_pos_a", mu=0, sigma=3)
        bna = pm.Normal("beta_dLC_neg_a", mu=0, sigma=3)
        bra = pm.Normal("beta_rhoT_a", mu=0, sigma=2)

        # --- Series B self terms ---
        b0b = pm.Normal("beta_0_b", mu=0, sigma=2)
        btb = pm.Normal("beta_time_b", mu=0, sigma=1)
        bpb = pm.Normal("beta_dLC_pos_b", mu=0, sigma=3)
        bnb = pm.Normal("beta_dLC_neg_b", mu=0, sigma=3)
        brb = pm.Normal("beta_rhoT_b", mu=0, sigma=2)

        # --- Cross interactions ---
        bi_ab_p = pm.Normal("beta_interact_ab_pos", mu=0, sigma=2)
        bi_ab_n = pm.Normal("beta_interact_ab_neg", mu=0, sigma=2)
        bi_ab_s = pm.Normal("beta_interact_ab_stress", mu=0, sigma=1.5)
        bi_ba_p = pm.Normal("beta_interact_ba_pos", mu=0, sigma=2)
        bi_ba_n = pm.Normal("beta_interact_ba_neg", mu=0, sigma=2)
        bi_ba_s = pm.Normal("beta_interact_ba_stress", mu=0, sigma=1.5)

        # --- Lag terms ---
        if len(da) > 1:
            lag_a = np.concatenate([[0], da[:-1]])
            lag_b = np.concatenate([[0], db[:-1]])
            bl_ab = pm.Normal("beta_lag_ab", mu=0, sigma=1)
            bl_ba = pm.Normal("beta_lag_ba", mu=0, sigma=1)
        else:
            lag_a = np.zeros_like(da)
            lag_b = np.zeros_like(db)
            bl_ab = bl_ba = 0

        mu_a = (b0a + bta * fa["time_trend"]
                + bpa * fa["delta_LambdaC_pos"]
                + bna * fa["delta_LambdaC_neg"]
                + bra * fa["rho_T"]
                + bi_ba_p * fb["delta_LambdaC_pos"]
                + bi_ba_n * fb["delta_LambdaC_neg"]
                + bi_ba_s * fb["rho_T"]
                + bl_ba * lag_b)

        mu_b = (b0b + btb * fb["time_trend"]
                + bpb * fb["delta_LambdaC_pos"]
                + bnb * fb["delta_LambdaC_neg"]
                + brb * fb["rho_T"]
                + bi_ab_p * fa["delta_LambdaC_pos"]
                + bi_ab_n * fa["delta_LambdaC_neg"]
                + bi_ab_s * fa["rho_T"]
                + bl_ab * lag_a)

        sa = pm.HalfNormal("sigma_a", sigma=1)
        sb = pm.HalfNormal("sigma_b", sigma=1)
        rho_ab = pm.Uniform("rho_ab", lower=-1, upper=1)
        cov = pm.math.stack([
            [sa ** 2, rho_ab * sa * sb],
            [rho_ab * sa * sb, sb ** 2],
        ])

        y_comb = pm.math.stack([da, db]).T
        mu_comb = pm.math.stack([mu_a, mu_b]).T
        pm.MvNormal("y_obs", mu=mu_comb, cov=cov, observed=y_comb)

        trace = pm.sample(draws=config.draws, tune=config.tune,
                          target_accept=config.target_accept,
                          return_inferencedata=True, cores=4, chains=4)

    if logger is not None:
        mid = f"pairwise_{na}_vs_{nb}"
        logger.log_trace(trace, mid, "pairwise_system",
                         series_names=[na, nb], verbose=False)
    return trace, model


def fit_hierarchical_bayesian(
    data: np.ndarray,
    h_features: Dict[str, np.ndarray],
    config: L3Config,
    logger: Optional[BayesianHDILogger] = None,
):
    """Hierarchical Bayesian model for multi-scale structural changes."""
    with pm.Model() as model:
        b0 = pm.Normal("beta_0", mu=0, sigma=2)
        bt = pm.Normal("beta_time", mu=0, sigma=1)
        bp = pm.Normal("beta_pos", mu=0, sigma=3)
        bn = pm.Normal("beta_neg", mu=0, sigma=3)
        br = pm.Normal("beta_rho", mu=0, sigma=2)
        al = pm.Normal("alpha_local", mu=0, sigma=1.5)
        ag = pm.Normal("alpha_global", mu=0, sigma=2)
        be = pm.Normal("beta_escalation", mu=0, sigma=1)
        bd = pm.Normal("beta_deescalation", mu=0, sigma=1)

        _zero = np.float64(0.0)
        mu = (b0 + bt * h_features["time_trend"]
              + bp * h_features["delta_LambdaC_pos"]
              + bn * h_features["delta_LambdaC_neg"]
              + br * h_features["rho_T"]
              + al * h_features.get("local_rho_T", _zero)
              + ag * h_features.get("global_rho_T", _zero)
              + be * h_features.get("escalation_indicator", _zero)
              + bd * h_features.get("deescalation_indicator", _zero))

        sig = pm.HalfNormal("sigma_obs", sigma=1)
        pm.Normal("y_obs", mu=mu, sigma=sig, observed=data)

        trace = pm.sample(draws=config.draws, tune=config.tune,
                          target_accept=config.target_accept,
                          return_inferencedata=True, cores=4, chains=4)
    if logger is not None:
        logger.log_trace(trace, "hierarchical", "hierarchical", verbose=True)
    return trace, model


# ===================================================================
#  SECTION 6: STRUCTURAL REGIME DETECTION (Domain-agnostic base)
# ===================================================================

class StructuralRegimeDetector:
    """Detect structural regimes via clustering on Λ³ features.

    This is the **domain-agnostic base class**.  Domain extensions
    (financial Bull/Bear, medical acute/chronic, etc.) should
    subclass and override ``label_regimes`` / ``fit``.
    """

    def __init__(self, n_regimes: int = 3, method: str = "adaptive"):
        self.n_regimes = n_regimes
        self.method = method
        self.regime_labels: Optional[np.ndarray] = None
        self.regime_features: Dict[int, Dict[str, Any]] = {}

    def fit(
        self,
        features: Dict[str, np.ndarray],
        raw_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Cluster time points into structural regimes.

        Parameters
        ----------
        features : dict
            Must contain ``delta_LambdaC_pos``, ``delta_LambdaC_neg``,
            ``rho_T``.
        raw_data : array, optional
            Original series values — used for additional statistics.
        """
        X = np.column_stack([
            features["delta_LambdaC_pos"],
            features["delta_LambdaC_neg"],
            features["rho_T"],
        ])

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        if self.method == "adaptive":
            best_labels, best_score = None, -np.inf
            for meth in ("kmeans", "gmm"):
                if meth == "kmeans":
                    m = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=20)
                    lab = m.fit_predict(Xs)
                else:
                    m = GaussianMixture(n_components=self.n_regimes, random_state=42)
                    lab = m.fit_predict(Xs)
                try:
                    sc = silhouette_score(Xs, lab)
                    if sc > best_score:
                        best_score, best_labels = sc, lab
                except Exception:
                    continue
            labels = best_labels if best_labels is not None else KMeans(
                n_clusters=self.n_regimes, random_state=42).fit_predict(Xs)
        else:
            labels = KMeans(n_clusters=self.n_regimes,
                            random_state=42).fit_predict(Xs)

        self.regime_labels = labels
        self._compute_stats(X, labels, raw_data)
        return labels

    def _compute_stats(self, X, labels, raw_data):
        for r in range(self.n_regimes):
            mask = labels == r
            n_pts = int(np.sum(mask))
            if n_pts == 0:
                self.regime_features[r] = {"n_points": 0, "frequency": 0.0}
                continue
            stats: Dict[str, Any] = {
                "n_points": n_pts,
                "frequency": n_pts / len(labels),
                "mean_rhoT": float(np.mean(X[mask, 2])),
                "std_rhoT": float(np.std(X[mask, 2])),
                "mean_pos_events": float(np.mean(X[mask, 0])),
                "mean_neg_events": float(np.mean(X[mask, 1])),
                "event_asymmetry": float(np.mean(X[mask, 0]) - np.mean(X[mask, 1])),
            }
            self.regime_features[r] = stats

    def label_regimes(self) -> Dict[int, str]:
        """Return human-readable labels for detected regimes.

        Base implementation assigns tension-based generic labels.
        Domain extensions should override this.
        """
        if not self.regime_features:
            return {r: f"Regime-{r + 1}" for r in range(self.n_regimes)}

        # Sort by tension → highest tension = "High-Activity"
        ranked = sorted(
            self.regime_features.items(),
            key=lambda kv: kv[1].get("mean_rhoT", 0),
            reverse=True,
        )
        generic_labels = ["High-Activity", "Transitional", "Low-Activity", "Stable"]
        result: Dict[int, str] = {}
        for i, (r, _) in enumerate(ranked):
            lbl = generic_labels[i] if i < len(generic_labels) else f"Regime-{i + 1}"
            result[r] = lbl
        return result


# ===================================================================
#  SECTION 7: INTERACTION ANALYSIS
# ===================================================================

def _safe_extract(summary: pd.DataFrame, param: str, default: float = 0.0) -> float:
    return summary.loc[param, "mean"] if param in summary.index else default


def extract_interaction_coefficients(
    trace, series_names: List[str]
) -> Dict[str, Any]:
    """Extract structural interaction coefficients from a pairwise trace."""
    summary = az.summary(trace)
    na, nb = series_names[:2]

    se = {}
    for nm, sfx in ((na, "a"), (nb, "b")):
        se[nm] = {
            "pos_jump": _safe_extract(summary, f"beta_dLC_pos_{sfx}"),
            "neg_jump": _safe_extract(summary, f"beta_dLC_neg_{sfx}"),
            "tension": _safe_extract(summary, f"beta_rhoT_{sfx}"),
        }

    ce = {}
    for direction, sfx in ((f"{na}_to_{nb}", "ab"), (f"{nb}_to_{na}", "ba")):
        ce[direction] = {
            "pos_jump": _safe_extract(summary, f"beta_interact_{sfx}_pos"),
            "neg_jump": _safe_extract(summary, f"beta_interact_{sfx}_neg"),
            "tension": _safe_extract(summary, f"beta_interact_{sfx}_stress"),
        }

    return {
        "self_effects": se,
        "cross_effects": ce,
        "lag_effects": {
            f"{na}_to_{nb}": _safe_extract(summary, "beta_lag_ab"),
            f"{nb}_to_{na}": _safe_extract(summary, "beta_lag_ba"),
        },
        "correlation": _safe_extract(summary, "rho_ab"),
    }


def predict_with_interactions(
    trace, features_dict, series_names
) -> Dict[str, np.ndarray]:
    """Predict series values from a fitted pairwise model."""
    summary = az.summary(trace)
    preds = {}
    for idx, nm in enumerate(series_names[:2]):
        other = series_names[1 - idx]
        s = "a" if idx == 0 else "b"
        os = "b" if idx == 0 else "a"
        preds[nm] = (
            _safe_extract(summary, f"beta_0_{s}")
            + _safe_extract(summary, f"beta_time_{s}") * features_dict[nm]["time_trend"]
            + _safe_extract(summary, f"beta_dLC_pos_{s}") * features_dict[nm]["delta_LambdaC_pos"]
            + _safe_extract(summary, f"beta_dLC_neg_{s}") * features_dict[nm]["delta_LambdaC_neg"]
            + _safe_extract(summary, f"beta_rhoT_{s}") * features_dict[nm]["rho_T"]
            + _safe_extract(summary, f"beta_interact_{os}{s}_pos") * features_dict[other]["delta_LambdaC_pos"]
            + _safe_extract(summary, f"beta_interact_{os}{s}_neg") * features_dict[other]["delta_LambdaC_neg"]
            + _safe_extract(summary, f"beta_interact_{os}{s}_stress") * features_dict[other]["rho_T"]
        )
    return preds


# ===================================================================
#  SECTION 8: CAUSALITY ANALYSIS
# ===================================================================

def detect_structural_causality(
    features_dict: Dict[str, Dict[str, np.ndarray]],
    series_names: List[str],
    lag_window: int = 5,
) -> Dict[str, Dict[int, float]]:
    """Detect ΔΛC(t) → ΔΛC(t+k) causal patterns between two series."""
    if len(series_names) < 2:
        return {}
    na, nb = series_names[:2]
    events = {
        na: {"pos": features_dict[na]["delta_LambdaC_pos"],
             "neg": features_dict[na]["delta_LambdaC_neg"]},
        nb: {"pos": features_dict[nb]["delta_LambdaC_pos"],
             "neg": features_dict[nb]["delta_LambdaC_neg"]},
    }
    patterns: Dict[str, Dict[int, float]] = {}
    for fn, tn in ((na, nb), (nb, na)):
        for ft in ("pos", "neg"):
            for tt in ("pos", "neg"):
                key = f"{fn}_{ft}_to_{tn}_{tt}"
                patterns[key] = _lagged_causality(
                    events[fn][ft], events[tn][tt], lag_window
                )
    return patterns


def _lagged_causality(cause, effect, lag_window):
    out: Dict[int, float] = {}
    for lag in range(1, min(lag_window + 1, len(cause))):
        cp = cause[:-lag]
        ef = effect[lag:]
        jp = float(np.mean(cp * ef))
        pr = float(np.mean(cp))
        out[lag] = jp / (pr + 1e-8)
    return out


def analyze_comprehensive_causality(
    features_dict, series_names, lag_window=5, verbose=True
) -> Dict[str, Any]:
    """Full causality analysis: basic patterns + asymmetry + decay."""
    basic = detect_structural_causality(features_dict, series_names, lag_window)

    dir_strengths: Dict[str, Dict[str, Any]] = {}
    all_c: List[float] = []
    for d, pat in basic.items():
        if pat:
            mx = max(pat.values())
            mn = float(np.mean(list(pat.values())))
            ol = max(pat, key=pat.get)
            dir_strengths[d] = {"max": mx, "mean": mn, "optimal_lag": ol}
            all_c.extend(pat.values())

    strongest_d, strongest_s, strongest_l = "", 0.0, 0
    for d, s in dir_strengths.items():
        if s["max"] > strongest_s:
            strongest_d, strongest_s, strongest_l = d, s["max"], s["optimal_lag"]

    summary = {
        "max_causality": max(all_c) if all_c else 0.0,
        "mean_causality": float(np.mean(all_c)) if all_c else 0.0,
        "strongest_direction": strongest_d,
        "strongest_strength": strongest_s,
        "strongest_lag": strongest_l,
        "causality_density": len([c for c in all_c if c > 0.1]) / max(len(all_c), 1),
    }

    if verbose:
        print(f"\nCausality Analysis:")
        print(f"  Strongest: {strongest_d} = {strongest_s:.4f} @ lag {strongest_l}")
        for d, s in list(dir_strengths.items())[:5]:
            print(f"  {d}: max={s['max']:.4f}, mean={s['mean']:.4f}")

    return {"basic_causality": basic, "directional_strengths": dir_strengths,
            "summary": summary}


# ===================================================================
#  SECTION 9: SYNCHRONIZATION ANALYSIS
# ===================================================================

def calculate_sync_profile(
    a: np.ndarray, b: np.ndarray, lag_window: int = LAG_WINDOW_DEFAULT
) -> Tuple[Dict[int, float], float, int]:
    lags, vals, mx, opt = _sync_profile(
        a.astype(np.float64), b.astype(np.float64), lag_window)
    profile = {int(l): float(v) for l, v in zip(lags, vals)}
    return profile, float(mx), int(opt)


def sync_matrix(
    event_series: Dict[str, np.ndarray],
    lag_window: int = LAG_WINDOW_DEFAULT,
) -> Tuple[np.ndarray, List[str]]:
    names = list(event_series.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                mat[i, j] = 1.0
            else:
                sa = event_series[a].astype(np.float64)
                sb = event_series[b].astype(np.float64)
                _, _, mx, _ = _sync_profile(sa, sb, lag_window)
                mat[i, j] = mx
    return mat, names


def build_sync_network(
    event_series: Dict[str, np.ndarray],
    lag_window: int = LAG_WINDOW_DEFAULT,
    threshold: float = SYNC_THRESHOLD_DEFAULT,
) -> nx.DiGraph:
    G = nx.DiGraph()
    names = list(event_series.keys())
    for s in names:
        G.add_node(s)
    for sa in names:
        for sb in names:
            if sa == sb:
                continue
            prof, mx, opt = calculate_sync_profile(
                event_series[sa].astype(np.float64),
                event_series[sb].astype(np.float64),
                lag_window)
            if mx >= threshold:
                G.add_edge(sa, sb, weight=mx, lag=opt, profile=prof)
    return G


# ===================================================================
#  SECTION 10: STRUCTURAL CRISIS DETECTION
# ===================================================================

def detect_structural_crisis(
    features_dict: Dict[str, Dict[str, np.ndarray]],
    crisis_threshold: float = 0.8,
    tension_weight: float = 0.6,
) -> Dict[str, Any]:
    """Detect crisis periods from combined ρT and ΔΛC indicators."""
    indicators: Dict[str, np.ndarray] = {}
    for nm, f in features_dict.items():
        t = f["rho_T"]
        t_score = (t - np.mean(t)) / (np.std(t) + 1e-8)
        j = f["delta_LambdaC_pos"] + f["delta_LambdaC_neg"]
        j_score = (j - np.mean(j)) / (np.std(j) + 1e-8)
        indicators[nm] = tension_weight * t_score + (1 - tension_weight) * j_score

    agg = np.mean(list(indicators.values()), axis=0)
    crisis = agg > crisis_threshold

    diff = np.diff(np.concatenate([[False], crisis, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    episodes = list(zip(starts.tolist(), ends.tolist()))

    return {"crisis_periods": crisis, "crisis_episodes": episodes,
            "crisis_indicators": indicators, "aggregate_crisis": agg}


# ===================================================================
#  SECTION 11: HIERARCHICAL ANALYSIS
# ===================================================================

def hierarchy_metrics(sc: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute structural-change hierarchy metrics."""
    tlp = float(np.sum(sc.get("local_pos", 0)))
    tln = float(np.sum(sc.get("local_neg", 0)))
    tgp = float(np.sum(sc.get("global_pos", 0)))
    tgn = float(np.sum(sc.get("global_neg", 0)))
    plp = float(np.sum(sc.get("pure_local_pos", 0)))
    pln = float(np.sum(sc.get("pure_local_neg", 0)))
    mp = float(np.sum(sc.get("mixed_pos", 0)))
    mn = float(np.sum(sc.get("mixed_neg", 0)))
    total = tlp + tln + tgp + tgn
    return {
        "local_dominance": (tlp + tln) / max(total, 1),
        "global_dominance": (tgp + tgn) / max(total, 1),
        "coupling_strength": (mp + mn) / max(total, 1),
        "escalation_rate": (mp + mn) / max(plp + pln, 1),
    }


def prepare_hierarchical_features(
    sc: Dict[str, np.ndarray],
    data: np.ndarray,
    config: L3Config,
    verbose: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Prepare hierarchical features for Bayesian estimation."""
    lp = sc.get("local_pos", np.zeros_like(data, dtype=np.float64))
    ln = sc.get("local_neg", np.zeros_like(data, dtype=np.float64))
    gp = sc.get("global_pos", np.zeros_like(data, dtype=np.float64))
    gn = sc.get("global_neg", np.zeros_like(data, dtype=np.float64))
    rho = sc.get("rho_T", _rho_t(data, config.window))
    tt = sc.get("time_trend", np.arange(len(data), dtype=np.float64))

    c_pos = np.maximum(lp, gp)
    c_neg = np.maximum(ln, gn)
    l_mask = (lp + ln) > 0
    g_mask = (gp + gn) > 0

    esc_pts = np.where(np.diff(np.concatenate([[0], g_mask.astype(int)])) > 0)[0]
    desc_pts = np.where(np.diff(np.concatenate([[0], l_mask.astype(int)])) > 0)[0]

    stats = {
        "n_local_events": int(np.sum(l_mask)),
        "n_global_events": int(np.sum(g_mask)),
        "n_mixed": int(np.sum(l_mask & g_mask)),
        "n_escalations": len(esc_pts),
        "n_deescalations": len(desc_pts),
        "mean_local_rhoT": float(np.mean(rho[l_mask])) if np.sum(l_mask) > 0 else 0,
        "mean_global_rhoT": float(np.mean(rho[g_mask])) if np.sum(g_mask) > 0 else 0,
    }

    if verbose:
        print(f"  Hierarchy — Local={stats['n_local_events']}, "
              f"Global={stats['n_global_events']}, Mixed={stats['n_mixed']}")

    feats = {
        "delta_LambdaC_pos": c_pos,
        "delta_LambdaC_neg": c_neg,
        "rho_T": rho,
        "time_trend": tt,
        "local_rho_T": rho * l_mask.astype(float),
        "global_rho_T": rho * g_mask.astype(float),
        "escalation_indicator": np.diff(np.concatenate([[0], g_mask.astype(float)])),
        "deescalation_indicator": np.diff(np.concatenate([[0], l_mask.astype(float)])),
    }
    return feats, stats


def analyze_hierarchical_separation(
    name: str,
    data: np.ndarray,
    sc: Dict[str, np.ndarray],
    config: L3Config,
    verbose: bool = True,
    logger: Optional[BayesianHDILogger] = None,
) -> Dict[str, Any]:
    """Analyse hierarchical separation dynamics for a single series."""
    if verbose:
        print(f"\nHierarchical separation — {name}")

    h_feats, h_stats = prepare_hierarchical_features(sc, data, config, verbose)
    l_mask = h_feats["local_rho_T"] > 0
    g_mask = h_feats["global_rho_T"] > 0
    nl, ng = int(np.sum(l_mask)), int(np.sum(g_mask))

    if nl < 10 or ng < 10:
        if verbose:
            print(f"  ⚠ Insufficient events (local={nl}, global={ng})")
        return {"insufficient_data": True, "hierarchical_stats": h_stats}

    total = nl + ng
    draws = min(4000, config.draws) if total < 100 else config.draws
    tune = min(4000, config.tune) if total < 100 else config.tune

    trace, model = fit_hierarchical_bayesian(
        data, h_feats,
        L3Config(draws=draws, tune=tune, target_accept=config.target_accept),
        logger=logger,
    )

    summary = az.summary(trace)
    coeffs = {
        "escalation": _safe_extract(summary, "beta_escalation"),
        "deescalation": _safe_extract(summary, "beta_deescalation"),
        "local_effect": _safe_extract(summary, "alpha_local"),
        "global_effect": _safe_extract(summary, "alpha_global"),
    }

    if verbose:
        print(f"  Escalation={coeffs['escalation']:.4f}, "
              f"De-esc={coeffs['deescalation']:.4f}")

    return {
        "trace": trace, "model": model,
        "separation_coefficients": coeffs,
        "hierarchical_stats": h_stats,
        "local_series": h_feats["local_rho_T"],
        "global_series": h_feats["global_rho_T"],
        "series_name": name,
    }


def complete_hierarchical_analysis(
    data_dict: Dict[str, np.ndarray],
    config: L3Config,
    series_names: Optional[List[str]] = None,
    verbose: bool = True,
    logger: Optional[BayesianHDILogger] = None,
) -> Dict[str, Any]:
    """Full hierarchical analysis across all series + pairwise sync."""
    if series_names is None:
        series_names = list(data_dict.keys())
    if logger is None:
        logger = BayesianHDILogger(hdi_prob=config.hdi_prob)

    results: Dict[str, Any] = {"bayes_logger": logger}

    if verbose:
        print("=" * 70)
        print("HIERARCHICAL STRUCTURAL ANALYSIS")
        print("=" * 70)

    h_config = L3Config(
        window=config.window, local_window=config.local_window,
        global_window=config.global_window, hierarchical=True,
        local_threshold_percentile=config.local_threshold_percentile,
        global_threshold_percentile=config.global_threshold_percentile,
        draws=config.draws, tune=config.tune,
        target_accept=config.target_accept, hdi_prob=config.hdi_prob,
    )

    for nm in series_names:
        if nm not in data_dict:
            continue
        d = data_dict[nm]
        sc = calc_lambda3_features(d, h_config)
        hm = hierarchy_metrics(sc)
        sep = analyze_hierarchical_separation(nm, d, sc, config, verbose, logger)
        results[nm] = {
            "structural_changes": sc,
            "hierarchy_metrics": hm,
            "hierarchical_separation": sep,
            "data": d,
        }

    # Pairwise hierarchical synchronization (first pair)
    if len(series_names) >= 2:
        a, b = series_names[0], series_names[1]
        if a in results and b in results:
            sc_a = results[a]["structural_changes"]
            sc_b = results[b]["structural_changes"]
            required = ("pure_local_pos", "pure_local_neg",
                        "pure_global_pos", "pure_global_neg")
            if all(k in sc_a and k in sc_b for k in required):
                try:
                    sync_res = _hierarchical_sync(
                        sc_a, sc_b, a, b, config, verbose, logger)
                    results["hierarchical_synchronization"] = sync_res

                    feats_caus = {
                        a: {k: sc_a[k] for k in ("delta_LambdaC_pos", "delta_LambdaC_neg",
                                                   "rho_T", "time_trend")},
                        b: {k: sc_b[k] for k in ("delta_LambdaC_pos", "delta_LambdaC_neg",
                                                   "rho_T", "time_trend")},
                    }
                    results["hierarchical_causality"] = detect_structural_causality(
                        feats_caus, [a, b], lag_window=5)
                except Exception as e:
                    if verbose:
                        print(f"  Hierarchical sync error: {e}")

    if verbose:
        print(f"\n{'=' * 70}")
        summary_df = logger.generate_summary_report()
        if not summary_df.empty:
            print(summary_df.to_string(index=False))

    return results


def _hierarchical_sync(sc1, sc2, n1, n2, config, verbose, logger):
    """Helper: pairwise hierarchical synchronization via Bayesian model."""
    if verbose:
        print(f"\nHierarchical sync — {n1} ⇄ {n2}")

    data_d = {n1: sc1.get("data", np.zeros(100)),
              n2: sc2.get("data", np.zeros(100))}
    feats_d = {n1: sc1, n2: sc2}

    trace, model = fit_pairwise_system(
        data_d, feats_d, config, series_pair=(n1, n2), logger=logger)
    ic = extract_interaction_coefficients(trace, [n1, n2])
    ce = ic["cross_effects"]

    s12 = sum(abs(v) for v in ce[f"{n1}_to_{n2}"].values()) / 3
    s21 = sum(abs(v) for v in ce[f"{n2}_to_{n1}"].values()) / 3

    if verbose:
        print(f"  {n1}→{n2}={s12:.3f}, {n2}→{n1}={s21:.3f}")

    return {
        "trace": trace, "model": model,
        "sync_strength_1_to_2": s12, "sync_strength_2_to_1": s21,
        "overall_sync_strength": (s12 + s21) / 2,
        "asymmetry": abs(s12 - s21),
        "interaction_coefficients": ic,
        "series_names": [n1, n2],
    }


# ===================================================================
#  SECTION 12: ALL-PAIRS INTERACTION ANALYSIS
# ===================================================================

def analyze_all_pairwise(
    series_dict: Dict[str, np.ndarray],
    features_dict: Dict[str, Dict[str, np.ndarray]],
    config: L3Config,
    max_pairs: Optional[int] = None,
    logger: Optional[BayesianHDILogger] = None,
) -> Dict[str, Any]:
    """Exhaustive pairwise structural interaction analysis."""
    if logger is None:
        logger = BayesianHDILogger(hdi_prob=config.hdi_prob)

    names = list(series_dict.keys())
    n = len(names)
    all_pairs = list(combinations(names, 2))
    if max_pairs and len(all_pairs) > max_pairs:
        all_pairs = all_pairs[:max_pairs]

    print(f"\n{'=' * 70}\nPairwise Interaction Analysis — {len(all_pairs)} pairs\n{'=' * 70}")

    results: Dict[str, Any] = {
        "pairs": {},
        "interaction_matrix": np.zeros((n, n)),
        "asymmetry_matrix": np.zeros((n, n)),
        "strongest_interactions": [],
    }

    for pi, (na, nb) in enumerate(all_pairs):
        print(f"\n{'─' * 50}\nPair {pi + 1}/{len(all_pairs)}: {na} ⇄ {nb}")
        try:
            trace, model = fit_pairwise_system(
                {na: series_dict[na], nb: series_dict[nb]},
                {na: features_dict[na], nb: features_dict[nb]},
                config, series_pair=(na, nb), logger=logger)

            ic = extract_interaction_coefficients(trace, [na, nb])
            preds = predict_with_interactions(
                trace, {na: features_dict[na], nb: features_dict[nb]}, [na, nb])
            caus = detect_structural_causality(
                {na: features_dict[na], nb: features_dict[nb]}, [na, nb])

            pk = f"{na}_vs_{nb}"
            results["pairs"][pk] = {
                "trace": trace, "model": model,
                "interaction_coefficients": ic,
                "predictions": preds,
                "causality_patterns": caus,
            }

            ce = ic["cross_effects"]
            s_ab = sum(abs(v) for v in ce[f"{na}_to_{nb}"].values())
            s_ba = sum(abs(v) for v in ce[f"{nb}_to_{na}"].values())

            ia, ib = names.index(na), names.index(nb)
            results["interaction_matrix"][ia, ib] = s_ab
            results["interaction_matrix"][ib, ia] = s_ba
            asym = abs(s_ab - s_ba)
            results["asymmetry_matrix"][ia, ib] = asym
            results["asymmetry_matrix"][ib, ia] = asym

            results["strongest_interactions"].append({
                "pair": pk, "total_strength": s_ab + s_ba,
                "asymmetry": asym,
                "dominant_direction": f"{na}→{nb}" if s_ab > s_ba else f"{nb}→{na}",
            })
            print(f"  {na}→{nb}: {s_ab:.3f}  |  {nb}→{na}: {s_ba:.3f}")
        except Exception as e:
            print(f"  ⚠ Error: {e}")

    results["strongest_interactions"].sort(
        key=lambda x: x["total_strength"], reverse=True)

    iv = results["interaction_matrix"][results["interaction_matrix"] > 0]
    results["summary"] = {
        "total_pairs": len(results["pairs"]),
        "max_interaction": float(np.max(iv)) if len(iv) else 0,
        "mean_interaction": float(np.mean(iv)) if len(iv) else 0,
        "strongest_pair": results["strongest_interactions"][0]
            if results["strongest_interactions"] else None,
        "bayes_logger": logger,
    }

    print(f"\n{'=' * 70}\nPairwise complete — "
          f"max interaction: {results['summary']['max_interaction']:.4f}")
    summary_df = logger.generate_summary_report()
    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    return results


# ===================================================================
#  SECTION 13: VISUALIZATION
# ===================================================================

class Lambda3Visualizer:
    """Integrated visualization for Λ³ structural tensor dynamics."""

    def __init__(self, style: str = "scientific"):
        self.style = style
        if style == "scientific":
            plt.style.use("seaborn-v0_8-darkgrid")
        self.C = {
            "pos": "#e74c3c", "neg": "#3498db", "tension": "#2ecc71",
            "local": "#9b59b6", "global": "#f39c12",
        }

    def plot_tensor_evolution(self, features_dict, series_names, figsize=(16, 8)):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

        # ΔΛC pulsation timeline
        ax1 = fig.add_subplot(gs[0])
        for i, nm in enumerate(series_names):
            f = features_dict[nm]
            pi = np.where(f["delta_LambdaC_pos"] > 0)[0]
            ni = np.where(f["delta_LambdaC_neg"] > 0)[0]
            if len(pi):
                ax1.vlines(pi, i, i + 0.4, colors=self.C["pos"], alpha=0.7, lw=2)
            if len(ni):
                ax1.vlines(ni, i, i - 0.4, colors=self.C["neg"], alpha=0.7, lw=2)
            ax1.text(-5, i, nm, ha="right", va="center", fontsize=10)
        ax1.set_ylim(-0.5, len(series_names) - 0.5)
        ax1.set_title("ΔΛC Pulsations")
        ax1.plot([], [], color=self.C["pos"], lw=2, label="ΔΛC⁺")
        ax1.plot([], [], color=self.C["neg"], lw=2, label="ΔΛC⁻")
        ax1.legend(loc="upper right")

        # ρT evolution
        ax2 = fig.add_subplot(gs[1])
        for nm in series_names:
            ax2.plot(features_dict[nm]["rho_T"], label=nm, alpha=0.8)
        ax2.set_ylabel("ρT")
        ax2.set_title("Tension scalar evolution")
        ax2.legend(loc="upper right", ncol=min(3, len(series_names)))

        # Coherence
        ax3 = fig.add_subplot(gs[2])
        w = 20
        n_w = len(features_dict[series_names[0]]["data"]) - w + 1
        coh = []
        for t in range(0, n_w, 5):
            vals = []
            for i in range(len(series_names)):
                for j in range(i + 1, len(series_names)):
                    ri = features_dict[series_names[i]]["rho_T"][t:t + w]
                    rj = features_dict[series_names[j]]["rho_T"][t:t + w]
                    if len(ri) > 1 and len(rj) > 1:
                        vals.append(abs(np.corrcoef(ri, rj)[0, 1]))
            coh.append(float(np.mean(vals)) if vals else 0)
        tp = range(0, n_w, 5)
        ax3.fill_between(tp, coh, alpha=0.3)
        ax3.plot(tp, coh, lw=2)
        ax3.set_ylabel("Coherence")
        ax3.set_ylim(0, 1)

        fig.suptitle("Structural Tensor Evolution", fontsize=16, fontweight="bold")
        return fig

    def plot_interaction_network(self, results, series_names, figsize=(12, 10)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        G = nx.DiGraph()
        for nm in series_names:
            G.add_node(nm)
        if "interaction_matrix" in results:
            mat = results["interaction_matrix"]
            for i, ni in enumerate(series_names):
                for j, nj in enumerate(series_names):
                    if i != j and mat[i, j] > 0:
                        G.add_edge(ni, nj, weight=mat[i, j])
        pos = nx.spring_layout(G, k=2, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_color="lightblue",
                               node_size=3000, alpha=0.9, ax=ax)
        edges = list(G.edges())
        weights = [G[u][v]["weight"] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w * 5 for w in weights],
                               alpha=0.6, arrows=True, arrowsize=20, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)
        el = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges if G[u][v]["weight"] > 0.1}
        nx.draw_networkx_edge_labels(G, pos, el, font_size=9, ax=ax)
        ax.set_title("Structural Interaction Network", fontsize=16)
        ax.axis("off")
        return fig

    def plot_hierarchical_dynamics(self, h_results, figsize=(14, 8)):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                        gridspec_kw={"height_ratios": [2, 1]})
        sd = None
        sn = None
        for k, v in h_results.items():
            if isinstance(v, dict) and "hierarchical_separation" in v:
                sd, sn = v, k
                break
        if sd is None:
            ax1.text(0.5, 0.5, "No hierarchical data", transform=ax1.transAxes,
                     ha="center", va="center")
            return fig
        sep = sd["hierarchical_separation"]
        ls = sep.get("local_series", np.zeros(100))
        gs = sep.get("global_series", np.zeros(100))
        t = np.arange(len(ls))
        ax1.fill_between(t, 0, ls, color=self.C["local"], alpha=0.3, label="Local")
        ax1.fill_between(t, 0, -gs, color=self.C["global"], alpha=0.3, label="Global")
        ax1.axhline(0, color="black", alpha=0.5)
        ax1.set_title(f"Local ⇄ Global — {sn}")
        ax1.legend()
        m = sd.get("hierarchy_metrics", {})
        labels = ["Local\nDominance", "Global\nDominance",
                  "Coupling", "Escalation"]
        vals = [m.get("local_dominance", 0), m.get("global_dominance", 0),
                m.get("coupling_strength", 0), m.get("escalation_rate", 0)]
        bars = ax2.bar(range(len(labels)), vals,
                       color=["#9b59b6", "#f39c12", "#2ecc71", "#e74c3c"])
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f"{v:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        return fig


# ===================================================================
#  SECTION 14: FULL ANALYSIS PIPELINE
# ===================================================================

def run_lambda3_analysis(
    data_source: Union[str, Dict[str, np.ndarray]],
    config: Optional[L3Config] = None,
    target_series: Optional[List[str]] = None,
    max_pairs: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Lambda³ full analysis pipeline.

    Executes all stages of structural tensor analysis:
      1. Feature extraction (ΔΛC, ρT, hierarchical)
      2. Hierarchical structural analysis
      3. Pairwise interaction analysis
      4. Structural regime detection
      5. Crisis detection
      6. Causality analysis
      7. Synchronization analysis

    Parameters
    ----------
    data_source : str or dict
        CSV file path or ``{name: np.ndarray}`` dictionary.
    config : L3Config, optional
        Analysis configuration. Uses defaults if None.
    target_series : list of str, optional
        Subset of series to analyse. None → all.
    max_pairs : int
        Maximum number of pairwise analyses.
    verbose : bool
        Print progress and summaries.

    Returns
    -------
    dict
        Complete analysis results with Bayesian HDI data.
    """
    if config is None:
        config = L3Config()

    # --- Load ---
    if isinstance(data_source, str):
        series_dict = load_csv_data(data_source, time_column="Date")
    else:
        series_dict = data_source

    if target_series:
        series_dict = {k: v for k, v in series_dict.items() if k in target_series}

    names = list(series_dict.keys())
    logger = BayesianHDILogger(hdi_prob=config.hdi_prob)

    if verbose:
        print("=" * 70)
        print("Λ³ CORE — Structural Tensor Analysis")
        print(f"Series: {len(names)}, Length: {len(next(iter(series_dict.values())))}")
        print("=" * 70)

    results: Dict[str, Any] = {
        "series_dict": series_dict,
        "series_names": names,
        "config": config,
        "bayes_logger": logger,
    }

    # --- Stage 1: Feature extraction ---
    if verbose:
        print("\n[Stage 1] Feature extraction")
    features_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for nm, d in series_dict.items():
        f = calc_lambda3_features(d, config)
        features_dict[nm] = f
        if verbose:
            print(f"  {nm}: ΔΛC⁺={np.sum(f['delta_LambdaC_pos']):.0f}, "
                  f"ΔΛC⁻={np.sum(f['delta_LambdaC_neg']):.0f}, "
                  f"ρT={np.mean(f['rho_T']):.3f}")
    results["features_dict"] = features_dict

    # --- Stage 2: Hierarchical analysis ---
    if verbose:
        print("\n[Stage 2] Hierarchical structural analysis")
    h_res = complete_hierarchical_analysis(
        series_dict, config, verbose=verbose, logger=logger)
    results["hierarchical_results"] = h_res

    # --- Stage 3: Pairwise interactions ---
    if len(names) >= 2:
        if verbose:
            print("\n[Stage 3] Pairwise interaction analysis")
        pw_res = analyze_all_pairwise(
            series_dict, features_dict, config,
            max_pairs=max_pairs, logger=logger)
        results["pairwise_results"] = pw_res

    # --- Stage 4: Regime detection ---
    if verbose:
        print("\n[Stage 4] Structural regime detection")
    regime_results: Dict[str, Any] = {}
    for nm in names:
        det = StructuralRegimeDetector(n_regimes=4)
        labels = det.fit(features_dict[nm], series_dict[nm])
        regime_results[nm] = {
            "labels": labels,
            "regime_names": det.label_regimes(),
            "features": det.regime_features,
        }
        if verbose:
            print(f"  {nm}: {len(set(labels))} regimes — "
                  f"{det.label_regimes()}")
    results["regime_results"] = regime_results

    # --- Stage 5: Crisis detection ---
    if verbose:
        print("\n[Stage 5] Structural crisis detection")
    crisis = detect_structural_crisis(features_dict)
    results["crisis_results"] = crisis
    if verbose:
        print(f"  Crisis episodes: {len(crisis['crisis_episodes'])}")

    # --- Stage 6: Causality ---
    if len(names) >= 2 and verbose:
        print("\n[Stage 6] Causality analysis")
    causality = analyze_comprehensive_causality(
        features_dict, names[:2], verbose=verbose)
    results["causality_results"] = causality

    # --- Stage 7: Synchronization ---
    if verbose:
        print("\n[Stage 7] Synchronization analysis")
    ev_series = {nm: f["delta_LambdaC_pos"].astype(np.float64)
                 for nm, f in features_dict.items()}
    s_mat, s_names = sync_matrix(ev_series)
    results["sync_matrix"] = s_mat
    results["sync_names"] = s_names

    # --- Summary ---
    if verbose:
        print("\n" + "=" * 70)
        print("Λ³ Analysis Summary")
        print("=" * 70)
        tp = sum(np.sum(f["delta_LambdaC_pos"]) for f in features_dict.values())
        tn = sum(np.sum(f["delta_LambdaC_neg"]) for f in features_dict.values())
        at = np.mean([np.mean(f["rho_T"]) for f in features_dict.values()])
        print(f"  Total ΔΛC⁺: {tp:.0f}, ΔΛC⁻: {tn:.0f}, Avg ρT: {at:.3f}")
        if "pairwise_results" in results:
            s = results["pairwise_results"]["summary"]
            print(f"  Max interaction: {s['max_interaction']:.4f}")
            if s["strongest_pair"]:
                print(f"  Strongest pair: {s['strongest_pair']['pair']}")
        c_s = causality.get("summary", {})
        print(f"  Causality: {c_s.get('strongest_direction', 'N/A')} "
              f"= {c_s.get('strongest_strength', 0):.4f}")
        print(f"  Crisis episodes: {len(crisis['crisis_episodes'])}")

        print(f"\n{'=' * 70}")
        print("BAYESIAN HDI SUMMARY")
        print(f"{'=' * 70}")
        df = logger.generate_summary_report()
        if not df.empty:
            print(df.to_string(index=False))
        print("\n" + "=" * 70)
        print("Analysis complete — structural tensor dynamics revealed")
        print("=" * 70)

    return results


# ===================================================================
#  SECTION 15: UTILITY / BATCH / STREAMING
# ===================================================================

def lambda3_batch(
    data_files: List[str],
    output_dir: str = "lambda3_results",
    config: Optional[L3Config] = None,
) -> Dict[str, Dict[str, Any]]:
    """Batch analysis across multiple CSV files."""
    Path(output_dir).mkdir(exist_ok=True)
    batch: Dict[str, Dict[str, Any]] = {}
    for f in data_files:
        print(f"\nProcessing: {f}")
        try:
            r = run_lambda3_analysis(f, config, verbose=False)
            out = Path(output_dir) / f"{Path(f).stem}_results.pkl"
            with open(out, "wb") as fh:
                pickle.dump(r, fh)
            batch[f] = r
        except Exception as e:
            print(f"  Error: {e}")
            batch[f] = {"error": str(e)}
    return batch


def lambda3_streaming(
    initial_data: Dict[str, np.ndarray],
    update_callback: Callable,
    window_size: int = 100,
    update_interval: int = 10,
    crisis_threshold: float = 0.8,
) -> None:
    """Real-time streaming analysis with crisis alerts."""
    config = L3Config(T=window_size, draws=4000, tune=4000)
    current = {k: v[-window_size:] for k, v in initial_data.items()}
    it = 0
    while True:
        try:
            new = update_callback()
            if new is None:
                break
            for nm, vals in new.items():
                if nm in current:
                    current[nm] = np.concatenate([current[nm][len(vals):], vals])
            if it % update_interval == 0:
                fd = {nm: calc_lambda3_features(d, config) for nm, d in current.items()}
                cr = detect_structural_crisis(fd)
                score = float(np.mean(cr["aggregate_crisis"]))
                msg = f"[iter {it}] Crisis score: {score:.3f}"
                if score > crisis_threshold:
                    msg += " ⚠ ALERT"
                print(msg)
            it += 1
        except KeyboardInterrupt:
            print("\nStreaming stopped")
            break
        except Exception as e:
            print(f"Error: {e}")


# ===================================================================
#  MODULE ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Λ³ CORE — Universal Structural Tensor Field Engine")
    print("=" * 60)
    print()
    print("Usage:")
    print("  from lambda3_core import run_lambda3_analysis, L3Config")
    print()
    print("  # Analyse CSV data")
    print('  results = run_lambda3_analysis("data.csv")')
    print()
    print("  # Analyse dict data")
    print("  results = run_lambda3_analysis({")
    print('      "Series_A": np.cumsum(np.random.randn(500)),')
    print('      "Series_B": np.cumsum(np.random.randn(500)),')
    print("  })")
    print()
    print("  # With custom config")
    print("  results = run_lambda3_analysis(data, config=L3Config(")
    print("      draws=4000, tune=4000, hierarchical=True")
    print("  ))")
    print()
    print("Domain extensions should subclass StructuralRegimeDetector")
    print("and override label_regimes() for domain-specific labelling.")
    print("=" * 60)
