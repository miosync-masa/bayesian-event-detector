# ==========================================================
# Λ³ABC: Lambda³ Analytics for Bayes & CausalJunction
# ----------------------------------------------------
# Refactored Framework for Structural Tensor Analysis
# No time causality - structural pulsations (∆ΛC)
#
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# Version: 2.0 (Refactored)
# ==========================================================

# ===============================
# IMPORTS
# ===============================
import pymc as pm
import numpy as np
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import networkx as nx
from numba import jit, njit, prange
from typing import Tuple, Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
from itertools import combinations
import yfinance as yf
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

# ===============================
# GLOBAL CONSTANTS
# ===============================
DELTA_PERCENTILE = 97.0
LOCAL_JUMP_PERCENTILE = 95.0
WINDOW_SIZE = 10
LOCAL_WINDOW_SIZE = 5
LAG_WINDOW_DEFAULT = 10
SYNC_THRESHOLD_DEFAULT = 0.3

# ===============================
# CONFIGURATION
# ===============================
@dataclass
class L3Config:
    """Configuration for Lambda³ analysis parameters."""
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
    var_names: list = ('beta_time_a', 'beta_time_b', 'beta_interact', 'beta_rhoT_a', 'beta_rhoT_b')
    hdi_prob: float = 0.94
    hierarchical: bool = True  # Enable hierarchical analysis by default

# ===============================
# SECTION 0: DATA HANDLING
# ===============================
def fetch_financial_data(
    start_date="2022-01-01",
    end_date="2024-12-31",
    tickers=None,
    desired_order=None,
    csv_filename="financial_data_2022to2024.csv",
    verbose=True
):
    """Fetch, preprocess, and save multi-market financial time series data."""
    if tickers is None:
        tickers = {
            "USD/JPY": "JPY=X",
            "OIL": "CL=F",
            "GOLD": "GC=F",
            "Nikkei 225": "^N225",
            "Dow Jones": "^DJI"
        }
    if desired_order is None:
        desired_order = ["USD/JPY", "OIL", "GOLD", "Nikkei 225", "Dow Jones"]

    if verbose:
        print(f"Fetching daily data from {start_date} to {end_date}...")

    try:
        all_data = {}
        for name, ticker in tickers.items():
            try:
                ticker_data = yf.download(ticker, start=start_date, end=end_date)['Close']
                if len(ticker_data) > 0:
                    all_data[name] = ticker_data
                    if verbose:
                        print(f"  {name}: {len(ticker_data)} data points")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to download {name} ({ticker}): {e}")
                continue

        if not all_data:
            raise ValueError("No data could be downloaded")

        common_dates = None
        for name, data in all_data.items():
            if common_dates is None:
                common_dates = set(data.index)
            else:
                common_dates = common_dates.intersection(set(data.index))

        common_dates = sorted(list(common_dates))

        if len(common_dates) < 50:
            raise ValueError(f"Insufficient common dates: {len(common_dates)}")

        final_data = pd.DataFrame(index=common_dates)
        for name in desired_order:
            if name in all_data:
                final_data[name] = all_data[name].reindex(common_dates)

        final_data = final_data.dropna()

        if verbose:
            print(f"\nFinal data shape: {final_data.shape}")
            print(f"Date range: {final_data.index[0]} to {final_data.index[-1]}")
            print(final_data.head())

        final_data.to_csv(csv_filename, index=True)
        if verbose:
            print(f"\nData saved to '{csv_filename}'.")

        return final_data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def load_csv_data(filepath: str, time_column: Optional[str] = None,
                  value_columns: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    df = pd.read_csv(filepath, parse_dates=True)

    print(f"Loaded CSV with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if time_column and time_column in df.columns:
        df = df.sort_values(by=time_column)

    if value_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_column and time_column in numeric_cols:
            numeric_cols.remove(time_column)
        value_columns = numeric_cols

    series_dict = {}
    for col in value_columns:
        if col in df.columns:
            data = df[col].values
            if pd.isna(data).any():
                data = pd.Series(data).ffill().bfill().values
            series_dict[col] = data.astype(np.float64)

    return series_dict

# ===============================
# SECTION 1: JIT-COMPILED CORE FUNCTIONS
# ===============================

@njit
def calculate_diff_and_threshold(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """JIT-compiled difference calculation and threshold computation."""
    diff = np.empty(len(data))
    diff[0] = 0
    for i in range(1, len(data)):
        diff[i] = data[i] - data[i-1]

    abs_diff = np.abs(diff)
    threshold = np.percentile(abs_diff, percentile)
    return diff, threshold

@njit
def detect_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """JIT-compiled jump detection based on threshold."""
    n = len(diff)
    pos_jumps = np.zeros(n, dtype=np.int32)
    neg_jumps = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if diff[i] > threshold:
            pos_jumps[i] = 1
        elif diff[i] < -threshold:
            neg_jumps[i] = 1

    return pos_jumps, neg_jumps

@njit
def calculate_local_std(data: np.ndarray, window: int) -> np.ndarray:
    """JIT-compiled local standard deviation calculation."""
    n = len(data)
    local_std = np.empty(n)

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)

        subset = data[start:end]
        mean = np.mean(subset)
        variance = np.sum((subset - mean) ** 2) / len(subset)
        local_std[i] = np.sqrt(variance)

    return local_std

@njit
def calculate_rho_t(data: np.ndarray, window: int) -> np.ndarray:
    """JIT-compiled tension scalar (ρT) calculation."""
    n = len(data)
    rho_t = np.empty(n)

    for i in range(n):
        start = max(0, i - window)
        end = i + 1

        subset = data[start:end]
        if len(subset) > 1:
            mean = np.mean(subset)
            variance = np.sum((subset - mean) ** 2) / len(subset)
            rho_t[i] = np.sqrt(variance)
        else:
            rho_t[i] = 0.0

    return rho_t

@njit
def sync_rate_at_lag(series_a: np.ndarray, series_b: np.ndarray, lag: int) -> float:
    """JIT-compiled synchronization rate calculation for a specific lag."""
    if lag < 0:
        if -lag < len(series_a):
            return np.mean(series_a[-lag:] * series_b[:lag])
        else:
            return 0.0
    elif lag > 0:
        if lag < len(series_b):
            return np.mean(series_a[:-lag] * series_b[lag:])
        else:
            return 0.0
    else:
        return np.mean(series_a * series_b)

@njit(parallel=True)
def calculate_sync_profile_jit(series_a: np.ndarray, series_b: np.ndarray,
                               lag_window: int) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """JIT-compiled synchronization profile calculation with parallelization."""
    n_lags = 2 * lag_window + 1
    lags = np.arange(-lag_window, lag_window + 1)
    sync_values = np.empty(n_lags)

    for i in prange(n_lags):
        lag = lags[i]
        sync_values[i] = sync_rate_at_lag(series_a, series_b, lag)

    max_sync = 0.0
    optimal_lag = 0
    for i in range(n_lags):
        if sync_values[i] > max_sync:
            max_sync = sync_values[i]
            optimal_lag = lags[i]

    return lags, sync_values, max_sync, optimal_lag

@njit
def detect_local_global_jumps(
    data: np.ndarray,
    local_window: int = 10,
    global_window: int = 50,
    local_percentile: float = 95.0,
    global_percentile: float = 97.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hierarchical structural change detection
    Lambda³ theory: Hierarchy in structural tensor changes
    """
    n = len(data)
    diff = np.empty(n)
    diff[0] = 0
    for i in range(1, n):
        diff[i] = data[i] - data[i-1]

    local_pos = np.zeros(n, dtype=np.int32)
    local_neg = np.zeros(n, dtype=np.int32)
    global_pos = np.zeros(n, dtype=np.int32)
    global_neg = np.zeros(n, dtype=np.int32)

    # Local criteria
    for i in range(n):
        local_start = max(0, i - local_window)
        local_end = min(n, i + local_window + 1)
        local_subset = np.abs(diff[local_start:local_end])

        if len(local_subset) > 0:
            local_threshold = np.percentile(local_subset, local_percentile)
            if diff[i] > local_threshold:
                local_pos[i] = 1
            elif diff[i] < -local_threshold:
                local_neg[i] = 1

    # Global criteria
    global_threshold_pos = np.percentile(np.abs(diff), global_percentile)
    global_threshold_neg = -global_threshold_pos

    for i in range(n):
        if diff[i] > global_threshold_pos:
            global_pos[i] = 1
        elif diff[i] < global_threshold_neg:
            global_neg[i] = 1

    return local_pos, local_neg, global_pos, global_neg

# ===============================
# SECTION 2: UNIFIED FEATURE EXTRACTION
# ===============================

def calc_lambda3_features(data: np.ndarray, config: L3Config) -> Dict[str, np.ndarray]:
    """
    Unified Lambda³ feature extraction with hierarchical support.
    Extracts structural change (ΔΛC) and tension scalar (ρT) features.
    """
    if config.hierarchical:
        # Hierarchical structural change detection
        local_pos, local_neg, global_pos, global_neg = detect_local_global_jumps(
            data,
            config.local_window,
            config.global_window,
            config.local_threshold_percentile,
            config.global_threshold_percentile
        )
        
        # Combined structural changes
        combined_pos = np.maximum(local_pos.astype(np.float64), global_pos.astype(np.float64))
        combined_neg = np.maximum(local_neg.astype(np.float64), global_neg.astype(np.float64))
        
        # Hierarchical classification
        n = len(data)
        pure_local_pos = np.zeros(n, dtype=np.float64)
        pure_local_neg = np.zeros(n, dtype=np.float64)
        pure_global_pos = np.zeros(n, dtype=np.float64)
        pure_global_neg = np.zeros(n, dtype=np.float64)
        mixed_pos = np.zeros(n, dtype=np.float64)
        mixed_neg = np.zeros(n, dtype=np.float64)

        for i in range(n):
            # Positive structural changes
            if local_pos[i] and global_pos[i]:
                mixed_pos[i] = 1.0
            elif local_pos[i] and not global_pos[i]:
                pure_local_pos[i] = 1.0
            elif not local_pos[i] and global_pos[i]:
                pure_global_pos[i] = 1.0

            # Negative structural changes
            if local_neg[i] and global_neg[i]:
                mixed_neg[i] = 1.0
            elif local_neg[i] and not global_neg[i]:
                pure_local_neg[i] = 1.0
            elif not local_neg[i] and global_neg[i]:
                pure_global_neg[i] = 1.0

        # Basic features
        rho_t = calculate_rho_t(data, config.window)
        time_trend = np.arange(len(data))
        
        # Local jump detection for compatibility
        local_std = calculate_local_std(data, config.local_window)
        diff, _ = calculate_diff_and_threshold(data, config.delta_percentile)
        score = np.abs(diff) / (local_std + 1e-8)
        local_threshold = np.percentile(score, config.local_jump_percentile)
        local_jump_detect = (score > local_threshold).astype(int)

        return {
            'data': data,
            'delta_LambdaC_pos': combined_pos,
            'delta_LambdaC_neg': combined_neg,
            'rho_T': rho_t,
            'time_trend': time_trend,
            'local_jump_detect': local_jump_detect,
            # Hierarchical features
            'local_pos': local_pos.astype(np.float64),
            'local_neg': local_neg.astype(np.float64),
            'global_pos': global_pos.astype(np.float64),
            'global_neg': global_neg.astype(np.float64),
            'pure_local_pos': pure_local_pos,
            'pure_local_neg': pure_local_neg,
            'pure_global_pos': pure_global_pos,
            'pure_global_neg': pure_global_neg,
            'mixed_pos': mixed_pos,
            'mixed_neg': mixed_neg
        }
    else:
        # Basic non-hierarchical extraction (backwards compatibility)
        diff, threshold = calculate_diff_and_threshold(data, config.delta_percentile)
        delta_pos, delta_neg = detect_jumps(diff, threshold)

        local_std = calculate_local_std(data, config.local_window)
        score = np.abs(diff) / (local_std + 1e-8)
        local_threshold = np.percentile(score, config.local_jump_percentile)
        local_jump_detect = (score > local_threshold).astype(int)

        rho_t = calculate_rho_t(data, config.window)
        time_trend = np.arange(len(data))

        return {
            'data': data,
            'delta_LambdaC_pos': delta_pos.astype(np.float64),
            'delta_LambdaC_neg': delta_neg.astype(np.float64),
            'rho_T': rho_t,
            'time_trend': time_trend,
            'local_jump_detect': local_jump_detect
        }

# ===============================
# SECTION 3: BAYESIAN MODELING
# ===============================

def fit_l3_bayesian_regression_asymmetric(
    data, features_dict, config,
    interaction_pos=None, interaction_neg=None, interaction_rhoT=None
):
    """
    Fit Bayesian regression model with asymmetric cross-series interactions.
    Models how one series influences another through structural changes.
    """
    with pm.Model() as model:
        # Prior distributions
        beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
        beta_time = pm.Normal('beta_time', mu=0, sigma=1)
        beta_dLC_pos = pm.Normal('beta_dLC_pos', mu=0, sigma=5)
        beta_dLC_neg = pm.Normal('beta_dLC_neg', mu=0, sigma=5)
        beta_rhoT = pm.Normal('beta_rhoT', mu=0, sigma=3)

        # Base model
        mu = (
            beta_0
            + beta_time * features_dict['time_trend']
            + beta_dLC_pos * features_dict['delta_LambdaC_pos']
            + beta_dLC_neg * features_dict['delta_LambdaC_neg']
            + beta_rhoT * features_dict['rho_T']
        )

        # Add asymmetric interactions
        if interaction_pos is not None:
            beta_interact_pos = pm.Normal('beta_interact_pos', mu=0, sigma=3)
            mu += beta_interact_pos * interaction_pos

        if interaction_neg is not None:
            beta_interact_neg = pm.Normal('beta_interact_neg', mu=0, sigma=3)
            mu += beta_interact_neg * interaction_neg

        if interaction_rhoT is not None:
            beta_interact_stress = pm.Normal('beta_interact_stress', mu=0, sigma=2)
            mu += beta_interact_stress * interaction_rhoT

        # Observation model
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=data)

        # Sample from posterior
        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            target_accept=config.target_accept,
            return_inferencedata=True,
            cores=4,
            chains=4
        )

    return trace

def fit_l3_pairwise_bayesian_system(
    data_dict: Dict[str, np.ndarray],
    features_dict: Dict[str, Dict[str, np.ndarray]],
    config,
    series_pair: Tuple[str, str] = None
):
    """
    Lambda³ pairwise system Bayesian estimation
    Models structural tensor interactions between two series
    """
    if series_pair is None:
        series_names = list(data_dict.keys())[:2]
    else:
        series_names = list(series_pair)

    name_a, name_b = series_names
    data_a = data_dict[name_a]
    data_b = data_dict[name_b]

    feats_a = features_dict[name_a]
    feats_b = features_dict[name_b]

    with pm.Model() as model:
        # Series A independent terms
        beta_0_a = pm.Normal('beta_0_a', mu=0, sigma=2)
        beta_time_a = pm.Normal('beta_time_a', mu=0, sigma=1)
        beta_dLC_pos_a = pm.Normal('beta_dLC_pos_a', mu=0, sigma=3)
        beta_dLC_neg_a = pm.Normal('beta_dLC_neg_a', mu=0, sigma=3)
        beta_rhoT_a = pm.Normal('beta_rhoT_a', mu=0, sigma=2)

        # Series B independent terms
        beta_0_b = pm.Normal('beta_0_b', mu=0, sigma=2)
        beta_time_b = pm.Normal('beta_time_b', mu=0, sigma=1)
        beta_dLC_pos_b = pm.Normal('beta_dLC_pos_b', mu=0, sigma=3)
        beta_dLC_neg_b = pm.Normal('beta_dLC_neg_b', mu=0, sigma=3)
        beta_rhoT_b = pm.Normal('beta_rhoT_b', mu=0, sigma=2)

        # Interaction terms
        # A → B influence
        beta_interact_ab_pos = pm.Normal('beta_interact_ab_pos', mu=0, sigma=2)
        beta_interact_ab_neg = pm.Normal('beta_interact_ab_neg', mu=0, sigma=2)
        beta_interact_ab_stress = pm.Normal('beta_interact_ab_stress', mu=0, sigma=1.5)

        # B → A influence
        beta_interact_ba_pos = pm.Normal('beta_interact_ba_pos', mu=0, sigma=2)
        beta_interact_ba_neg = pm.Normal('beta_interact_ba_neg', mu=0, sigma=2)
        beta_interact_ba_stress = pm.Normal('beta_interact_ba_stress', mu=0, sigma=1.5)

        # Time lag terms
        if len(data_a) > 1:
            lag_data_a = np.concatenate([[0], data_a[:-1]])
            lag_data_b = np.concatenate([[0], data_b[:-1]])

            beta_lag_ab = pm.Normal('beta_lag_ab', mu=0, sigma=1)
            beta_lag_ba = pm.Normal('beta_lag_ba', mu=0, sigma=1)
        else:
            lag_data_a = np.zeros_like(data_a)
            lag_data_b = np.zeros_like(data_b)
            beta_lag_ab = 0
            beta_lag_ba = 0

        # Mean model for series A
        mu_a = (
            beta_0_a
            + beta_time_a * feats_a['time_trend']
            + beta_dLC_pos_a * feats_a['delta_LambdaC_pos']
            + beta_dLC_neg_a * feats_a['delta_LambdaC_neg']
            + beta_rhoT_a * feats_a['rho_T']
            # B → A interaction
            + beta_interact_ba_pos * feats_b['delta_LambdaC_pos']
            + beta_interact_ba_neg * feats_b['delta_LambdaC_neg']
            + beta_interact_ba_stress * feats_b['rho_T']
            # Lag effect
            + beta_lag_ba * lag_data_b
        )

        # Mean model for series B
        mu_b = (
            beta_0_b
            + beta_time_b * feats_b['time_trend']
            + beta_dLC_pos_b * feats_b['delta_LambdaC_pos']
            + beta_dLC_neg_b * feats_b['delta_LambdaC_neg']
            + beta_rhoT_b * feats_b['rho_T']
            # A → B interaction
            + beta_interact_ab_pos * feats_a['delta_LambdaC_pos']
            + beta_interact_ab_neg * feats_a['delta_LambdaC_neg']
            + beta_interact_ab_stress * feats_a['rho_T']
            # Lag effect
            + beta_lag_ab * lag_data_a
        )

        # Observation model
        sigma_a = pm.HalfNormal('sigma_a', sigma=1)
        sigma_b = pm.HalfNormal('sigma_b', sigma=1)

        # Correlation structure
        rho_ab = pm.Uniform('rho_ab', lower=-1, upper=1)
        cov_matrix = pm.math.stack([
            [sigma_a**2, rho_ab * sigma_a * sigma_b],
            [rho_ab * sigma_a * sigma_b, sigma_b**2]
        ])

        # Joint observation
        y_combined = pm.math.stack([data_a, data_b]).T
        mu_combined = pm.math.stack([mu_a, mu_b]).T

        y_obs = pm.MvNormal('y_obs', mu=mu_combined, cov=cov_matrix, observed=y_combined)

        # Sampling
        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            target_accept=config.target_accept,
            return_inferencedata=True,
            cores=4,
            chains=4
        )

    return trace, model

def fit_hierarchical_bayesian(
    data: np.ndarray,
    hierarchical_features: Dict[str, np.ndarray],
    config
) -> Tuple[any, any]:
    """
    Hierarchical Bayesian model for short-term/long-term structural changes
    Lambda³ theory: Efficient modeling of structural tensor hierarchy
    """
    with pm.Model() as model:
        # Basic terms
        beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
        beta_time = pm.Normal('beta_time', mu=0, sigma=1)

        # Basic structural change terms
        beta_pos = pm.Normal('beta_pos', mu=0, sigma=3)
        beta_neg = pm.Normal('beta_neg', mu=0, sigma=3)
        beta_rho = pm.Normal('beta_rho', mu=0, sigma=2)

        # Hierarchical effect coefficients
        alpha_local = pm.Normal('alpha_local', mu=0, sigma=1.5)
        alpha_global = pm.Normal('alpha_global', mu=0, sigma=2)

        # Hierarchical transition coefficients
        beta_escalation = pm.Normal('beta_escalation', mu=0, sigma=1)
        beta_deescalation = pm.Normal('beta_deescalation', mu=0, sigma=1)

        # Mean structural tensor model
        mu = (
            beta_0
            + beta_time * hierarchical_features['time_trend']
            # Basic structural changes
            + beta_pos * hierarchical_features['delta_LambdaC_pos']
            + beta_neg * hierarchical_features['delta_LambdaC_neg']
            + beta_rho * hierarchical_features['rho_T']
            # Hierarchical effects
            + alpha_local * hierarchical_features.get('local_rho_T', 0)
            + alpha_global * hierarchical_features.get('global_rho_T', 0)
            # Hierarchical transitions
            + beta_escalation * hierarchical_features.get('escalation_indicator', 0)
            + beta_deescalation * hierarchical_features.get('deescalation_indicator', 0)
        )

        # Observation model
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=data)

        # Bayesian estimation
        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            target_accept=config.target_accept,
            return_inferencedata=True,
            cores=4,
            chains=4
        )

    return trace, model

# ===============================
# SECTION 4: REGIME DETECTION
# ===============================

class Lambda3RegimeDetector:
    """Base regime detector using Lambda³ features."""

    def __init__(self, n_regimes=3, method='kmeans'):
        self.n_regimes = n_regimes
        self.method = method
        self.regime_labels = None
        self.regime_features = None

    def fit(self, features_dict):
        """Estimate regimes using clustering on Lambda³ features."""
        X = np.column_stack([
            features_dict['delta_LambdaC_pos'],
            features_dict['delta_LambdaC_neg'],
            features_dict['rho_T']
        ])

        km = KMeans(n_clusters=self.n_regimes, random_state=42)
        labels = km.fit_predict(X)
        self.regime_labels = labels

        self.regime_features = {
            r: {
                'frequency': np.mean(labels == r),
                'mean_rhoT': np.mean(X[labels == r, 2])
            }
            for r in range(self.n_regimes)
        }
        return labels

    def label_regimes(self):
        """Assign descriptive labels to each regime."""
        return {r: f"Regime-{r+1}" for r in range(self.n_regimes)}

class Lambda3FinancialRegimeDetector(Lambda3RegimeDetector):
    """Financial market regime detector with Bull/Bear/Crisis classification."""

    def __init__(self, n_regimes=4, method='adaptive'):
        super().__init__(n_regimes, method)

    def fit(self, features_dict, market_data=None):
        """Enhanced fit method for financial markets."""
        X = np.column_stack([
            features_dict['delta_LambdaC_pos'],
            features_dict['delta_LambdaC_neg'],
            features_dict['rho_T']
        ])

        # Add market-specific features
        if market_data is not None:
            returns = np.diff(market_data, prepend=market_data[0]) / market_data[0]
            rolling_returns = np.array([np.mean(returns[max(0, i-20):i+1]) for i in range(len(returns))])
            X = np.column_stack([X, rolling_returns])

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Adaptive clustering
        if self.method == 'adaptive':
            best_labels = None
            best_score = -np.inf

            for method in ['kmeans', 'gmm']:
                if method == 'kmeans':
                    km = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=20)
                    labels = km.fit_predict(X_scaled)
                else:
                    gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
                    labels = gmm.fit_predict(X_scaled)

                try:
                    score = silhouette_score(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                except:
                    continue

            labels = best_labels
        else:
            labels = super().fit(features_dict)

        self.regime_labels = labels
        self._calculate_financial_stats(X, labels, market_data)
        return labels

    def _calculate_financial_stats(self, X, labels, market_data):
        """Calculate financial-specific regime statistics."""
        self.regime_features = {}

        for r in range(self.n_regimes):
            mask = (labels == r)
            n_points = np.sum(mask)

            if n_points > 0:
                stats = {
                    'frequency': n_points / len(labels),
                    'mean_rhoT': np.mean(X[mask, 2]),
                    'std_rhoT': np.std(X[mask, 2]),
                    'mean_pos_jumps': np.mean(X[mask, 0]),
                    'mean_neg_jumps': np.mean(X[mask, 1]),
                    'jump_asymmetry': np.mean(X[mask, 0]) - np.mean(X[mask, 1])
                }

                if market_data is not None:
                    regime_returns = np.diff(market_data[mask], prepend=market_data[mask][0]) / market_data[mask][0]
                    stats.update({
                        'mean_return': np.mean(regime_returns),
                        'volatility': np.std(regime_returns),
                        'sharpe_ratio': np.mean(regime_returns) / (np.std(regime_returns) + 1e-8)
                    })

                self.regime_features[r] = stats

    def label_financial_regimes(self):
        """Assign financial market labels to each regime."""
        if not self.regime_features:
            return {r: f"Regime-{r+1}" for r in range(self.n_regimes)}

        labels = {}
        regime_chars = []

        for r in range(self.n_regimes):
            stats = self.regime_features[r]

            high_vol = stats['mean_rhoT'] > np.median([s['mean_rhoT'] for s in self.regime_features.values()])
            positive_bias = stats['jump_asymmetry'] > 0

            if 'mean_return' in stats:
                positive_return = stats['mean_return'] > 0
                high_sharpe = stats['sharpe_ratio'] > 0.5
            else:
                positive_return = positive_bias
                high_sharpe = not high_vol

            # Classify regime
            if high_vol and not high_sharpe:
                regime_type = "Crisis"
            elif positive_return and not high_vol:
                regime_type = "Bull"
            elif not positive_return and high_vol:
                regime_type = "Bear"
            else:
                regime_type = "Sideways"

            regime_chars.append((r, regime_type, stats['frequency']))

        regime_chars.sort(key=lambda x: x[2], reverse=True)

        for i, (r, regime_type, freq) in enumerate(regime_chars):
            labels[r] = f"{regime_type}-{i+1}"

        return labels

# ===============================
# SECTION 5: INTERACTION ANALYSIS
# ===============================

def extract_interaction_coefficients(
    trace,
    series_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Extract structural tensor interaction coefficients from Bayesian estimation
    Unified function replacing both extract_interaction_structure and extract_interaction_coefficients
    """
    summary = az.summary(trace)
    name_a, name_b = series_names[:2]
    
    # Parameter name mapping
    param_map = {
        'self': {
            name_a: ('a', ['pos', 'neg', 'rhoT']),
            name_b: ('b', ['pos', 'neg', 'rhoT'])
        },
        'cross': {
            f'{name_a}_to_{name_b}': ('ab', ['pos', 'neg', 'stress']),
            f'{name_b}_to_{name_a}': ('ba', ['pos', 'neg', 'stress'])
        }
    }
    
    results = {'self_effects': {}, 'cross_effects': {}, 'lag_effects': {}}
    
    # Extract self effects
    for series, (suffix, types) in param_map['self'].items():
        results['self_effects'][series] = {
            'pos_jump': _safe_extract(summary, f'beta_dLC_pos_{suffix}'),
            'neg_jump': _safe_extract(summary, f'beta_dLC_neg_{suffix}'),
            'tension': _safe_extract(summary, f'beta_rhoT_{suffix}')
        }
    
    # Extract cross effects
    for direction, (suffix, types) in param_map['cross'].items():
        results['cross_effects'][direction] = {
            'pos_jump': _safe_extract(summary, f'beta_interact_{suffix}_pos'),
            'neg_jump': _safe_extract(summary, f'beta_interact_{suffix}_neg'),
            'tension': _safe_extract(summary, f'beta_interact_{suffix}_stress')
        }
    
    # Lag effects
    results['lag_effects'] = {
        f'{name_a}_to_{name_b}': _safe_extract(summary, 'beta_lag_ab'),
        f'{name_b}_to_{name_a}': _safe_extract(summary, 'beta_lag_ba')
    }
    
    # Correlation
    results['correlation'] = _safe_extract(summary, 'rho_ab')
    
    return results

def predict_with_interactions(
    trace,
    features_dict: Dict[str, Dict[str, np.ndarray]],
    series_names: List[str]
) -> Dict[str, np.ndarray]:
    """
    Unified prediction function for structural tensor evolution
    Replaces both predict_with_interactions and predict_structural_evolution
    """
    summary = az.summary(trace)
    predictions = {}
    
    for idx, name in enumerate(series_names[:2]):
        other_idx = 1 - idx
        other_name = series_names[other_idx]
        suffix = 'a' if idx == 0 else 'b'
        other_suffix = 'b' if idx == 0 else 'a'
        
        # Extract parameters
        params = {
            'intercept': _safe_extract(summary, f'beta_0_{suffix}'),
            'time': _safe_extract(summary, f'beta_time_{suffix}'),
            'self_pos': _safe_extract(summary, f'beta_dLC_pos_{suffix}'),
            'self_neg': _safe_extract(summary, f'beta_dLC_neg_{suffix}'),
            'self_tension': _safe_extract(summary, f'beta_rhoT_{suffix}'),
            'cross_pos': _safe_extract(summary, f'beta_interact_{other_suffix}{suffix}_pos'),
            'cross_neg': _safe_extract(summary, f'beta_interact_{other_suffix}{suffix}_neg'),
            'cross_tension': _safe_extract(summary, f'beta_interact_{other_suffix}{suffix}_stress')
        }
        
        # Calculate prediction
        predictions[name] = (
            params['intercept']
            + params['time'] * features_dict[name]['time_trend']
            + params['self_pos'] * features_dict[name]['delta_LambdaC_pos']
            + params['self_neg'] * features_dict[name]['delta_LambdaC_neg']
            + params['self_tension'] * features_dict[name]['rho_T']
            + params['cross_pos'] * features_dict[other_name]['delta_LambdaC_pos']
            + params['cross_neg'] * features_dict[other_name]['delta_LambdaC_neg']
            + params['cross_tension'] * features_dict[other_name]['rho_T']
        )
    
    return predictions

def _safe_extract(summary: pd.DataFrame, param_name: str, default: float = 0.0) -> float:
    """Safe parameter extraction from summary"""
    return summary.loc[param_name, 'mean'] if param_name in summary.index else default

# ===============================
# SECTION 6: CAUSALITY ANALYSIS
# ===============================

def detect_basic_structural_causality(
    features_dict: Dict[str, Dict[str, np.ndarray]],
    series_names: List[str],
    lag_window: int = 5
) -> Dict[str, Dict[int, float]]:
    """
    Unified structural tensor causality detection
    Lambda³ theory: ΔΛC(t) → ΔΛC(t+k) causal pattern analysis
    """
    if len(series_names) < 2:
        return {}

    name_a, name_b = series_names[:2]
    
    # Get structural change events
    events = {
        name_a: {
            'pos': features_dict[name_a]['delta_LambdaC_pos'],
            'neg': features_dict[name_a]['delta_LambdaC_neg']
        },
        name_b: {
            'pos': features_dict[name_b]['delta_LambdaC_pos'],
            'neg': features_dict[name_b]['delta_LambdaC_neg']
        }
    }
    
    causality_patterns = {}
    
    # Calculate all causal patterns efficiently
    for from_name, to_name in [(name_a, name_b), (name_b, name_a)]:
        for from_type in ['pos', 'neg']:
            for to_type in ['pos', 'neg']:
                pattern_key = f'{from_name}_{from_type}_to_{to_name}_{to_type}'
                causality_patterns[pattern_key] = _compute_lagged_causality(
                    events[from_name][from_type],
                    events[to_name][to_type],
                    lag_window
                )
    
    return causality_patterns

def _compute_lagged_causality(
    cause_events: np.ndarray,
    effect_events: np.ndarray,
    lag_window: int
) -> Dict[int, float]:
    """Compute lagged causality probabilities"""
    causality_by_lag = {}
    
    for lag in range(1, min(lag_window + 1, len(cause_events))):
        cause_past = cause_events[:-lag]
        effect_future = effect_events[lag:]
        
        joint_prob = np.mean(cause_past * effect_future)
        cause_prob = np.mean(cause_past)
        
        causality_by_lag[lag] = joint_prob / (cause_prob + 1e-8)
    
    return causality_by_lag

def analyze_comprehensive_causality(
    features_dict: Dict[str, Dict[str, np.ndarray]],
    series_names: List[str],
    lag_window: int = 5,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Comprehensive causality analysis with advanced metrics
    Lambda³ theory: Complete causal structure analysis including asymmetry and decay
    """
    # Basic causality detection
    basic_causality = detect_basic_structural_causality(features_dict, series_names, lag_window)
    
    # Build causality matrix
    causality_matrix = _build_causality_matrix(basic_causality, series_names)
    
    # Analyze temporal patterns
    temporal_patterns = _analyze_temporal_causality_patterns(basic_causality, lag_window)
    
    # Compute directional strengths
    directional_strengths = {}
    all_causalities = []
    
    for direction, pattern in basic_causality.items():
        if pattern:
            max_causality = max(pattern.values())
            mean_causality = np.mean(list(pattern.values()))
            optimal_lag = max(pattern, key=pattern.get)
            
            directional_strengths[direction] = {
                'max': max_causality,
                'mean': mean_causality,
                'optimal_lag': optimal_lag
            }
            all_causalities.extend(pattern.values())
    
    # Compute structural causality metrics
    structural_metrics = _compute_structural_causality_metrics(
        features_dict, series_names, basic_causality
    )
    
    # Compute summary statistics
    causality_summary = _compute_causality_summary(
        all_causalities, directional_strengths, basic_causality
    )
    
    # Display results if verbose
    if verbose:
        _display_causality_results(causality_summary, directional_strengths, structural_metrics)
    
    return {
        'basic_causality': basic_causality,
        'causality_matrix': causality_matrix,
        'temporal_patterns': temporal_patterns,
        'directional_strengths': directional_strengths,
        'structural_metrics': structural_metrics,
        'summary': causality_summary
    }

def _build_causality_matrix(
    basic_causality: Dict[str, Dict[int, float]],
    series_names: List[str]
) -> np.ndarray:
    """Build causality matrix"""
    n_series = len(series_names)
    causality_matrix = np.zeros((n_series, n_series))

    for i, series_a in enumerate(series_names):
        for j, series_b in enumerate(series_names):
            if i != j:
                # Search for A → B causality
                for direction, pattern in basic_causality.items():
                    if (series_a in direction and series_b in direction and
                        direction.find(series_a) < direction.find(series_b)):
                        if pattern:
                            causality_matrix[i, j] = max(pattern.values())
                        break

    return causality_matrix

def _analyze_temporal_causality_patterns(
    basic_causality: Dict[str, Dict[int, float]],
    lag_window: int
) -> Dict[str, any]:
    """Analyze time-dependent causality patterns"""
    temporal_patterns = {
        'lag_distribution': {},
        'causality_decay': {},
        'peak_causality_lags': {}
    }

    for direction, pattern in basic_causality.items():
        if pattern and isinstance(pattern, dict):
            lags = list(pattern.keys())
            causalities = list(pattern.values())

            # Lag distribution
            temporal_patterns['lag_distribution'][direction] = {
                'lags': lags,
                'causalities': causalities
            }

            # Peak causality lag
            peak_lag = max(pattern, key=pattern.get)
            temporal_patterns['peak_causality_lags'][direction] = {
                'lag': peak_lag,
                'causality': pattern[peak_lag]
            }

            # Causality decay pattern
            if len(causalities) > 1:
                decay_rate = _compute_causality_decay_rate(lags, causalities)
                temporal_patterns['causality_decay'][direction] = decay_rate

    return temporal_patterns

def _compute_causality_decay_rate(lags: List[int], causalities: List[float]) -> float:
    """Compute causality decay rate"""
    if len(causalities) < 2:
        return 0.0

    # Linear regression for decay rate estimation
    try:
        slope = np.polyfit(lags, causalities, 1)[0]
        return abs(slope)  # Decay rate as absolute value
    except:
        return 0.0

def _compute_causality_summary(
    all_causalities: List[float],
    directional_strengths: Dict[str, Dict],
    basic_causality: Dict[str, Dict[int, float]]
) -> Dict[str, any]:
    """Compute causality summary statistics"""

    # Identify strongest causality
    strongest_strength = 0.0
    strongest_direction = ""
    strongest_lag = 0

    for direction, strength_info in directional_strengths.items():
        if strength_info['max'] > strongest_strength:
            strongest_strength = strength_info['max']
            strongest_direction = direction
            strongest_lag = strength_info['optimal_lag']

    # Asymmetry metrics
    asymmetry_metrics = _compute_causality_asymmetry(directional_strengths)

    causality_summary = {
        'max_causality': max(all_causalities) if all_causalities else 0.0,
        'mean_causality': np.mean(all_causalities) if all_causalities else 0.0,
        'std_causality': np.std(all_causalities) if all_causalities else 0.0,
        'total_causality_patterns': len(basic_causality),
        'active_patterns': len([d for d, p in basic_causality.items() if p]),
        'strongest_direction': strongest_direction,
        'strongest_strength': strongest_strength,
        'strongest_lag': strongest_lag,
        'asymmetry_metrics': asymmetry_metrics,
        'causality_density': len([c for c in all_causalities if c > 0.1]) / max(len(all_causalities), 1)
    }

    return causality_summary

def _compute_causality_asymmetry(
    directional_strengths: Dict[str, Dict]
) -> Dict[str, float]:
    """Compute causality asymmetry"""
    asymmetry_metrics = {
        'total_asymmetry': 0.0,
        'max_directional_difference': 0.0,
        'asymmetry_patterns': {}
    }

    # Calculate asymmetry for each direction pair
    directions = list(directional_strengths.keys())
    total_asymmetry = 0.0
    max_diff = 0.0

    for i, dir_a in enumerate(directions):
        for j, dir_b in enumerate(directions[i+1:], i+1):
            # Search for reverse relationships
            if _are_reverse_directions(dir_a, dir_b):
                strength_a = directional_strengths[dir_a]['max']
                strength_b = directional_strengths[dir_b]['max']

                asymmetry = abs(strength_a - strength_b)
                total_asymmetry += asymmetry
                max_diff = max(max_diff, asymmetry)

                asymmetry_metrics['asymmetry_patterns'][f"{dir_a}_vs_{dir_b}"] = {
                    'asymmetry': asymmetry,
                    'dominant_direction': dir_a if strength_a > strength_b else dir_b
                }

    asymmetry_metrics['total_asymmetry'] = total_asymmetry
    asymmetry_metrics['max_directional_difference'] = max_diff

    return asymmetry_metrics

def _are_reverse_directions(dir_a: str, dir_b: str) -> bool:
    """Check if two directions are reverse of each other"""
    # Simple implementation: compare series names in directions
    parts_a = dir_a.split('_to_')
    parts_b = dir_b.split('_to_')

    if len(parts_a) == 2 and len(parts_b) == 2:
        return (parts_a[0] == parts_b[1] and parts_a[1] == parts_b[0])

    return False

def _compute_structural_causality_metrics(
    features_dict: Dict[str, Dict[str, np.ndarray]],
    series_names: List[str],
    basic_causality: Dict[str, Dict[int, float]]
) -> Dict[str, any]:
    """Compute structural tensor causality metrics"""

    structural_metrics = {
        'structural_change_causality': {},
        'tension_causality': {},
        'overall_structural_influence': {}
    }

    for name in series_names:
        if name in features_dict:
            features = features_dict[name]

            # Structural change intensity
            pos_changes = np.sum(features.get('delta_LambdaC_pos', np.array([])))
            neg_changes = np.sum(features.get('delta_LambdaC_neg', np.array([])))
            avg_tension = np.mean(features.get('rho_T', np.array([0])))

            # Causality strength related to this series
            related_causality = []
            for direction, pattern in basic_causality.items():
                if name in direction and pattern:
                    related_causality.extend(pattern.values())

            avg_causality_involvement = np.mean(related_causality) if related_causality else 0.0

            structural_metrics['structural_change_causality'][name] = pos_changes + neg_changes
            structural_metrics['tension_causality'][name] = avg_tension
            structural_metrics['overall_structural_influence'][name] = avg_causality_involvement

    return structural_metrics

def _display_causality_results(
    causality_summary: Dict[str, any],
    directional_strengths: Dict[str, Dict],
    structural_metrics: Dict[str, any]
):
    """Display causality results"""

    print(f"\nIntegrated Causality Analysis Results:")
    print(f"  Max causality strength: {causality_summary['max_causality']:.4f}")
    print(f"  Mean causality strength: {causality_summary['mean_causality']:.4f}")
    print(f"  Causality strength std: {causality_summary['std_causality']:.4f}")
    print(f"  Total causality patterns: {causality_summary['total_causality_patterns']}")
    print(f"  Active causality patterns: {causality_summary['active_patterns']}")
    print(f"  Causality density: {causality_summary['causality_density']:.3f}")

    print(f"\nStrongest Causality:")
    print(f"  Direction: {causality_summary['strongest_direction']}")
    print(f"  Strength: {causality_summary['strongest_strength']:.4f}")
    print(f"  Optimal lag: {causality_summary['strongest_lag']}")

    print(f"\nCausality Asymmetry:")
    asymmetry = causality_summary['asymmetry_metrics']
    print(f"  Total asymmetry: {asymmetry['total_asymmetry']:.4f}")
    print(f"  Max directional difference: {asymmetry['max_directional_difference']:.4f}")

    print(f"\nDirectional Causality Strength:")
    for direction, strength in directional_strengths.items():
        print(f"  {direction}: max={strength['max']:.4f}, "
              f"mean={strength['mean']:.4f}, optimal_lag={strength['optimal_lag']}")

    print(f"\nStructural Tensor Causality Characteristics:")
    for series_name in structural_metrics['overall_structural_influence'].keys():
        structural_change = structural_metrics['structural_change_causality'][series_name]
        tension = structural_metrics['tension_causality'][series_name]
        influence = structural_metrics['overall_structural_influence'][series_name]
        print(f"  {series_name}: structural_change={structural_change}, "
              f"avg_tension={tension:.3f}, causality_involvement={influence:.4f}")

# ===============================
# SECTION 7: SYNCHRONIZATION ANALYSIS
# ===============================

def calculate_sync_profile(series_a: np.ndarray, series_b: np.ndarray,
                          lag_window: int = LAG_WINDOW_DEFAULT) -> Tuple[Dict[int, float], float, int]:
    """Calculate synchronization profile using JIT-compiled function."""
    lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_jit(
        series_a.astype(np.float64),
        series_b.astype(np.float64),
        lag_window
    )

    sync_profile = {int(lag): float(sync) for lag, sync in zip(lags, sync_values)}
    return sync_profile, float(max_sync), int(optimal_lag)

def sync_matrix(event_series_dict: Dict[str, np.ndarray], lag_window: int = LAG_WINDOW_DEFAULT) -> Tuple[np.ndarray, List[str]]:
    """Create synchronization rate matrix for all series pairs."""
    series_names = list(event_series_dict.keys())
    n = len(series_names)
    mat = np.zeros((n, n))

    for i, a in enumerate(series_names):
        for j, b in enumerate(series_names):
            if i == j:
                mat[i, j] = 1.0
                continue

            series_a = event_series_dict[a].astype(np.float64)
            series_b = event_series_dict[b].astype(np.float64)

            _, _, max_sync, _ = calculate_sync_profile_jit(series_a, series_b, lag_window)
            mat[i, j] = max_sync

    return mat, series_names

def build_sync_network(event_series_dict: Dict[str, np.ndarray],
                      lag_window: int = LAG_WINDOW_DEFAULT,
                      sync_threshold: float = SYNC_THRESHOLD_DEFAULT) -> nx.DiGraph:
    """Build directed synchronization network from event series."""
    series_names = list(event_series_dict.keys())
    G = nx.DiGraph()

    for series in series_names:
        G.add_node(series)

    print(f"\nBuilding sync network with threshold={sync_threshold}")

    edge_count = 0
    for series_a in series_names:
        for series_b in series_names:
            if series_a == series_b:
                continue

            sync_profile, max_sync, optimal_lag = calculate_sync_profile(
                event_series_dict[series_a].astype(np.float64),
                event_series_dict[series_b].astype(np.float64),
                lag_window
            )

            print(f"{series_a} → {series_b}: max_sync={max_sync:.4f}, lag={optimal_lag}")

            if max_sync >= sync_threshold:
                G.add_edge(series_a, series_b,
                          weight=max_sync,
                          lag=optimal_lag,
                          profile=sync_profile)
                edge_count += 1
                print(f"  ✓ Edge added!")

    print(f"\nNetwork summary: {G.number_of_nodes()} nodes, {edge_count} edges")
    return G

# ===============================
# SECTION 8: CRISIS DETECTION
# ===============================

def detect_structural_crisis(
    features_dict: Dict[str, Dict],
    crisis_threshold: float = 0.8,
    tension_weight: float = 0.6
) -> Dict[str, any]:
    """
    Structural crisis detection
    Lambda³ theory: Combined indicator of tension scalar (ρT) and structural changes (ΔΛC)
    """
    crisis_indicators = {}
    
    # Calculate crisis score for each series
    for name, features in features_dict.items():
        # Normalize tension scalar
        tension = features['rho_T']
        tension_score = (tension - np.mean(tension)) / (np.std(tension) + 1e-8)
        
        # Normalize structural changes
        total_jumps = features['delta_LambdaC_pos'] + features['delta_LambdaC_neg']
        jump_score = (total_jumps - np.mean(total_jumps)) / (np.std(total_jumps) + 1e-8)
        
        # Combined crisis score
        crisis_indicators[name] = tension_weight * tension_score + (1 - tension_weight) * jump_score
    
    # Aggregate crisis score
    aggregate_crisis = np.mean(list(crisis_indicators.values()), axis=0)
    
    # Identify crisis periods
    crisis_periods = aggregate_crisis > crisis_threshold
    crisis_episodes = _identify_episodes(crisis_periods)
    
    return {
        'crisis_periods': crisis_periods,
        'crisis_episodes': crisis_episodes,
        'crisis_indicators': crisis_indicators,
        'aggregate_crisis': aggregate_crisis
    }

def _identify_episodes(binary_series: np.ndarray) -> List[Tuple[int, int]]:
    """Identify episodes of consecutive True values"""
    diff = np.diff(np.concatenate([[False], binary_series, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    
    return list(zip(starts, ends))

# ===============================
# SECTION 9: HIERARCHICAL ANALYSIS
# ===============================

def calculate_structural_hierarchy_metrics(
    structural_changes: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Calculate structural change hierarchy metrics
    """
    total_local_pos = np.sum(structural_changes.get('local_pos', 0))
    total_local_neg = np.sum(structural_changes.get('local_neg', 0))
    total_global_pos = np.sum(structural_changes.get('global_pos', 0))
    total_global_neg = np.sum(structural_changes.get('global_neg', 0))

    pure_local_pos = np.sum(structural_changes.get('pure_local_pos', 0))
    pure_local_neg = np.sum(structural_changes.get('pure_local_neg', 0))
    pure_global_pos = np.sum(structural_changes.get('pure_global_pos', 0))
    pure_global_neg = np.sum(structural_changes.get('pure_global_neg', 0))

    mixed_pos = np.sum(structural_changes.get('mixed_pos', 0))
    mixed_neg = np.sum(structural_changes.get('mixed_neg', 0))

    # Hierarchy indices
    total_events = total_local_pos + total_local_neg + total_global_pos + total_global_neg

    metrics = {
        'local_dominance': (total_local_pos + total_local_neg) / max(total_events, 1),
        'global_dominance': (total_global_pos + total_global_neg) / max(total_events, 1),
        'hierarchy_ratio': (pure_local_pos + pure_local_neg) / max(total_local_pos + total_local_neg, 1),
        'coupling_strength': (mixed_pos + mixed_neg) / max(total_events, 1),
        'asymmetry_local': (pure_local_pos - pure_local_neg) / max(pure_local_pos + pure_local_neg, 1),
        'asymmetry_global': (pure_global_pos - pure_global_neg) / max(pure_global_pos + pure_global_neg, 1),
        'escalation_rate': (mixed_pos + mixed_neg) / max(pure_local_pos + pure_local_neg, 1)
    }

    return metrics

def prepare_hierarchical_features_for_bayesian(
    structural_changes: Dict[str, np.ndarray],
    original_data: np.ndarray,
    config,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Prepare hierarchical structural change features for Bayesian model
    """
    # Use existing hierarchical structural changes directly
    local_pos = structural_changes.get('local_pos', np.zeros_like(original_data, dtype=np.float64))
    local_neg = structural_changes.get('local_neg', np.zeros_like(original_data, dtype=np.float64))
    global_pos = structural_changes.get('global_pos', np.zeros_like(original_data, dtype=np.float64))
    global_neg = structural_changes.get('global_neg', np.zeros_like(original_data, dtype=np.float64))

    rho_T = structural_changes.get('rho_T', calculate_rho_t(original_data, config.window))
    time_trend = structural_changes.get('time_trend', np.arange(len(original_data)))

    # Combined structural changes
    combined_pos = np.maximum(local_pos, global_pos)
    combined_neg = np.maximum(local_neg, global_neg)

    # Hierarchical event masks
    local_events_mask = (local_pos + local_neg) > 0
    global_events_mask = (global_pos + global_neg) > 0

    if verbose:
        print(f"  Hierarchical structural changes: Short-term={np.sum(local_events_mask)}, Long-term={np.sum(global_events_mask)}")

    # Bayesian model features
    hierarchical_features = {
        'delta_LambdaC_pos': combined_pos,
        'delta_LambdaC_neg': combined_neg,
        'rho_T': rho_T,
        'time_trend': time_trend,
        # Hierarchical ρT (non-zero only at structural change positions)
        'local_rho_T': rho_T * local_events_mask.astype(float),
        'global_rho_T': rho_T * global_events_mask.astype(float),
        # Hierarchy transition indicators
        'escalation_indicator': np.diff(np.concatenate([[0], global_events_mask.astype(float)])),
        'deescalation_indicator': np.diff(np.concatenate([[0], local_events_mask.astype(float)]))
    }

    return hierarchical_features

def analyze_hierarchical_separation_dynamics(
    series_name: str,
    original_data: np.ndarray,
    structural_changes: Dict[str, np.ndarray],
    config,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Hierarchical separation dynamics analysis
    """
    if verbose:
        print(f"\nHierarchical separation dynamics analysis: {series_name}")

    # Prepare features for Bayesian model
    hierarchical_features = prepare_hierarchical_features_for_bayesian(
        structural_changes, original_data, config, verbose=verbose
    )

    # Calculate hierarchical event counts
    local_mask = hierarchical_features['local_rho_T'] > 0
    global_mask = hierarchical_features['global_rho_T'] > 0

    N_local = np.sum(local_mask)
    N_global = np.sum(global_mask)

    # Check for sufficient events
    MIN_EVENTS = 10
    if N_local < MIN_EVENTS or N_global < MIN_EVENTS:
        if verbose:
            print(f"  Warning: Insufficient events (short-term: {N_local}, long-term: {N_global})")
        return {}

    # Adjust Bayesian estimation parameters
    total_events = N_local + N_global
    if total_events < 100:
        draws, tune = 4000, 4000
    else:
        draws, tune = config.draws, config.tune

    # Execute hierarchical Bayesian estimation
    trace, model = fit_hierarchical_bayesian(
        original_data,
        hierarchical_features,
        L3Config(draws=draws, tune=tune, target_accept=config.target_accept)
    )

    # Extract coefficients
    summary = az.summary(trace)

    # Calculate inter-hierarchy correlation
    hierarchy_correlation = 0.0
    if N_local > 0 and N_global > 0:
        local_values = hierarchical_features['local_rho_T'][local_mask]
        global_values = hierarchical_features['global_rho_T'][global_mask]

        if len(local_values) > 1 and len(global_values) > 1:
            min_length = min(len(local_values), len(global_values))
            if min_length > 1:
                local_sample = local_values[:min_length]
                global_sample = global_values[:min_length]
                try:
                    correlation_matrix = np.corrcoef(local_sample, global_sample)
                    if correlation_matrix.size > 1:
                        hierarchy_correlation = correlation_matrix[0, 1]
                except:
                    hierarchy_correlation = 0.0

    separation_coefficients = {
        'escalation': {
            'coefficient': summary.loc['beta_escalation', 'mean'],
            'hdi_lower': summary.loc['beta_escalation', 'hdi_3%'],
            'hdi_upper': summary.loc['beta_escalation', 'hdi_97%']
        },
        'deescalation': {
            'coefficient': summary.loc['beta_deescalation', 'mean'],
            'hdi_lower': summary.loc['beta_deescalation', 'hdi_3%'],
            'hdi_upper': summary.loc['beta_deescalation', 'hdi_97%']
        },
        'local_effect': {
            'coefficient': summary.loc['alpha_local', 'mean'],
            'hdi_lower': summary.loc['alpha_local', 'hdi_3%'],
            'hdi_upper': summary.loc['alpha_local', 'hdi_97%']
        },
        'global_effect': {
            'coefficient': summary.loc['alpha_global', 'mean'],
            'hdi_lower': summary.loc['alpha_global', 'hdi_3%'],
            'hdi_upper': summary.loc['alpha_global', 'hdi_97%']
        },
        'hierarchy_correlation': hierarchy_correlation
    }

    # Asymmetry metrics
    asymmetry_metrics = {
        'transition_asymmetry': abs(separation_coefficients['escalation']['coefficient']) - abs(separation_coefficients['deescalation']['coefficient']),
        'escalation_dominance': abs(separation_coefficients['escalation']['coefficient']) / (abs(separation_coefficients['escalation']['coefficient']) + abs(separation_coefficients['deescalation']['coefficient']) + 1e-8),
        'deescalation_dominance': abs(separation_coefficients['deescalation']['coefficient']) / (abs(separation_coefficients['escalation']['coefficient']) + abs(separation_coefficients['deescalation']['coefficient']) + 1e-8),
    }

    if verbose:
        print(f"  Hierarchical separation coefficients:")
        print(f"    Escalation: {separation_coefficients['escalation']['coefficient']:.4f}")
        print(f"    De-escalation: {separation_coefficients['deescalation']['coefficient']:.4f}")
        print(f"    Inter-hierarchy correlation: {hierarchy_correlation:.4f}")

    return {
        'trace': trace,
        'model': model,
        'separation_coefficients': separation_coefficients,
        'asymmetry_metrics': asymmetry_metrics,
        'hierarchy_stats': {
            'local_mean': np.mean(hierarchical_features['local_rho_T'][local_mask]) if N_local > 0 else 0.0,
            'global_mean': np.mean(hierarchical_features['global_rho_T'][global_mask]) if N_global > 0 else 0.0,
            'local_std': np.std(hierarchical_features['local_rho_T'][local_mask]) if N_local > 0 else 0.0,
            'global_std': np.std(hierarchical_features['global_rho_T'][global_mask]) if N_global > 0 else 0.0,
        },
        'local_series': hierarchical_features['local_rho_T'],
        'global_series': hierarchical_features['global_rho_T'],
        'series_name': series_name
    }

# ===============================
# SECTION 10: COMPREHENSIVE ANALYSIS
# ===============================

def analyze_all_pairwise_interactions(
    series_dict: Dict[str, np.ndarray],
    features_dict: Dict[str, Dict[str, np.ndarray]],
    config: L3Config,
    max_pairs: int = None
) -> Dict[str, any]:
    """
    Comprehensive pairwise interaction analysis
    Lambda³ theory: Exhaustive structural tensor interaction analysis
    """
    from itertools import combinations
    
    series_names = list(series_dict.keys())
    n_series = len(series_names)
    
    print(f"\n{'='*80}")
    print("All Pairwise Interaction Analysis")
    print(f"{'='*80}")
    print(f"Number of series: {n_series}")
    print(f"Total pairs: {n_series * (n_series - 1) // 2}")
    
    # Generate pair combinations
    all_pairs = list(combinations(series_names, 2))
    
    # Limit pairs if specified
    if max_pairs and len(all_pairs) > max_pairs:
        all_pairs = all_pairs[:max_pairs]
        print(f"Limited analysis to {max_pairs} pairs")
    
    print(f"Analyzing {len(all_pairs)} pairs")
    
    # Result storage
    all_pairwise_results = {
        'pairs': {},
        'interaction_matrix': np.zeros((n_series, n_series)),
        'asymmetry_matrix': np.zeros((n_series, n_series)),
        'strongest_interactions': [],
        'summary': {}
    }
    
    # Analyze each pair
    for pair_idx, (name_a, name_b) in enumerate(all_pairs):
        print(f"\n{'─'*60}")
        print(f"Pair {pair_idx + 1}/{len(all_pairs)}: {name_a} ⇄ {name_b}")
        
        try:
            # Pairwise Bayesian estimation
            trace, model = fit_l3_pairwise_bayesian_system(
                {name_a: series_dict[name_a], name_b: series_dict[name_b]},
                {name_a: features_dict[name_a], name_b: features_dict[name_b]},
                config,
                series_pair=(name_a, name_b)
            )
            
            # Extract interaction coefficients
            interaction_coeffs = extract_interaction_coefficients(trace, [name_a, name_b])
            
            # Predictions
            predictions = predict_with_interactions(trace, 
                {name_a: features_dict[name_a], name_b: features_dict[name_b]},
                [name_a, name_b]
            )
            
            # Causality inference
            causality_patterns = detect_basic_structural_causality(
                {name_a: features_dict[name_a], name_b: features_dict[name_b]},
                [name_a, name_b]
            )
            
            # Save results
            pair_key = f"{name_a}_vs_{name_b}"
            pair_results = {
                'trace': trace,
                'model': model,
                'interaction_coefficients': interaction_coeffs,
                'predictions': predictions,
                'causality_patterns': causality_patterns,
                'series_names': [name_a, name_b]
            }
            all_pairwise_results['pairs'][pair_key] = pair_results
            
            # Extract interaction strengths
            cross_effects = interaction_coeffs['cross_effects']
            
            # A → B total strength
            strength_a_to_b = sum(abs(v) for v in cross_effects[f'{name_a}_to_{name_b}'].values())
            # B → A total strength
            strength_b_to_a = sum(abs(v) for v in cross_effects[f'{name_b}_to_{name_a}'].values())
            
            # Store in matrix
            idx_a = series_names.index(name_a)
            idx_b = series_names.index(name_b)
            all_pairwise_results['interaction_matrix'][idx_a, idx_b] = strength_a_to_b
            all_pairwise_results['interaction_matrix'][idx_b, idx_a] = strength_b_to_a
            
            # Asymmetry
            asymmetry = abs(strength_a_to_b - strength_b_to_a)
            all_pairwise_results['asymmetry_matrix'][idx_a, idx_b] = asymmetry
            all_pairwise_results['asymmetry_matrix'][idx_b, idx_a] = asymmetry
            
            # Record strong interactions
            total_strength = strength_a_to_b + strength_b_to_a
            all_pairwise_results['strongest_interactions'].append({
                'pair': pair_key,
                'total_strength': total_strength,
                'asymmetry': asymmetry,
                'dominant_direction': f'{name_a}→{name_b}' if strength_a_to_b > strength_b_to_a else f'{name_b}→{name_a}',
                'strength_a_to_b': strength_a_to_b,
                'strength_b_to_a': strength_b_to_a
            })
            
            # Display results
            print(f"  {name_a} → {name_b}: {strength_a_to_b:.3f}")
            print(f"  {name_b} → {name_a}: {strength_b_to_a:.3f}")
            print(f"  Asymmetry: {asymmetry:.3f}")
            
        except Exception as e:
            print(f"  Warning: Error in pair analysis: {str(e)}")
            continue
    
    # Sort strongest interactions
    all_pairwise_results['strongest_interactions'].sort(
        key=lambda x: x['total_strength'], reverse=True
    )
    
    # Summary statistics
    interaction_values = all_pairwise_results['interaction_matrix'][
        all_pairwise_results['interaction_matrix'] > 0
    ]
    asymmetry_values = all_pairwise_results['asymmetry_matrix'][
        np.triu_indices_from(all_pairwise_results['asymmetry_matrix'], k=1)
    ]
    
    all_pairwise_results['summary'] = {
        'total_pairs_analyzed': len(all_pairwise_results['pairs']),
        'max_interaction_strength': np.max(interaction_values) if len(interaction_values) > 0 else 0,
        'mean_interaction_strength': np.mean(interaction_values) if len(interaction_values) > 0 else 0,
        'max_asymmetry': np.max(asymmetry_values) if len(asymmetry_values) > 0 else 0,
        'mean_asymmetry': np.mean(asymmetry_values) if len(asymmetry_values) > 0 else 0,
        'strongest_pair': all_pairwise_results['strongest_interactions'][0] if all_pairwise_results['strongest_interactions'] else None
    }
    
    # Display results
    print(f"\n{'='*80}")
    print("All Pairwise Analysis Complete")
    print(f"{'='*80}")
    print(f"Pairs analyzed: {all_pairwise_results['summary']['total_pairs_analyzed']}")
    print(f"Max interaction strength: {all_pairwise_results['summary']['max_interaction_strength']:.4f}")
    print(f"Mean interaction strength: {all_pairwise_results['summary']['mean_interaction_strength']:.4f}")
    print(f"Max asymmetry: {all_pairwise_results['summary']['max_asymmetry']:.4f}")
    
    if all_pairwise_results['summary']['strongest_pair']:
        strongest = all_pairwise_results['summary']['strongest_pair']
        print(f"\nStrongest interaction pair: {strongest['pair']}")
        print(f"  Total strength: {strongest['total_strength']:.4f}")
        print(f"  Asymmetry: {strongest['asymmetry']:.4f}")
        print(f"  Dominant direction: {strongest['dominant_direction']}")
    
    # Display top 5 pairs
    print(f"\nTop 5 interaction pairs:")
    for i, interaction in enumerate(all_pairwise_results['strongest_interactions'][:5]):
        print(f"  {i+1}. {interaction['pair']}: "
              f"strength={interaction['total_strength']:.3f}, "
              f"asymmetry={interaction['asymmetry']:.3f}")
    
    # Visualization placeholder
    # visualize_all_pairwise_results(all_pairwise_results, series_names)
    
    return all_pairwise_results

def complete_hierarchical_analysis(
    data_dict: Dict[str, np.ndarray],
    config,
    local_window: int = 5,
    global_window: int = 30,
    series_names: List[str] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Complete hierarchical structural change analysis (Lambda³ theory compliant)
    
    Detects hierarchical ΔΛC changes in structural tensor (Λ) and analyzes
    multi-scale interactions of progression vectors (ΛF) and tension scalars (ρT).
    Time-independent structural space ΔΛC pulsation analysis.
    """
    if series_names is None:
        series_names = list(data_dict.keys())

    results = {}

    if verbose:
        print("=" * 80)
        print("LAMBDA³ HIERARCHICAL STRUCTURAL ANALYSIS")
        print("=" * 80)
        print(f"Series count: {len(series_names)}, Windows: short-term={local_window}, long-term={global_window}")

    # STAGE 1: Hierarchical structural change detection for each series
    for name, data in data_dict.items():
        if name not in series_names:
            continue

        if verbose:
            print(f"\n{name}: ", end="")

        # Extract hierarchical structural changes using unified feature extraction
        config_hier = L3Config(
            window=config.window,
            local_window=local_window,
            global_window=global_window,
            hierarchical=True,
            local_threshold_percentile=config.local_threshold_percentile,
            global_threshold_percentile=config.global_threshold_percentile
        )
        structural_changes = calc_lambda3_features(data, config_hier)

        # Calculate hierarchy metrics
        hierarchy_metrics = calculate_structural_hierarchy_metrics(structural_changes)

        if verbose:
            print(f"Local dominance={hierarchy_metrics['local_dominance']:.2f}, "
                  f"Escalation rate={hierarchy_metrics['escalation_rate']:.2f}")

        # Hierarchical separation dynamics analysis
        separation_results = analyze_hierarchical_separation_dynamics(
            name, data, structural_changes, config, verbose=verbose
        )

        # Integrate results
        results[name] = {
            'structural_changes': structural_changes,
            'hierarchy_metrics': hierarchy_metrics,
            'hierarchical_separation': separation_results,
            'data': data,
            'series_name': name
        }

        # Visualization placeholder
        # if separation_results and 'trace' in separation_results and verbose:
        #     plot_hierarchical_separation_analysis(separation_results, structural_changes)

    # STAGE 2: Pairwise hierarchical synchronization analysis
    if len(series_names) >= 2:
        if verbose:
            print(f"\n{'=' * 80}")
            print("Pairwise Hierarchical Synchronization Analysis")

        first_series = series_names[0]
        second_series = series_names[1]

        # Verify hierarchical features exist
        first_changes = results[first_series]['structural_changes']
        second_changes = results[second_series]['structural_changes']

        required_features = ['pure_local_pos', 'pure_local_neg',
                           'pure_global_pos', 'pure_global_neg']

        has_hierarchical_features = all(
            feature in first_changes and feature in second_changes
            for feature in required_features
        )

        if has_hierarchical_features:
            try:
                # Detect basic causality
                features_for_causality = {}
                for name in series_names[:2]:
                    if name in results:
                        structural_changes = results[name]['structural_changes']
                        features_for_causality[name] = {
                            'delta_LambdaC_pos': structural_changes.get('delta_LambdaC_pos', np.zeros(100)),
                            'delta_LambdaC_neg': structural_changes.get('delta_LambdaC_neg', np.zeros(100)),
                            'rho_T': structural_changes.get('rho_T', np.zeros(100)),
                            'time_trend': structural_changes.get('time_trend', np.arange(100))
                        }

                if len(features_for_causality) >= 2:
                    causality_results = {
                        'causality_results': detect_basic_structural_causality(
                            features_for_causality,
                            series_names[:2],
                            lag_window=5
                        ),
                        'analysis_type': 'basic_causality_for_hierarchical'
                    }
                    results['hierarchical_causality'] = causality_results
                    if verbose:
                        print(f"  Hierarchical causality detection complete")

            except Exception as e:
                if verbose:
                    print(f"  Hierarchical causality detection error: {e}")
                results['hierarchical_causality'] = {}

    # STAGE 3: Overall hierarchical structure summary
    if verbose:
        print(f"\n{'=' * 80}")
        print("Hierarchical Structure Summary")
        print(f"{'=' * 80}")

        # Series-wise hierarchy summary
        for name in series_names:
            if name in results:
                metrics = results[name].get('hierarchy_metrics', {})
                print(f"{name}: Local={metrics.get('local_dominance', 0):.2f}, "
                      f"Global={metrics.get('global_dominance', 0):.2f}, "
                      f"Coupling={metrics.get('coupling_strength', 0):.2f}")

    return results

def analyze_hierarchical_synchronization(
    structural_changes_1: Dict[str, np.ndarray],
    structural_changes_2: Dict[str, np.ndarray],
    series_name_1: str = "Series_A",
    series_name_2: str = "Series_B",
    config = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Hierarchical structural change synchronization analysis (simplified)
    
    Uses fit_l3_pairwise_bayesian_system directly for hierarchical synchronization
    """
    if verbose:
        print(f"\nHierarchical synchronization analysis: {series_name_1} ⇄ {series_name_2}")

    # Prepare data and features
    data_dict = {
        series_name_1: structural_changes_1.get('data', np.zeros(100)),
        series_name_2: structural_changes_2.get('data', np.zeros(100))
    }
    
    features_dict = {
        series_name_1: structural_changes_1,
        series_name_2: structural_changes_2
    }
    
    # Default config
    if config is None:
        config = L3Config(draws=8000, tune=8000, target_accept=0.95)

    try:
        # Pairwise Bayesian estimation
        trace, model = fit_l3_pairwise_bayesian_system(
            data_dict, features_dict, config,
            series_pair=(series_name_1, series_name_2)
        )
        
        # Extract interaction coefficients
        interaction_coeffs = extract_interaction_coefficients(
            trace, [series_name_1, series_name_2]
        )
        
        # Calculate synchronization strength
        cross_effects = interaction_coeffs['cross_effects']
        
        sync_strength_1_to_2 = sum(
            abs(v) for v in cross_effects[f'{series_name_1}_to_{series_name_2}'].values()
        ) / 3
        
        sync_strength_2_to_1 = sum(
            abs(v) for v in cross_effects[f'{series_name_2}_to_{series_name_1}'].values()
        ) / 3
        
        overall_sync_strength = (sync_strength_1_to_2 + sync_strength_2_to_1) / 2
        total_asymmetry = abs(sync_strength_1_to_2 - sync_strength_2_to_1)
        
        if verbose:
            print(f"  Sync strength: {series_name_1}→{series_name_2}={sync_strength_1_to_2:.3f}, "
                  f"{series_name_2}→{series_name_1}={sync_strength_2_to_1:.3f}, "
                  f"asymmetry={total_asymmetry:.3f}")
        
        return {
            'trace': trace,
            'model': model,
            'sync_strength_1_to_2': sync_strength_1_to_2,
            'sync_strength_2_to_1': sync_strength_2_to_1,
            'overall_sync_strength': overall_sync_strength,
            'asymmetry': total_asymmetry,
            'interaction_coefficients': interaction_coeffs,
            'series_names': [series_name_1, series_name_2]
        }
        
    except Exception as e:
        if verbose:
            print(f"  Hierarchical synchronization analysis error: {e}")
        return {
            'error': str(e),
            'sync_strength_1_to_2': 0.0,
            'sync_strength_2_to_1': 0.0,
            'overall_sync_strength': 0.0,
            'series_names': [series_name_1, series_name_2]
        }

# ===============================
# SECTION 11: VISUALIZATION
# ===============================
class Lambda3Visualizer:
    """
    Lambda³理論に基づく統合可視化システム
    構造テンソル空間の本質的ダイナミクスを表現
    """
    
    def __init__(self, style: str = 'scientific'):
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """可視化スタイルの設定"""
        if self.style == 'scientific':
            plt.style.use('seaborn-v0_8-darkgrid')
            self.colors = {
                'pos_jump': '#e74c3c',
                'neg_jump': '#3498db',
                'tension': '#2ecc71',
                'hierarchy_local': '#9b59b6',
                'hierarchy_global': '#f39c12',
                'background': '#ecf0f1'
            }
        else:
            self.colors = plt.cm.Set1.colors
            
    def plot_structural_tensor_evolution(
        self,
        features_dict: Dict[str, Dict[str, np.ndarray]],
        series_names: List[str],
        figsize: Tuple[int, int] = (16, 8)
    ) -> plt.Figure:
        """
        構造テンソル(Λ)の時間発展を統合可視化
        ΔΛC pulsationとρTの本質的ダイナミクス
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        # === 上段: ΔΛC Pulsation Timeline ===
        ax1 = fig.add_subplot(gs[0])
        self._plot_pulsation_timeline(ax1, features_dict, series_names)
        
        # === 中段: Tension Scalar Evolution ===
        ax2 = fig.add_subplot(gs[1])
        self._plot_tension_evolution(ax2, features_dict, series_names)
        
        # === 下段: Structural Coherence ===
        ax3 = fig.add_subplot(gs[2])
        self._plot_structural_coherence(ax3, features_dict, series_names)
        
        fig.suptitle('Structural Tensor Evolution in Semantic Space', 
                     fontsize=16, fontweight='bold')
        
        return fig
    
    def _plot_pulsation_timeline(self, ax, features_dict, series_names):
        """ΔΛC pulsationの統合タイムライン"""
        n_series = len(series_names)
        
        for i, name in enumerate(series_names):
            y_base = i
            features = features_dict[name]
            
            # Positive pulsations
            pos_indices = np.where(features['delta_LambdaC_pos'] > 0)[0]
            pos_magnitudes = features['delta_LambdaC_pos'][pos_indices]
            
            # Negative pulsations
            neg_indices = np.where(features['delta_LambdaC_neg'] > 0)[0]
            neg_magnitudes = features['delta_LambdaC_neg'][neg_indices]
            
            # Plot as vertical lines with magnitude
            if len(pos_indices) > 0:
                ax.vlines(pos_indices, y_base, y_base + 0.4 * pos_magnitudes,
                         colors=self.colors['pos_jump'], alpha=0.7, linewidth=2)
                         
            if len(neg_indices) > 0:
                ax.vlines(neg_indices, y_base, y_base - 0.4 * neg_magnitudes,
                         colors=self.colors['neg_jump'], alpha=0.7, linewidth=2)
            
            # Series label
            ax.text(-5, y_base, name, ha='right', va='center', fontsize=10)
        
        ax.set_xlim(0, len(features_dict[series_names[0]]['data']))
        ax.set_ylim(-0.5, n_series - 0.5)
        ax.set_xlabel('Structural Time τ')
        ax.set_title('ΔΛC Pulsations across Series')
        ax.grid(True, alpha=0.3)
        
        # Legend
        ax.plot([], [], color=self.colors['pos_jump'], linewidth=2, label='ΔΛC⁺')
        ax.plot([], [], color=self.colors['neg_jump'], linewidth=2, label='ΔΛC⁻')
        ax.legend(loc='upper right')
        
    def _plot_tension_evolution(self, ax, features_dict, series_names):
        """張力スカラー(ρT)の進化"""
        for i, name in enumerate(series_names):
            rho_t = features_dict[name]['rho_T']
            time = np.arange(len(rho_t))
            
            # Smooth tension curve
            ax.plot(time, rho_t, label=name, alpha=0.8, linewidth=1.5)
            
        ax.set_xlabel('Structural Time τ')
        ax.set_ylabel('Tension Scalar ρT')
        ax.set_title('Evolution of Structural Tension')
        ax.legend(loc='upper right', ncol=min(3, len(series_names)))
        ax.grid(True, alpha=0.3)
        
    def _plot_structural_coherence(self, ax, features_dict, series_names):
        """構造的一貫性の時間発展"""
        window = 20
        n_windows = len(features_dict[series_names[0]]['data']) - window + 1
        
        coherence_evolution = []
        
        for t in range(0, n_windows, 5):  # Skip every 5 for efficiency
            # Calculate instantaneous coherence
            coherence = 0
            count = 0
            
            for i in range(len(series_names)):
                for j in range(i+1, len(series_names)):
                    rho_i = features_dict[series_names[i]]['rho_T'][t:t+window]
                    rho_j = features_dict[series_names[j]]['rho_T'][t:t+window]
                    
                    if len(rho_i) > 1 and len(rho_j) > 1:
                        corr = np.corrcoef(rho_i, rho_j)[0, 1]
                        coherence += abs(corr)
                        count += 1
                        
            if count > 0:
                coherence_evolution.append(coherence / count)
            else:
                coherence_evolution.append(0)
                
        time_points = range(0, n_windows, 5)
        ax.fill_between(time_points, coherence_evolution, alpha=0.3)
        ax.plot(time_points, coherence_evolution, linewidth=2)
        
        ax.set_xlabel('Structural Time τ')
        ax.set_ylabel('Coherence')
        ax.set_title('Structural Coherence Evolution')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
    def plot_hierarchical_dynamics(
        self,
        hierarchical_results: Dict[str, any],
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """階層的ダイナミクスの本質的可視化"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # Extract first series with hierarchical data
        series_name = None
        series_data = None
        for name, data in hierarchical_results.items():
            if 'hierarchical_separation' in data and data['hierarchical_separation']:
                series_name = name
                series_data = data
                break
                
        if series_data is None:
            ax1.text(0.5, 0.5, 'No hierarchical data available', 
                    transform=ax1.transAxes, ha='center', va='center')
            return fig
            
        # === Upper: Hierarchical Flow Diagram ===
        self._plot_hierarchical_flow(ax1, series_data, series_name)
        
        # === Lower: Transition Dynamics ===
        self._plot_transition_dynamics(ax2, series_data)
        
        fig.suptitle(f'Hierarchical Structural Dynamics: {series_name}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
        
    def _plot_hierarchical_flow(self, ax, series_data, series_name):
        """階層間フローの可視化"""
        if 'hierarchical_separation' not in series_data:
            return
            
        sep_data = series_data['hierarchical_separation']
        local_series = sep_data.get('local_series', np.zeros(100))
        global_series = sep_data.get('global_series', np.zeros(100))
        
        time = np.arange(len(local_series))
        
        # Create flow visualization
        ax.fill_between(time, 0, local_series, 
                       color=self.colors['hierarchy_local'], alpha=0.3, 
                       label='Local Structure')
        ax.fill_between(time, 0, -global_series, 
                       color=self.colors['hierarchy_global'], alpha=0.3,
                       label='Global Structure')
                       
        # Add flow arrows at transition points
        coeffs = sep_data.get('separation_coefficients', {})
        escalation = abs(coeffs.get('escalation', {}).get('coefficient', 0))
        deescalation = abs(coeffs.get('deescalation', {}).get('coefficient', 0))
        
        # Annotate key transitions
        ax.annotate(f'Escalation: {escalation:.3f}', 
                   xy=(0.7, 0.8), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
        ax.annotate(f'De-escalation: {deescalation:.3f}',
                   xy=(0.7, 0.2), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
                   
        ax.set_ylabel('Hierarchical Amplitude')
        ax.set_title('Local ⇄ Global Structural Flow')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
    def _plot_transition_dynamics(self, ax, series_data):
        """遷移ダイナミクスの可視化"""
        metrics = series_data.get('hierarchy_metrics', {})
        
        labels = ['Local\nDominance', 'Global\nDominance', 
                 'Coupling\nStrength', 'Escalation\nRate']
        values = [
            metrics.get('local_dominance', 0),
            metrics.get('global_dominance', 0),
            metrics.get('coupling_strength', 0),
            metrics.get('escalation_rate', 0)
        ]
        
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=['#9b59b6', '#f39c12', '#2ecc71', '#e74c3c'])
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
                   
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Metric Value')
        ax.set_title('Hierarchical Transition Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        
    def plot_interaction_network(
        self,
        interaction_results: Dict[str, any],
        series_names: List[str],
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """相互作用ネットワークの本質的可視化"""
        fig = plt.figure(figsize=figsize)
        
        # Create main axis for network
        ax_main = fig.add_subplot(111)
        
        # Build interaction graph
        G = self._build_interaction_graph(interaction_results, series_names)
        
        # Network layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=3000, alpha=0.9, ax=ax_main)
        
        # Draw edges with varying widths based on interaction strength
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights],
                              alpha=0.6, edge_color='gray', 
                              arrows=True, arrowsize=20, ax=ax_main)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax_main)
        
        # Add edge labels for significant interactions
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            if data['weight'] > 0.1:  # Only show significant interactions
                edge_labels[(u, v)] = f"{data['weight']:.2f}"
                
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, ax=ax_main)
        
        ax_main.set_title('Structural Tensor Interaction Network', 
                         fontsize=16, fontweight='bold')
        ax_main.axis('off')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='gray', label='Interaction Strength'),
            mpatches.Patch(color='lightblue', label='Series Node')
        ]
        ax_main.legend(handles=legend_elements, loc='upper right')
        
        return fig
        
    def _build_interaction_graph(self, interaction_results, series_names):
        """相互作用グラフの構築"""
        G = nx.DiGraph()
        
        # Add nodes
        for name in series_names:
            G.add_node(name)
            
        # Add edges based on interaction strengths
        if 'interaction_matrix' in interaction_results:
            matrix = interaction_results['interaction_matrix']
            for i, name_i in enumerate(series_names):
                for j, name_j in enumerate(series_names):
                    if i != j and matrix[i, j] > 0:
                        G.add_edge(name_i, name_j, weight=matrix[i, j])
                        
        elif 'pairs' in interaction_results:
            # Extract from pairwise results
            for pair_key, pair_data in interaction_results['pairs'].items():
                if 'interaction_coefficients' in pair_data:
                    coeffs = pair_data['interaction_coefficients']
                    if 'cross_effects' in coeffs:
                        for direction, effects in coeffs['cross_effects'].items():
                            parts = direction.split('_to_')
                            if len(parts) == 2:
                                strength = sum(abs(v) for v in effects.values())
                                G.add_edge(parts[0], parts[1], weight=strength)
                                
        return G
        
    def plot_lambda3_core_dashboard(
        self,
        results: Dict[str, any],
        figsize: Tuple[int, int] = (18, 12)
    ) -> plt.Figure:
        """Lambda³コアダイナミクスの統合ダッシュボード"""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3,
                             height_ratios=[1.5, 1, 1],
                             width_ratios=[2, 1, 1])
        
        features_dict = results.get('features_dict', {})
        series_names = results.get('series_names', [])
        
        # === 1. Main: Structural Evolution Timeline ===
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_integrated_timeline(ax1, features_dict, series_names)
        
        # === 2. Interaction Matrix ===
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_interaction_matrix(ax2, results)
        
        # === 3. Synchronization Profile ===
        ax3 = fig.add_subplot(gs[1, 1:])
        self._plot_synchronization_profile(ax3, results)
        
        # === 4. Crisis/Regime Indicators ===
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_structural_indicators(ax4, results)
        
        fig.suptitle('Lambda³ Structural Tensor Dynamics Dashboard',
                     fontsize=18, fontweight='bold')
                     
        return fig
        
    def _plot_integrated_timeline(self, ax, features_dict, series_names):
        """統合構造変化タイムライン"""
        if not features_dict or not series_names:
            return
            
        # Aggregate structural changes
        time_length = len(features_dict[series_names[0]]['data'])
        aggregate_pos = np.zeros(time_length)
        aggregate_neg = np.zeros(time_length)
        
        for name in series_names:
            aggregate_pos += features_dict[name]['delta_LambdaC_pos']
            aggregate_neg += features_dict[name]['delta_LambdaC_neg']
            
        time = np.arange(time_length)
        
        # Plot aggregated pulsations
        ax.fill_between(time, 0, aggregate_pos, color=self.colors['pos_jump'], 
                       alpha=0.5, label='Aggregate ΔΛC⁺')
        ax.fill_between(time, 0, -aggregate_neg, color=self.colors['neg_jump'],
                       alpha=0.5, label='Aggregate ΔΛC⁻')
                       
        # Overlay average tension
        avg_tension = np.mean([features_dict[name]['rho_T'] for name in series_names], axis=0)
        ax2 = ax.twinx()
        ax2.plot(time, avg_tension, color=self.colors['tension'], 
                linewidth=2, label='Average ρT')
        ax2.set_ylabel('Average Tension ρT', color=self.colors['tension'])
        ax2.tick_params(axis='y', labelcolor=self.colors['tension'])
        
        ax.set_xlabel('Structural Time τ')
        ax.set_ylabel('Aggregate Structural Change')
        ax.set_title('Integrated Structural Evolution')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
    def _plot_interaction_matrix(self, ax, results):
        """相互作用行列の可視化"""
        if 'sync_matrix' in results:
            matrix = results['sync_matrix']
            series_names = results.get('series_names', [])
            
            im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(len(series_names)))
            ax.set_yticks(range(len(series_names)))
            ax.set_xticklabels(series_names, rotation=45, ha='right')
            ax.set_yticklabels(series_names)
            ax.set_title('Synchronization Matrix')
            
            # Add text annotations
            for i in range(len(series_names)):
                for j in range(len(series_names)):
                    text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                  ha="center", va="center",
                                  color="white" if matrix[i, j] > 0.5 else "black")
                                  
    def _plot_synchronization_profile(self, ax, results):
        """同期プロファイルの可視化"""
        if 'causality_results' in results:
            causality = results['causality_results']
            if 'basic_causality' in causality:
                # Plot causality patterns
                for direction, pattern in causality['basic_causality'].items():
                    if pattern:
                        lags = list(pattern.keys())
                        probs = list(pattern.values())
                        ax.plot(lags, probs, 'o-', label=direction.replace('_', '→'),
                               markersize=6, alpha=0.8)
                               
                ax.set_xlabel('Lag')
                ax.set_ylabel('Causality Strength')
                ax.set_title('Structural Causality Profile')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
    def _plot_structural_indicators(self, ax, results):
        """構造的指標の時系列"""
        if 'crisis_results' in results:
            crisis = results['crisis_results']
            aggregate_crisis = crisis.get('aggregate_crisis', [])
            
            if len(aggregate_crisis) > 0:
                time = np.arange(len(aggregate_crisis))
                ax.plot(time, aggregate_crisis, 'k-', linewidth=2)
                ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.7,
                          label='Crisis Threshold')
                          
                # Highlight crisis episodes
                for start, end in crisis.get('crisis_episodes', []):
                    ax.axvspan(start, end, alpha=0.2, color='red')
                    
                ax.set_xlabel('Structural Time τ')
                ax.set_ylabel('Crisis Indicator')
                ax.set_title('Structural Crisis Detection')
                ax.legend()
                ax.grid(True, alpha=0.3)

# ===============================
# Standalone plotting functions
# ===============================

def plot_lambda3_summary(
    results: Dict[str, any],
    focus: str = 'comprehensive',
    save_path: Optional[str] = None
) -> None:
    """
    Lambda³分析結果の要約可視化
    
    Parameters:
    -----------
    results : Dict
        Lambda³分析結果
    focus : str
        'comprehensive', 'structural', 'interaction', 'hierarchical'
    save_path : Optional[str]
        保存パス（Noneの場合は表示のみ）
    """
    visualizer = Lambda3Visualizer()
    
    if focus == 'comprehensive':
        fig = visualizer.plot_lambda3_core_dashboard(results)
    elif focus == 'structural':
        fig = visualizer.plot_structural_tensor_evolution(
            results['features_dict'], 
            results['series_names']
        )
    elif focus == 'interaction':
        if 'pairwise_results' in results:
            fig = visualizer.plot_interaction_network(
                results['pairwise_results'],
                results['series_names']
            )
        else:
            print("No interaction results available")
            return
    elif focus == 'hierarchical':
        if 'hierarchical_results' in results:
            fig = visualizer.plot_hierarchical_dynamics(
                results['hierarchical_results']
            )
        else:
            print("No hierarchical results available")
            return
            
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()

def quick_lambda3_plot(
    features_dict: Dict[str, Dict[str, np.ndarray]],
    series_names: List[str],
    plot_type: str = 'pulsation'
) -> None:
    """
    Lambda³特徴量の簡易可視化
    
    Parameters:
    -----------
    features_dict : Dict
        特徴量辞書
    series_names : List[str]
        系列名リスト
    plot_type : str
        'pulsation', 'tension', 'phase_space'
    """
    if plot_type == 'pulsation':
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, name in enumerate(series_names):
            y_offset = i * 0.5
            pos_events = np.where(features_dict[name]['delta_LambdaC_pos'] > 0)[0]
            neg_events = np.where(features_dict[name]['delta_LambdaC_neg'] > 0)[0]
            
            ax.scatter(pos_events, [y_offset] * len(pos_events),
                      marker='^', s=100, c='red', alpha=0.7, label=f'{name} ΔΛC⁺' if i == 0 else '')
            ax.scatter(neg_events, [y_offset] * len(neg_events),
                      marker='v', s=100, c='blue', alpha=0.7, label=f'{name} ΔΛC⁻' if i == 0 else '')
                      
        ax.set_yticks([i * 0.5 for i in range(len(series_names))])
        ax.set_yticklabels(series_names)
        ax.set_xlabel('Structural Time τ')
        ax.set_title('ΔΛC Pulsation Pattern')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    elif plot_type == 'tension':
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name in series_names:
            ax.plot(features_dict[name]['rho_T'], label=name, alpha=0.8)
            
        ax.set_xlabel('Structural Time τ')
        ax.set_ylabel('Tension Scalar ρT')
        ax.set_title('Tension Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    elif plot_type == 'phase_space':
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, name in enumerate(series_names):
            colors = plt.cm.Set1(i)
            ax.scatter(
                features_dict[name]['delta_LambdaC_pos'],
                features_dict[name]['delta_LambdaC_neg'],
                features_dict[name]['rho_T'],
                c=[colors], s=30, alpha=0.6, label=name
            )
            
        ax.set_xlabel('ΔΛC⁺')
        ax.set_ylabel('ΔΛC⁻')
        ax.set_zlabel('ρT')
        ax.set_title('Structural Tensor Phase Space')
        ax.legend()
        
    plt.tight_layout()
    plt.show()

# ===============================
# SECTION 15: MAIN ANALYSIS PIPELINES - ENHANCED
# ===============================

def main_lambda3_comprehensive_analysis(
    csv_path: str = "financial_data_2022-2024.csv",
    config: L3Config = None,
    analysis_modes: Dict[str, bool] = None
) -> Dict[str, any]:
    """
    Lambda³ Comprehensive Analysis - 全機能統合パイプライン
    構造テンソル空間における完全分析システム
    """
    if config is None:
        config = L3Config()

    if analysis_modes is None:
        analysis_modes = {
            'hierarchical_analysis': True,
            'separation_dynamics': True,
            'pairwise_analysis': True,
            'asymmetric_analysis': True,
            'regime_analysis': True,
            'crisis_detection': True,
            'advanced_visualization': True,
            'multi_scale_analysis': True,
            'coherence_analysis': True
        }

    print("="*80)
    print("LAMBDA³ COMPREHENSIVE ANALYSIS - ULTIMATE PIPELINE")
    print("Lambda³理論に基づく構造テンソル完全分析システム")
    print("="*80)

    # Step 1: データロード
    print("\n=== Step 1: データロードと前処理 ===")
    series_dict = load_csv_data(csv_path, time_column="Date")

    data_length = len(next(iter(series_dict.values())))
    config.T = data_length
    series_names = list(series_dict.keys())

    print(f"データ長: {data_length}, 系列数: {len(series_dict)}")
    print(f"分析対象: {', '.join(series_names)}")

    # Step 2: Lambda³特徴量抽出
    print("\n=== Step 2: Lambda³特徴量抽出 ===")
    features_dict = {}

    for name, data in series_dict.items():
        print(f"  {name}の構造テンソル特徴量を抽出中...")

        if analysis_modes['hierarchical_analysis']:
            features = calc_lambda3_features_hierarchical(data, config)
        else:
            feats = calc_lambda3_features_v2(data, config)
            features = {
                'data': data,
                'delta_LambdaC_pos': feats[0],
                'delta_LambdaC_neg': feats[1],
                'rho_T': feats[2],
                'time_trend': feats[3],
                'local_jump': feats[4]
            }

        features_dict[name] = features

        # 基本統計の表示
        pos_events = np.sum(features['delta_LambdaC_pos'])
        neg_events = np.sum(features['delta_LambdaC_neg'])
        avg_tension = np.mean(features['rho_T'])
        print(f"    ΔΛC⁺: {pos_events}, ΔΛC⁻: {neg_events}, 平均ρT: {avg_tension:.3f}")

    # 結果格納
    comprehensive_results = {
        'series_dict': series_dict,
        'features_dict': features_dict,
        'config': config,
        'series_names': series_names
    }

    # Step 3: 階層的構造変化分析
    hierarchical_results = None
    if analysis_modes['hierarchical_analysis']:
        print("\n=== Step 3: 階層的構造変化分析 ===")
        hierarchical_results = complete_hierarchical_analysis(
            series_dict, config, series_names=series_names
        )
        comprehensive_results['hierarchical_results'] = hierarchical_results

    # Step 5: ペアワイズ相互作用分析
    pairwise_results = None
    if analysis_modes['pairwise_analysis'] and len(series_names) >= 2:
        print("\n=== Step 5: ペアワイズ相互作用分析 ===")

        if analysis_modes['asymmetric_analysis']:
            pairwise_results = complete_asymmetric_pairwise_analysis(
                series_dict, features_dict, config, tuple(series_names[:2])
            )
        else:
            pairwise_results = complete_pairwise_analysis(
                series_dict, features_dict, config, tuple(series_names[:2])
            )

        comprehensive_results['pairwise_results'] = pairwise_results

    # Step 6: 金融レジーム分析
    regime_results = None
    if analysis_modes['regime_analysis']:
        print("\n=== Step 6: 金融レジーム分析 ===")
        regime_results = analyze_multi_asset_regimes(
            features_dict, series_dict, config, n_regimes=4
        )
        comprehensive_results['regime_results'] = regime_results

    # Step 7: 危機検出
    crisis_results = None
    if analysis_modes['crisis_detection']:
        print("\n=== Step 7: 金融危機検出 ===")
        crisis_results = detect_financial_crises(
            features_dict, series_dict, crisis_threshold=0.8
        )
        plot_crisis_detection(crisis_results, series_names)
        comprehensive_results['crisis_results'] = crisis_results

    # Step 8: 同期性ネットワーク分析
    print("\n=== Step 8: 同期性ネットワーク分析 ===")
    event_series_dict = {
        name: features_dict[name]['delta_LambdaC_pos'].astype(np.float64)
        for name in series_names
    }

    sync_mat, names = sync_matrix(event_series_dict, lag_window=10)

    # 同期行列の可視化
    plt.figure(figsize=(10, 8))
    sns.heatmap(sync_mat, annot=True, fmt='.3f',
                xticklabels=names, yticklabels=names,
                cmap="Blues", vmin=0, vmax=1, square=True)
    plt.title("構造変化同期率行列 (σₛ)")
    plt.tight_layout()
    plt.show()

    # 同期ネットワーク構築
    threshold = np.percentile([sync_mat[i, j] for i in range(len(names))
                              for j in range(len(names)) if i != j], 75)
    G = build_sync_network(event_series_dict, sync_threshold=threshold)

    if G.number_of_edges() > 0:
        plot_sync_network(G)

    comprehensive_results['sync_matrix'] = sync_mat
    comprehensive_results['sync_network'] = G

    # Step 9: 高度可視化分析
    if analysis_modes['advanced_visualization']:
        print("\n=== Step 9: 高度可視化分析 ===")

        # 統合ダッシュボード
        print("統合ダッシュボードを生成中...")
        plot_comprehensive_lambda3_dashboard(comprehensive_results, series_names)

    # Step 10: マルチスケール分析
    if analysis_modes['multi_scale_analysis']:
        print("\n=== Step 10: マルチスケール分析 ===")
        plot_multi_scale_analysis(features_dict, series_names[:3])

        # 動的相関分析
        print("動的相関を分析中...")
        plot_dynamic_correlation_heatmap(features_dict, series_names[:4])

    # Step 11: 構造的一貫性分析
    if analysis_modes['coherence_analysis']:
        print("\n=== Step 11: 構造的一貫性分析 ===")
        plot_structural_coherence_analysis(features_dict, series_names)

    # Step 12: 統合因果分析
    print("\n=== Step 12: 統合因果分析 ===")
    causality_results = analyze_complete_causality(
        features_dict, series_names[:2], lag_window=5,
        include_hierarchical=analysis_modes['hierarchical_analysis']
    )

    # 因果関係の可視化
    plot_complete_causality_analysis(causality_results, series_names[:2])

    comprehensive_results['causality_results'] = causality_results

    # Step 13: 統合サマリー生成
    print(f"\n{'='*80}")
    print("LAMBDA³ COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"{'='*80}")

    # データサマリー
    print(f"\nデータサマリー:")
    print(f"  分析期間: {data_length}期間")
    print(f"  分析系列: {len(series_names)}系列")
    print(f"  分析モード: {sum(analysis_modes.values())}/{len(analysis_modes)}項目")

    # 構造テンソル統計
    print(f"\n構造テンソル統計:")
    total_pos_events = sum(np.sum(features_dict[name]['delta_LambdaC_pos']) for name in series_names)
    total_neg_events = sum(np.sum(features_dict[name]['delta_LambdaC_neg']) for name in series_names)
    avg_tension = np.mean([np.mean(features_dict[name]['rho_T']) for name in series_names])

    print(f"  総正構造変化: {total_pos_events}")
    print(f"  総負構造変化: {total_neg_events}")
    print(f"  平均張力スカラー: {avg_tension:.3f}")

    # 階層的構造変化
    if hierarchical_results:
        print(f"\n階層的構造変化:")
        for name in series_names[:3]:
            if name in hierarchical_results:
                metrics = hierarchical_results[name]['hierarchy_metrics']
                print(f"  {name}:")
                print(f"    ローカル優勢度: {metrics['local_dominance']:.3f}")
                print(f"    グローバル優勢度: {metrics['global_dominance']:.3f}")
                print(f"    エスカレーション率: {metrics['escalation_rate']:.3f}")

                if 'hierarchical_separation' in hierarchical_results[name]:
                    sep_metrics = hierarchical_results[name]['hierarchical_separation']['asymmetry_metrics']
                    print(f"    エスカレーション優勢度: {sep_metrics['escalation_dominance']:.3f}")
                    print(f"    デエスカレーション優勢度: {sep_metrics['deescalation_dominance']:.3f}")

    # レジーム分析
    if regime_results:
        print(f"\n金融レジーム分析:")
        asset_regimes = regime_results['asset_regimes']
        for asset_name in series_names:
            if asset_name in asset_regimes:
                regime_info = asset_regimes[asset_name]
                print(f"  {asset_name}:")
                for regime_id, label in regime_info['labels'].items():
                    stats = regime_info['detector'].regime_features[regime_id]
                    print(f"    {label}: {stats['frequency']:.1%}")

    # 危機分析
    if crisis_results:
        print(f"\n危機検出結果:")
        print(f"  危機エピソード: {len(crisis_results['crisis_episodes'])}回")
        for i, (start, end) in enumerate(crisis_results['crisis_episodes']):
            duration = end - start + 1
            print(f"    Episode {i+1}: ステップ{start}-{end} (継続{duration}期間)")

    # 相互作用分析
    if pairwise_results:
        print(f"\n相互作用分析:")
        if 'asymmetric_results' in pairwise_results:
            interaction_coeffs = pairwise_results['asymmetric_results']['interaction_coefficients']
            for direction, effects in interaction_coeffs.items():
                total_effect = sum(abs(v) for v in effects.values())
                print(f"  {direction}: 総合強度 {total_effect:.3f}")
        elif 'interaction_coefficients' in pairwise_results:
            cross_effects = pairwise_results['interaction_coefficients']['cross_effects']
            for direction, effects in cross_effects.items():
                total_effect = sum(abs(v) for v in effects.values())
                print(f"  {direction}: 総合強度 {total_effect:.3f}")

    # 同期性分析
    print(f"\n同期性分析:")
    max_sync = np.max(sync_mat[sync_mat < 1.0])
    min_sync = np.min(sync_mat[sync_mat < 1.0])
    avg_sync = np.mean(sync_mat[sync_mat < 1.0])

    print(f"  最大同期率: {max_sync:.3f}")
    print(f"  最小同期率: {min_sync:.3f}")
    print(f"  平均同期率: {avg_sync:.3f}")
    print(f"  同期ネットワークエッジ数: {G.number_of_edges()}")

    # 因果関係分析
    if causality_results:
        print(f"\n因果関係分析:")
        basic_causality = causality_results.get('basic_causality', {})
        max_causality = 0
        max_direction = ""

        for direction, pattern in basic_causality.items():
            if pattern:
                max_prob = max(pattern.values())
                if max_prob > max_causality:
                    max_causality = max_prob
                    max_direction = direction

        if max_direction:
            print(f"  最強因果関係: {max_direction}")
            print(f"  最大因果確率: {max_causality:.3f}")

    print(f"\n{'='*80}")
    print("Lambda³ Comprehensive Analysis Complete!")
    print("• 構造テンソル相互作用を完全解析")
    print("• 階層分離ダイナミクスを高速分析")
    print("• 非対称相互作用を詳細分析")
    print("• 市場レジームを検出・同期化")
    print("• 危機エピソードを特定")
    print("• マルチスケール構造変化を解析")
    print("• 構造的一貫性を定量化")
    print("• 統合因果関係を解明")
    print(f"{'='*80}")

    return comprehensive_results

def main_lambda3_custom_analysis(
    csv_path: str,
    target_series: List[str] = None,
    analysis_focus: str = "comprehensive",
    config: L3Config = None
) -> Dict[str, any]:
    """
    カスタマイズ可能なLambda³分析パイプライン
    特定の分析に焦点を当てた柔軟な実行
    """
    if config is None:
        config = L3Config()

    # 分析フォーカスに応じた設定
    focus_configs = {
        "hierarchical": {
            'hierarchical_analysis': True,
            'separation_dynamics': True,
            'pairwise_analysis': False,
            'asymmetric_analysis': False,
            'regime_analysis': False,
            'crisis_detection': False,
            'advanced_visualization': True,
            'multi_scale_analysis': False,
            'coherence_analysis': False
        },
        "pairwise": {
            'hierarchical_analysis': False,
            'separation_dynamics': False,
            'pairwise_analysis': True,
            'asymmetric_analysis': True,
            'regime_analysis': False,
            'crisis_detection': False,
            'advanced_visualization': True,
            'multi_scale_analysis': False,
            'coherence_analysis': False
        },
        "regime": {
            'hierarchical_analysis': False,
            'separation_dynamics': False,
            'pairwise_analysis': False,
            'asymmetric_analysis': False,
            'regime_analysis': True,
            'crisis_detection': True,
            'advanced_visualization': True,
            'multi_scale_analysis': False,
            'coherence_analysis': False
        },
        "comprehensive": {
            'hierarchical_analysis': True,
            'separation_dynamics': True,
            'pairwise_analysis': True,
            'asymmetric_analysis': True,
            'regime_analysis': True,
            'crisis_detection': True,
            'advanced_visualization': True,
            'multi_scale_analysis': True,
            'coherence_analysis': True
        }
    }

    analysis_modes = focus_configs.get(analysis_focus, focus_configs["comprehensive"])

    print(f"Lambda³ Custom Analysis - Focus: {analysis_focus}")

    # データロード
    series_dict = load_csv_data(csv_path, time_column="Date")

    # 対象系列の絞り込み
    if target_series:
        filtered_series_dict = {name: data for name, data in series_dict.items()
                               if name in target_series}
        series_dict = filtered_series_dict

    # メイン分析実行
    results = main_lambda3_comprehensive_analysis(
        csv_path=csv_path,
        config=config,
        analysis_modes=analysis_modes
    )

    return results

def main_lambda3_rapid_analysis(
    csv_path: str,
    max_series: int = 3,
    rapid_config: Dict[str, int] = None
) -> Dict[str, any]:
    """
    高速Lambda³分析パイプライン
    迅速な分析のための軽量版
    """
    if rapid_config is None:
        rapid_config = {
            'draws': 2000,
            'tune': 2000,
            'target_accept': 0.90
        }

    # 高速設定
    config = L3Config(
        draws=rapid_config['draws'],
        tune=rapid_config['tune'],
        target_accept=rapid_config['target_accept']
    )

    # 高速分析モード
    analysis_modes = {
        'hierarchical_analysis': True,
        'separation_dynamics': True,  # 高速版を使用
        'pairwise_analysis': True,
        'asymmetric_analysis': False,  # スキップして高速化
        'regime_analysis': True,
        'crisis_detection': True,
        'advanced_visualization': False,  # 基本可視化のみ
        'multi_scale_analysis': False,
        'coherence_analysis': False
    }

    print("Lambda³ Rapid Analysis - 高速分析モード")
    print(f"設定: draws={config.draws}, tune={config.tune}")

    # データロード（系列数制限）
    series_dict = load_csv_data(csv_path, time_column="Date")

    # 系列数制限
    if len(series_dict) > max_series:
        limited_series = dict(list(series_dict.items())[:max_series])
        series_dict = limited_series
        print(f"系列数を{max_series}に制限しました")

    # メイン分析実行
    results = main_lambda3_comprehensive_analysis(
        csv_path=csv_path,
        config=config,
        analysis_modes=analysis_modes
    )

    return results

# ===============================
# SECTION 12: EXECUTION AND MAIN
# ===============================

if __name__ == '__main__':
    # Lambda³設定
    config = L3Config()

    print("Lambda³ Analytics Framework - Ultimate Edition")
    print("構造テンソル空間における完全分析システム")
    print("="*60)

    # 実行モード選択
    execution_mode = "comprehensive"  # "comprehensive", "custom", "rapid"

    if execution_mode == "comprehensive":
        print("実行モード: 包括的分析 (全機能)")

        # 金融データ取得
        print("\n金融データの取得...")
        fetch_financial_data(
            start_date="2022-01-01",
            end_date="2024-12-31",
            csv_filename="financial_data_2022-2024.csv"
        )

        # Lambda³包括的分析実行
        results = main_lambda3_comprehensive_analysis(
            csv_path="financial_data_2022-2024.csv",
            config=config
        )

    elif execution_mode == "custom":
        print("実行モード: カスタム分析")

        # カスタム分析実行例
        results = main_lambda3_custom_analysis(
            csv_path="financial_data_2024.csv",
            target_series=["USD/JPY", "Nikkei 225"],
            analysis_focus="pairwise",  # "hierarchical", "pairwise", "regime", "comprehensive"
            config=config
        )

    elif execution_mode == "rapid":
        print("実行モード: 高速分析")

        # 高速分析実行
        results = main_lambda3_rapid_analysis(
            csv_path="financial_data_2024.csv",
            max_series=3,
            rapid_config={'draws': 4000, 'tune': 4000, 'target_accept': 0.95}
        )

    print("\n" + "="*60)
    print("Lambda³ Analytics Framework - Analysis Complete!")
    print("="*60)

    # 結果の簡易表示
    if results:
        print(f"\n分析結果サマリー:")
        print(f"  分析系列数: {len(results['series_names'])}")
        print(f"  データ期間: {results['config'].T}")

        if 'hierarchical_results' in results:
            print(f"  階層的分析: 完了")

        if 'pairwise_results' in results:
            print(f"  ペアワイズ分析: 完了")

        if 'regime_results' in results:
            print(f"  レジーム分析: 完了")

        if 'crisis_results' in results:
            crisis_episodes = len(results['crisis_results']['crisis_episodes'])
            print(f"  危機検出: {crisis_episodes}エピソード")

        print(f"\n構造テンソル空間における全∆ΛC pulsationsが解析され、")
        print(f"progression vectors (ΛF) と tension scalars (ρT) の")
        print(f"完全な相互作用パターンが明らかになりました。")

        print(f"\nLambda³理論に基づく semantic structure space の")
        print(f"多次元的理解が達成されました。")
