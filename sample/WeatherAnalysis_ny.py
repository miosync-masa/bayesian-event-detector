# ==========================================================
# Λ³ABC: Lambda³ Analytics for Weather Analysis
# ----------------------------------------------------
# Weather phenomena as structural tensor transactions
# No time causality - only structural pulsations (∆ΛC)
#
# Author: Modified for Weather Analysis
# License: MIT
# ----------------------------------------------------

# ===============================
#  import
# ===============================
import pymc as pm
import numpy as np
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from numba import jit, njit, prange
from typing import Tuple, Dict, List, Optional
import pandas as pd
from pathlib import Path

# ===============================
# Global Constants for JIT
# ===============================
DELTA_PERCENTILE = 97.0  # Percentile threshold for jump detection
LOCAL_JUMP_PERCENTILE = 97.0  # Percentile for local jump detection
WINDOW_SIZE = 10  # Window size for tension scalar calculation
LOCAL_WINDOW_SIZE = 10  # Window for local standard deviation
LAG_WINDOW_DEFAULT = 10  # Default lag window for synchronization
SYNC_THRESHOLD_DEFAULT = 0.3  # Default threshold for sync network edges

# ===============================
# Lambda³ Config Class
# ===============================
@dataclass
class L3Config:
    """Configuration for Lambda³ analysis parameters."""
    T: int = 150  # Time series length
    # Feature extraction parameters (uses globals for JIT compatibility)
    window: int = WINDOW_SIZE
    local_window: int = LOCAL_WINDOW_SIZE
    delta_percentile: float = DELTA_PERCENTILE
    local_jump_percentile: float = LOCAL_JUMP_PERCENTILE
    # Bayesian sampling parameters
    draws: int = 8000  # Number of MCMC draws
    tune: int = 8000  # Number of tuning/warmup steps
    target_accept: float = 0.97  # Target acceptance probability
    # Posterior visualization parameters
    var_names: list = ('beta_time_a', 'beta_time_b', 'beta_interact', 'beta_rhoT_a', 'beta_rhoT_b')
    hdi_prob: float = 0.94  # Highest density interval probability

# ===============================
# Lambda³ Weather Data API
# ===============================

def fetch_weather_data(
    csv_path="tokyo_weather_days.csv",
    start_date=None,
    end_date=None,
    weather_params=None,
    desired_order=None,
    output_filename="weather_lambda3_data.csv",
    verbose=True
):
    """
    Load and preprocess weather time series data for Lambda³ analysis.

    In Lambda³ Theory:
    - Weather parameters are structural components (Λ), not temporal measurements
    - Each measurement is a structural state snapshot in semantic space
    - Changes are ∆ΛC pulsations, not time-based evolution

    Args:
        csv_path (str): Path to weather CSV file.
        start_date (str): Optional start date filter.
        end_date (str): Optional end date filter.
        weather_params (dict): Dictionary mapping display names to column names.
        desired_order (list): List of parameters for column ordering.
        output_filename (str): Output CSV file name.
        verbose (bool): Whether to print progress and sample data.

    Returns:
        pd.DataFrame: Preprocessed weather structural data.
    """
    # Default weather parameters mapping
    if weather_params is None:
        weather_params = {
            "Temperature": "temperature_2m",
            "Humidity": "relative_humidity_2m",
            "DewPoint": "dew_point_2m",
            "Precipitation": "precipitation",
            "WindSpeed": "wind_speed_10m",
            "Pressure": "surface_pressure"
        }

    if desired_order is None:
        desired_order = ["Temperature", "Humidity", "DewPoint", "Precipitation", "WindSpeed", "Pressure"]

    if verbose:
        print(f"Loading weather structural data from {csv_path}...")

    try:
        # Load weather data
        df = pd.read_csv(csv_path)

        # Optional date filtering
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            if start_date:
                df = df[df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['date'] <= pd.to_datetime(end_date)]
            df = df.sort_values('date')
            df = df.set_index('date')

        # Select and rename weather parameters
        reversed_params = {v: k for k, v in weather_params.items()}
        weather_data = df[list(weather_params.values())].rename(columns=reversed_params)

        # Normalize pressure to compatible scale
        if "Pressure" in weather_data.columns:
            weather_data["Pressure"] = (weather_data["Pressure"] - 1000) / 10

        # Reorder columns
        weather_data = weather_data[desired_order]

        # Drop any rows with missing values
        weather_data = weather_data.dropna()

        if verbose:
            print("\nFirst 5 structural snapshots:")
            print(weather_data.head())
            print("\nLast 5 structural snapshots:")
            print(weather_data.tail())
            print(f"\nStructural components: {list(weather_data.columns)}")
            print(f"Total snapshots: {len(weather_data)}")

        # Save to CSV
        weather_data.to_csv(output_filename, index=True)
        if verbose:
            print(f"\nData successfully saved to '{output_filename}'.")

        return weather_data

    except Exception as e:
        print(f"\nError occurred while loading weather data: {e}")
        return None

# ===============================
# JIT-compiled Core Functions (Same as original)
# ===============================
@njit
def calculate_diff_and_threshold(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """
    JIT-compiled difference calculation and threshold computation.

    Args:
        data: Input time series data
        percentile: Percentile for threshold calculation

    Returns:
        diff: First differences of the data
        threshold: Calculated threshold value
    """
    diff = np.empty(len(data))
    diff[0] = 0
    for i in range(1, len(data)):
        diff[i] = data[i] - data[i-1]

    abs_diff = np.abs(diff)
    threshold = np.percentile(abs_diff, percentile)
    return diff, threshold

@njit
def detect_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled jump detection based on threshold.

    Args:
        diff: First differences of time series
        threshold: Jump detection threshold

    Returns:
        pos_jumps: Binary array indicating positive jumps
        neg_jumps: Binary array indicating negative jumps
    """
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
    """
    JIT-compiled local standard deviation calculation.

    Args:
        data: Input time series
        window: Window size for local calculation

    Returns:
        local_std: Array of local standard deviations
    """
    n = len(data)
    local_std = np.empty(n)

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)

        # Calculate std manually for JIT compatibility
        subset = data[start:end]
        mean = np.mean(subset)
        variance = np.sum((subset - mean) ** 2) / len(subset)
        local_std[i] = np.sqrt(variance)

    return local_std

@njit
def calculate_rho_t(data: np.ndarray, window: int) -> np.ndarray:
    """
    JIT-compiled tension scalar (ρT) calculation.
    Represents local volatility/tension in the time series.

    Args:
        data: Input time series
        window: Window size for calculation

    Returns:
        rho_t: Array of tension scalar values
    """
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
    """
    JIT-compiled synchronization rate calculation for a specific lag.

    Args:
        series_a: First binary event series
        series_b: Second binary event series
        lag: Time lag (positive = b lags a, negative = a lags b)

    Returns:
        Synchronization rate at the specified lag
    """
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
    """
    JIT-compiled synchronization profile calculation with parallelization.

    Args:
        series_a: First binary event series
        series_b: Second binary event series
        lag_window: Maximum lag to consider

    Returns:
        lags: Array of lag values
        sync_values: Synchronization rates at each lag
        max_sync: Maximum synchronization rate
        optimal_lag: Lag with maximum synchronization
    """
    n_lags = 2 * lag_window + 1
    lags = np.arange(-lag_window, lag_window + 1)
    sync_values = np.empty(n_lags)

    # Parallel computation of sync rates
    for i in prange(n_lags):
        lag = lags[i]
        sync_values[i] = sync_rate_at_lag(series_a, series_b, lag)

    # Find maximum synchronization
    max_sync = 0.0
    optimal_lag = 0
    for i in range(n_lags):
        if sync_values[i] > max_sync:
            max_sync = sync_values[i]
            optimal_lag = lags[i]

    return lags, sync_values, max_sync, optimal_lag

# ===============================
# Feature Extraction Wrapper
# ===============================
def calc_lambda3_features_v2(data: np.ndarray, config: L3Config) -> Tuple[np.ndarray, ...]:
    """
    Wrapper for Lambda³ feature extraction using JIT-compiled functions.
    Extracts structural change (ΔΛC) and tension scalar (ρT) features.

    Args:
        data: Input time series
        config: Lambda³ configuration

    Returns:
        delta_pos: Positive structural changes (jumps)
        delta_neg: Negative structural changes (jumps)
        rho_t: Tension scalar time series
        time_trend: Linear time trend
        local_jump_detect: Local jump detection indicator
    """
    # Use JIT functions with global constants for performance
    diff, threshold = calculate_diff_and_threshold(data, DELTA_PERCENTILE)
    delta_pos, delta_neg = detect_jumps(diff, threshold)

    # Local jump detection using normalized score
    local_std = calculate_local_std(data, LOCAL_WINDOW_SIZE)
    score = np.abs(diff) / (local_std + 1e-8)  # Avoid division by zero
    local_threshold = np.percentile(score, LOCAL_JUMP_PERCENTILE)
    local_jump_detect = (score > local_threshold).astype(int)

    # Calculate tension scalar (local volatility measure)
    rho_t = calculate_rho_t(data, WINDOW_SIZE)

    # Simple linear time trend
    time_trend = np.arange(len(data))

    return delta_pos, delta_neg, rho_t, time_trend, local_jump_detect

# ===============================
# Bayesian Model (PyMC) - Same as original
# ===============================
def fit_l3_bayesian_regression_asymmetric(
    data, features_dict, config,
    interaction_pos=None, interaction_neg=None, interaction_rhoT=None
):
    """
    Fit Bayesian regression model with asymmetric cross-series interactions.
    Models how one series influences another through structural changes.

    Args:
        data: Target time series
        features_dict: Dictionary of Lambda³ features for target series
        config: Model configuration
        interaction_pos: Positive jumps from influencing series
        interaction_neg: Negative jumps from influencing series
        interaction_rhoT: Tension scalar from influencing series

    Returns:
        trace: PyMC InferenceData object with posterior samples
    """
    with pm.Model() as model:
        # Prior distributions for base parameters
        beta_0 = pm.Normal('beta_0', mu=0, sigma=2)  # Intercept
        beta_time = pm.Normal('beta_time', mu=0, sigma=1)  # Time trend
        beta_dLC_pos = pm.Normal('beta_dLC_pos', mu=0, sigma=5)  # Own positive jumps
        beta_dLC_neg = pm.Normal('beta_dLC_neg', mu=0, sigma=5)  # Own negative jumps
        beta_rhoT = pm.Normal('beta_rhoT', mu=0, sigma=3)  # Own tension

        # Base linear model
        mu = (
            beta_0
            + beta_time * features_dict['time_trend']
            + beta_dLC_pos * features_dict['delta_LambdaC_pos']
            + beta_dLC_neg * features_dict['delta_LambdaC_neg']
            + beta_rhoT * features_dict['rho_T']
        )

        # Add asymmetric cross-series interactions if provided
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
            return_inferencedata=True
        )
    return trace

def fit_l3_dynamic_bayesian(data, features_dict, config,
                           change_points=None,  # Candidate structural change points
                           window_size=50):
    """
    Fit dynamic Bayesian model with time-varying parameters.
    Allows for regime changes and structural breaks.

    Args:
        data: Time series data
        features_dict: Lambda³ features
        config: Model configuration
        change_points: List of potential change point locations
        window_size: Window for local parameter estimation

    Returns:
        trace: Posterior samples
    """
    n = len(data)
    time_idx = np.arange(n)

    with pm.Model() as model:
        # Time-varying parameter using Gaussian Random Walk
        beta_time_series = pm.GaussianRandomWalk(
            'beta_time_series',
            mu=0,
            sigma=0.1,
            shape=n
        )

        # Structural change jumps at specified points
        jump_total = 0
        if change_points:
            for i, cp in enumerate(change_points):
                jump = pm.Normal(f'jump_{i}', mu=0, sigma=5)
                jump_total += jump * (time_idx >= cp)
        else:
            jump_total = 0

        # Dynamic model specification
        mu = (
            beta_time_series
            + features_dict['delta_LambdaC_pos']
            + features_dict['delta_LambdaC_neg']
            + features_dict['rho_T']
            + jump_total
        )

        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=data)

        trace = pm.sample(draws=config.draws, tune=config.tune, target_accept=config.target_accept, return_inferencedata=True)

    return trace

# ===============================
# Weather-Specific Regime Detection
# ===============================
class Lambda3WeatherRegimeDetector:
    """
    Detect weather regimes using Lambda³ features.
    Clusters structural states based on meteorological characteristics.
    """
    def __init__(self, n_regimes=4, method='kmeans'):
        self.n_regimes = n_regimes
        self.method = method
        self.regime_labels = None
        self.regime_features = None

    def fit(self, features_dict):
        """
        Estimate weather regimes using clustering on Lambda³ features.

        Args:
            features_dict: Dictionary containing jump and tension features

        Returns:
            regime_labels: Array of regime assignments
        """
        # Stack features for clustering
        X = np.column_stack([
            features_dict['delta_LambdaC_pos'],
            features_dict['delta_LambdaC_neg'],
            features_dict['rho_T']
        ])

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_regimes, random_state=42)
        labels = km.fit_predict(X)
        self.regime_labels = labels

        # Calculate regime statistics
        self.regime_features = {
            r: {
                'frequency': np.mean(labels == r),
                'mean_rhoT': np.mean(X[labels == r, 2]) if np.sum(labels == r) > 0 else 0,
                'mean_pos_jumps': np.mean(X[labels == r, 0]) if np.sum(labels == r) > 0 else 0,
                'mean_neg_jumps': np.mean(X[labels == r, 1]) if np.sum(labels == r) > 0 else 0
            }
            for r in range(self.n_regimes)
        }
        return labels

    def label_weather_regimes(self):
        """
        Assign descriptive weather labels to each regime.
        """
        labels = {}
        for r in range(self.n_regimes):
            stats = self.regime_features[r]
            # Assign weather-specific labels based on characteristics
            if stats['mean_rhoT'] > 0.5:
                if stats['mean_pos_jumps'] > stats['mean_neg_jumps']:
                    labels[r] = f"Unstable-Rising"
                else:
                    labels[r] = f"Unstable-Falling"
            else:
                if stats['mean_pos_jumps'] > stats['mean_neg_jumps']:
                    labels[r] = f"Stable-Warming"
                else:
                    labels[r] = f"Stable-Cooling"
        return labels

# ===============================
# Enhanced Weather Regime Detection with Adaptive Methods
# ===============================
class AdaptiveLambda3WeatherRegimeDetector(Lambda3WeatherRegimeDetector):
    """
    Enhanced regime detector with better initialization and adaptive clustering.
    Handles imbalanced seasonal data more effectively.
    """

    def __init__(self, n_regimes=4, method='kmeans', min_regime_size=20):
        super().__init__(n_regimes, method)
        self.min_regime_size = min_regime_size

    def fit(self, features_dict, temporal_hint=None):
        """
        Enhanced fit method with temporal hints and better initialization.

        Args:
            features_dict: Dictionary containing jump and tension features
            temporal_hint: Optional array indicating temporal progression (e.g., day of year)

        Returns:
            regime_labels: Array of regime assignments
        """
        # Stack features for clustering
        X = np.column_stack([
            features_dict['delta_LambdaC_pos'],
            features_dict['delta_LambdaC_neg'],
            features_dict['rho_T']
        ])

        # Normalize features for better clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self.method == 'kmeans' and temporal_hint is not None:
            # Use temporal information for better initialization
            n_points = len(X)
            init_centers = []

            # Divide data into temporal segments
            for i in range(self.n_regimes):
                start_idx = int(i * n_points / self.n_regimes)
                end_idx = int((i + 1) * n_points / self.n_regimes)
                segment_mean = np.mean(X_scaled[start_idx:end_idx], axis=0)
                init_centers.append(segment_mean)

            init_centers = np.array(init_centers)

            # Use k-means++ with temporal initialization
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.n_regimes, init=init_centers, n_init=1, random_state=42)
            labels = km.fit_predict(X_scaled)

        elif self.method == 'gmm':
            # Use Gaussian Mixture Model for better handling of overlapping regimes
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=self.n_regimes,
                                  covariance_type='full',
                                  n_init=10,
                                  random_state=42)
            labels = gmm.fit_predict(X_scaled)

        else:
            # Default k-means with better initialization
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.n_regimes, init='k-means++', n_init=20, random_state=42)
            labels = km.fit_predict(X_scaled)

        # Post-processing: merge small regimes
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if count < self.min_regime_size:
                # Find nearest large regime
                mask = labels == label
                if np.any(mask):
                    # Reassign to most common neighboring regime
                    for i in np.where(mask)[0]:
                        # Look at neighbors
                        window = 10
                        start = max(0, i - window)
                        end = min(len(labels), i + window + 1)
                        neighbor_labels = labels[start:end]
                        neighbor_labels = neighbor_labels[neighbor_labels != label]
                        if len(neighbor_labels) > 0:
                            most_common = np.bincount(neighbor_labels).argmax()
                            labels[i] = most_common

        self.regime_labels = labels

        # Calculate enhanced regime statistics
        self.regime_features = {}
        for r in range(self.n_regimes):
            mask = (labels == r)
            n_points = np.sum(mask)

            if n_points > 0:
                self.regime_features[r] = {
                    'frequency': n_points / len(labels),
                    'mean_rhoT': np.mean(X[mask, 2]),
                    'std_rhoT': np.std(X[mask, 2]),
                    'mean_pos_jumps': np.mean(X[mask, 0]),
                    'mean_neg_jumps': np.mean(X[mask, 1]),
                    'temporal_span': self._get_temporal_span(mask) if temporal_hint is not None else None
                }
            else:
                self.regime_features[r] = {
                    'frequency': 0,
                    'mean_rhoT': 0,
                    'std_rhoT': 0,
                    'mean_pos_jumps': 0,
                    'mean_neg_jumps': 0,
                    'temporal_span': None
                }

        return labels

    def _get_temporal_span(self, mask):
        """Get the temporal span of a regime."""
        indices = np.where(mask)[0]
        if len(indices) > 0:
            return (indices[0], indices[-1])
        return None

    def detect_regime_transitions(self, smooth_window=5):
        """
        Detect transition points between regimes.

        Args:
            smooth_window: Window size for smoothing regime assignments

        Returns:
            transitions: List of (index, from_regime, to_regime) tuples
        """
        if self.regime_labels is None:
            raise ValueError("Must fit the detector first")

        # Smooth labels to reduce noise
        from scipy.ndimage import median_filter
        smoothed_labels = median_filter(self.regime_labels, size=smooth_window)

        # Find transitions
        transitions = []
        for i in range(1, len(smoothed_labels)):
            if smoothed_labels[i] != smoothed_labels[i-1]:
                transitions.append((i, smoothed_labels[i-1], smoothed_labels[i]))

        return transitions


# ===============================
# Multi-Scale Feature Extraction (Same as original)
# ===============================
class Lambda3MultiScaleAnalyzer:
    """
    Analyze Lambda³ features across multiple time scales.
    Detects scale-dependent structural changes.
    """
    def __init__(self, scales=[5, 10, 20, 50]):
        self.scales = scales
        self.scale_features = {}

    def extract_features(self, data):
        """
        Extract features at each time scale.

        Args:
            data: Input time series

        Returns:
            scale_features: Dictionary of features by scale
        """
        self.scale_features = {}
        for w in self.scales:
            # Rolling standard deviation at scale w
            rolling_std = np.array([np.std(data[max(0, i-w):i+1]) for i in range(len(data))])

            # Jump detection at this scale
            diff = np.diff(data, prepend=data[0])
            jumps = np.abs(diff) > (np.percentile(np.abs(diff), 97))

            self.scale_features[w] = {
                'rolling_std': rolling_std,
                'jumps': jumps
            }
        return self.scale_features

    def detect_scale_breaks(self, threshold=1.5):
        """
        Detect scale breaks - sudden increases in volatility.

        Args:
            threshold: Number of standard deviations for break detection

        Returns:
            breaks: List of (scale, time_indices) tuples
        """
        breaks = []
        for w, feats in self.scale_features.items():
            std = feats['rolling_std']
            # Detect peaks that are far from average
            mean, std_dev = np.mean(std), np.std(std)
            peaks = np.where(std > mean + threshold * std_dev)[0]
            if len(peaks) > 0:
                breaks.append((w, peaks.tolist()))
        return breaks

# ===============================
# Weather-Specific Conditional Synchronization
# ===============================
def lambda3_weather_conditional_sync(series_a, series_b, condition_series, condition_threshold):
    """
    Calculate conditional synchronization rate for weather data.
    Only considers periods where weather condition is met.

    Args:
        series_a: First weather event series
        series_b: Second weather event series
        condition_series: Conditioning variable (e.g., temperature, pressure)
        condition_threshold: Threshold for condition

    Returns:
        Conditional synchronization rate
    """
    mask = condition_series > condition_threshold
    # Calculate sync only for specific weather conditions
    sync = np.mean(series_a[mask] * series_b[mask]) if np.sum(mask) > 0 else 0.0
    return sync

# ===============================
# Integrated Advanced Weather Analysis
# ===============================
def lambda3_advanced_weather_analysis(data, features_dict, weather_type='Temperature'):
    """
    Comprehensive Lambda³ analysis for weather data.

    Args:
        data: Weather time series data
        features_dict: Pre-computed Lambda³ features
        weather_type: Type of weather parameter being analyzed

    Returns:
        Dictionary with analysis results
    """
    # 1. Weather Regime Detection
    regime_detector = Lambda3WeatherRegimeDetector(n_regimes=4)
    regimes = regime_detector.fit(features_dict)
    regime_labels = regime_detector.label_weather_regimes()

    print(f"\n{weather_type} Weather Regime Detection:")
    for regime, label in regime_labels.items():
        stats = regime_detector.regime_features[regime]
        print(f"  {label}: {stats['frequency']:.1%} (Mean ρT: {stats['mean_rhoT']:.2f})")

    # 2. Multi-scale Analysis
    ms_analyzer = Lambda3MultiScaleAnalyzer(scales=[5, 10, 20, 50])
    ms_features = ms_analyzer.extract_features(data)
    scale_breaks = ms_analyzer.detect_scale_breaks()

    print(f"\nScale Break Locations in {weather_type}: {scale_breaks}")

    # 3. Conditional Synchronization
    # Example: sync only during high-tension (unstable) weather periods
    if 'rho_T' in features_dict:
        sync_cond = lambda3_weather_conditional_sync(
            series_a=features_dict['delta_LambdaC_pos'],
            series_b=features_dict['delta_LambdaC_neg'],
            condition_series=features_dict['rho_T'],
            condition_threshold=np.median(features_dict['rho_T'])
        )
        print(f"\nConditional Sync Rate (unstable weather): {sync_cond:.3f}")

    return {
        'regimes': regimes,
        'regime_labels': regime_labels,
        'multi_scale_features': ms_features,
        'scale_breaks': scale_breaks
    }

# ===============================
# Lambda³ Extended Analysis (Same core with weather context)
# ===============================
class Lambda3BayesianExtended:
    """
    Extended Lambda³ analysis with event memory and causality detection.
    Tracks structural evolution and causal relationships.
    """

    def __init__(self, config: L3Config, series_names: Optional[List[str]] = None):
        self.config = config
        self.series_names = series_names or ['A']
        self.event_memory = []  # Store event history
        self.structure_evolution = []  # Track structural changes

    def update_event_memory(self, events_dict: Dict[str, Dict[str, int]]):
        """
        Update event memory with new structural changes.

        Args:
            events_dict: Dictionary of events by series
        """
        if len(self.event_memory) == 0:
            self.series_names = list(events_dict.keys())
        self.event_memory.append(events_dict)

    def detect_causality_chain(self, series: str = 'A') -> Optional[float]:
        """
        Detect causality chains: positive jump followed by negative.

        Args:
            series: Series name to analyze

        Returns:
            Probability of causality chain
        """
        if len(self.event_memory) < 2:
            return None

        count_pairs = 0
        count_pos = 0

        for i in range(len(self.event_memory) - 1):
            if self.event_memory[i][series]['pos']:
                count_pos += 1
                if self.event_memory[i + 1][series]['neg']:
                    count_pairs += 1

        return count_pairs / max(count_pos, 1)

    def detect_time_dependent_causality(self, series: str = 'A', lag_window: int = LAG_WINDOW_DEFAULT) -> Dict[int, float]:
        """
        Calculate time-dependent causality probabilities at different lags.

        Args:
            series: Series to analyze
            lag_window: Maximum lag to consider

        Returns:
            Dictionary of causality probabilities by lag
        """
        causality_by_lag = {}

        for lag in range(1, lag_window + 1):
            count_pairs, count_pos = 0, 0

            for i in range(len(self.event_memory) - lag):
                if self.event_memory[i][series]['pos']:
                    count_pos += 1
                    if self.event_memory[i + lag][series]['neg']:
                        count_pairs += 1

            causality_by_lag[lag] = count_pairs / max(count_pos, 1)

        return causality_by_lag

    def detect_cross_causality(self, from_series: str, to_series: str, lag: int = 1) -> float:
        """
        Detect cross-causality between different series.

        Args:
            from_series: Source series
            to_series: Target series
            lag: Time lag

        Returns:
            Cross-causality probability
        """
        count_pairs, count_from = 0, 0

        for i in range(len(self.event_memory) - lag):
            if self.event_memory[i][from_series]['pos']:
                count_from += 1
                if self.event_memory[i + lag][to_series]['pos']:
                    count_pairs += 1

        return count_pairs / max(count_from, 1)

# ===============================
# Synchronization Analysis (Same as original)
# ===============================
def calculate_sync_profile(series_a: np.ndarray, series_b: np.ndarray,
                          lag_window: int = LAG_WINDOW_DEFAULT) -> Tuple[Dict[int, float], float, int]:
    """
    Calculate synchronization profile using JIT-compiled function.

    Args:
        series_a: First binary event series
        series_b: Second binary event series
        lag_window: Maximum lag to consider

    Returns:
        sync_profile: Dictionary of sync rates by lag
        max_sync: Maximum synchronization rate
        optimal_lag: Lag with maximum synchronization
    """
    lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_jit(
        series_a.astype(np.float64),
        series_b.astype(np.float64),
        lag_window
    )

    # Convert to dictionary for easier access
    sync_profile = {int(lag): float(sync) for lag, sync in zip(lags, sync_values)}

    return sync_profile, float(max_sync), int(optimal_lag)

def calculate_sync_rate(series_a_events, series_b_events, lag_window=10):
    """
    Calculate synchronization rate σₛ between two event series.

    Args:
        series_a_events: Binary event series A
        series_b_events: Binary event series B
        lag_window: Maximum lag to consider

    Returns:
        max_sync: Maximum synchronization rate
        optimal_lag: Lag achieving maximum sync
    """
    max_sync, optimal_lag = 0, 0
    for lag in range(-lag_window, lag_window+1):
        if lag < 0:
            sync = np.mean(series_a_events[-lag:] * series_b_events[:lag])
        elif lag > 0:
            sync = np.mean(series_a_events[:-lag] * series_b_events[lag:])
        else:
            sync = np.mean(series_a_events * series_b_events)

        if sync > max_sync:
            max_sync, optimal_lag = sync, lag

    return max_sync, optimal_lag

def calculate_dynamic_sync(series_a_events, series_b_events, window=20, lag_window=10):
    """
    Calculate time-varying synchronization rate.

    Args:
        series_a_events: Binary event series A
        series_b_events: Binary event series B
        window: Sliding window size
        lag_window: Maximum lag within each window

    Returns:
        time_points: Time indices
        sync_rates: Synchronization rates over time
        optimal_lags: Optimal lag at each time
    """
    T = len(series_a_events)
    sync_rates, optimal_lags = [], []

    for t in range(T - window + 1):
        sync, lag = calculate_sync_rate(
            series_a_events[t:t+window],
            series_b_events[t:t+window],
            lag_window
        )
        sync_rates.append(sync)
        optimal_lags.append(lag)

    time_points = np.arange(window//2, T - window//2 + 1)
    return time_points, sync_rates, optimal_lags

def sync_matrix(event_series_dict: Dict[str, np.ndarray], lag_window: int = LAG_WINDOW_DEFAULT) -> Tuple[np.ndarray, List[str]]:
    """
    Create synchronization rate matrix for all series pairs.

    Args:
        event_series_dict: Dictionary of event series
        lag_window: Maximum lag to consider

    Returns:
        mat: Synchronization matrix
        series_names: List of series names
    """
    series_names = list(event_series_dict.keys())
    n = len(series_names)
    mat = np.zeros((n, n))

    for i, a in enumerate(series_names):
        for j, b in enumerate(series_names):
            if i == j:
                mat[i, j] = 1.0  # Self-sync is perfect
                continue

            # Ensure float64 for JIT function
            series_a = event_series_dict[a].astype(np.float64)
            series_b = event_series_dict[b].astype(np.float64)

            _, _, max_sync, _ = calculate_sync_profile_jit(series_a, series_b, lag_window)
            mat[i, j] = max_sync

    return mat, series_names

def cluster_series_by_sync(event_series_dict, lag_window=10, n_clusters=2):
    """
    Cluster time series based on synchronization patterns.

    Args:
        event_series_dict: Dictionary of event series
        lag_window: Maximum lag for sync calculation
        n_clusters: Number of clusters

    Returns:
        clusters: Cluster assignments
        mat: Synchronization matrix
    """
    mat, names = sync_matrix(event_series_dict, lag_window)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(1 - mat)  # Use 1-sync as distance
    clusters = {name: label for name, label in zip(names, labels)}
    return clusters, mat

def build_sync_network(event_series_dict: Dict[str, np.ndarray],
                      lag_window: int = LAG_WINDOW_DEFAULT,
                      sync_threshold: float = SYNC_THRESHOLD_DEFAULT) -> nx.DiGraph:
    """
    Build directed synchronization network from event series.

    Args:
        event_series_dict: Dictionary of event series
        lag_window: Maximum lag to consider
        sync_threshold: Minimum sync rate for edge creation

    Returns:
        G: Directed graph with sync relationships
    """
    series_names = list(event_series_dict.keys())
    G = nx.DiGraph()

    # Add nodes
    for series in series_names:
        G.add_node(series)

    # Debug: print sync calculations
    print(f"\nBuilding sync network with threshold={sync_threshold}")

    # Add edges based on synchronization
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
# Visualization Functions (Adapted for Weather)
# ===============================
def plot_posterior(trace, var_names: Optional[List[str]] = None, hdi_prob: float = 0.94):
    """
    Visualize posterior distributions from Bayesian analysis.

    Args:
        trace: PyMC InferenceData object
        var_names: Variables to plot
        hdi_prob: Highest density interval probability
    """
    if var_names is None:
        var_names = list(trace.posterior.data_vars)
    az.plot_posterior(trace, var_names=var_names, hdi_prob=hdi_prob)
    plt.tight_layout()
    plt.show()

def plot_l3_weather_prediction(
    data_dict: Dict[str, np.ndarray],
    mu_pred_dict: Dict[str, np.ndarray],
    jump_pos_dict: Dict[str, np.ndarray],
    jump_neg_dict: Dict[str, np.ndarray],
    local_jump_dict: Optional[Dict[str, np.ndarray]] = None,
    series_names: Optional[List[str]] = None,
    titles: Optional[List[str]] = None,
    weather_units: Optional[Dict[str, str]] = None
):
    """
    Plot weather data, model predictions, and detected structural events.

    Args:
        data_dict: Original weather time series data
        mu_pred_dict: Model predictions
        jump_pos_dict: Positive jump indicators
        jump_neg_dict: Negative jump indicators
        local_jump_dict: Local jump indicators
        series_names: Names of series to plot
        titles: Custom titles for subplots
        weather_units: Units for each weather parameter
    """
    if series_names is None:
        series_names = list(data_dict.keys())

    if weather_units is None:
        weather_units = {
            'Temperature': '°C',
            'Humidity': '%',
            'Pressure': 'hPa (normalized)',
            'WindSpeed': 'm/s',
            'Precipitation': 'mm',
            'DewPoint': '°C'
        }

    n_series = len(series_names)
    fig, axes = plt.subplots(n_series, 1, figsize=(15, 5 * n_series), sharex=True)

    if n_series == 1:
        axes = [axes]

    weather_colors = {
        'Temperature': 'darkred',
        'Humidity': 'darkblue',
        'Pressure': 'darkgreen',
        'WindSpeed': 'darkorange',
        'Precipitation': 'skyblue',
        'DewPoint': 'purple'
    }

    for i, series in enumerate(series_names):
        ax = axes[i]
        data = data_dict[series]
        mu_pred = mu_pred_dict[series]
        jump_pos = jump_pos_dict[series]
        jump_neg = jump_neg_dict[series]
        local_jump = local_jump_dict[series] if local_jump_dict else None

        # Get color for this weather parameter
        base_color = weather_colors.get(series, 'gray')

        # Plot data and prediction
        ax.plot(data, 'o', color=base_color, markersize=3, alpha=0.4, label='Observed')
        ax.plot(mu_pred, color=base_color, lw=2, alpha=0.8, label='Lambda³ Model')

        # Plot structural events
        jump_pos_idx = np.where(jump_pos > 0)[0]
        if len(jump_pos_idx):
            ax.scatter(jump_pos_idx, data[jump_pos_idx],
                      color='red', s=100, marker='^',
                      label='Positive ∆ΛC', zorder=5)
            for idx in jump_pos_idx:
                ax.axvline(x=idx, color='red', linestyle='--', alpha=0.3)

        jump_neg_idx = np.where(jump_neg > 0)[0]
        if len(jump_neg_idx):
            ax.scatter(jump_neg_idx, data[jump_neg_idx],
                      color='blue', s=100, marker='v',
                      label='Negative ∆ΛC', zorder=5)
            for idx in jump_neg_idx:
                ax.axvline(x=idx, color='blue', linestyle='-.', alpha=0.3)

        if local_jump is not None:
            local_jump_idx = np.where(local_jump > 0)[0]
            if len(local_jump_idx):
                ax.scatter(local_jump_idx, data[local_jump_idx],
                          color='orange', s=70, marker='o', alpha=0.7,
                          label='Local Jump', zorder=4)

        # Formatting
        plot_title = titles[i] if titles and i < len(titles) else f"{series}: Lambda³ Structural Analysis"
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel('Structural Snapshot Index', fontsize=12)

        unit = weather_units.get(series, '')
        ax.set_ylabel(f'{series} ({unit})', fontsize=12)

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Weather Parameters: Lambda³ Structural Evolution', fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_sync_profile(sync_profile: Dict[int, float], title: str = "Sync Profile (σₛ vs Lag)"):
    """
    Plot synchronization profile showing sync rate vs lag.

    Args:
        sync_profile: Dictionary of sync rates by lag
        title: Plot title
    """
    lags, syncs = zip(*sorted(sync_profile.items()))
    plt.figure(figsize=(8, 4))
    plt.plot(lags, syncs, marker='o')
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Sync Rate σₛ')
    plt.grid(alpha=0.5)
    plt.show()

def plot_dynamic_sync(time_points, sync_rates, optimal_lags):
    """
    Plot time-varying synchronization rate and optimal lag.

    Args:
        time_points: Time indices
        sync_rates: Synchronization rates over time
        optimal_lags: Optimal lags over time
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(time_points, sync_rates, label='σₛ Sync Rate', color='royalblue')
    ax1.set_ylabel('σₛ Sync Rate')
    ax1.set_xlabel('Time Step')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(time_points, optimal_lags, label='Optimal Lag', color='darkorange', linestyle='--')
    ax2.set_ylabel('Optimal Lag')
    ax2.legend(loc='upper right')

    plt.title("Dynamic Synchronization (σₛ) and Optimal Lag")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_multi_causality_lags(
    causality_dicts,
    labels=None,
    colors=None,
    title="Lagged Causality Profiles",
    xlabel="Lag (steps)",
    ylabel="Causality Probability",
    figsize=(9,5),
    alpha=0.7
):
    """
    Plot multiple lagged causality profiles together.

    Args:
        causality_dicts: List of causality dictionaries
        labels: Labels for each profile
        colors: Colors for each profile
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        alpha: Line transparency
    """
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(causality_dicts))]
    if colors is None:
        base_colors = ['royalblue', 'darkorange', 'forestgreen', 'crimson']
        colors = base_colors[:len(causality_dicts)]

    plt.figure(figsize=figsize)

    for i, (causality_by_lag, label, color) in enumerate(zip(causality_dicts, labels, colors)):
        lags, probs = zip(*sorted(causality_by_lag.items()))
        plt.plot(lags, probs, marker='o', label=label, color=color, alpha=alpha, lw=2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_sync_network(G: nx.DiGraph):
    """
    Plot synchronization network graph.

    Args:
        G: NetworkX directed graph
    """
    pos = nx.spring_layout(G)
    edge_labels = {
        (u, v): f"σₛ:{d['weight']:.2f},lag:{d['lag']}"
        for u, v, d in G.edges(data=True)
    }

    nx.draw(G, pos, with_labels=True, node_color='skyblue',
            node_size=1500, font_size=10, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Weather Parameter Synchronization (σₛ) Network")
    plt.show()

# ===============================
# Updated Main Analysis Function with Regime Detection Option
# ===============================
def main_weather_analysis(
    csv_path: str = "tokyo_weather_days.csv",
    csv_paths: List[str] = None,
    time_column: str = "date",
    value_columns: List[str] = None,
    series_names: List[str] = None,
    config: L3Config = None,
    analyze_all_pairs: bool = True,
    max_pairs: int = None,
    use_regime_detection: bool = True,  # New parameter
    n_regimes: int = 3,  # New parameter
    regime_names: Optional[List[str]] = None  # New parameter
):
    """
    Main pipeline for analyzing weather CSV data using Lambda³ framework.

    This treats weather as structural tensor transactions in semantic space,
    not as time-dependent phenomena.

    Args:
        csv_path: Path to weather CSV file
        csv_paths: List of paths to CSVs (one series each)
        time_column: Time column name for sorting
        value_columns: Columns to analyze (None = all numeric)
        series_names: Names for the series
        config: L3Config instance (uses default if None)
        analyze_all_pairs: Whether to analyze all pairs (default: True)
        max_pairs: Maximum number of pairs to analyze (None = all)
        use_regime_detection: Whether to use structural regime detection
        n_regimes: Number of structural regimes to detect
        regime_names: Optional names for regimes (e.g., ['Spring', 'Transition', 'Rainy'])

    Returns:
        features_dict: Extracted Lambda³ features
        sync_mat: Synchronization matrix
        regime_analysis_results: Results from regime analysis (if enabled)
    """
    if config is None:
        config = L3Config()

    # Load weather data
    if csv_path:
        print(f"Loading weather data from: {csv_path}")
        series_dict = load_csv_data(csv_path, time_column, value_columns)
    elif csv_paths:
        print(f"Loading data from {len(csv_paths)} files")
        series_dict = load_multiple_csv_files(csv_paths, series_names)
    else:
        raise ValueError("Must provide either csv_path or csv_paths")

    # Rename columns for weather context
    weather_mapping = {
        'temperature_2m': 'Temperature',
        'relative_humidity_2m': 'Humidity',
        'dew_point_2m': 'DewPoint',
        'precipitation': 'Precipitation',
        'wind_speed_10m': 'WindSpeed',
        'surface_pressure': 'Pressure'
    }

    # Apply renaming if original column names exist
    renamed_dict = {}
    for col, data in series_dict.items():
        new_name = weather_mapping.get(col, col)
        renamed_dict[new_name] = data
        # Normalize pressure if needed
        if new_name == 'Pressure':
            renamed_dict[new_name] = (data - 1000) / 10

    series_dict = renamed_dict

    # Validate series
    series_dict = validate_series_lengths(series_dict)

    if len(series_dict) < 2:
        print("Warning: Need at least 2 series for cross-analysis")
        return

    # Update config with actual data length
    data_length = len(next(iter(series_dict.values())))
    config.T = data_length
    print(f"\nAnalyzing {len(series_dict)} weather parameters with {data_length} structural snapshots each")

    # Extract features for all series
    features_dict = {}
    for name, data in series_dict.items():
        print(f"\nExtracting Lambda³ features for: {name}")
        feats = calc_lambda3_features_v2(data, config)
        features_dict[name] = {
            'data': data,
            'delta_LambdaC_pos': feats[0],
            'delta_LambdaC_neg': feats[1],
            'rho_T': feats[2],
            'time_trend': feats[3],
            'local_jump': feats[4]
        }

        # Report weather-specific statistics
        n_pos = np.sum(feats[0])
        n_neg = np.sum(feats[1])
        mean_tension = np.mean(feats[2])
        print(f"  Positive ∆ΛC: {n_pos}, Negative ∆ΛC: {n_neg}")
        print(f"  Mean tension ρT: {mean_tension:.3f}")

    # Analyze pairs
    series_list = list(series_dict.keys())
    n_series = len(series_list)

    # Store regime analysis results if enabled
    regime_analysis_results = {}

    if analyze_all_pairs and n_series > 2:
        # Analyze all unique pairs
        from itertools import combinations
        pairs = list(combinations(series_list, 2))

        if max_pairs and len(pairs) > max_pairs:
            print(f"\nNote: Limiting analysis to first {max_pairs} pairs out of {len(pairs)} total")
            pairs = pairs[:max_pairs]

        print(f"\n{'='*60}")
        print(f"ANALYZING ALL {len(pairs)} WEATHER PARAMETER PAIRS")
        print(f"{'='*60}")

        # Store interaction effects for summary
        interaction_effects = {}

        for i, (name_a, name_b) in enumerate(pairs, 1):
            print(f"\n[{i}/{len(pairs)}] Analyzing: {name_a} ↔ {name_b}")

            try:
                if use_regime_detection:
                    # Use enhanced analysis with regime detection
                    regime_results = analyze_weather_pair_with_regimes(
                        name_a, name_b, features_dict, config,
                        n_regimes=n_regimes,
                        regime_names=regime_names,
                        show_all_plots=(i <= 3),  # Show detailed plots only for first 3 pairs
                        analyze_regimes_separately=True
                    )

                    # Store regime analysis results
                    regime_analysis_results[(name_a, name_b)] = regime_results

                    # Extract overall interaction effects
                    beta_ab = regime_results['overall_results']['beta_b_on_a']
                    beta_ba = regime_results['overall_results']['beta_a_on_b']
                else:
                    # Use standard analysis without regime detection
                    beta_ab, beta_ba = analyze_weather_pair(
                        name_a, name_b, features_dict, config,
                        show_all_plots=(i <= 3)
                    )

                # Store interaction effects
                interaction_effects[(name_a, name_b)] = beta_ab
                interaction_effects[(name_b, name_a)] = beta_ba

            except Exception as e:
                print(f"Error analyzing pair {name_a} ↔ {name_b}: {e}")
                continue

        # Summary of all interaction effects
        plot_weather_interaction_summary(interaction_effects, series_list)

        # If regime detection was used, create regime summary
        if use_regime_detection and regime_analysis_results:
            create_regime_summary(regime_analysis_results, series_list)

    else:
        # Analyze just the first two series
        if len(series_list) >= 2:
            if use_regime_detection:
                regime_results = analyze_weather_pair_with_regimes(
                    series_list[0], series_list[1],
                    features_dict, config,
                    n_regimes=n_regimes,
                    regime_names=regime_names,
                    show_all_plots=True,
                    analyze_regimes_separately=True
                )
                regime_analysis_results[(series_list[0], series_list[1])] = regime_results
            else:
                analyze_weather_pair(
                    series_list[0], series_list[1],
                    features_dict, config,
                    show_all_plots=True
                )

    # Multi-series synchronization analysis
    print("\n" + "="*50)
    print("WEATHER PARAMETER SYNCHRONIZATION ANALYSIS")
    print("="*50)

    # Build event series dictionary
    event_series_dict = {
        name: features_dict[name]['delta_LambdaC_pos'].astype(np.float64)
        for name in series_dict.keys()
    }

    # Synchronization matrix
    sync_mat, names = sync_matrix(event_series_dict, lag_window=10)

    # Plot sync matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sync_mat, annot=True, fmt='.3f',
                xticklabels=names,
                yticklabels=names,
                cmap="Blues", vmin=0, vmax=1,
                square=True, cbar_kws={'label': 'Sync Rate σₛ'})
    plt.title("Weather Parameter Synchronization Matrix (σₛ)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Build and plot sync network
    print("\n=== Building Weather Synchronization Network ===")

    # Find appropriate threshold
    non_diag_values = []
    n = len(names)
    for i in range(n):
        for j in range(n):
            if i != j:
                non_diag_values.append(sync_mat[i, j])

    if non_diag_values:
        threshold = np.percentile(non_diag_values, 25)  # Use 25th percentile
        print(f"Using threshold: {threshold:.4f}")

        G = build_sync_network(event_series_dict, lag_window=10, sync_threshold=threshold)
        if G.number_of_edges() > 0:
            plt.figure(figsize=(12, 10))
            plot_sync_network(G)

    # Clustering analysis
    if len(series_dict) > 2:
        print("\n=== Weather Parameter Clustering ===")
        n_clusters = min(3, len(series_dict) // 2)
        clusters, _ = cluster_series_by_sync(event_series_dict, lag_window=10, n_clusters=n_clusters)
        print(f"Weather Clusters: {clusters}")

        # Plot clustered series
        plot_weather_clusters(series_dict, clusters)

    # Create weather summary report
    create_weather_summary(series_list, sync_mat, features_dict)

    return features_dict, sync_mat, regime_analysis_results

# ===============================
# Weather-Specific Pairwise Analysis
# ===============================
def analyze_weather_pair(
    name_a: str, name_b: str,
    features_dict: Dict[str, Dict[str, np.ndarray]],
    config: L3Config,
    show_all_plots: bool = True
) -> Tuple[float, float]:
    """
    Detailed analysis of a pair of weather parameters including cross-interactions.

    Args:
        name_a: First weather parameter name
        name_b: Second weather parameter name
        features_dict: Pre-computed Lambda³ features
        config: Analysis configuration
        show_all_plots: Whether to show all plots

    Returns:
        Tuple of (beta_B_on_A, beta_A_on_B) interaction coefficients
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING WEATHER PAIR: {name_a} ↔ {name_b}")
    print(f"{'='*50}")

    # Get features
    feats_a = features_dict[name_a]
    feats_b = features_dict[name_b]

    # Fit models with asymmetric interaction
    print(f"\nFitting Bayesian model for {name_a} (with {name_b} interaction)...")
    trace_a = fit_l3_bayesian_regression_asymmetric(
        data=feats_a['data'],
        features_dict={
            'delta_LambdaC_pos': feats_a['delta_LambdaC_pos'],
            'delta_LambdaC_neg': feats_a['delta_LambdaC_neg'],
            'rho_T': feats_a['rho_T'],
            'time_trend': feats_a['time_trend']
        },
        config=config,
        interaction_pos=feats_b['delta_LambdaC_pos'],
        interaction_neg=feats_b['delta_LambdaC_neg'],
        interaction_rhoT=feats_b['rho_T']
    )

    print(f"\nFitting Bayesian model for {name_b} (with {name_a} interaction)...")
    trace_b = fit_l3_bayesian_regression_asymmetric(
        data=feats_b['data'],
        features_dict={
            'delta_LambdaC_pos': feats_b['delta_LambdaC_pos'],
            'delta_LambdaC_neg': feats_b['delta_LambdaC_neg'],
            'rho_T': feats_b['rho_T'],
            'time_trend': feats_b['time_trend']
        },
        config=config,
        interaction_pos=feats_a['delta_LambdaC_pos'],
        interaction_neg=feats_a['delta_LambdaC_neg'],
        interaction_rhoT=feats_a['rho_T']
    )

    # Get summaries with HDI
    summary_a = az.summary(trace_a, hdi_prob=config.hdi_prob)
    summary_b = az.summary(trace_b, hdi_prob=config.hdi_prob)

    # Extract interaction coefficients
    beta_b_on_a_pos = summary_a.loc['beta_interact_pos', 'mean'] if 'beta_interact_pos' in summary_a.index else 0
    beta_b_on_a_neg = summary_a.loc['beta_interact_neg', 'mean'] if 'beta_interact_neg' in summary_a.index else 0
    beta_a_on_b_pos = summary_b.loc['beta_interact_pos', 'mean'] if 'beta_interact_pos' in summary_b.index else 0
    beta_a_on_b_neg = summary_b.loc['beta_interact_neg', 'mean'] if 'beta_interact_neg' in summary_b.index else 0

    # Extract HDI values
    def get_hdi_bounds(summary, param_name, hdi_prob):
        """Extract HDI bounds from summary table."""
        if param_name in summary.index:
            # ArviZ summary provides HDI with column names like 'hdi_3%' and 'hdi_97%' for 94% HDI
            hdi_lower_col = f'hdi_{(1-hdi_prob)/2*100:.0f}%'
            hdi_upper_col = f'hdi_{(1+hdi_prob)/2*100:.0f}%'

            # Check for exact column names
            if hdi_lower_col in summary.columns and hdi_upper_col in summary.columns:
                return summary.loc[param_name, hdi_lower_col], summary.loc[param_name, hdi_upper_col]
            else:
                # Look for any HDI columns
                hdi_cols = [col for col in summary.columns if 'hdi' in col.lower()]
                if len(hdi_cols) >= 2:
                    return summary.loc[param_name, hdi_cols[0]], summary.loc[param_name, hdi_cols[1]]
        return None, None

    # Use positive interaction as primary metric
    beta_b_on_a = beta_b_on_a_pos
    beta_a_on_b = beta_a_on_b_pos

    print(f"\nAsymmetric Weather Interaction Effects:")
    print(f"  {name_b} → {name_a} (pos): β = {beta_b_on_a_pos:.3f}")
    print(f"  {name_b} → {name_a} (neg): β = {beta_b_on_a_neg:.3f}")
    print(f"  {name_a} → {name_b} (pos): β = {beta_a_on_b_pos:.3f}")
    print(f"  {name_a} → {name_b} (neg): β = {beta_a_on_b_neg:.3f}")

    # Print HDI information
    print(f"\nHDI ({config.hdi_prob*100:.0f}%) for Interaction Effects:")

    # HDI for B → A interactions
    hdi_low, hdi_high = get_hdi_bounds(summary_a, 'beta_interact_pos', config.hdi_prob)
    if hdi_low is not None:
        print(f"  {name_b} → {name_a} (pos): [{hdi_low:.3f}, {hdi_high:.3f}]")

    hdi_low, hdi_high = get_hdi_bounds(summary_a, 'beta_interact_neg', config.hdi_prob)
    if hdi_low is not None:
        print(f"  {name_b} → {name_a} (neg): [{hdi_low:.3f}, {hdi_high:.3f}]")

    # HDI for A → B interactions
    hdi_low, hdi_high = get_hdi_bounds(summary_b, 'beta_interact_pos', config.hdi_prob)
    if hdi_low is not None:
        print(f"  {name_a} → {name_b} (pos): [{hdi_low:.3f}, {hdi_high:.3f}]")

    hdi_low, hdi_high = get_hdi_bounds(summary_b, 'beta_interact_neg', config.hdi_prob)
    if hdi_low is not None:
        print(f"  {name_a} → {name_b} (neg): [{hdi_low:.3f}, {hdi_high:.3f}]")

    if show_all_plots:
        # Calculate predictions including interactions
        mu_pred_a = (
            summary_a.loc['beta_0', 'mean']
            + summary_a.loc['beta_time', 'mean'] * feats_a['time_trend']
            + summary_a.loc['beta_dLC_pos', 'mean'] * feats_a['delta_LambdaC_pos']
            + summary_a.loc['beta_dLC_neg', 'mean'] * feats_a['delta_LambdaC_neg']
            + summary_a.loc['beta_rhoT', 'mean'] * feats_a['rho_T']
            + beta_b_on_a_pos * feats_b['delta_LambdaC_pos']
            + beta_b_on_a_neg * feats_b['delta_LambdaC_neg']
        )

        mu_pred_b = (
            summary_b.loc['beta_0', 'mean']
            + summary_b.loc['beta_time', 'mean'] * feats_b['time_trend']
            + summary_b.loc['beta_dLC_pos', 'mean'] * feats_b['delta_LambdaC_pos']
            + summary_b.loc['beta_dLC_neg', 'mean'] * feats_b['delta_LambdaC_neg']
            + summary_b.loc['beta_rhoT', 'mean'] * feats_b['rho_T']
            + beta_a_on_b_pos * feats_a['delta_LambdaC_pos']
            + beta_a_on_b_neg * feats_a['delta_LambdaC_neg']
        )

        # Plot results
        plot_l3_weather_prediction(
            data_dict={name_a: feats_a['data'], name_b: feats_b['data']},
            mu_pred_dict={name_a: mu_pred_a, name_b: mu_pred_b},
            jump_pos_dict={name_a: feats_a['delta_LambdaC_pos'], name_b: feats_b['delta_LambdaC_pos']},
            jump_neg_dict={name_a: feats_a['delta_LambdaC_neg'], name_b: feats_b['delta_LambdaC_neg']},
            local_jump_dict={name_a: feats_a['local_jump'], name_b: feats_b['local_jump']},
            titles=[f'{name_a}: Structural Evolution', f'{name_b}: Structural Evolution']
        )

        # Plot posterior distributions
        print(f"\nPosterior for {name_a} (with {name_b} interaction):")
        plot_posterior(
            trace_a,
            var_names=['beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT',
                      'beta_interact_pos', 'beta_interact_neg', 'beta_interact_stress'],
            hdi_prob=config.hdi_prob
        )

        # Causality analysis
        lambda3_ext = Lambda3BayesianExtended(config, series_names=[name_a, name_b])

        # Build event memory
        for i in range(config.T):
            lambda3_ext.update_event_memory({
                name_a: {'pos': int(feats_a['delta_LambdaC_pos'][i]),
                         'neg': int(feats_a['delta_LambdaC_neg'][i])},
                name_b: {'pos': int(feats_b['delta_LambdaC_pos'][i]),
                         'neg': int(feats_b['delta_LambdaC_neg'][i])}
            })

        # Compute lagged causality profiles
        causality_by_lag_a = lambda3_ext.detect_time_dependent_causality(series=name_a, lag_window=10)
        causality_by_lag_b = lambda3_ext.detect_time_dependent_causality(series=name_b, lag_window=10)
        causality_ab = {lag: lambda3_ext.detect_cross_causality(name_a, name_b, lag=lag) for lag in range(1, 11)}
        causality_ba = {lag: lambda3_ext.detect_cross_causality(name_b, name_a, lag=lag) for lag in range(1, 11)}

        # Plot causality profiles
        plot_multi_causality_lags(
            [causality_by_lag_a, causality_by_lag_b, causality_ab, causality_ba],
            labels=[name_a, name_b, f'{name_a}→{name_b}', f'{name_b}→{name_a}'],
            title=f'Weather Causality Profiles: {name_a} ↔ {name_b}'
        )

        # Dynamic synchronization
        time_points, sync_rates, optimal_lags = calculate_dynamic_sync(
            feats_a['delta_LambdaC_pos'],
            feats_b['delta_LambdaC_pos'],
            window=20, lag_window=10
        )
        plot_dynamic_sync(time_points, sync_rates, optimal_lags)

    # Always calculate sync profile (even if not plotting)
    sync_profile, sync_rate, optimal_lag = calculate_sync_profile(
        feats_a['delta_LambdaC_pos'].astype(np.float64),
        feats_b['delta_LambdaC_pos'].astype(np.float64),
        lag_window=10
    )

    print(f"\nWeather Sync Rate σₛ ({name_a}↔{name_b}): {sync_rate:.3f}")
    print(f"Optimal Lag: {optimal_lag} steps")

    if show_all_plots:
        plot_sync_profile(sync_profile, title=f"Weather Sync Profile ({name_a}↔{name_b})")

    return beta_b_on_a, beta_a_on_b

# ===============================
# Weather-Specific Pairwise Analysis with Regime Detection
# ===============================
def analyze_weather_pair_with_regimes(
    name_a: str, name_b: str,
    features_dict: Dict[str, Dict[str, np.ndarray]],
    config: L3Config,
    n_regimes: int = 4,
    regime_names: Optional[List[str]] = None,
    show_all_plots: bool = True,
    analyze_regimes_separately: bool = True,
    use_adaptive_detection: bool = True,
    detection_method: str = 'auto',
    min_regime_size: int = 20
) -> Dict:
    """
    Enhanced analysis of weather parameter pairs with structural regime detection.
    Handles seasonal transitions by identifying distinct structural states.

    Args:
        name_a: First weather parameter name
        name_b: Second weather parameter name
        features_dict: Pre-computed Lambda³ features
        config: Analysis configuration
        n_regimes: Number of structural regimes to detect
        regime_names: Optional names for regimes (e.g., ['Spring', 'Transition', 'Rainy'])
        show_all_plots: Whether to show all plots
        analyze_regimes_separately: Whether to analyze each regime separately
        use_adaptive_detection: Whether to use adaptive regime detection
        detection_method: Method for regime detection ('auto', 'kmeans', 'gmm')
        min_regime_size: Minimum size for a valid regime

    Returns:
        Dictionary containing:
        - overall_results: Full dataset analysis results
        - regime_results: Per-regime analysis results
        - regime_labels: Detected regime assignments
        - regime_statistics: Statistics for each regime
        - detection_method_used: Actual method used for detection
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING WEATHER PAIR WITH REGIME DETECTION: {name_a} ↔ {name_b}")
    print(f"{'='*60}")

    # Get features
    feats_a = features_dict[name_a]
    feats_b = features_dict[name_b]

    # ===============================
    # Step 1: Structural Regime Detection
    # ===============================
    print(f"\n1. Detecting {n_regimes} structural regimes...")

    # Combine features for both parameters to detect common regimes
    combined_features = {
        'delta_LambdaC_pos': np.concatenate([feats_a['delta_LambdaC_pos'], feats_b['delta_LambdaC_pos']]),
        'delta_LambdaC_neg': np.concatenate([feats_a['delta_LambdaC_neg'], feats_b['delta_LambdaC_neg']]),
        'rho_T': np.concatenate([feats_a['rho_T'], feats_b['rho_T']])
    }

    # Create temporal hint for better initialization
    temporal_hint = None
    if use_adaptive_detection:
        temporal_hint = np.concatenate([feats_a['time_trend'], feats_b['time_trend']])

    # Choose detection method
    detection_method_used = detection_method

    if use_adaptive_detection and detection_method == 'auto':
        # Try multiple methods and choose best
        print("  Using adaptive detection with automatic method selection...")

        methods_to_try = ['gmm', 'kmeans']
        best_method = None
        best_score = -np.inf
        best_detector = None
        best_regimes = None

        for method in methods_to_try:
            print(f"    Trying {method}...")

            detector = AdaptiveLambda3WeatherRegimeDetector(
                n_regimes=n_regimes,
                method=method,
                min_regime_size=min_regime_size
            )

            combined_regimes = detector.fit(combined_features, temporal_hint=temporal_hint)
            regimes = combined_regimes[:len(feats_a['data'])]

            # Calculate quality score
            unique, counts = np.unique(regimes, return_counts=True)
            proportions = counts / len(regimes)
            entropy = -np.sum(proportions * np.log(proportions + 1e-10))
            imbalance_penalty = np.std(proportions) * 2
            score = entropy - imbalance_penalty

            print(f"      Score: {score:.3f}, Distribution: {dict(zip(unique, counts))}")

            if score > best_score:
                best_score = score
                best_method = method
                best_detector = detector
                best_regimes = regimes

        print(f"  Selected method: {best_method} (score: {best_score:.3f})")
        regime_detector = best_detector
        regimes = best_regimes
        detection_method_used = best_method

    elif use_adaptive_detection:
        # Use specified adaptive method
        print(f"  Using adaptive detection with {detection_method} method...")

        regime_detector = AdaptiveLambda3WeatherRegimeDetector(
            n_regimes=n_regimes,
            method=detection_method if detection_method != 'auto' else 'kmeans',
            min_regime_size=min_regime_size
        )
        combined_regimes = regime_detector.fit(combined_features, temporal_hint=temporal_hint)
        regimes = combined_regimes[:len(feats_a['data'])]

        # Show distribution
        unique, counts = np.unique(regimes, return_counts=True)
        print(f"  Regime distribution: {dict(zip(unique, counts))}")

    else:
        # Use standard detection
        print("  Using standard regime detection...")

        regime_detector = Lambda3WeatherRegimeDetector(n_regimes=n_regimes)
        combined_regimes = regime_detector.fit(combined_features)
        regimes = combined_regimes[:len(feats_a['data'])]

        # Show distribution
        unique, counts = np.unique(regimes, return_counts=True)
        print(f"  Regime distribution: {dict(zip(unique, counts))}")

    # Detect transitions if using adaptive detection
    transitions = []
    if use_adaptive_detection and hasattr(regime_detector, 'detect_regime_transitions'):
        transitions = regime_detector.detect_regime_transitions()
        if transitions:
            print(f"\n  Detected {len(transitions)} regime transitions:")
            for idx, from_r, to_r in transitions[:3]:  # Show first 3
                print(f"    Index {idx}: Regime {from_r} → Regime {to_r}")

    # Assign regime names if provided
    if regime_names is None:
        # Use automatic weather-specific labeling
        auto_labels = regime_detector.label_weather_regimes()
        regime_names = [auto_labels[i] for i in range(n_regimes)]

    # Calculate regime statistics
    regime_stats = {}
    for i in range(n_regimes):
        mask = (regimes == i)
        n_points = np.sum(mask)
        if n_points > 0:
            regime_stats[regime_names[i]] = {
                'count': n_points,
                'percentage': n_points / len(regimes) * 100,
                f'mean_{name_a}': np.mean(feats_a['data'][mask]),
                f'mean_{name_b}': np.mean(feats_b['data'][mask]),
                f'mean_rhoT_{name_a}': np.mean(feats_a['rho_T'][mask]),
                f'mean_rhoT_{name_b}': np.mean(feats_b['rho_T'][mask]),
                f'jumps_{name_a}': np.sum(feats_a['delta_LambdaC_pos'][mask] + feats_a['delta_LambdaC_neg'][mask]),
                f'jumps_{name_b}': np.sum(feats_b['delta_LambdaC_pos'][mask] + feats_b['delta_LambdaC_neg'][mask])
            }

    print("\nDetected Structural Regimes:")
    for regime_name, stats in regime_stats.items():
        print(f"\n{regime_name}:")
        print(f"  - Points: {stats['count']} ({stats['percentage']:.1f}%)")
        print(f"  - Mean ρT ({name_a}): {stats[f'mean_rhoT_{name_a}']:.3f}")
        print(f"  - Mean ρT ({name_b}): {stats[f'mean_rhoT_{name_b}']:.3f}")
        print(f"  - Total jumps: {name_a}={stats[f'jumps_{name_a}']}, {name_b}={stats[f'jumps_{name_b}']}")

    # ===============================
    # Step 2: Overall Analysis (Full Dataset)
    # ===============================
    print(f"\n2. Overall analysis across all regimes...")

    # Use the existing analyze_weather_pair function for overall analysis
    overall_beta_b_on_a, overall_beta_a_on_b = analyze_weather_pair(
        name_a, name_b, features_dict, config,
        show_all_plots=show_all_plots and not analyze_regimes_separately
    )

    overall_results = {
        'beta_b_on_a': overall_beta_b_on_a,
        'beta_a_on_b': overall_beta_a_on_b
    }

    # ===============================
    # Step 3: Regime-Specific Analysis
    # ===============================
    regime_results = {}

    if analyze_regimes_separately:
        print(f"\n3. Analyzing each structural regime separately...")

        for regime_idx, regime_name in enumerate(regime_names):
            mask = (regimes == regime_idx)
            n_regime_points = np.sum(mask)

            # Skip if regime has too few points
            if n_regime_points < 20:
                print(f"\nSkipping {regime_name}: insufficient data ({n_regime_points} points)")
                continue

            print(f"\n{'='*50}")
            print(f"REGIME: {regime_name} ({n_regime_points} points)")
            print(f"{'='*50}")

            # Create regime-specific features
            regime_feats_a = {
                'data': feats_a['data'][mask],
                'delta_LambdaC_pos': feats_a['delta_LambdaC_pos'][mask],
                'delta_LambdaC_neg': feats_a['delta_LambdaC_neg'][mask],
                'rho_T': feats_a['rho_T'][mask],
                'time_trend': np.arange(n_regime_points),  # Reset time trend
                'local_jump': feats_a['local_jump'][mask]
            }

            regime_feats_b = {
                'data': feats_b['data'][mask],
                'delta_LambdaC_pos': feats_b['delta_LambdaC_pos'][mask],
                'delta_LambdaC_neg': feats_b['delta_LambdaC_neg'][mask],
                'rho_T': feats_b['rho_T'][mask],
                'time_trend': np.arange(n_regime_points),  # Reset time trend
                'local_jump': feats_b['local_jump'][mask]
            }

            # Update config for regime-specific analysis
            regime_config = L3Config(
                T=n_regime_points,
                draws=min(config.draws, 8000),  # Reduce for computational efficiency
                tune=min(config.tune, 8000),
                target_accept=config.target_accept,
                hdi_prob=config.hdi_prob
            )

            try:
                # Fit regime-specific models
                print(f"\nFitting models for {regime_name}...")

                trace_a_regime = fit_l3_bayesian_regression_asymmetric(
                    data=regime_feats_a['data'],
                    features_dict={
                        'delta_LambdaC_pos': regime_feats_a['delta_LambdaC_pos'],
                        'delta_LambdaC_neg': regime_feats_a['delta_LambdaC_neg'],
                        'rho_T': regime_feats_a['rho_T'],
                        'time_trend': regime_feats_a['time_trend']
                    },
                    config=regime_config,
                    interaction_pos=regime_feats_b['delta_LambdaC_pos'],
                    interaction_neg=regime_feats_b['delta_LambdaC_neg'],
                    interaction_rhoT=regime_feats_b['rho_T']
                )

                trace_b_regime = fit_l3_bayesian_regression_asymmetric(
                    data=regime_feats_b['data'],
                    features_dict={
                        'delta_LambdaC_pos': regime_feats_b['delta_LambdaC_pos'],
                        'delta_LambdaC_neg': regime_feats_b['delta_LambdaC_neg'],
                        'rho_T': regime_feats_b['rho_T'],
                        'time_trend': regime_feats_b['time_trend']
                    },
                    config=regime_config,
                    interaction_pos=regime_feats_a['delta_LambdaC_pos'],
                    interaction_neg=regime_feats_a['delta_LambdaC_neg'],
                    interaction_rhoT=regime_feats_a['rho_T']
                )

                # Extract results with HDI
                summary_a_regime = az.summary(trace_a_regime, hdi_prob=config.hdi_prob)
                summary_b_regime = az.summary(trace_b_regime, hdi_prob=config.hdi_prob)

                # Get interaction coefficients for this regime
                beta_b_on_a_regime_pos = summary_a_regime.loc['beta_interact_pos', 'mean'] if 'beta_interact_pos' in summary_a_regime.index else 0
                beta_b_on_a_regime_neg = summary_a_regime.loc['beta_interact_neg', 'mean'] if 'beta_interact_neg' in summary_a_regime.index else 0
                beta_a_on_b_regime_pos = summary_b_regime.loc['beta_interact_pos', 'mean'] if 'beta_interact_pos' in summary_b_regime.index else 0
                beta_a_on_b_regime_neg = summary_b_regime.loc['beta_interact_neg', 'mean'] if 'beta_interact_neg' in summary_b_regime.index else 0

                # Extract HDI values
                def get_hdi_bounds_regime(summary, param_name, hdi_prob):
                    """Extract HDI bounds from summary table."""
                    if param_name in summary.index:
                        hdi_cols = [col for col in summary.columns if 'hdi' in col.lower()]
                        if len(hdi_cols) >= 2:
                            return summary.loc[param_name, hdi_cols[0]], summary.loc[param_name, hdi_cols[1]]
                    return None, None

                # Store regime results
                regime_results[regime_name] = {
                    'beta_b_on_a_pos': beta_b_on_a_regime_pos,
                    'beta_b_on_a_neg': beta_b_on_a_regime_neg,
                    'beta_a_on_b_pos': beta_a_on_b_regime_pos,
                    'beta_a_on_b_neg': beta_a_on_b_regime_neg,
                    'hdi': {}
                }

                # Print regime-specific results
                print(f"\n{regime_name} - Asymmetric Weather Interaction Effects:")
                print(f"  {name_b} → {name_a} (pos): β = {beta_b_on_a_regime_pos:.3f}")
                print(f"  {name_b} → {name_a} (neg): β = {beta_b_on_a_regime_neg:.3f}")
                print(f"  {name_a} → {name_b} (pos): β = {beta_a_on_b_regime_pos:.3f}")
                print(f"  {name_a} → {name_b} (neg): β = {beta_a_on_b_regime_neg:.3f}")

                # Print HDI for this regime
                print(f"\n{regime_name} - HDI ({config.hdi_prob*100:.0f}%) for Interaction Effects:")

                hdi_low, hdi_high = get_hdi_bounds_regime(summary_a_regime, 'beta_interact_pos', config.hdi_prob)
                if hdi_low is not None:
                    print(f"  {name_b} → {name_a} (pos): [{hdi_low:.3f}, {hdi_high:.3f}]")
                    regime_results[regime_name]['hdi'][f'{name_b}_to_{name_a}_pos'] = (hdi_low, hdi_high)

                hdi_low, hdi_high = get_hdi_bounds_regime(summary_a_regime, 'beta_interact_neg', config.hdi_prob)
                if hdi_low is not None:
                    print(f"  {name_b} → {name_a} (neg): [{hdi_low:.3f}, {hdi_high:.3f}]")
                    regime_results[regime_name]['hdi'][f'{name_b}_to_{name_a}_neg'] = (hdi_low, hdi_high)

                hdi_low, hdi_high = get_hdi_bounds_regime(summary_b_regime, 'beta_interact_pos', config.hdi_prob)
                if hdi_low is not None:
                    print(f"  {name_a} → {name_b} (pos): [{hdi_low:.3f}, {hdi_high:.3f}]")
                    regime_results[regime_name]['hdi'][f'{name_a}_to_{name_b}_pos'] = (hdi_low, hdi_high)

                hdi_low, hdi_high = get_hdi_bounds_regime(summary_b_regime, 'beta_interact_neg', config.hdi_prob)
                if hdi_low is not None:
                    print(f"  {name_a} → {name_b} (neg): [{hdi_low:.3f}, {hdi_high:.3f}]")
                    regime_results[regime_name]['hdi'][f'{name_a}_to_{name_b}_neg'] = (hdi_low, hdi_high)

                # Calculate synchronization for this regime
                sync_profile_regime, sync_rate_regime, optimal_lag_regime = calculate_sync_profile(
                    regime_feats_a['delta_LambdaC_pos'].astype(np.float64),
                    regime_feats_b['delta_LambdaC_pos'].astype(np.float64),
                    lag_window=10
                )

                regime_results[regime_name]['sync_rate'] = sync_rate_regime
                regime_results[regime_name]['optimal_lag'] = optimal_lag_regime

                print(f"\n{regime_name} - Weather Sync Rate σₛ ({name_a}↔{name_b}): {sync_rate_regime:.3f}")
                print(f"{regime_name} - Optimal Lag: {optimal_lag_regime} steps")

            except Exception as e:
                print(f"Error analyzing regime {regime_name}: {e}")
                continue

    # ===============================
    # Step 4: Visualize Regime Comparison
    # ===============================
    if show_all_plots and len(regime_results) > 0:
        plot_regime_comparison(regime_results, overall_results, regime_names, name_a, name_b)

    # Return comprehensive results
    return {
        'overall_results': overall_results,
        'regime_results': regime_results,
        'regime_labels': regimes,
        'regime_statistics': regime_stats,
        'regime_detector': regime_detector,
        'detection_method_used': detection_method_used,
        'transitions': transitions
    }

# ===============================
# Additional Weather-Specific Plotting Functions
# ===============================
def plot_weather_clusters(series_dict: Dict[str, np.ndarray], clusters: Dict[str, int]):
    """
    Plot weather parameters colored by cluster membership.

    Args:
        series_dict: Dictionary of weather time series
        clusters: Cluster assignments for each parameter
    """
    n_clusters = len(set(clusters.values()))
    if n_clusters <= 10:
        colors = plt.cm.tab10(np.arange(n_clusters))
    else:
        colors = plt.cm.tab20(np.arange(n_clusters))

    plt.figure(figsize=(14, 8))

    for name, data in series_dict.items():
        cluster = clusters[name]
        # Normalize data for visualization
        data_norm = (data - np.mean(data)) / np.std(data)
        plt.plot(data_norm, label=f"{name} (Cluster {cluster})",
                 color=colors[cluster], alpha=0.8, linewidth=2)

    plt.xlabel("Structural Snapshot Index")
    plt.ylabel("Normalized Value")
    plt.title("Weather Parameters Grouped by Synchronization Clusters")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weather_interaction_summary(interaction_effects: Dict[Tuple[str, str], float],
                                   series_names: List[str]):
    """
    Plot summary of all weather parameter interaction effects as a heatmap.

    Args:
        interaction_effects: Dictionary of pairwise interaction coefficients
        series_names: List of weather parameter names
    """
    n = len(series_names)
    interaction_matrix = np.zeros((n, n))

    # Fill interaction matrix
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i != j:
                key = (name_b, name_a)  # B's effect on A
                if key in interaction_effects:
                    interaction_matrix[i, j] = interaction_effects[key]

    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix,
                xticklabels=series_names,
                yticklabels=series_names,
                annot=True, fmt='.3f',
                cmap='RdBu_r', center=0,
                square=True,
                cbar_kws={'label': 'Interaction Coefficient β'})
    plt.title("Weather Parameter Cross-Interactions\n(Column → Row)", fontsize=16)
    plt.xlabel("From Parameter", fontsize=12)
    plt.ylabel("To Parameter", fontsize=12)
    plt.tight_layout()
    plt.show()

def create_weather_summary(series_names: List[str],
                          sync_mat: np.ndarray,
                          features_dict: Dict[str, Dict[str, np.ndarray]]):
    """
    Create a summary report of the weather analysis.

    Args:
        series_names: List of analyzed weather parameters
        sync_mat: Synchronization matrix
        features_dict: Extracted Lambda³ features
    """
    print("\n" + "="*60)
    print("WEATHER ANALYSIS SUMMARY - Lambda³ Theory")
    print("="*60)

    # Jump event statistics
    print("\nStructural Jump Event Statistics:")
    print("-" * 50)
    print(f"{'Parameter':15s} | {'Pos ∆ΛC':>8s} | {'Neg ∆ΛC':>8s} | {'Local':>8s} | {'Mean ρT':>8s}")
    print("-" * 50)

    for name in series_names:
        pos_jumps = np.sum(features_dict[name]['delta_LambdaC_pos'])
        neg_jumps = np.sum(features_dict[name]['delta_LambdaC_neg'])
        local_jumps = np.sum(features_dict[name]['local_jump'])
        mean_tension = np.mean(features_dict[name]['rho_T'])
        print(f"{name:15s} | {pos_jumps:8.0f} | {neg_jumps:8.0f} | {local_jumps:8.0f} | {mean_tension:8.3f}")

    # Top synchronizations
    print("\nTop Weather Parameter Synchronizations:")
    print("-" * 50)
    sync_pairs = []
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i < j:  # Only unique pairs
                sync_pairs.append((sync_mat[i, j], name_a, name_b))

    sync_pairs.sort(reverse=True)
    for sync_rate, name_a, name_b in sync_pairs[:5]:
        print(f"{name_a:15s} ↔ {name_b:15s} | σₛ = {sync_rate:.3f}")

    # Weather-specific insights
    print("\nLambda³ Weather Insights:")
    print("-" * 50)
    print("• Weather parameters exhibit discrete structural states (regimes)")
    print("• Synchronization reveals structural coupling between parameters")
    print("• ∆ΛC events represent discrete weather state transitions")
    print("• Tension scalar ρT indicates atmospheric instability")
    print("• No temporal causality assumed - only structural resonance")

    print("\n" + "="*60)


def plot_regime_comparison(regime_results: Dict, overall_results: Dict,
                          regime_names: List[str], name_a: str, name_b: str):
    """
    Visualize comparison of interaction effects across different structural regimes.

    Args:
        regime_results: Dictionary of results for each regime
        overall_results: Overall analysis results
        regime_names: List of regime names
        name_a: First parameter name
        name_b: Second parameter name
    """
    # Prepare data for plotting
    regimes = ['Overall'] + list(regime_results.keys())

    # Interaction coefficients
    beta_b_on_a_pos = [overall_results['beta_b_on_a']]
    beta_a_on_b_pos = [overall_results['beta_a_on_b']]
    sync_rates = [None]  # Overall doesn't have separate sync rate

    for regime in regime_results.keys():
        beta_b_on_a_pos.append(regime_results[regime]['beta_b_on_a_pos'])
        beta_a_on_b_pos.append(regime_results[regime]['beta_a_on_b_pos'])
        sync_rates.append(regime_results[regime]['sync_rate'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Bar plot of interaction coefficients
    ax1 = axes[0, 0]
    x = np.arange(len(regimes))
    width = 0.35

    bars1 = ax1.bar(x - width/2, beta_b_on_a_pos, width,
                     label=f'{name_b} → {name_a}', color='royalblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, beta_a_on_b_pos, width,
                     label=f'{name_a} → {name_b}', color='darkorange', alpha=0.8)

    ax1.set_xlabel('Structural Regime')
    ax1.set_ylabel('Interaction Coefficient β (pos)')
    ax1.set_title('Cross-Parameter Interactions by Regime')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=8)

    # 2. Sync rates by regime (excluding overall)
    ax2 = axes[0, 1]
    regime_sync_data = [(r, s) for r, s in zip(regimes[1:], sync_rates[1:]) if s is not None]
    if regime_sync_data:
        regime_labels_plot, sync_values = zip(*regime_sync_data)
        bars = ax2.bar(regime_labels_plot, sync_values, color='forestgreen', alpha=0.8)
        ax2.set_xlabel('Structural Regime')
        ax2.set_ylabel('Synchronization Rate σₛ')
        ax2.set_title(f'Sync Rates by Regime ({name_a} ↔ {name_b})')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)

    # 3. HDI width comparison
    ax3 = axes[1, 0]
    hdi_widths_b_on_a = []
    hdi_widths_a_on_b = []
    regime_labels_hdi = []

    for regime_name in regime_results.keys():
        if 'hdi' in regime_results[regime_name]:
            hdi_dict = regime_results[regime_name]['hdi']

            # Calculate HDI widths
            key_b_on_a = f'{name_b}_to_{name_a}_pos'
            key_a_on_b = f'{name_a}_to_{name_b}_pos'

            if key_b_on_a in hdi_dict:
                low, high = hdi_dict[key_b_on_a]
                hdi_widths_b_on_a.append(high - low)
                regime_labels_hdi.append(regime_name)

            if key_a_on_b in hdi_dict:
                low, high = hdi_dict[key_a_on_b]
                hdi_widths_a_on_b.append(high - low)

    if hdi_widths_b_on_a:
        x_hdi = np.arange(len(regime_labels_hdi))
        bars1 = ax3.bar(x_hdi - width/2, hdi_widths_b_on_a, width,
                        label=f'{name_b} → {name_a}', color='royalblue', alpha=0.8)
        bars2 = ax3.bar(x_hdi + width/2, hdi_widths_a_on_b, width,
                        label=f'{name_a} → {name_b}', color='darkorange', alpha=0.8)

        ax3.set_xlabel('Structural Regime')
        ax3.set_ylabel('HDI Width')
        ax3.set_title('Uncertainty (HDI Width) by Regime')
        ax3.set_xticks(x_hdi)
        ax3.set_xticklabels(regime_labels_hdi, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Regime characteristics summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = "Regime Characteristics Summary\n" + "="*40 + "\n\n"

    for regime_name in regime_results.keys():
        if regime_name in regime_results:
            res = regime_results[regime_name]
            summary_text += f"{regime_name}:\n"
            summary_text += f"  • {name_b}→{name_a}: β={res['beta_b_on_a_pos']:.3f}\n"
            summary_text += f"  • {name_a}→{name_b}: β={res['beta_a_on_b_pos']:.3f}\n"
            summary_text += f"  • Sync rate: σₛ={res['sync_rate']:.3f}\n"
            summary_text += f"  • Optimal lag: {res['optimal_lag']} steps\n\n"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.suptitle(f'Structural Regime Analysis: {name_a} ↔ {name_b}', fontsize=16)
    plt.tight_layout()
    plt.show()

# ===============================
# Regime Summary Creation Function
# ===============================
def create_regime_summary(regime_analysis_results: Dict, series_names: List[str]):
    """
    Create a comprehensive summary of regime analysis results across all parameter pairs.

    Args:
        regime_analysis_results: Dictionary of regime analysis results by parameter pair
        series_names: List of all weather parameter names
    """
    print("\n" + "="*70)
    print("STRUCTURAL REGIME ANALYSIS SUMMARY - Lambda³ Theory")
    print("="*70)

    # Collect all unique regimes across all analyses
    all_regimes = set()
    for pair_key, results in regime_analysis_results.items():
        if 'regime_results' in results:
            all_regimes.update(results['regime_results'].keys())

    all_regimes = sorted(list(all_regimes))

    print(f"\nDetected {len(all_regimes)} unique structural regimes across all analyses:")
    for regime in all_regimes:
        print(f"  • {regime}")

    # Summary table of interaction effects by regime
    print("\n" + "-"*70)
    print("INTERACTION EFFECTS BY REGIME")
    print("-"*70)

    for pair_key, results in regime_analysis_results.items():
        name_a, name_b = pair_key
        print(f"\n{name_a} ↔ {name_b}:")

        # Overall results
        overall = results['overall_results']
        print(f"  Overall: {name_b}→{name_a} β={overall['beta_b_on_a']:.3f}, "
              f"{name_a}→{name_b} β={overall['beta_a_on_b']:.3f}")

        # Regime-specific results
        if 'regime_results' in results:
            for regime_name, regime_data in results['regime_results'].items():
                print(f"  {regime_name}: {name_b}→{name_a} β={regime_data['beta_b_on_a_pos']:.3f}, "
                      f"{name_a}→{name_b} β={regime_data['beta_a_on_b_pos']:.3f}, "
                      f"σₛ={regime_data['sync_rate']:.3f}")

    # Key insights about regimes
    print("\n" + "-"*70)
    print("KEY REGIME INSIGHTS")
    print("-"*70)

    # Find regime with strongest interactions
    max_interaction = 0
    max_regime = None
    max_pair = None

    for pair_key, results in regime_analysis_results.items():
        if 'regime_results' in results:
            for regime_name, regime_data in results['regime_results'].items():
                interaction_strength = abs(regime_data['beta_b_on_a_pos']) + abs(regime_data['beta_a_on_b_pos'])
                if interaction_strength > max_interaction:
                    max_interaction = interaction_strength
                    max_regime = regime_name
                    max_pair = pair_key

    if max_regime:
        print(f"\n• Strongest structural coupling in {max_regime} for {max_pair[0]} ↔ {max_pair[1]}")

    # Compare HDI widths across regimes
    print("\n• Regime-specific uncertainty (HDI width) patterns:")
    for pair_key, results in regime_analysis_results.items():
        if 'regime_results' in results:
            name_a, name_b = pair_key
            hdi_summary = []

            for regime_name, regime_data in results['regime_results'].items():
                if 'hdi' in regime_data and regime_data['hdi']:
                    avg_hdi_width = np.mean([h[1] - h[0] for h in regime_data['hdi'].values()])
                    hdi_summary.append((regime_name, avg_hdi_width))

            if hdi_summary:
                hdi_summary.sort(key=lambda x: x[1])
                print(f"  {name_a} ↔ {name_b}: "
                      f"Most certain in {hdi_summary[0][0]} (HDI width={hdi_summary[0][1]:.2f}), "
                      f"Least certain in {hdi_summary[-1][0]} (HDI width={hdi_summary[-1][1]:.2f})")

    # Lambda³ theoretical interpretation
    print("\n" + "-"*70)
    print("LAMBDA³ THEORETICAL INTERPRETATION")
    print("-"*70)
    print("• Structural regimes represent distinct configurations in semantic space")
    print("• Regime transitions are not temporal but structural phase changes")
    print("• Different β values across regimes indicate context-dependent resonance")
    print("• Narrower HDI in specific regimes suggests structural coherence")
    print("• Synchronization variations reveal regime-specific coupling patterns")

    print("\n" + "="*70)

# ===============================
# Data Loading Functions (Same as original)
# ===============================
def load_csv_data(filepath: str,
                  time_column: Optional[str] = None,
                  value_columns: Optional[List[str]] = None,
                  delimiter: str = ',',
                  parse_dates: bool = True) -> Dict[str, np.ndarray]:
    """
    Load time series data from CSV file.

    Args:
        filepath: Path to CSV file
        time_column: Column name for time/date
        value_columns: Columns to analyze
        delimiter: CSV delimiter
        parse_dates: Whether to parse date columns

    Returns:
        Dictionary mapping column names to numpy arrays
    """
    # Load data
    df = pd.read_csv(filepath, delimiter=delimiter, parse_dates=parse_dates)

    print(f"Loaded CSV with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    # If time column is specified, sort by it
    if time_column and time_column in df.columns:
        df = df.sort_values(by=time_column)
        print(f"\nData sorted by {time_column}")

    # Select value columns
    if value_columns is None:
        # Use all numeric columns except time column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_column and time_column in numeric_cols:
            numeric_cols.remove(time_column)
        value_columns = numeric_cols

    print(f"\nUsing columns: {value_columns}")

    # Extract series data
    series_dict = {}
    for col in value_columns:
        if col in df.columns:
            # Handle missing values
            data = df[col].values
            if pd.isna(data).any():
                print(f"Warning: Column '{col}' has {pd.isna(data).sum()} missing values")
                # Fill missing values using forward/backward fill
                data = pd.Series(data).ffill().bfill().values
            series_dict[col] = data.astype(np.float64)
        else:
            print(f"Warning: Column '{col}' not found in CSV")

    return series_dict


def load_multiple_csv_files(filepaths: List[str],
                           series_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Load multiple CSV files, each containing a single time series.

    Args:
        filepaths: List of CSV file paths
        series_names: Names for each series (defaults to filenames)

    Returns:
        Dictionary mapping series names to numpy arrays
    """
    if series_names is None:
        series_names = [Path(fp).stem for fp in filepaths]

    series_dict = {}
    for filepath, name in zip(filepaths, series_names):
        df = pd.read_csv(filepath)

        # Assume first numeric column is the data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            data = df[numeric_cols[0]].values
            series_dict[name] = data.astype(np.float64)
            print(f"Loaded {name} from {filepath}: {len(data)} points")
        else:
            print(f"Warning: No numeric data found in {filepath}")

    return series_dict

def validate_series_lengths(series_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Validate and align series lengths to ensure compatibility.

    Args:
        series_dict: Dictionary of time series

    Returns:
        Aligned series dictionary
    """
    lengths = {name: len(data) for name, data in series_dict.items()}
    print(f"\nSeries lengths: {lengths}")

    if len(set(lengths.values())) > 1:
        print("Warning: Series have different lengths!")
        min_length = min(lengths.values())
        print(f"Truncating all series to minimum length: {min_length}")

        aligned_dict = {}
        for name, data in series_dict.items():
            aligned_dict[name] = data[:min_length]
        return aligned_dict

    return series_dict

# ===============================
# Main Execution Block
# ===============================
if __name__ == '__main__':
    print("="*70)
    print("Lambda³ Weather Analysis - Structural Tensor Framework")
    print("Weather as semantic space pulsations, not time series")
    print("="*70)

    # 1. Load and preprocess Tokyo weather data
    weather_data = fetch_weather_data(
        csv_path="tokyo_weather_days.csv",
        output_filename="weather_lambda3_data.csv"
    )

    # 2. Run Lambda³ analysis with structural regime detection
    features, sync_mat, regime_results = main_weather_analysis(
        csv_path="tokyo_weather_days.csv",
        time_column="date",
        value_columns=["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                      "precipitation", "wind_speed_10m", "surface_pressure"],
        config=L3Config(),
        analyze_all_pairs=True,
        max_pairs=10,  # Limit pairs for computational efficiency
        use_regime_detection=True,  # Enable structural regime detection
        n_regimes=4,  # Detect 4 regimes (e.g., Spring, Transition, Rainy)
        regime_names=['Late Spring', 'Pre-Monsoon', 'Rainy Season', 'Early Summer', 'Early Summer']
    )

    # 3. Advanced analysis for Temperature with regime context
    if 'Temperature' in features:
        print("\n" + "="*50)
        print("ADVANCED TEMPERATURE ANALYSIS")
        print("="*50)

        temp_data = features['Temperature']['data']
        temp_features_dict = {
            'delta_LambdaC_pos': features['Temperature']['delta_LambdaC_pos'],
            'delta_LambdaC_neg': features['Temperature']['delta_LambdaC_neg'],
            'rho_T': features['Temperature']['rho_T']
        }

        result = lambda3_advanced_weather_analysis(temp_data, temp_features_dict, 'Temperature')

        # If regime results exist for temperature pairs, show regime-specific statistics
        if regime_results:
            print("\n--- Temperature Regime-Specific Analysis ---")
            for pair_key, results in regime_results.items():
                if 'Temperature' in pair_key and 'regime_statistics' in results:
                    other_param = pair_key[0] if pair_key[1] == 'Temperature' else pair_key[1]
                    print(f"\nTemperature ↔ {other_param} regime statistics:")
                    for regime_name, stats in results['regime_statistics'].items():
                        print(f"  {regime_name}: {stats['percentage']:.1f}% of data, "
                              f"Mean Temp={stats.get('mean_Temperature', 'N/A'):.1f}°C")

    # 4. Precipitation event analysis with regime context
    if 'Precipitation' in features:
        print("\n" + "="*50)
        print("PRECIPITATION STRUCTURAL ANALYSIS")
        print("="*50)

        precip = features['Precipitation']['data']
        rain_events = precip > 0
        rain_indices = np.where(rain_events)[0]

        print(f"Total precipitation events: {len(rain_indices)}")
        print(f"Precipitation frequency: {len(rain_indices)/len(precip):.1%}")

        # Analyze atmospheric structure during rain
        if len(rain_indices) > 0 and 'Temperature' in features and 'Humidity' in features:
            temp_tension_rain = np.mean(features['Temperature']['rho_T'][rain_events])
            humid_tension_rain = np.mean(features['Humidity']['rho_T'][rain_events])

            temp_tension_dry = np.mean(features['Temperature']['rho_T'][~rain_events])
            humid_tension_dry = np.mean(features['Humidity']['rho_T'][~rain_events])

            print(f"\nStructural tension (ρT) during precipitation:")
            print(f"  Temperature: Rain={temp_tension_rain:.3f}, Dry={temp_tension_dry:.3f}")
            print(f"  Humidity: Rain={humid_tension_rain:.3f}, Dry={humid_tension_dry:.3f}")

            # Regime-specific precipitation analysis
            if regime_results and ('Temperature', 'Humidity') in regime_results:
                th_results = regime_results[('Temperature', 'Humidity')]
                if 'regime_labels' in th_results:
                    regimes = th_results['regime_labels']
                    print("\nPrecipitation frequency by structural regime:")

                    for regime_idx, regime_name in enumerate(['Spring', 'Transition', 'Rainy Season']):
                        regime_mask = (regimes == regime_idx)
                        regime_rain_freq = np.mean(rain_events[regime_mask]) if np.sum(regime_mask) > 0 else 0
                        print(f"  {regime_name}: {regime_rain_freq:.1%}")

    # 5. Regime-specific insights summary
    if regime_results:
        print("\n" + "="*50)
        print("STRUCTURAL REGIME INSIGHTS")
        print("="*50)

        # Find the most structurally different regimes
        max_diff = 0
        max_diff_pair = None
        max_diff_regimes = None

        for pair_key, results in regime_results.items():
            if 'regime_results' in results and len(results['regime_results']) > 1:
                regime_betas = []
                for regime_name, regime_data in results['regime_results'].items():
                    regime_betas.append((
                        regime_name,
                        regime_data['beta_b_on_a_pos'],
                        regime_data['beta_a_on_b_pos']
                    ))

                # Calculate maximum difference between regimes
                for i in range(len(regime_betas)):
                    for j in range(i+1, len(regime_betas)):
                        diff = abs(regime_betas[i][1] - regime_betas[j][1]) + \
                               abs(regime_betas[i][2] - regime_betas[j][2])
                        if diff > max_diff:
                            max_diff = diff
                            max_diff_pair = pair_key
                            max_diff_regimes = (regime_betas[i][0], regime_betas[j][0])

        if max_diff_pair:
            print(f"\nLargest structural difference found between {max_diff_regimes[0]} and {max_diff_regimes[1]}")
            print(f"for the parameter pair: {max_diff_pair[0]} ↔ {max_diff_pair[1]}")
            print(f"This indicates a fundamental shift in atmospheric coupling mechanisms")

    print("\n" + "="*70)
    print("Lambda³ Weather Analysis Complete")
    print("Weather understood as structural transactions in semantic space")
    print("with regime-specific coupling patterns revealed")
    print("="*70)
