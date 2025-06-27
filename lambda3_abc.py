# ==========================================================
# Λ³ABC: Lambda³ Analytics for Bayes & CausalJunction
# ----------------------------------------------------------
# To install required packages, run this command FIRST:
# !pip install yfinance pandas
# ==========================================================
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
# YFinanceAPI
# ===============================
import yfinance as yf

# ===============================
# Global Constants for JIT
# ===============================
DELTA_PERCENTILE = 97.0  # Percentile threshold for jump detection
LOCAL_JUMP_PERCENTILE = 97.0  # Percentile for local jump detection
WINDOW_SIZE = 10  # Window size for tension scalar calculation
LOCAL_WINDOW_SIZE = 10  # Window for local standard deviation
LAG_WINDOW_DEFAULT = 10  # Default lag window for synchronization
SYNC_THRESHOLD_DEFAULT = 0.3  # Default threshold for sync network edges
NOISE_STD_DEFAULT = 0.5  # Default noise standard deviation
NOISE_STD_HIGH = 1.5  # High noise standard deviation

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
    target_accept: float = 0.95  # Target acceptance probability
    # Posterior visualization parameters
    var_names: list = ('beta_time_a', 'beta_time_b', 'beta_interact', 'beta_rhoT_a', 'beta_rhoT_b')
    hdi_prob: float = 0.94  # Highest density interval probability

# ===============================
# Lambda³ YFinanceAPI
# ===============================

def fetch_financial_data(
    start_date="2024-01-01",
    end_date="2024-12-31",
    tickers=None,
    desired_order=None,
    csv_filename="financial_data_2024.csv",
    verbose=True
):
    """
    Fetch, preprocess, and save multi-market financial time series data using Yahoo Finance.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        tickers (dict): Dictionary mapping label to Yahoo Finance ticker.
        desired_order (list): List of labels for column ordering.
        csv_filename (str): Output CSV file name.
        verbose (bool): Whether to print progress and sample data.

    Returns:
        pd.DataFrame: Preprocessed financial time series.
    """
    # Default ticker symbols (can be customized)
    if tickers is None:
        tickers = {
            "USD/JPY": "JPY=X",
            "GBP/USD": "GBPUSD=X",
            "GBP/JPY": "GBPJPY=X",  # Temporary, used for JPY/GBP calculation
            "Nikkei 225": "^N225",
            "Dow Jones": "^DJI"
        }
    if desired_order is None:
        desired_order = ["USD/JPY", "JPY/GBP", "GBP/USD", "Nikkei 225", "Dow Jones"]

    if verbose:
        print(f"Fetching daily data from {start_date} to {end_date}...")

    try:
        # Download historical close prices for all tickers
        data_close = yf.download(list(tickers.values()), start=start_date, end=end_date)['Close']

        # Calculate JPY/GBP as the reciprocal of GBP/JPY
        data_close['JPY/GBP'] = 1 / data_close['GBPJPY=X']

        # Drop the temporary GBP/JPY column (not needed after calculation)
        data_close = data_close.drop(columns=['GBPJPY=X'])

        # Rename columns from ticker symbols to descriptive labels
        reversed_tickers = {v: k for k, v in tickers.items()}
        final_data = data_close.rename(columns=reversed_tickers)

        # Reorder columns according to desired_order
        final_data = final_data[desired_order]

        # Drop any rows with missing values (e.g., market holidays)
        final_data = final_data.dropna()

        if verbose:
            print("\nFirst 5 rows of the processed data:")
            print(final_data.head())
            print("\nLast 5 rows of the processed data:")
            print(final_data.tail())

        # Save to CSV (index=True ensures date is the first column)
        final_data.to_csv(csv_filename, index=True)
        if verbose:
            print(f"\nData successfully saved to '{csv_filename}'.")

        return final_data

    except Exception as e:
        print(f"\nError occurred while fetching financial data: {e}")
        print("Check ticker symbols or network connectivity.")
        return None

# ===============================
# JIT-compiled Core Functions
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
# Bayesian Model (PyMC)
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

#===============================
# Regime Detection (Clustering/Switching)
# ===============================
class Lambda3RegimeDetector:
    """
    Detect market regimes using Lambda³ features.
    Clusters time periods based on structural characteristics.
    """
    def __init__(self, n_regimes=3, method='kmeans'):
        self.n_regimes = n_regimes
        self.method = method
        self.regime_labels = None
        self.regime_features = None
    
    def fit(self, features_dict):
        """
        Estimate market regimes using clustering on Lambda³ features.
        
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
                'mean_rhoT': np.mean(X[labels == r, 2])
            }
            for r in range(self.n_regimes)
        }
        return labels

    def label_regimes(self):
        """
        Assign descriptive labels to each regime.
        """
        return {r: f"Regime-{r+1}" for r in range(self.n_regimes)}


# ===============================
# Multi-Scale Feature Extraction
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
# Conditional Synchronization Analysis
# ===============================
def lambda3_conditional_sync(series_a, series_b, condition_series, condition_threshold):
    """
    Calculate conditional synchronization rate.
    Only considers periods where condition is met.
    
    Args:
        series_a: First event series
        series_b: Second event series
        condition_series: Conditioning variable (e.g., tension)
        condition_threshold: Threshold for condition
        
    Returns:
        Conditional synchronization rate
    """
    mask = condition_series > condition_threshold
    # Calculate sync only for high-tension periods
    sync = np.mean(series_a[mask] * series_b[mask]) if np.sum(mask) > 0 else 0.0
    return sync

# ===============================
# Integrated Advanced Lambda³ Analysis
# ===============================
def lambda3_advanced_analysis(data, features_dict):
    """
    Comprehensive Lambda³ analysis combining multiple techniques.
    
    Args:
        data: Time series data
        features_dict: Pre-computed Lambda³ features
        
    Returns:
        Dictionary with analysis results
    """
    # 1. Regime Detection
    regime_detector = Lambda3RegimeDetector(n_regimes=3)
    regimes = regime_detector.fit(features_dict)
    regime_labels = regime_detector.label_regimes()

    print("Market Regime Detection:")
    for regime, label in regime_labels.items():
        stats = regime_detector.regime_features[regime]
        print(f"  {label}: {stats['frequency']:.1%} (Mean ρT: {stats['mean_rhoT']:.2f})")

    # 2. Multi-scale Analysis
    ms_analyzer = Lambda3MultiScaleAnalyzer(scales=[5, 10, 20, 50])
    ms_features = ms_analyzer.extract_features(data)
    scale_breaks = ms_analyzer.detect_scale_breaks()

    print(f"\nScale Break Locations: {scale_breaks}")

    # 3. Conditional Synchronization
    # Example: sync only during high-tension periods
    if 'rho_T' in features_dict:
        sync_cond = lambda3_conditional_sync(
            series_a=features_dict['delta_LambdaC_pos'],
            series_b=features_dict['delta_LambdaC_neg'],
            condition_series=features_dict['rho_T'],
            condition_threshold=np.median(features_dict['rho_T'])
        )
        print(f"\nConditional Sync Rate (high tension): {sync_cond:.3f}")

    return {
        'regimes': regimes,
        'regime_labels': regime_labels,
        'multi_scale_features': ms_features,
        'scale_breaks': scale_breaks
    }


# ===============================
# Lambda³ Extended Analysis
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
# Synchronization Analysis
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
# Visualization Functions
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

def plot_l3_prediction_dual(
    data_dict: Dict[str, np.ndarray],
    mu_pred_dict: Dict[str, np.ndarray],
    jump_pos_dict: Dict[str, np.ndarray],
    jump_neg_dict: Dict[str, np.ndarray],
    local_jump_dict: Optional[Dict[str, np.ndarray]] = None,
    series_names: Optional[List[str]] = None,
    titles: Optional[List[str]] = None
):
    """
    Plot observed data, model predictions, and detected events for multiple series.
    
    Args:
        data_dict: Original time series data
        mu_pred_dict: Model predictions
        jump_pos_dict: Positive jump indicators
        jump_neg_dict: Negative jump indicators
        local_jump_dict: Local jump indicators
        series_names: Names of series to plot
        titles: Custom titles for subplots
    """
    if series_names is None:
        series_names = list(data_dict.keys())

    n_series = len(series_names)
    fig, axes = plt.subplots(n_series, 1, figsize=(15, 5 * n_series), sharex=True)

    if n_series == 1:
        axes = [axes]

    for i, series in enumerate(series_names):
        ax = axes[i]
        data = data_dict[series]
        mu_pred = mu_pred_dict[series]
        jump_pos = jump_pos_dict[series]
        jump_neg = jump_neg_dict[series]
        local_jump = local_jump_dict[series] if local_jump_dict else None

        # Plot data and prediction
        ax.plot(data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
        ax.plot(mu_pred, color='C2', lw=2, label='Model Prediction')

        # Plot jump events
        jump_pos_idx = np.where(jump_pos > 0)[0]
        if len(jump_pos_idx):
            ax.plot(jump_pos_idx, data[jump_pos_idx], 'o', color='dodgerblue',
                   markersize=10, label='Positive Jump')
            for idx in jump_pos_idx:
                ax.axvline(x=idx, color='dodgerblue', linestyle='--', alpha=0.5)

        jump_neg_idx = np.where(jump_neg > 0)[0]
        if len(jump_neg_idx):
            ax.plot(jump_neg_idx, data[jump_neg_idx], 'o', color='orange',
                   markersize=10, label='Negative Jump')
            for idx in jump_neg_idx:
                ax.axvline(x=idx, color='orange', linestyle='-.', alpha=0.5)

        if local_jump is not None:
            local_jump_idx = np.where(local_jump > 0)[0]
            if len(local_jump_idx):
                ax.plot(local_jump_idx, data[local_jump_idx], 'o', color='magenta',
                       markersize=7, alpha=0.7, label='Local Jump')

        # Formatting
        plot_title = titles[i] if titles and i < len(titles) else f"Series {series}: Fit + Events"
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
        ax.grid(axis='y', linestyle=':', alpha=0.7)

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
    plt.title("Synchronization (σₛ) Network")
    plt.show()

# ===============================
# Data Loading Functions
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
# Main Execution Pipeline for CSV
# ===============================
def main_csv_analysis(
    csv_path: str = None,
    csv_paths: List[str] = None,
    time_column: str = None,
    value_columns: List[str] = None,
    series_names: List[str] = None,
    config: L3Config = None,
    analyze_all_pairs: bool = True,
    max_pairs: int = None
):
    """
    Main pipeline for analyzing CSV data using Lambda³ framework.

    Args:
        csv_path: Path to single CSV with multiple series
        csv_paths: List of paths to CSVs (one series each)
        time_column: Time column name for sorting
        value_columns: Columns to analyze (None = all numeric)
        series_names: Names for the series
        config: L3Config instance (uses default if None)
        analyze_all_pairs: Whether to analyze all pairs (default: True)
        max_pairs: Maximum number of pairs to analyze (None = all)
        
    Returns:
        features_dict: Extracted Lambda³ features
        sync_mat: Synchronization matrix
    """
    if config is None:
        config = L3Config()

    # Load data
    if csv_path:
        print(f"Loading data from: {csv_path}")
        series_dict = load_csv_data(csv_path, time_column, value_columns)
    elif csv_paths:
        print(f"Loading data from {len(csv_paths)} files")
        series_dict = load_multiple_csv_files(csv_paths, series_names)
    else:
        raise ValueError("Must provide either csv_path or csv_paths")

    # Validate series
    series_dict = validate_series_lengths(series_dict)

    if len(series_dict) < 2:
        print("Warning: Need at least 2 series for cross-analysis")
        return

    # Update config with actual data length
    data_length = len(next(iter(series_dict.values())))
    config.T = data_length
    print(f"\nAnalyzing {len(series_dict)} series with {data_length} time points each")

    # Extract features for all series
    features_dict = {}
    for name, data in series_dict.items():
        print(f"\nExtracting features for series: {name}")
        feats = calc_lambda3_features_v2(data, config)
        features_dict[name] = {
            'data': data,
            'delta_LambdaC_pos': feats[0],
            'delta_LambdaC_neg': feats[1],
            'rho_T': feats[2],
            'time_trend': feats[3],
            'local_jump': feats[4]
        }

    # Analyze pairs
    series_list = list(series_dict.keys())
    n_series = len(series_list)

    if analyze_all_pairs and n_series > 2:
        # Analyze all unique pairs
        from itertools import combinations
        pairs = list(combinations(series_list, 2))

        if max_pairs and len(pairs) > max_pairs:
            print(f"\nNote: Limiting analysis to first {max_pairs} pairs out of {len(pairs)} total")
            pairs = pairs[:max_pairs]

        print(f"\n{'='*60}")
        print(f"ANALYZING ALL {len(pairs)} PAIRS")
        print(f"{'='*60}")

        # Store interaction effects for summary
        interaction_effects = {}

        for i, (name_a, name_b) in enumerate(pairs, 1):
            print(f"\n[{i}/{len(pairs)}] Analyzing: {name_a} ↔ {name_b}")

            try:
                # Analyze this pair
                beta_ab, beta_ba = analyze_series_pair(
                    name_a, name_b, features_dict, config,
                    show_all_plots=(i <= 3)  # Show detailed plots only for first 3 pairs
                )

                # Store interaction effects
                interaction_effects[(name_a, name_b)] = beta_ab
                interaction_effects[(name_b, name_a)] = beta_ba

            except Exception as e:
                print(f"Error analyzing pair {name_a} ↔ {name_b}: {e}")
                continue

        # Summary of all interaction effects
        plot_interaction_summary(interaction_effects, series_list)

    else:
        # Analyze just the first two series
        if len(series_list) >= 2:
            analyze_series_pair(
                series_list[0], series_list[1],
                features_dict, config,
                show_all_plots=True
            )

    # Multi-series synchronization analysis
    print("\n" + "="*50)
    print("MULTI-SERIES SYNCHRONIZATION ANALYSIS")
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
    plt.title("Synchronization Rate Matrix (σₛ)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Build and plot sync network
    print("\n=== Building Synchronization Network ===")

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
        print("\n=== Clustering Analysis ===")
        n_clusters = min(3, len(series_dict) // 2)
        clusters, _ = cluster_series_by_sync(event_series_dict, lag_window=10, n_clusters=n_clusters)
        print(f"Clusters: {clusters}")

        # Plot clustered series
        plot_clustered_series(series_dict, clusters)

    # Create summary report
    create_analysis_summary(series_list, sync_mat, features_dict)

    return features_dict, sync_mat

# ===============================
# Pairwise Series Analysis
# ===============================
def analyze_series_pair(
    name_a: str, name_b: str,
    features_dict: Dict[str, Dict[str, np.ndarray]],
    config: L3Config,
    show_all_plots: bool = True
) -> Tuple[float, float]:
    """
    Detailed analysis of a pair of series including cross-interactions.
    
    Args:
        name_a: First series name
        name_b: Second series name
        features_dict: Pre-computed Lambda³ features
        config: Analysis configuration
        show_all_plots: Whether to show all plots
        
    Returns:
        Tuple of (beta_B_on_A, beta_A_on_B) interaction coefficients
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING PAIR: {name_a} ↔ {name_b}")
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

    # Get summaries
    summary_a = az.summary(trace_a)
    summary_b = az.summary(trace_b)

    # Extract interaction coefficients
    beta_b_on_a_pos = summary_a.loc['beta_interact_pos', 'mean'] if 'beta_interact_pos' in summary_a.index else 0
    beta_b_on_a_neg = summary_a.loc['beta_interact_neg', 'mean'] if 'beta_interact_neg' in summary_a.index else 0
    beta_a_on_b_pos = summary_b.loc['beta_interact_pos', 'mean'] if 'beta_interact_pos' in summary_b.index else 0
    beta_a_on_b_neg = summary_b.loc['beta_interact_neg', 'mean'] if 'beta_interact_neg' in summary_b.index else 0

    # Use positive interaction as primary metric
    beta_b_on_a = beta_b_on_a_pos
    beta_a_on_b = beta_a_on_b_pos

    print(f"\nAsymmetric Interaction Effects:")
    print(f"  {name_b} → {name_a} (pos): β = {beta_b_on_a_pos:.3f}")
    print(f"  {name_b} → {name_a} (neg): β = {beta_b_on_a_neg:.3f}")
    print(f"  {name_a} → {name_b} (pos): β = {beta_a_on_b_pos:.3f}")
    print(f"  {name_a} → {name_b} (neg): β = {beta_a_on_b_neg:.3f}")

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
        plot_l3_prediction_dual(
            data_dict={name_a: feats_a['data'], name_b: feats_b['data']},
            mu_pred_dict={name_a: mu_pred_a, name_b: mu_pred_b},
            jump_pos_dict={name_a: feats_a['delta_LambdaC_pos'], name_b: feats_b['delta_LambdaC_pos']},
            jump_neg_dict={name_a: feats_a['delta_LambdaC_neg'], name_b: feats_b['delta_LambdaC_neg']},
            local_jump_dict={name_a: feats_a['local_jump'], name_b: feats_b['local_jump']},
            titles=[f'{name_a}: Fit + Events', f'{name_b}: Fit + Events']
        )

        # Plot posterior distributions
        print(f"\nPosterior for {name_a} (with {name_b} interaction):")
        plot_posterior(
            trace_a,
            var_names=['beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT', 
                      'beta_interact_pos', 'beta_interact_neg', 'beta_interact_stress'],
            hdi_prob=config.hdi_prob
        )

        print(f"\nPosterior for {name_b} (with {name_a} interaction):")
        plot_posterior(
            trace_b,
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
            title=f'Lagged Causality Profiles: {name_a} ↔ {name_b}'
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

    print(f"\nSync Rate σₛ ({name_a}↔{name_b}): {sync_rate:.3f}")
    print(f"Optimal Lag: {optimal_lag} steps")

    if show_all_plots:
        plot_sync_profile(sync_profile, title=f"Sync Profile ({name_a}↔{name_b})")

    return beta_b_on_a, beta_a_on_b

# ===============================
# Additional Plotting Functions
# ===============================
def plot_clustered_series(series_dict: Dict[str, np.ndarray], 
                         clusters: Dict[str, int]):
    """
    Plot time series colored by cluster membership.
    
    Args:
        series_dict: Dictionary of time series
        clusters: Cluster assignments for each series
    """
    n_clusters = len(set(clusters.values()))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    plt.figure(figsize=(12, 6))
    for name, data in series_dict.items():
        cluster = clusters[name]
        plt.plot(data, label=f"{name} (Cluster {cluster})", 
                color=colors[cluster], alpha=0.7, linewidth=1.5)
    
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Time Series Grouped by Synchronization Clusters")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_interaction_summary(interaction_effects: Dict[Tuple[str, str], float],
                           series_names: List[str]):
    """
    Plot summary of all interaction effects as a heatmap.
    
    Args:
        interaction_effects: Dictionary of pairwise interaction coefficients
        series_names: List of series names
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
    plt.title("Cross-Series Interaction Effects\n(Column → Row)", fontsize=16)
    plt.xlabel("From Series", fontsize=12)
    plt.ylabel("To Series", fontsize=12)
    plt.tight_layout()
    plt.show()

def create_analysis_summary(series_names: List[str],
                          sync_mat: np.ndarray,
                          features_dict: Dict[str, Dict[str, np.ndarray]]):
    """
    Create a summary report of the analysis.
    
    Args:
        series_names: List of analyzed series
        sync_mat: Synchronization matrix
        features_dict: Extracted Lambda³ features
    """
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    # Jump event statistics
    print("\nJump Event Statistics:")
    print("-" * 40)
    for name in series_names:
        pos_jumps = np.sum(features_dict[name]['delta_LambdaC_pos'])
        neg_jumps = np.sum(features_dict[name]['delta_LambdaC_neg'])
        local_jumps = np.sum(features_dict[name]['local_jump'])
        print(f"{name:15s} | Pos: {pos_jumps:3.0f} | Neg: {neg_jumps:3.0f} | Local: {local_jumps:3.0f}")

    # Top synchronizations
    print("\nTop Synchronization Pairs:")
    print("-" * 40)
    sync_pairs = []
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i < j:  # Only unique pairs
                sync_pairs.append((sync_mat[i, j], name_a, name_b))

    sync_pairs.sort(reverse=True)
    for sync_rate, name_a, name_b in sync_pairs[:5]:
        print(f"{name_a:15s} ↔ {name_b:15s} | σₛ = {sync_rate:.3f}")

    print("\n" + "="*60)

# ===============================
# Main Execution Block
# ===============================
if __name__ == '__main__':

    # 1. Download and preprocess financial data, then save to CSV
    fetch_financial_data(
        start_date="2024-01-01",
        end_date="2024-12-31",
        csv_filename="financial_data_2024.csv"
    )

    # 2. Load real financial data and extract Lambda³ features from CSV
    features, sync_mat = main_csv_analysis(
        csv_path="financial_data_2024.csv",  # Adjust path if necessary
        time_column="Date",
        value_columns=["USD/JPY", "JPY/GBP", "GBP/USD", "Nikkei 225", "Dow Jones"],
        config=L3Config(
            draws=8000,
            tune=8000
        )
    )

    # 3. Example: Advanced Lambda³ analysis for USD/JPY
    usd_jpy_data = features['USD/JPY']['data']
    usd_jpy_features_dict = {
        'delta_LambdaC_pos': features['USD/JPY']['delta_LambdaC_pos'],
        'delta_LambdaC_neg': features['USD/JPY']['delta_LambdaC_neg'],
        'rho_T': features['USD/JPY']['rho_T']
    }
    result = lambda3_advanced_analysis(usd_jpy_data, usd_jpy_features_dict)

    # 4. Example: Multi-file analysis (can be commented out)
    # main_csv_analysis(
    #     csv_paths=["data/series1.csv", "data/series2.csv", "data/series3.csv"],
    #     series_names=["Tokyo", "Osaka", "Nagoya"]
    # )
