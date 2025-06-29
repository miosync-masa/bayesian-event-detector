"""
Feature extraction module for Lambda³ framework.

This module implements JIT-compiled functions for efficient
extraction of structural features from time series data.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Dict, List, Optional, Union
import networkx as nx

from .config import (
    L3Config, FeatureConfig,
    DELTA_PERCENTILE, LOCAL_JUMP_PERCENTILE,
    WINDOW_SIZE, LOCAL_WINDOW_SIZE, LAG_WINDOW_DEFAULT
)
from .types import Lambda3FeatureSet, SyncProfile


# ===============================
# JIT-compiled Core Functions
# ===============================

@njit
def calculate_diff_and_threshold(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """
    JIT-compiled difference calculation and threshold computation.
    
    Args:
        data: Time series data
        percentile: Percentile for threshold calculation
        
    Returns:
        diff: First differences
        threshold: Threshold value at given percentile
    """
    n = len(data)
    diff = np.empty(n)
    diff[0] = 0
    
    for i in range(1, n):
        diff[i] = data[i] - data[i-1]
    
    abs_diff = np.abs(diff)
    threshold = np.percentile(abs_diff, percentile)
    
    return diff, threshold


@njit
def detect_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled jump detection based on threshold.
    
    Args:
        diff: First differences
        threshold: Detection threshold
        
    Returns:
        pos_jumps: Binary array of positive jumps
        neg_jumps: Binary array of negative jumps
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
        data: Time series data
        window: Window size for local calculation
        
    Returns:
        local_std: Local standard deviation at each point
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
    
    The tension scalar measures local volatility/stress in the time series,
    representing the structural tension at each point in Lambda³ theory.
    
    Args:
        data: Time series data
        window: Window size for calculation
        
    Returns:
        rho_t: Tension scalar values (ρT)
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
    
    Computes σₛ (synchronization rate) in Lambda³ theory.
    
    Args:
        series_a: First series (binary events)
        series_b: Second series (binary events)
        lag: Time lag (positive = b leads a, negative = a leads b)
        
    Returns:
        sync_rate: Synchronization rate (σₛ) at given lag
    """
    if lag < 0:
        # A leads B
        if -lag < len(series_a):
            return np.mean(series_a[-lag:] * series_b[:lag])
        else:
            return 0.0
    elif lag > 0:
        # B leads A
        if lag < len(series_b):
            return np.mean(series_a[:-lag] * series_b[lag:])
        else:
            return 0.0
    else:
        # No lag
        return np.mean(series_a * series_b)


@njit(parallel=True)
def calculate_sync_profile_jit(
    series_a: np.ndarray, 
    series_b: np.ndarray,
    lag_window: int
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    JIT-compiled synchronization profile calculation with parallelization.
    
    Args:
        series_a: First series (binary events)
        series_b: Second series (binary events)
        lag_window: Maximum lag to consider
        
    Returns:
        lags: Array of lag values
        sync_values: Synchronization rate (σₛ) at each lag
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


@njit
def calculate_local_jumps(diff: np.ndarray, local_std: np.ndarray, percentile: float) -> np.ndarray:
    """
    Detect local jumps using normalized score.
    
    Local jumps represent localized ΔΛC pulsations in Lambda³ theory.
    
    Args:
        diff: First differences
        local_std: Local standard deviation
        percentile: Detection percentile
        
    Returns:
        local_jumps: Binary array of local jump detections
    """
    # Normalized score
    score = np.abs(diff) / (local_std + 1e-8)  # Avoid division by zero
    threshold = np.percentile(score, percentile)
    
    return (score > threshold).astype(np.int32)


# ===============================
# High-level Feature Extraction
# ===============================

def extract_lambda3_features(
    data: np.ndarray,
    config: Optional[Union[L3Config, FeatureConfig]] = None,
    series_name: Optional[str] = None
) -> Lambda3FeatureSet:
    """
    Extract Lambda³ features from time series data.
    
    This function computes:
    - Jump events (ΔΛC±: structural changes)
    - Tension scalar (ρT: local volatility)
    - Local jump detections (localized ΔΛC pulsations)
    - Time trend (ΛF: progression vector)
    
    Args:
        data: Time series data
        config: Configuration object (L3Config or FeatureConfig)
        series_name: Optional name for the series
        
    Returns:
        Lambda3FeatureSet: Extracted features
    """
    # Handle configuration
    if config is None:
        feat_config = FeatureConfig()
    elif isinstance(config, L3Config):
        feat_config = config.feature
    else:
        feat_config = config
    
    # Validate input
    data = np.asarray(data, dtype=np.float64)
    if len(data) < feat_config.window:
        raise ValueError(f"Data length ({len(data)}) must be >= window size ({feat_config.window})")
    
    # Calculate differences and threshold
    diff, threshold = calculate_diff_and_threshold(data, feat_config.delta_percentile)
    
    # Detect jumps (ΔΛC±)
    delta_pos, delta_neg = detect_jumps(diff, threshold)
    
    # Calculate local standard deviation
    local_std = calculate_local_std(data, feat_config.local_window)
    
    # Detect local jumps (localized ΔΛC pulsations)
    local_jumps = calculate_local_jumps(diff, local_std, feat_config.local_jump_percentile)
    
    # Calculate tension scalar (ρT)
    rho_t = calculate_rho_t(data, feat_config.window)
    
    # Simple linear time trend (ΛF: progression vector)
    time_trend = np.arange(len(data), dtype=np.float64)
    
    # Create metadata with Lambda³ annotations
    metadata = {
        'series_name': series_name,
        'length': len(data),
        'config': feat_config.__dict__,
        'lambda3_components': {
            'delta_LambdaC_pos': 'ΔΛC⁺ - positive structural jumps',
            'delta_LambdaC_neg': 'ΔΛC⁻ - negative structural jumps',
            'rho_T': 'ρT - tension scalar (local volatility)',
            'time_trend': 'ΛF - progression vector',
            'local_jump': 'Localized ΔΛC pulsations'
        }
    }
    
    return Lambda3FeatureSet(
        data=data,
        delta_LambdaC_pos=delta_pos,
        delta_LambdaC_neg=delta_neg,
        rho_T=rho_t,
        time_trend=time_trend,
        local_jump=local_jumps,
        metadata=metadata
    )


def extract_features_dict(
    series_dict: Dict[str, np.ndarray],
    config: Optional[Union[L3Config, FeatureConfig]] = None,
    show_progress: bool = True
) -> Dict[str, Lambda3FeatureSet]:
    """
    Extract features from multiple time series.
    
    Args:
        series_dict: Dictionary of series {name: data}
        config: Configuration object
        show_progress: Whether to show progress messages
        
    Returns:
        Dict[str, Lambda3FeatureSet]: Features for each series
    """
    features = {}
    
    for i, (name, data) in enumerate(series_dict.items(), 1):
        if show_progress:
            print(f"[{i}/{len(series_dict)}] Extracting Lambda³ features for {name}...")
        
        try:
            features[name] = extract_lambda3_features(data, config, series_name=name)
            if show_progress:
                feats = features[name]
                print(f"  ✓ Length: {feats.length}, "
                      f"ΔΛC⁺: {feats.n_pos_jumps}, "
                      f"ΔΛC⁻: {feats.n_neg_jumps}, "
                      f"Mean ρT: {feats.mean_tension:.3f}")
        except Exception as e:
            print(f"  ✗ Error extracting features for {name}: {e}")
            continue
    
    return features


# ===============================
# Synchronization Analysis
# ===============================

def calculate_sync_profile(
    series_a: np.ndarray,
    series_b: np.ndarray,
    lag_window: int = LAG_WINDOW_DEFAULT,
    series_names: Optional[Tuple[str, str]] = None
) -> SyncProfile:
    """
    Calculate synchronization profile between two event series.
    
    Computes σₛ (synchronization rate) profile in Lambda³ theory.
    
    Args:
        series_a: First event series (binary)
        series_b: Second event series (binary)
        lag_window: Maximum lag to consider
        series_names: Optional names for the series
        
    Returns:
        SyncProfile: Synchronization profile results
    """
    # Ensure float64 for JIT function
    series_a = series_a.astype(np.float64)
    series_b = series_b.astype(np.float64)
    
    # Calculate using JIT function
    lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_jit(
        series_a, series_b, lag_window
    )
    
    # Convert to dictionary
    profile = {int(lag): float(sync) for lag, sync in zip(lags, sync_values)}
    
    return SyncProfile(
        profile=profile,
        max_sync_rate=float(max_sync),
        optimal_lag=int(optimal_lag),
        series_names=series_names
    )


def calculate_dynamic_sync(
    series_a: np.ndarray,
    series_b: np.ndarray,
    window: int = 20,
    lag_window: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate time-varying synchronization rate (dynamic σₛ).
    
    Args:
        series_a: First event series
        series_b: Second event series
        window: Sliding window size
        lag_window: Maximum lag within each window
        
    Returns:
        time_points: Time indices for sync values
        sync_rates: Synchronization rates over time
        optimal_lags: Optimal lag at each time point
    """
    T = len(series_a)
    if T < window:
        raise ValueError(f"Series length ({T}) must be >= window size ({window})")
    
    n_windows = T - window + 1
    sync_rates = np.zeros(n_windows)
    optimal_lags = np.zeros(n_windows, dtype=np.int32)
    
    for i in range(n_windows):
        # Extract window
        window_a = series_a[i:i+window].astype(np.float64)
        window_b = series_b[i:i+window].astype(np.float64)
        
        # Calculate sync for this window
        _, _, max_sync, opt_lag = calculate_sync_profile_jit(
            window_a, window_b, lag_window
        )
        
        sync_rates[i] = max_sync
        optimal_lags[i] = opt_lag
    
    # Time points are window centers
    time_points = np.arange(window//2, T - window//2 + 1)
    
    return time_points, sync_rates, optimal_lags


def sync_matrix(
    event_series_dict: Dict[str, np.ndarray],
    lag_window: int = LAG_WINDOW_DEFAULT,
    features_dict: Optional[Dict[str, Lambda3FeatureSet]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Create synchronization rate matrix (σₛ matrix) for all series pairs.
    
    Args:
        event_series_dict: Dictionary of event series
        lag_window: Maximum lag for synchronization
        features_dict: Optional pre-computed features
        
    Returns:
        sync_mat: N×N synchronization matrix (σₛ matrix)
        series_names: List of series names
    """
    series_names = list(event_series_dict.keys())
    n = len(series_names)
    sync_mat = np.zeros((n, n))
    
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i == j:
                sync_mat[i, j] = 1.0  # Perfect self-synchronization
            else:
                # Use features if available, otherwise use raw events
                if features_dict:
                    events_a = features_dict[name_a].delta_LambdaC_pos
                    events_b = features_dict[name_b].delta_LambdaC_pos
                else:
                    events_a = event_series_dict[name_a]
                    events_b = event_series_dict[name_b]
                
                profile = calculate_sync_profile(events_a, events_b, lag_window)
                sync_mat[i, j] = profile.max_sync_rate
    
    return sync_mat, series_names


def build_sync_network(
    features_dict: Dict[str, Lambda3FeatureSet],
    lag_window: int = LAG_WINDOW_DEFAULT,
    sync_threshold: float = 0.3,
    event_type: str = 'pos'
) -> nx.DiGraph:
    """
    Build directed synchronization network from features.
    
    Creates a network where edges represent significant synchronization (σₛ > threshold).
    
    Args:
        features_dict: Dictionary of Lambda³ features
        lag_window: Maximum lag for synchronization
        sync_threshold: Minimum sync rate for edge creation
        event_type: 'pos', 'neg', or 'both' for event selection
        
    Returns:
        G: NetworkX directed graph
    """
    G = nx.DiGraph()
    series_names = list(features_dict.keys())
    
    # Add nodes with feature statistics
    for name, features in features_dict.items():
        G.add_node(
            name,
            n_pos_jumps=features.n_pos_jumps,
            n_neg_jumps=features.n_neg_jumps,
            mean_tension=features.mean_tension,
            lambda3_stats={
                'ΔΛC⁺': features.n_pos_jumps,
                'ΔΛC⁻': features.n_neg_jumps,
                'mean_ρT': features.mean_tension
            }
        )
    
    # Add edges based on synchronization
    edge_count = 0
    for name_a in series_names:
        for name_b in series_names:
            if name_a == name_b:
                continue
            
            # Select event type
            if event_type == 'pos':
                events_a = features_dict[name_a].delta_LambdaC_pos
                events_b = features_dict[name_b].delta_LambdaC_pos
            elif event_type == 'neg':
                events_a = features_dict[name_a].delta_LambdaC_neg
                events_b = features_dict[name_b].delta_LambdaC_neg
            else:  # both
                events_a = (features_dict[name_a].delta_LambdaC_pos + 
                           features_dict[name_a].delta_LambdaC_neg)
                events_b = (features_dict[name_b].delta_LambdaC_pos + 
                           features_dict[name_b].delta_LambdaC_neg)
            
            # Calculate synchronization
            profile = calculate_sync_profile(
                events_a, events_b, lag_window,
                series_names=(name_a, name_b)
            )
            
            if profile.max_sync_rate >= sync_threshold:
                G.add_edge(
                    name_a, name_b,
                    weight=profile.max_sync_rate,
                    lag=profile.optimal_lag,
                    sync_profile=profile,
                    sigma_s=profile.max_sync_rate  # σₛ annotation
                )
                edge_count += 1
    
    # Add graph attributes with Lambda³ annotations
    G.graph['sync_threshold'] = sync_threshold
    G.graph['lag_window'] = lag_window
    G.graph['event_type'] = event_type
    G.graph['n_edges'] = edge_count
    G.graph['lambda3_description'] = f'Synchronization network (σₛ > {sync_threshold})'
    
    return G


# ===============================
# Multi-scale Analysis
# ===============================

def extract_multiscale_features(
    data: np.ndarray,
    scales: List[int] = [5, 10, 20, 50],
    config: Optional[FeatureConfig] = None
) -> Dict[int, Lambda3FeatureSet]:
    """
    Extract Lambda³ features at multiple time scales.
    
    Multi-scale analysis reveals scale-dependent structural patterns in Λ.
    
    Args:
        data: Original time series
        scales: List of window sizes for multi-scale analysis
        config: Feature configuration
        
    Returns:
        Dict mapping scale to features
    """
    if config is None:
        config = FeatureConfig()
    
    multiscale_features = {}
    
    for scale in scales:
        # Create scale-specific config
        scale_config = FeatureConfig(
            window=scale,
            local_window=scale,
            delta_percentile=config.delta_percentile,
            local_jump_percentile=config.local_jump_percentile,
            lag_window=config.lag_window
        )
        
        # Extract features at this scale
        features = extract_lambda3_features(data, scale_config)
        features.metadata['scale'] = scale
        features.metadata['scale_description'] = f'Lambda³ features at scale τ={scale}'
        multiscale_features[scale] = features
    
    return multiscale_features


def detect_scale_breaks(
    multiscale_features: Dict[int, Lambda3FeatureSet],
    threshold_std: float = 2.0
) -> Dict[int, List[int]]:
    """
    Detect scale-dependent structural breaks.
    
    Identifies scale-specific ΔΛC singularities based on ρT outliers.
    
    Args:
        multiscale_features: Features at different scales
        threshold_std: Number of standard deviations for break detection
        
    Returns:
        Dict mapping scale to list of break indices
    """
    scale_breaks = {}
    
    for scale, features in multiscale_features.items():
        # Use tension scalar (ρT) for break detection
        rho_t = features.rho_T
        
        # Detect outliers in tension
        mean_tension = np.mean(rho_t)
        std_tension = np.std(rho_t)
        threshold = mean_tension + threshold_std * std_tension
        
        # Find break points (ρT singularities)
        breaks = np.where(rho_t > threshold)[0]
        
        # Filter consecutive breaks (keep first)
        if len(breaks) > 0:
            filtered_breaks = [breaks[0]]
            for b in breaks[1:]:
                if b - filtered_breaks[-1] > scale:  # Minimum separation
                    filtered_breaks.append(b)
            scale_breaks[scale] = filtered_breaks
        else:
            scale_breaks[scale] = []
    
    return scale_breaks
