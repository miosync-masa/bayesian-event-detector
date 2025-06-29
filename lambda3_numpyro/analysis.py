"""
Analysis module for Lambda³ framework.

This module provides high-level analysis functions including
pairwise analysis, causality detection, and regime identification.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering, KMeans
import networkx as nx

from .config import L3Config
from .types import (
    Lambda3FeatureSet, AnalysisResult, CrossAnalysisResult,
    SyncProfile, CausalityProfile, BayesianResults, RegimeInfo
)
from .feature import calculate_sync_profile, sync_matrix, build_sync_network
from .bayes import fit_bayesian_model, check_convergence


# ===============================
# Pairwise Analysis
# ===============================

def analyze_pair(
    name_a: str,
    name_b: str, 
    features_a: Lambda3FeatureSet,
    features_b: Lambda3FeatureSet,
    config: L3Config,
    seed: int = 0,
    check_convergence_flag: bool = True
) -> AnalysisResult:
    """
    Comprehensive analysis of a series pair.
    
    This function:
    1. Fits Bayesian models with cross-interactions
    2. Calculates synchronization profiles
    3. Detects causality patterns
    4. Extracts interaction effects
    
    Args:
        name_a, name_b: Series names
        features_a, features_b: Extracted features
        config: Configuration
        seed: Random seed
        check_convergence_flag: Whether to check MCMC convergence
        
    Returns:
        AnalysisResult: Complete analysis results
    """
    print(f"\nAnalyzing pair: {name_a} ↔ {name_b}")
    
    # Fit model for series A (with B's influence)
    print(f"  Fitting model for {name_a} (with {name_b} interactions)...")
    results_a = fit_bayesian_model(
        features=features_a,
        config=config,
        interaction_features=features_b,
        model_type='interaction',
        seed=seed * 1000
    )
    
    if check_convergence_flag:
        check_convergence(results_a, verbose=False)
    
    # Fit model for series B (with A's influence)
    print(f"  Fitting model for {name_b} (with {name_a} interactions)...")
    results_b = fit_bayesian_model(
        features=features_b,
        config=config,
        interaction_features=features_a,
        model_type='interaction',
        seed=seed * 1000 + 1
    )
    
    if check_convergence_flag:
        check_convergence(results_b, verbose=False)
    
    # Calculate synchronization profile
    sync_profile = calculate_sync_profile(
        features_a.delta_LambdaC_pos,
        features_b.delta_LambdaC_pos,
        config.feature.lag_window,
        series_names=(name_a, name_b)
    )
    
    # Extract interaction effects
    interaction_effects = _extract_interaction_effects(
        results_a, results_b, name_a, name_b
    )
    
    # Calculate causality profiles
    causality_profiles = {
        name_a: _calculate_causality_profile(features_a, features_b, name_a, name_b),
        name_b: _calculate_causality_profile(features_b, features_a, name_b, name_a)
    }
    
    # Create metadata
    metadata = {
        'name_a': name_a,
        'name_b': name_b,
        'seed': seed,
        'sync_rate': sync_profile.max_sync_rate,
        'optimal_lag': sync_profile.optimal_lag,
        'primary_effect': max(interaction_effects.items(), key=lambda x: abs(x[1]))[0] if interaction_effects else None
    }
    
    return AnalysisResult(
        trace_a=results_a,
        trace_b=results_b,
        sync_profile=sync_profile,
        interaction_effects=interaction_effects,
        causality_profiles=causality_profiles,
        metadata=metadata
    )


def analyze_multiple_series(
    features_dict: Dict[str, Lambda3FeatureSet],
    config: L3Config,
    show_progress: bool = True,
    parallel: bool = False
) -> CrossAnalysisResult:
    """
    Analyze all pairs in a collection of series.
    
    Args:
        features_dict: Dictionary of features for all series
        config: Configuration
        show_progress: Whether to show progress
        parallel: Whether to use parallel processing
        
    Returns:
        CrossAnalysisResult: Complete cross-analysis results
    """
    series_names = list(features_dict.keys())
    n_series = len(series_names)
    
    if n_series < 2:
        raise ValueError("Need at least 2 series for cross-analysis")
    
    # Generate all pairs
    pairs = list(combinations(series_names, 2))
    
    # Limit pairs if specified
    if config.max_pairs and len(pairs) > config.max_pairs:
        print(f"Limiting analysis to {config.max_pairs} pairs (from {len(pairs)} total)")
        pairs = pairs[:config.max_pairs]
    
    print(f"\n{'='*60}")
    print(f"CROSS-SERIES ANALYSIS")
    print(f"Series: {', '.join(series_names)}")
    print(f"Total pairs: {len(pairs)}")
    print(f"{'='*60}")
    
    # Analyze pairs
    if parallel and config.parallel_pairs:
        pairwise_results = _analyze_pairs_parallel(
            pairs, features_dict, config, show_progress
        )
    else:
        pairwise_results = _analyze_pairs_sequential(
            pairs, features_dict, config, show_progress
        )
    
    # Build synchronization matrix
    print("\nBuilding synchronization matrix...")
    sync_mat, _ = sync_matrix(
        {name: feat.delta_LambdaC_pos for name, feat in features_dict.items()},
        config.feature.lag_window
    )
    
    # Build interaction matrix
    interaction_matrix = _build_interaction_matrix(pairwise_results, series_names)
    
    # Build synchronization network
    print("Building synchronization network...")
    network = build_sync_network(
        features_dict,
        config.feature.lag_window,
        sync_threshold=0.3
    )
    
    # Cluster series
    clusters = None
    if n_series > 2:
        clusters = _cluster_series(sync_mat, series_names, n_clusters=min(3, n_series // 2))
    
    # Create metadata
    metadata = {
        'series_names': series_names,
        'n_pairs_analyzed': len(pairwise_results),
        'config': config.to_dict()
    }
    
    return CrossAnalysisResult(
        pairwise_results=pairwise_results,
        sync_matrix=sync_mat,
        interaction_matrix=interaction_matrix,
        network=network,
        clusters=clusters,
        metadata=metadata
    )


# ===============================
# Causality Analysis
# ===============================

def _calculate_causality_profile(
    features_self: Lambda3FeatureSet,
    features_other: Lambda3FeatureSet,
    name_self: str,
    name_other: str,
    max_lag: int = 10
) -> CausalityProfile:
    """
    Calculate causality profile for a series.
    
    Args:
        features_self: Features of primary series
        features_other: Features of other series
        name_self: Name of primary series
        name_other: Name of other series
        max_lag: Maximum lag to consider
        
    Returns:
        CausalityProfile: Causality analysis results
    """
    # Self-causality: P(negative jump | positive jump)
    self_causality = {}
    pos_jumps = features_self.delta_LambdaC_pos
    neg_jumps = features_self.delta_LambdaC_neg
    
    for lag in range(1, max_lag + 1):
        count_pos = 0
        count_pairs = 0
        
        for t in range(len(pos_jumps) - lag):
            if pos_jumps[t] > 0:
                count_pos += 1
                if neg_jumps[t + lag] > 0:
                    count_pairs += 1
        
        self_causality[lag] = count_pairs / max(count_pos, 1)
    
    # Cross-causality: P(self event | other event)
    cross_causality = {}
    other_pos = features_other.delta_LambdaC_pos
    
    for lag in range(1, max_lag + 1):
        count_other = 0
        count_cross = 0
        
        for t in range(len(other_pos) - lag):
            if other_pos[t] > 0:
                count_other += 1
                if pos_jumps[t + lag] > 0:
                    count_cross += 1
        
        cross_causality[lag] = count_cross / max(count_other, 1)
    
    return CausalityProfile(
        self_causality=self_causality,
        cross_causality=cross_causality,
        series_names=(name_self, name_other)
    )


def calculate_causality_matrix(
    features_dict: Dict[str, Lambda3FeatureSet],
    lag: int = 1,
    event_type: str = 'pos'
) -> Tuple[np.ndarray, List[str]]:
    """
    Calculate causality matrix for all series.
    
    Args:
        features_dict: Features for all series
        lag: Time lag for causality
        event_type: 'pos' or 'neg' jumps
        
    Returns:
        causality_matrix: N×N matrix of causality strengths
        series_names: List of series names
    """
    series_names = list(features_dict.keys())
    n = len(series_names)
    causality_mat = np.zeros((n, n))
    
    for i, name_i in enumerate(series_names):
        for j, name_j in enumerate(series_names):
            if i == j:
                # Self-causality on diagonal
                features = features_dict[name_i]
                if event_type == 'pos':
                    events = features.delta_LambdaC_pos
                    response = features.delta_LambdaC_neg
                else:
                    events = features.delta_LambdaC_neg
                    response = features.delta_LambdaC_pos
                
                count_events = 0
                count_response = 0
                
                for t in range(len(events) - lag):
                    if events[t] > 0:
                        count_events += 1
                        if response[t + lag] > 0:
                            count_response += 1
                
                causality_mat[i, j] = count_response / max(count_events, 1)
            else:
                # Cross-causality
                features_cause = features_dict[name_j]
                features_effect = features_dict[name_i]
                
                if event_type == 'pos':
                    cause_events = features_cause.delta_LambdaC_pos
                    effect_events = features_effect.delta_LambdaC_pos
                else:
                    cause_events = features_cause.delta_LambdaC_neg
                    effect_events = features_effect.delta_LambdaC_neg
                
                count_cause = 0
                count_effect = 0
                
                for t in range(len(cause_events) - lag):
                    if cause_events[t] > 0:
                        count_cause += 1
                        if effect_events[t + lag] > 0:
                            count_effect += 1
                
                causality_mat[i, j] = count_effect / max(count_cause, 1)
    
    return causality_mat, series_names


# ===============================
# Regime Detection
# ===============================

def detect_regimes(
    features: Lambda3FeatureSet,
    n_regimes: int = 3,
    method: str = 'kmeans',
    features_to_use: Optional[List[str]] = None
) -> RegimeInfo:
    """
    Detect market regimes using Lambda³ features.
    
    Args:
        features: Lambda³ features for a series
        n_regimes: Number of regimes to detect
        method: 'kmeans' or 'hierarchical'
        features_to_use: List of feature names to use
        
    Returns:
        RegimeInfo: Regime detection results
    """
    # Select features for clustering
    if features_to_use is None:
        features_to_use = ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T']
    
    # Build feature matrix
    X = []
    for feat_name in features_to_use:
        if feat_name == 'delta_LambdaC_pos':
            X.append(features.delta_LambdaC_pos)
        elif feat_name == 'delta_LambdaC_neg':
            X.append(features.delta_LambdaC_neg)
        elif feat_name == 'rho_T':
            X.append(features.rho_T)
        elif feat_name == 'local_jump':
            X.append(features.local_jump)
    
    X = np.column_stack(X)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    else:  # hierarchical
        clusterer = AgglomerativeClustering(n_clusters=n_regimes)
    
    labels = clusterer.fit_predict(X_scaled)
    
    # Calculate regime statistics
    regime_stats = {}
    for regime in range(n_regimes):
        mask = labels == regime
        if np.any(mask):
            regime_stats[regime] = {
                'frequency': float(np.mean(mask)),
                'mean_tension': float(np.mean(features.rho_T[mask])),
                'jump_rate_pos': float(np.mean(features.delta_LambdaC_pos[mask])),
                'jump_rate_neg': float(np.mean(features.delta_LambdaC_neg[mask])),
                'avg_value': float(np.mean(features.data[mask]))
            }
        else:
            regime_stats[regime] = {
                'frequency': 0.0,
                'mean_tension': 0.0,
                'jump_rate_pos': 0.0,
                'jump_rate_neg': 0.0,
                'avg_value': 0.0
            }
    
    # Create descriptive names based on characteristics
    regime_names = _name_regimes(regime_stats)
    
    return RegimeInfo(
        labels=labels,
        n_regimes=n_regimes,
        regime_stats=regime_stats,
        regime_names=regime_names
    )


def detect_regime_changes(
    features_dict: Dict[str, Lambda3FeatureSet],
    n_regimes: int = 3,
    min_regime_length: int = 10
) -> Dict[str, RegimeInfo]:
    """
    Detect regime changes across multiple series.
    
    Args:
        features_dict: Features for all series
        n_regimes: Number of regimes
        min_regime_length: Minimum consecutive points in a regime
        
    Returns:
        Dict mapping series name to RegimeInfo
    """
    regime_results = {}
    
    for name, features in features_dict.items():
        # Detect regimes
        regime_info = detect_regimes(features, n_regimes)
        
        # Filter short regimes
        labels = regime_info.labels.copy()
        for i in range(1, len(labels) - 1):
            if labels[i] != labels[i-1] and labels[i] != labels[i+1]:
                # Single point regime - assign to previous
                labels[i] = labels[i-1]
        
        # Update regime info with filtered labels
        regime_info.labels = labels
        regime_results[name] = regime_info
    
    return regime_results


# ===============================
# Helper Functions
# ===============================

def _extract_interaction_effects(
    results_a: BayesianResults,
    results_b: BayesianResults,
    name_a: str,
    name_b: str
) -> Dict[str, float]:
    """Extract interaction coefficients from results."""
    effects = {}
    
    # Extract from summary dataframes
    summary_a = results_a.summary
    summary_b = results_b.summary
    
    # B's effect on A
    if 'beta_interact_pos' in summary_a.index:
        effects[f'{name_b}_to_{name_a}_pos'] = summary_a.loc['beta_interact_pos', 'mean']
    if 'beta_interact_neg' in summary_a.index:
        effects[f'{name_b}_to_{name_a}_neg'] = summary_a.loc['beta_interact_neg', 'mean']
    if 'beta_interact_stress' in summary_a.index:
        effects[f'{name_b}_to_{name_a}_stress'] = summary_a.loc['beta_interact_stress', 'mean']
    
    # A's effect on B
    if 'beta_interact_pos' in summary_b.index:
        effects[f'{name_a}_to_{name_b}_pos'] = summary_b.loc['beta_interact_pos', 'mean']
    if 'beta_interact_neg' in summary_b.index:
        effects[f'{name_a}_to_{name_b}_neg'] = summary_b.loc['beta_interact_neg', 'mean']
    if 'beta_interact_stress' in summary_b.index:
        effects[f'{name_a}_to_{name_b}_stress'] = summary_b.loc['beta_interact_stress', 'mean']
    
    return effects


def _analyze_pairs_sequential(
    pairs: List[Tuple[str, str]],
    features_dict: Dict[str, Lambda3FeatureSet],
    config: L3Config,
    show_progress: bool
) -> Dict[Tuple[str, str], AnalysisResult]:
    """Sequential pairwise analysis."""
    results = {}
    
    for i, (name_a, name_b) in enumerate(pairs, 1):
        if show_progress:
            print(f"\n[{i}/{len(pairs)}] Analyzing: {name_a} ↔ {name_b}")
        
        try:
            result = analyze_pair(
                name_a, name_b,
                features_dict[name_a],
                features_dict[name_b],
                config,
                seed=i
            )
            results[(name_a, name_b)] = result
            
            if show_progress:
                print(f"  ✓ Sync rate: {result.sync_profile.max_sync_rate:.3f}")
                print(f"  ✓ Primary effect: {result.metadata.get('primary_effect', 'None')}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return results


def _analyze_pairs_parallel(
    pairs: List[Tuple[str, str]],
    features_dict: Dict[str, Lambda3FeatureSet],
    config: L3Config,
    show_progress: bool
) -> Dict[Tuple[str, str], AnalysisResult]:
    """Parallel pairwise analysis (placeholder for future implementation)."""
    # For now, fall back to sequential
    print("Note: Parallel analysis not yet implemented, using sequential")
    return _analyze_pairs_sequential(pairs, features_dict, config, show_progress)


def _build_interaction_matrix(
    pairwise_results: Dict[Tuple[str, str], AnalysisResult],
    series_names: List[str]
) -> np.ndarray:
    """Build interaction effect matrix from pairwise results."""
    n = len(series_names)
    matrix = np.zeros((n, n))
    
    for i, name_i in enumerate(series_names):
        for j, name_j in enumerate(series_names):
            if i == j:
                continue
            
            # Find the pair result
            if (name_i, name_j) in pairwise_results:
                result = pairwise_results[(name_i, name_j)]
                # j's effect on i
                key = f'{name_j}_to_{name_i}_pos'
                if key in result.interaction_effects:
                    matrix[i, j] = result.interaction_effects[key]
            elif (name_j, name_i) in pairwise_results:
                result = pairwise_results[(name_j, name_i)]
                # j's effect on i
                key = f'{name_j}_to_{name_i}_pos'
                if key in result.interaction_effects:
                    matrix[i, j] = result.interaction_effects[key]
    
    return matrix


def _cluster_series(
    sync_mat: np.ndarray,
    series_names: List[str],
    n_clusters: int
) -> Dict[str, int]:
    """Cluster series based on synchronization patterns."""
    # Use 1 - sync as distance
    distance_mat = 1 - sync_mat
    
    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    
    labels = clustering.fit_predict(distance_mat)
    
    return {name: int(label) for name, label in zip(series_names, labels)}


def _name_regimes(regime_stats: Dict[int, Dict[str, float]]) -> Dict[int, str]:
    """Generate descriptive names for regimes based on characteristics."""
    names = {}
    
    for regime_id, stats in regime_stats.items():
        characteristics = []
        
        # Tension level
        if stats['mean_tension'] > np.mean([s['mean_tension'] for s in regime_stats.values()]) * 1.5:
            characteristics.append("High-Stress")
        elif stats['mean_tension'] < np.mean([s['mean_tension'] for s in regime_stats.values()]) * 0.5:
            characteristics.append("Low-Stress")
        
        # Jump activity
        total_jumps = stats['jump_rate_pos'] + stats['jump_rate_neg']
        if total_jumps > 0.1:
            characteristics.append("Volatile")
        elif total_jumps < 0.02:
            characteristics.append("Stable")
        
        # Direction bias
        if stats['jump_rate_pos'] > stats['jump_rate_neg'] * 1.5:
            characteristics.append("Bullish")
        elif stats['jump_rate_neg'] > stats['jump_rate_pos'] * 1.5:
            characteristics.append("Bearish")
        
        # Create name
        if characteristics:
            names[regime_id] = "-".join(characteristics)
        else:
            names[regime_id] = f"Regime-{regime_id + 1}"
    
    return names


def generate_analysis_summary(
    cross_results: CrossAnalysisResult,
    features_dict: Dict[str, Lambda3FeatureSet]
) -> List[str]:
    """
    Generate key findings from analysis results.
    
    Args:
        cross_results: Cross-analysis results
        features_dict: Original features
        
    Returns:
        List of key findings
    """
    findings = []
    
    # Find strongest synchronization
    sync_mat = cross_results.sync_matrix
    series_names = cross_results.get_series_names()
    
    # Get off-diagonal max
    np.fill_diagonal(sync_mat, 0)
    max_sync_idx = np.unravel_index(np.argmax(sync_mat), sync_mat.shape)
    max_sync = sync_mat[max_sync_idx]
    
    if max_sync > 0.3:
        findings.append(
            f"Strongest synchronization: {series_names[max_sync_idx[0]]} ↔ "
            f"{series_names[max_sync_idx[1]]} (σₛ = {max_sync:.3f})"
        )
    
    # Find strongest interaction
    int_mat = cross_results.interaction_matrix
    np.fill_diagonal(int_mat, 0)
    max_int_idx = np.unravel_index(np.argmax(np.abs(int_mat)), int_mat.shape)
    max_int = int_mat[max_int_idx]
    
    if abs(max_int) > 0.1:
        findings.append(
            f"Strongest interaction: {series_names[max_int_idx[1]]} → "
            f"{series_names[max_int_idx[0]]} (β = {max_int:.3f})"
        )
    
    # Network statistics
    if cross_results.network:
        n_edges = cross_results.network.number_of_edges()
        n_nodes = cross_results.network.number_of_nodes()
        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        
        findings.append(
            f"Network density: {density:.1%} ({n_edges} edges)"
        )
        
        # Find hub nodes
        in_degrees = dict(cross_results.network.in_degree())
        if in_degrees:
            hub = max(in_degrees.items(), key=lambda x: x[1])
            if hub[1] > 1:
                findings.append(f"Primary hub: {hub[0]} (in-degree: {hub[1]})")
    
    # Clustering insights
    if cross_results.clusters:
        n_clusters = len(set(cross_results.clusters.values()))
        findings.append(f"Series grouped into {n_clusters} synchronization clusters")
    
    return findings
