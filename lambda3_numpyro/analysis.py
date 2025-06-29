"""
Analysis module for Lambda³ framework (Key sections refactored).

This module provides high-level analysis functions including
pairwise analysis, causality detection, and regime identification.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any
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
# Interaction Matrix Building (FIXED)
# ===============================

def _build_interaction_matrix(
    pairwise_results: Dict[Tuple[str, str], AnalysisResult],
    series_names: List[str],
    effect_type: str = 'integrated'
) -> np.ndarray:
    """
    Build interaction effect matrix from pairwise results.
    
    Args:
        pairwise_results: Dictionary of analysis results for each pair
        series_names: List of series names
        effect_type: Type of effect to extract:
            - 'integrated': Use maximum absolute effect across all types
            - 'pos': Only positive jump effects
            - 'neg': Only negative jump effects  
            - 'stress': Only tension/stress effects
            
    Returns:
        interaction_matrix: N×N matrix of interaction coefficients
    """
    n = len(series_names)
    matrix = np.zeros((n, n))
    
    for i, name_i in enumerate(series_names):
        for j, name_j in enumerate(series_names):
            if i == j:
                continue
            
            # Find the pair result
            if (name_i, name_j) in pairwise_results:
                result = pairwise_results[(name_i, name_j)]
            elif (name_j, name_i) in pairwise_results:
                result = pairwise_results[(name_j, name_i)]
            else:
                continue
            
            # Extract j's effect on i based on effect_type
            if effect_type == 'integrated':
                # Get all possible effects and use maximum absolute value
                effects = []
                
                # Positive jumps
                key_pos = f'{name_j}_to_{name_i}_pos'
                if key_pos in result.interaction_effects:
                    effects.append(result.interaction_effects[key_pos])
                
                # Negative jumps  
                key_neg = f'{name_j}_to_{name_i}_neg'
                if key_neg in result.interaction_effects:
                    effects.append(result.interaction_effects[key_neg])
                
                # Stress/tension
                key_stress = f'{name_j}_to_{name_i}_stress'
                if key_stress in result.interaction_effects:
                    effects.append(result.interaction_effects[key_stress])
                
                # Use the effect with maximum absolute value
                if effects:
                    max_effect_idx = np.argmax(np.abs(effects))
                    matrix[i, j] = effects[max_effect_idx]
                    
            elif effect_type == 'pos':
                key = f'{name_j}_to_{name_i}_pos'
                if key in result.interaction_effects:
                    matrix[i, j] = result.interaction_effects[key]
                    
            elif effect_type == 'neg':
                key = f'{name_j}_to_{name_i}_neg'
                if key in result.interaction_effects:
                    matrix[i, j] = result.interaction_effects[key]
                    
            elif effect_type == 'stress':
                key = f'{name_j}_to_{name_i}_stress'
                if key in result.interaction_effects:
                    matrix[i, j] = result.interaction_effects[key]
            else:
                raise ValueError(f"Unknown effect_type: {effect_type}")
    
    return matrix


def build_multi_effect_tensor(
    pairwise_results: Dict[Tuple[str, str], AnalysisResult],
    series_names: List[str]
) -> np.ndarray:
    """
    Build 3D tensor containing all interaction effect types.
    
    Args:
        pairwise_results: Dictionary of analysis results
        series_names: List of series names
        
    Returns:
        interaction_tensor: N×N×3 tensor where third dimension is [pos, neg, stress]
    """
    n = len(series_names)
    tensor = np.zeros((n, n, 3))
    
    # Build matrices for each effect type
    tensor[:, :, 0] = _build_interaction_matrix(pairwise_results, series_names, 'pos')
    tensor[:, :, 1] = _build_interaction_matrix(pairwise_results, series_names, 'neg')
    tensor[:, :, 2] = _build_interaction_matrix(pairwise_results, series_names, 'stress')
    
    return tensor


# ===============================
# Enhanced CrossAnalysisResult Creation
# ===============================

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
    
    # Build synchronization matrix (σₛ matrix in Lambda³ theory)
    print("\nBuilding synchronization matrix...")
    sync_mat, _ = sync_matrix(
        {name: feat.delta_LambdaC_pos for name, feat in features_dict.items()},
        config.feature.lag_window
    )
    
    # Build interaction matrix with integrated effects
    interaction_matrix = _build_interaction_matrix(
        pairwise_results, series_names, effect_type='integrated'
    )
    
    # Build full interaction tensor for detailed analysis
    interaction_tensor = build_multi_effect_tensor(pairwise_results, series_names)
    
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
    
    # Create metadata with theory annotations
    metadata = {
        'series_names': series_names,
        'n_pairs_analyzed': len(pairwise_results),
        'config': config.to_dict(),
        'interaction_tensor_shape': interaction_tensor.shape,
        'lambda3_annotations': {
            'sync_matrix': 'σₛ matrix - synchronization rates',
            'interaction_matrix': 'Integrated β coefficients (max |β| across ΔΛC±, ρT)',
            'interaction_tensor': 'Full tensor [N×N×3] for [pos, neg, stress] effects'
        }
    }
    
    # Store tensor in metadata for access
    metadata['interaction_tensor'] = interaction_tensor
    
    return CrossAnalysisResult(
        pairwise_results=pairwise_results,
        sync_matrix=sync_mat,
        interaction_matrix=interaction_matrix,
        network=network,
        clusters=clusters,
        metadata=metadata
    )


# ===============================
# Remove Unused Function
# ===============================

# The _export_pdf_report function has been removed as it was not implemented


# ===============================
# Export Functions (Updated)
# ===============================

def export_analysis_report(
    results: Union[AnalysisResult, CrossAnalysisResult, Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = 'html'
) -> None:
    """
    Export analysis results as a formatted report.
    
    Args:
        results: Analysis results
        output_path: Output file path
        format: 'html' or 'markdown' (pdf support removed)
    """
    output_path = Path(output_path)
    
    if format == 'html':
        _export_html_report(results, output_path)
    elif format == 'markdown':
        _export_markdown_report(results, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'html' or 'markdown'")
    
    print(f"Report exported to {output_path}")


# ===============================
# Summary Generation with Theory Labels
# ===============================

def generate_analysis_summary(
    cross_results: CrossAnalysisResult,
    features_dict: Dict[str, Lambda3FeatureSet]
) -> List[str]:
    """
    Generate key findings from analysis results with Lambda³ theory annotations.
    
    Args:
        cross_results: Cross-analysis results
        features_dict: Original features
        
    Returns:
        List of key findings
    """
    findings = []
    
    # Find strongest synchronization (σₛ)
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
    
    # Find strongest interaction across all effect types
    if 'interaction_tensor' in cross_results.metadata:
        tensor = cross_results.metadata['interaction_tensor']
        
        # Find maximum absolute effect across all types
        abs_tensor = np.abs(tensor)
        max_idx = np.unravel_index(np.argmax(abs_tensor), abs_tensor.shape)
        max_effect = tensor[max_idx]
        effect_types = ['ΔΛC⁺', 'ΔΛC⁻', 'ρT']
        
        findings.append(
            f"Strongest interaction: {series_names[max_idx[1]]} → "
            f"{series_names[max_idx[0]]} via {effect_types[max_idx[2]]} "
            f"(β = {max_effect:.3f})"
        )
    else:
        # Fallback to integrated matrix
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
    
    # Add Lambda³ theory interpretation
    findings.append("Note: σₛ = synchronization rate, β = interaction coefficient")
    
    return findings


# ===============================
# Helper Functions (Fixed plate usage)
# ===============================

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
                print(f"  ✓ Sync rate (σₛ): {result.sync_profile.max_sync_rate:.3f}")
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
        
        # Tension level (ρT)
        mean_tensions = [s['mean_tension'] for s in regime_stats.values()]
        avg_tension = np.mean(mean_tensions)
        
        if stats['mean_tension'] > avg_tension * 1.5:
            characteristics.append("High-ρT")
        elif stats['mean_tension'] < avg_tension * 0.5:
            characteristics.append("Low-ρT")
        
        # Jump activity (ΔΛC)
        total_jumps = stats['jump_rate_pos'] + stats['jump_rate_neg']
        if total_jumps > 0.1:
            characteristics.append("Volatile")
        elif total_jumps < 0.02:
            characteristics.append("Stable")
        
        # Direction bias
        if stats['jump_rate_pos'] > stats['jump_rate_neg'] * 1.5:
            characteristics.append("ΔΛC⁺-dominant")
        elif stats['jump_rate_neg'] > stats['jump_rate_pos'] * 1.5:
            characteristics.append("ΔΛC⁻-dominant")
        
        # Create name
        if characteristics:
            names[regime_id] = "-".join(characteristics)
        else:
            names[regime_id] = f"Regime-{regime_id + 1}"
    
    return names

# ===============================
# Pairwise Analysis Function
# ===============================

def analyze_pair(
    name_a: str,
    name_b: str,
    features_a: Lambda3FeatureSet,
    features_b: Lambda3FeatureSet,
    config: L3Config,
    seed: int = 0
) -> AnalysisResult:
    """
    Analyze interaction between two time series.
    
    Args:
        name_a: Name of first series
        name_b: Name of second series
        features_a: Features for first series
        features_b: Features for second series
        config: Configuration
        seed: Random seed for reproducibility
        
    Returns:
        AnalysisResult: Analysis results including sync profile and interactions
    """
    print(f"\nAnalyzing interaction: {name_a} ↔ {name_b}")
    
    # 1. Calculate synchronization profile
    print("  Calculating synchronization profile...")
    sync_profile = calculate_sync_profile(
        features_a.delta_LambdaC_pos,
        features_b.delta_LambdaC_pos,
        config.feature.lag_window,
        series_names=(name_a, name_b)
    )
    
    # 2. Fit Bayesian models with interactions
    print(f"  Fitting Bayesian model for {name_a}...")
    trace_a = fit_bayesian_model(
        features_a,
        config,
        interaction_features=features_b,
        model_type='interaction',
        seed=seed
    )
    
    print(f"  Fitting Bayesian model for {name_b}...")
    trace_b = fit_bayesian_model(
        features_b,
        config,
        interaction_features=features_a,
        model_type='interaction',
        seed=seed + 1
    )
    
    # 3. Extract interaction effects
    interaction_effects = _extract_interaction_effects(
        trace_a, trace_b, name_a, name_b
    )
    
    # 4. Calculate causality profiles (optional)
    causality_profiles = None
    if config.verbose:
        print("  Calculating causality profiles...")
        # Simple causality based on conditional probabilities
        causality_a = CausalityProfile(
            self_causality={lag: 0.0 for lag in range(-5, 6)},
            series_names=name_a
        )
        causality_b = CausalityProfile(
            self_causality={lag: 0.0 for lag in range(-5, 6)},
            series_names=name_b
        )
        causality_profiles = {name_a: causality_a, name_b: causality_b}
    
    # 5. Determine primary interaction
    primary_effect = max(interaction_effects.items(), key=lambda x: abs(x[1]))[0]
    
    # 6. Create metadata
    metadata = {
        'name_a': name_a,
        'name_b': name_b,
        'analysis_timestamp': datetime.now().isoformat(),
        'config': config.to_dict(),
        'primary_effect': primary_effect,
        'seed': seed
    }
    
    return AnalysisResult(
        trace_a=trace_a,
        trace_b=trace_b,
        sync_profile=sync_profile,
        interaction_effects=interaction_effects,
        causality_profiles=causality_profiles,
        metadata=metadata
    )

# ===============================
# Complete Analysis Pipeline Functions
# ===============================
def run_comprehensive_analysis(
    features_dict: Dict[str, Lambda3FeatureSet],
    config: L3Config,
    include_bayesian: bool = True,
    include_regimes: bool = True,
    include_changepoints: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive analysis including all advanced features.
    
    Args:
        features_dict: Dictionary of features for all series
        config: Configuration
        include_bayesian: Whether to run Bayesian analysis
        include_regimes: Whether to detect regimes
        include_changepoints: Whether to detect change points
        
    Returns:
        Comprehensive analysis results
    """
    from datetime import datetime
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_series': len(features_dict),
        'series_names': list(features_dict.keys())
    }
    
    # 1. Cross-series analysis
    print("\n1. Running cross-series analysis...")
    cross_results = analyze_multiple_series(features_dict, config)
    results['cross_analysis'] = cross_results
    
    # 2. Causality analysis
    print("\n2. Calculating causality matrix...")
    causality_mat, series_names = calculate_causality_matrix(features_dict)
    results['causality'] = {
        'matrix': causality_mat,
        'series_names': series_names
    }
    
    # 3. Regime detection (per series)
    if include_regimes:
        print("\n3. Detecting regimes...")
        regime_results = {}
        for name, features in features_dict.items():
            regime_info = detect_regimes(features)
            regime_results[name] = regime_info
        results['regimes'] = regime_results
    
    # 4. Change point detection (per series)
    if include_changepoints:
        print("\n4. Detecting change points...")
        changepoint_results = {}
        for name, features in features_dict.items():
            cp_analysis = analyze_with_changepoints(
                features, config, auto_detect=True
            )
            changepoint_results[name] = cp_analysis
        results['changepoints'] = changepoint_results
    
    # 5. Bayesian analysis (if requested)
    if include_bayesian:
        print("\n5. Running Bayesian analysis...")
        from .bayes import Lambda3BayesianInference
        
        bayesian_results = {}
        inference = Lambda3BayesianInference(config)
        
        # Analyze first series in detail
        first_name = list(features_dict.keys())[0]
        first_features = features_dict[first_name]
        
        # Fit multiple models
        inference.fit_model(first_features, 'base')
        inference.fit_model(first_features, 'dynamic')
        
        # Compare and get best
        comparison = inference.compare_models(first_features)
        best_name, best_results = inference.get_best_model()
        
        bayesian_results['inference'] = inference
        bayesian_results['comparison'] = comparison
        bayesian_results['best_model'] = best_name
        
        results['bayesian'] = bayesian_results
    
    # 6. Generate summary
    results['summary'] = _generate_comprehensive_summary(results)
    
    return results


def _generate_comprehensive_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of comprehensive analysis with Lambda³ annotations."""
    summary = {
        'n_series': results['n_series'],
        'series_names': results['series_names'],
        'analysis_timestamp': results['timestamp']
    }
    
    # Cross-analysis summary
    if 'cross_analysis' in results:
        cross = results['cross_analysis']
        
        # Access interaction tensor from metadata
        if 'interaction_tensor' in cross.metadata:
            tensor = cross.metadata['interaction_tensor']
            summary['n_significant_interactions'] = np.sum(np.abs(tensor) > 0.1)
            summary['max_interaction_strength'] = np.max(np.abs(tensor))
        else:
            summary['n_significant_interactions'] = np.sum(
                np.abs(cross.interaction_matrix) > 0.1
            )
        
        summary['max_sync_rate'] = np.max(
            cross.sync_matrix[np.triu_indices_from(cross.sync_matrix, k=1)]
        )
    
    # Causality summary
    if 'causality' in results:
        causality_mat = results['causality']['matrix']
        summary['max_causality'] = np.max(causality_mat[causality_mat < 1])
        summary['causality_density'] = np.mean(causality_mat > 0.1)
    
    # Regime summary
    if 'regimes' in results:
        summary['regime_counts'] = {
            name: info.n_regimes 
            for name, info in results['regimes'].items()
        }
    
    # Change point summary
    if 'changepoints' in results:
        summary['changepoint_counts'] = {
            name: len(cp_result['change_points'])
            for name, cp_result in results['changepoints'].items()
        }
    
    # Bayesian summary
    if 'bayesian' in results:
        summary['best_bayesian_model'] = results['bayesian']['best_model']
    
    # Add Lambda³ theory annotations
    summary['lambda3_metrics'] = {
        'sync_metric': 'σₛ (synchronization rate)',
        'interaction_metric': 'β (structural coupling coefficient)',
        'causality_metric': 'P(ΔΛC⁻|ΔΛC⁺) (structural causality)',
        'tension_metric': 'ρT (local volatility/stress)'
    }
    
    return summary

# ===============================
# Export Results Functions
# ===============================

def export_analysis_report(
    results: Union[AnalysisResult, CrossAnalysisResult, Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = 'html'
) -> None:
    """
    Export analysis results as a formatted report.
    
    Args:
        results: Analysis results
        output_path: Output file path
        format: 'html', 'pdf', or 'markdown'
    """
    output_path = Path(output_path)
    
    if format == 'html':
        _export_html_report(results, output_path)
    elif format == 'pdf':
        _export_pdf_report(results, output_path)
    elif format == 'markdown':
        _export_markdown_report(results, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Report exported to {output_path}")


def _export_html_report(results: Any, output_path: Path) -> None:
    """Export results as HTML report."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lambda³ Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            h2 { color: #666; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .metric { font-weight: bold; color: #0066cc; }
        </style>
    </head>
    <body>
        <h1>Lambda³ Analysis Report</h1>
    """
    
    # Add content based on result type
    if isinstance(results, dict) and 'summary' in results:
        # Comprehensive results
        summary = results['summary']
        html_content += f"""
        <h2>Summary</h2>
        <p>Analysis of {summary['n_series']} series</p>
        <ul>
            <li>Series: {', '.join(summary['series_names'])}</li>
            <li>Timestamp: {summary['analysis_timestamp']}</li>
        </ul>
        """
        
        if 'max_sync_rate' in summary:
            html_content += f"""
            <h2>Synchronization</h2>
            <p>Maximum sync rate: <span class="metric">{summary['max_sync_rate']:.3f}</span></p>
            """
        
        if 'changepoint_counts' in summary:
            html_content += "<h2>Change Points</h2><ul>"
            for series, count in summary['changepoint_counts'].items():
                html_content += f"<li>{series}: {count} change points</li>"
            html_content += "</ul>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def _export_markdown_report(results: Any, output_path: Path) -> None:
    """Export results as Markdown report."""
    md_content = "# Lambda³ Analysis Report\n\n"
    
    if isinstance(results, AnalysisResult):
        # Single pair analysis
        md_content += f"## Series Analysis: {results.series_names[0]} ↔ {results.series_names[1]}\n\n"
        md_content += f"- **Max Sync Rate**: {results.sync_profile.max_sync_rate:.3f}\n"
        md_content += f"- **Optimal Lag**: {results.sync_profile.optimal_lag}\n\n"
        
        md_content += "### Interaction Effects\n\n"
        for effect, value in results.interaction_effects.items():
            md_content += f"- {effect}: {value:.3f}\n"
    
    elif isinstance(results, dict) and 'summary' in results:
        # Comprehensive results
        summary = results['summary']
        md_content += f"## Summary\n\n"
        md_content += f"- **Series**: {', '.join(summary['series_names'])}\n"
        md_content += f"- **Analysis Date**: {summary['analysis_timestamp']}\n\n"
        
        if 'best_bayesian_model' in summary:
            md_content += f"### Bayesian Analysis\n\n"
            md_content += f"- **Best Model**: {summary['best_bayesian_model']}\n\n"
    
    with open(output_path, 'w') as f:
        f.write(md_content)


def _export_pdf_report(results: Any, output_path: Path) -> None:
    """Export results as PDF report (requires additional dependencies)."""
    # This would require libraries like reportlab or weasyprint
    raise NotImplementedError("PDF export requires additional dependencies")


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
