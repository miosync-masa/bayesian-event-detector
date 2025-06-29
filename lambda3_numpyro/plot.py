"""
Visualization module for Lambda³ framework.

This module provides plotting functions for features, analysis results,
and network visualizations. Imports are isolated to make plotting optional.
"""

# Check if plotting libraries are available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

from .types import (
    Lambda3FeatureSet, AnalysisResult, CrossAnalysisResult,
    SyncProfile, RegimeInfo
)
from .config import PlottingConfig


def _check_plotting():
    """Check if plotting libraries are available."""
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "Plotting libraries not available. "
            "Install with: pip install matplotlib seaborn"
        )


# ===============================
# Feature Visualization
# ===============================

def plot_features(
    features: Union[Lambda3FeatureSet, Dict[str, Lambda3FeatureSet]],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    Plot Lambda³ features for one or more series.
    
    Args:
        features: Single feature set or dictionary
        title: Plot title
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    # Set style
    plt.style.use(config.style)
    
    # Handle single vs multiple features
    if isinstance(features, Lambda3FeatureSet):
        features_dict = {'Series': features}
    else:
        features_dict = features
    
    n_series = len(features_dict)
    
    # Create subplots
    fig, axes = plt.subplots(
        n_series, 3,
        figsize=(config.figure_size[0], config.figure_size[1] * n_series / 2),
        sharex=True
    )
    
    if n_series == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each series
    for idx, (name, feat) in enumerate(features_dict.items()):
        # Data and events
        ax = axes[idx, 0]
        time = np.arange(len(feat.data))
        
        # Plot data
        ax.plot(time, feat.data, color='gray', alpha=0.6, linewidth=1)
        
        # Mark jump events
        pos_idx = np.where(feat.delta_LambdaC_pos > 0)[0]
        neg_idx = np.where(feat.delta_LambdaC_neg > 0)[0]
        
        if len(pos_idx) > 0:
            ax.scatter(pos_idx, feat.data[pos_idx], 
                      color='dodgerblue', s=50, alpha=0.8, 
                      label=f'Pos jumps ({len(pos_idx)})')
        
        if len(neg_idx) > 0:
            ax.scatter(neg_idx, feat.data[neg_idx],
                      color='orangered', s=50, alpha=0.8,
                      label=f'Neg jumps ({len(neg_idx)})')
        
        ax.set_ylabel(name)
        if idx == 0:
            ax.set_title('Data & Jump Events')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Tension scalar
        ax = axes[idx, 1]
        ax.plot(time, feat.rho_T, color='green', alpha=0.8)
        ax.fill_between(time, 0, feat.rho_T, color='green', alpha=0.2)
        
        # Mark high tension periods
        high_tension = feat.rho_T > np.percentile(feat.rho_T, 90)
        if np.any(high_tension):
            ax.scatter(time[high_tension], feat.rho_T[high_tension],
                      color='red', s=20, alpha=0.6)
        
        if idx == 0:
            ax.set_title('Tension Scalar (ρT)')
        ax.grid(True, alpha=0.3)
        
        # Event timeline
        ax = axes[idx, 2]
        
        # Create event timeline
        events = np.zeros(len(time))
        events[feat.delta_LambdaC_pos > 0] = 1
        events[feat.delta_LambdaC_neg > 0] = -1
        events[feat.local_jump > 0] = np.where(events[feat.local_jump > 0] == 0, 0.5, events[feat.local_jump > 0])
        
        # Plot as stem plot
        colors = ['orangered' if e < 0 else 'dodgerblue' if e > 0 else 'purple' 
                 for e in events]
        
        for t, e, c in zip(time[events != 0], events[events != 0], 
                          [colors[i] for i in range(len(colors)) if events[i] != 0]):
            ax.vlines(t, 0, e, colors=c, alpha=0.7)
        
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Neg', '0', 'Pos'])
        
        if idx == 0:
            ax.set_title('Event Timeline')
        ax.grid(True, alpha=0.3)
    
    # Set common x-label
    axes[-1, 1].set_xlabel('Time')
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===============================
# Analysis Results Visualization
# ===============================

def plot_analysis_results(
    results: AnalysisResult,
    features_a: Lambda3FeatureSet,
    features_b: Lambda3FeatureSet,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    Plot comprehensive analysis results for a series pair.
    
    Args:
        results: Analysis results
        features_a: Features for first series
        features_b: Features for second series
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    plt.style.use(config.style)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Get series names
    name_a, name_b = results.series_names
    
    # 1. Data with predictions (top row)
    ax1 = fig.add_subplot(gs[0, :])
    _plot_data_with_predictions(ax1, features_a, results.trace_a, name_a)
    
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    _plot_data_with_predictions(ax2, features_b, results.trace_b, name_b)
    
    # 2. Synchronization profile (bottom left)
    ax3 = fig.add_subplot(gs[2, 0])
    _plot_sync_profile(ax3, results.sync_profile)
    
    # 3. Interaction effects (bottom middle)
    ax4 = fig.add_subplot(gs[2, 1])
    _plot_interaction_effects(ax4, results.interaction_effects, name_a, name_b)
    
    # 4. Causality profiles (bottom right)
    ax5 = fig.add_subplot(gs[2, 2])
    if results.causality_profiles:
        _plot_causality_profiles(ax5, results.causality_profiles)
    
    # Overall title
    fig.suptitle(f'Lambda³ Analysis: {name_a} ↔ {name_b}', fontsize=16)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _plot_data_with_predictions(
    ax: Any,
    features: Lambda3FeatureSet,
    bayes_results: Any,
    name: str
) -> None:
    """Plot data with model predictions and events."""
    time = np.arange(len(features.data))
    
    # Plot data
    ax.plot(time, features.data, 'o', color='gray', markersize=3, 
            alpha=0.5, label='Data')
    
    # Plot predictions
    if bayes_results and bayes_results.predictions is not None:
        ax.plot(time, bayes_results.predictions, color='green', 
                linewidth=2, label='Model')
        
        # Confidence interval if available
        if hasattr(bayes_results, 'prediction_interval'):
            lower, upper = bayes_results.prediction_interval
            ax.fill_between(time, lower, upper, color='green', alpha=0.2)
    
    # Mark events
    pos_idx = np.where(features.delta_LambdaC_pos > 0)[0]
    neg_idx = np.where(features.delta_LambdaC_neg > 0)[0]
    
    for idx in pos_idx:
        ax.axvline(x=idx, color='dodgerblue', linestyle='--', alpha=0.3)
    
    for idx in neg_idx:
        ax.axvline(x=idx, color='orangered', linestyle='--', alpha=0.3)
    
    ax.set_ylabel(name)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{name}: Data, Model Fit, and Events')


def _plot_sync_profile(ax: Any, sync_profile: SyncProfile) -> None:
    """Plot synchronization profile."""
    lags = sorted(sync_profile.profile.keys())
    sync_values = [sync_profile.profile[lag] for lag in lags]
    
    # Plot profile
    ax.plot(lags, sync_values, 'o-', color='royalblue', markersize=6)
    
    # Mark optimal lag
    ax.axvline(x=sync_profile.optimal_lag, color='red', linestyle='--', 
               label=f'Optimal lag: {sync_profile.optimal_lag}')
    
    # Mark max sync rate
    ax.axhline(y=sync_profile.max_sync_rate, color='red', linestyle=':', 
               alpha=0.5, label=f'Max σₛ: {sync_profile.max_sync_rate:.3f}')
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Sync Rate (σₛ)')
    ax.set_title('Synchronization Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_interaction_effects(
    ax: Any,
    effects: Dict[str, float],
    name_a: str,
    name_b: str
) -> None:
    """Plot interaction effects as a bar chart."""
    if not effects:
        ax.text(0.5, 0.5, 'No interaction effects detected',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Interaction Effects')
        return
    
    # Extract and organize effects
    labels = []
    values = []
    colors = []
    
    for key, value in effects.items():
        if abs(value) > 0.01:  # Only show significant effects
            # Parse the key
            if f'{name_b}_to_{name_a}' in key:
                direction = f'{name_b}→{name_a}'
            elif f'{name_a}_to_{name_b}' in key:
                direction = f'{name_a}→{name_b}'
            else:
                direction = key
            
            effect_type = key.split('_')[-1]  # pos, neg, or stress
            
            labels.append(f'{direction}\n({effect_type})')
            values.append(value)
            
            # Color based on sign
            colors.append('green' if value > 0 else 'red')
    
    # Create bar plot
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Coefficient (β)')
    ax.set_title('Interaction Effects')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom' if value > 0 else 'top',
                fontsize=8)


def _plot_causality_profiles(
    ax: Any,
    causality_profiles: Dict[str, Any]
) -> None:
    """Plot causality profiles."""
    colors = plt.cm.tab10(np.arange(len(causality_profiles)))
    
    for idx, (name, profile) in enumerate(causality_profiles.items()):
        if hasattr(profile, 'self_causality'):
            lags = sorted(profile.self_causality.keys())
            probs = [profile.self_causality[lag] for lag in lags]
            
            ax.plot(lags, probs, 'o-', color=colors[idx], 
                   label=f'{name} self', markersize=4)
            
            # Also plot cross-causality if available
            if profile.cross_causality:
                cross_probs = [profile.cross_causality[lag] for lag in lags]
                ax.plot(lags, cross_probs, 's--', color=colors[idx],
                       label=f'{name} cross', markersize=4, alpha=0.7)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Causality Probability')
    ax.set_title('Causality Profiles')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)


# ===============================
# Network Visualization
# ===============================

def plot_sync_network(
    network: Any,  # networkx.DiGraph
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None,
    layout: str = 'spring'
) -> None:
    """
    Plot synchronization network.
    
    Args:
        network: NetworkX directed graph
        save_path: Path to save figure
        config: Plotting configuration
        layout: Network layout algorithm
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    import networkx as nx
    
    plt.figure(figsize=config.figure_size)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(network, k=2, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(network)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(network)
    else:
        pos = nx.spring_layout(network)
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in network.nodes():
        # Color by in-degree
        in_deg = network.in_degree(node)
        node_colors.append(in_deg)
        # Size by out-degree
        out_deg = network.out_degree(node)
        node_sizes.append(300 + 200 * out_deg)
    
    nx.draw_networkx_nodes(
        network, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap='viridis',
        alpha=0.8
    )
    
    # Draw edges with varying width based on weight
    edges = network.edges()
    weights = [network[u][v].get('weight', 1.0) for u, v in edges]
    
    # Normalize weights for visualization
    if weights:
        max_weight = max(weights)
        min_weight = min(weights)
        if max_weight > min_weight:
            edge_widths = [1 + 4 * (w - min_weight) / (max_weight - min_weight) 
                          for w in weights]
        else:
            edge_widths = [2] * len(weights)
    else:
        edge_widths = []
    
    nx.draw_networkx_edges(
        network, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        network, pos,
        font_size=10,
        font_weight='bold'
    )
    
    # Draw edge labels (weight and lag)
    edge_labels = {}
    for u, v, d in network.edges(data=True):
        weight = d.get('weight', 0)
        lag = d.get('lag', 0)
        edge_labels[(u, v)] = f'{weight:.2f}\n(lag:{lag})'
    
    nx.draw_networkx_edge_labels(
        network, pos,
        edge_labels,
        font_size=8,
        alpha=0.7
    )
    
    plt.title('Synchronization Network', fontsize=16)
    plt.axis('off')
    
    # Add colorbar for node colors
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=min(node_colors), 
                                                vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('In-degree', rotation=270, labelpad=15)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_interaction_matrix(
    interaction_matrix: np.ndarray,
    series_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    Plot interaction effect matrix as a heatmap.
    
    Args:
        interaction_matrix: N×N matrix of interaction coefficients
        series_names: List of series names
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    mask = np.eye(len(series_names), dtype=bool)  # Mask diagonal
    
    sns.heatmap(
        interaction_matrix,
        mask=mask,
        xticklabels=series_names,
        yticklabels=series_names,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Interaction Coefficient (β)'}
    )
    
    plt.title('Cross-Series Interaction Effects\n(Column → Row)', fontsize=16)
    plt.xlabel('From Series', fontsize=12)
    plt.ylabel('To Series', fontsize=12)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===============================
# Time Series Comparison
# ===============================

def plot_series_comparison(
    features_dict: Dict[str, Lambda3FeatureSet],
    normalize: bool = True,
    highlight_events: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    Plot multiple time series for comparison.
    
    Args:
        features_dict: Dictionary of Lambda³ features
        normalize: Whether to normalize series
        highlight_events: Whether to highlight jump events
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    n_series = len(features_dict)
    colors = sns.color_palette(config.color_palette, n_series)
    
    plt.figure(figsize=config.figure_size)
    
    # Plot each series
    for idx, (name, features) in enumerate(features_dict.items()):
        data = features.data.copy()
        
        # Normalize if requested
        if normalize:
            data = (data - np.mean(data)) / np.std(data)
        
        # Plot data
        plt.plot(data, color=colors[idx], label=name, 
                linewidth=config.line_width, alpha=config.alpha)
        
        # Highlight events
        if highlight_events:
            pos_idx = np.where(features.delta_LambdaC_pos > 0)[0]
            if len(pos_idx) > 0:
                plt.scatter(pos_idx, data[pos_idx], 
                          color=colors[idx], marker='^', s=50,
                          alpha=0.8, edgecolors='black', linewidth=0.5)
            
            neg_idx = np.where(features.delta_LambdaC_neg > 0)[0]
            if len(neg_idx) > 0:
                plt.scatter(neg_idx, data[neg_idx],
                          color=colors[idx], marker='v', s=50,
                          alpha=0.8, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value' + (' (normalized)' if normalize else ''), fontsize=12)
    plt.title('Time Series Comparison', fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===============================
# Regime Visualization
# ===============================

def plot_regimes(
    features: Lambda3FeatureSet,
    regime_info: RegimeInfo,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    Plot time series with regime coloring.
    
    Args:
        features: Lambda³ features
        regime_info: Regime detection results
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.figure_size,
                                   sharex=True, height_ratios=[3, 1])
    
    # Get regime colors
    n_regimes = regime_info.n_regimes
    regime_colors = sns.color_palette('husl', n_regimes)
    
    # Plot data with regime background
    time = np.arange(len(features.data))
    
    # Add regime backgrounds
    for regime in range(n_regimes):
        mask = regime_info.labels == regime
        if np.any(mask):
            # Find continuous segments
            segments = []
            start = None
            for i, m in enumerate(mask):
                if m and start is None:
                    start = i
                elif not m and start is not None:
                    segments.append((start, i-1))
                    start = None
            if start is not None:
                segments.append((start, len(mask)-1))
            
            # Plot segments
            for start, end in segments:
                ax1.axvspan(start, end, color=regime_colors[regime],
                           alpha=0.2, label=regime_info.regime_names.get(regime, f'Regime {regime}')
                           if start == segments[0][0] else '')
    
    # Plot data
    ax1.plot(time, features.data, color='black', linewidth=1.5, alpha=0.8)
    
    # Mark transition points
    for tp in regime_info.transition_points:
        ax1.axvline(x=tp, color='red', linestyle='--', alpha=0.5)
    
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Time Series with Market Regimes', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot regime sequence
    ax2.plot(time, regime_info.labels, color='darkblue', linewidth=2)
    ax2.fill_between(time, 0, regime_info.labels, alpha=0.3)
    ax2.set_ylabel('Regime', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_yticks(range(n_regimes))
    ax2.set_yticklabels([regime_info.regime_names.get(i, f'R{i}') 
                        for i in range(n_regimes)])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===============================
# Summary Dashboard
# ===============================

def create_analysis_dashboard(
    results: CrossAnalysisResult,
    features_dict: Dict[str, Lambda3FeatureSet],
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    Create a comprehensive dashboard of analysis results.
    
    Args:
        results: Cross-analysis results
        features_dict: Dictionary of features
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Synchronization matrix (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    series_names = results.get_series_names()
    sns.heatmap(
        results.sync_matrix,
        xticklabels=series_names,
        yticklabels=series_names,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Sync Rate'},
        ax=ax1
    )
    ax1.set_title('Synchronization Matrix')
    
    # 2. Interaction matrix (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    mask = np.eye(len(series_names), dtype=bool)
    sns.heatmap(
        results.interaction_matrix,
        mask=mask,
        xticklabels=series_names,
        yticklabels=series_names,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        cbar_kws={'label': 'Interaction β'},
        ax=ax2
    )
    ax2.set_title('Interaction Effects')
    
    # 3. Network visualization (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if results.network and results.network.number_of_edges() > 0:
        import networkx as nx
        pos = nx.spring_layout(results.network)
        nx.draw(results.network, pos, ax=ax3, with_labels=True,
                node_color='lightblue', node_size=1000,
                font_size=10, font_weight='bold',
                edge_color='gray', arrows=True)
        ax3.set_title('Synchronization Network')
    else:
        ax3.text(0.5, 0.5, 'No significant network edges',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Synchronization Network')
    
    # 4. Time series comparison (middle row)
    ax4 = fig.add_subplot(gs[1, :])
    for idx, (name, features) in enumerate(features_dict.items()):
        data_norm = (features.data - np.mean(features.data)) / np.std(features.data)
        ax4.plot(data_norm + idx * 3, label=name, alpha=0.8)
    ax4.set_ylabel('Normalized Value (offset)')
    ax4.set_xlabel('Time')
    ax4.set_title('Time Series Comparison')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Event statistics (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    event_data = []
    for name, features in features_dict.items():
        event_data.append({
            'Series': name,
            'Pos Jumps': features.n_pos_jumps,
            'Neg Jumps': features.n_neg_jumps,
            'Mean Tension': features.mean_tension
        })
    event_df = pd.DataFrame(event_data)
    event_df.set_index('Series').plot(kind='bar', ax=ax5)
    ax5.set_title('Event Statistics')
    ax5.set_ylabel('Count / Value')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Cluster dendrogram (bottom middle)
    ax6 = fig.add_subplot(gs[2, 1])
    if results.clusters and len(set(results.clusters.values())) > 1:
        from scipy.cluster.hierarchy import dendrogram, linkage
        # Use sync distance for clustering
        distance_matrix = 1 - results.sync_matrix
        np.fill_diagonal(distance_matrix, 0)
        linkage_matrix = linkage(distance_matrix[np.triu_indices_from(distance_matrix, k=1)], 
                               method='average')
        dendrogram(linkage_matrix, labels=series_names, ax=ax6)
        ax6.set_title('Hierarchical Clustering')
        ax6.set_ylabel('Sync Distance')
    else:
        ax6.text(0.5, 0.5, 'No clustering performed',
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Hierarchical Clustering')
    
    # 7. Summary statistics (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # Calculate summary stats
    n_significant_sync = np.sum(results.sync_matrix > 0.3) - len(series_names)
    n_significant_interact = np.sum(np.abs(results.interaction_matrix) > 0.1)
    avg_sync = np.mean(results.sync_matrix[np.triu_indices_from(results.sync_matrix, k=1)])
    
    summary_text = f"""
    Analysis Summary
    ================
    Series analyzed: {len(series_names)}
    Total pairs: {len(results.pairwise_results)}
    
    Synchronization:
    - Significant pairs: {n_significant_sync//2}
    - Average sync rate: {avg_sync:.3f}
    
    Interactions:
    - Significant effects: {n_significant_interact}
    - Strongest effect: {np.max(np.abs(results.interaction_matrix)):.3f}
    
    Network:
    - Edges: {results.network.number_of_edges() if results.network else 0}
    - Density: {results.network.number_of_edges() / (len(series_names)*(len(series_names)-1)) if results.network else 0:.2%}
    """
    
    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    # 8. Key findings (bottom row)
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # Generate findings
    from .analysis import generate_analysis_summary
    findings = generate_analysis_summary(results, features_dict)
    
    findings_text = "Key Findings:\n" + "\n".join([f"• {finding}" for finding in findings[:5]])
    ax8.text(0.05, 0.8, findings_text, transform=ax8.transAxes,
            fontsize=12, verticalalignment='top')
    
    # Overall title
    fig.suptitle('Lambda³ Analysis Dashboard', fontsize=20, y=0.98)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Import pandas for some functions
try:
    import pandas as pd
except ImportError:
    pd = None
