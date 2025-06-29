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
from datetime import datetime

from .types import (
    Lambda3FeatureSet, AnalysisResult, CrossAnalysisResult,
    SyncProfile, CausalityProfile, RegimeInfo
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
    Plot Lambda³ features for one or more series (PyMC/lambda3_abc style).
    
    Args:
        features: Single feature set or dictionary
        title: Plot title
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    # Handle single vs multiple features
    if isinstance(features, Lambda3FeatureSet):
        features_dict = {'Series': features}
    else:
        features_dict = features
    
    n_series = len(features_dict)
    
    # PyMCスタイルの大きな図
    fig, axes = plt.subplots(n_series, 1, figsize=(15, 5 * n_series), sharex=True)
    
    if n_series == 1:
        axes = [axes]
    
    # Plot each series in PyMC style
    for i, (name, feat) in enumerate(features_dict.items()):
        ax = axes[i]
        time = np.arange(len(feat.data))
        
        # データプロット（グレーの点）
        ax.plot(feat.data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
        
        # グリーンの滑らかな線（ここではρTベースの移動平均を使用）
        window = min(10, len(feat.data) // 10)
        if window > 1:
            smoothed = np.convolve(feat.data, np.ones(window)/window, mode='same')
            ax.plot(smoothed, color='C2', lw=2, label='Smoothed')
        
        # ジャンプイベントの強調表示
        # 正のジャンプ（青い大きな○）
        pos_idx = np.where(feat.delta_LambdaC_pos > 0)[0]
        if len(pos_idx) > 0:
            ax.plot(pos_idx, feat.data[pos_idx], 'o', color='dodgerblue',
                   markersize=10, label=f'Positive ΔΛC ({len(pos_idx)})')
            # 垂直線で強調
            for idx in pos_idx:
                ax.axvline(x=idx, color='dodgerblue', linestyle='--', alpha=0.5)
        
        # 負のジャンプ（オレンジの大きな○）
        neg_idx = np.where(feat.delta_LambdaC_neg > 0)[0]
        if len(neg_idx) > 0:
            ax.plot(neg_idx, feat.data[neg_idx], 'o', color='orange',
                   markersize=10, label=f'Negative ΔΛC ({len(neg_idx)})')
            # 垂直線で強調
            for idx in neg_idx:
                ax.axvline(x=idx, color='orange', linestyle='-.', alpha=0.5)
        
        # ローカルジャンプ（マゼンタの星）
        local_idx = np.where(feat.local_jump > 0)[0]
        if len(local_idx) > 0:
            # グローバルジャンプと重複しないローカルジャンプのみ表示
            global_jumps = set(pos_idx) | set(neg_idx)
            local_only = [idx for idx in local_idx if idx not in global_jumps]
            if local_only:
                ax.plot(local_only, feat.data[local_only], '*', color='magenta',
                       markersize=12, alpha=0.7, label=f'Local Jump ({len(local_only)})')
        
        # タイトルとラベル
        plot_title = f"{name}: Lambda³ Features" if title is None else f"{title} - {name}"
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        
        # 凡例（重複除去）
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
        
        # グリッド（Y軸のみ）
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        
        # テンション情報を右上に表示
        mean_tension = np.mean(feat.rho_T)
        max_tension = np.max(feat.rho_T)
        info_text = f'Mean ρT: {mean_tension:.3f}\nMax ρT: {max_tension:.3f}'
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# 追加：テンションとイベントの関係を示す補助プロット関数
def plot_features_with_tension(
    features: Lambda3FeatureSet,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    データとテンションを上下に表示するプロット（lambda3_abc.py スタイル）
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    time = np.arange(len(features.data))
    
    # 上段：データとイベント
    ax1.plot(features.data, 'o', color='gray', markersize=3, alpha=0.5, label='Data')
    
    # ジャンプイベント
    pos_idx = np.where(features.delta_LambdaC_pos > 0)[0]
    neg_idx = np.where(features.delta_LambdaC_neg > 0)[0]
    local_idx = np.where(features.local_jump > 0)[0]
    
    if len(pos_idx) > 0:
        ax1.scatter(pos_idx, features.data[pos_idx], color='dodgerblue', 
                   s=100, marker='^', label='Positive ΔΛC', zorder=5)
    if len(neg_idx) > 0:
        ax1.scatter(neg_idx, features.data[neg_idx], color='orange',
                   s=100, marker='v', label='Negative ΔΛC', zorder=5)
    if len(local_idx) > 0:
        ax1.scatter(local_idx, features.data[local_idx], color='magenta',
                   s=80, marker='*', label='Local Jump', zorder=4, alpha=0.7)
    
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    if title:
        ax1.set_title(title, fontsize=16)
    
    # 下段：テンションスカラー
    ax2.fill_between(time, 0, features.rho_T, color='green', alpha=0.3)
    ax2.plot(time, features.rho_T, color='green', linewidth=2, label='Tension Scalar ρT')
    
    # 高テンション期間をハイライト
    high_tension_threshold = np.percentile(features.rho_T, 90)
    high_tension = features.rho_T > high_tension_threshold
    if np.any(high_tension):
        ax2.fill_between(time, 0, features.rho_T, where=high_tension,
                        color='red', alpha=0.3, label='High Tension')
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('ρT', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
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
    Plot comprehensive analysis results for a series pair (PyMC style).
    
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
    
    # Get series names
    name_a, name_b = results.series_names
    
    # 1. メインの予測プロット（PyMCスタイル）
    print(f"\nPlotting predictions for {name_a} and {name_b}...")
    
    # 予測データの準備
    data_dict = {name_a: features_a.data, name_b: features_b.data}
    features_dict = {name_a: features_a, name_b: features_b}
    
    # 予測値の抽出
    mu_pred_dict = {}
    if results.trace_a and hasattr(results.trace_a, 'predictions'):
        mu_pred_dict[name_a] = results.trace_a.predictions
    else:
        mu_pred_dict[name_a] = features_a.data  # フォールバック
        
    if results.trace_b and hasattr(results.trace_b, 'predictions'):
        mu_pred_dict[name_b] = results.trace_b.predictions
    else:
        mu_pred_dict[name_b] = features_b.data  # フォールバック
    
    # PyMCスタイルのデュアルプロット
    plot_l3_prediction_dual(
        data_dict,
        mu_pred_dict,
        features_dict,
        series_names=[name_a, name_b],
        titles=[f"{name_a}: Fit + Events", f"{name_b}: Fit + Events"]
    )
    
    # 2. 解析結果のサマリープロット（別の図として）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 同期プロファイル（左上）
    ax = axes[0, 0]
    _plot_sync_profile(ax, results.sync_profile)
    ax.set_title('Synchronization Profile (σₛ)', fontsize=14)
    
    # 相互作用効果（右上）
    ax = axes[0, 1]
    _plot_interaction_effects(ax, results.interaction_effects, name_a, name_b)
    ax.set_title('Interaction Effects (β)', fontsize=14)
    
    # 因果関係プロファイル（左下）
    ax = axes[1, 0]
    if results.causality_profiles:
        _plot_causality_profiles(ax, results.causality_profiles)
        ax.set_title('Causality Profiles', fontsize=14)
    else:
        ax.text(0.5, 0.5, 'No causality analysis performed',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Causality Profiles', fontsize=14)
    
    # サマリー統計（右下）
    ax = axes[1, 1]
    ax.axis('off')
    
    # 主要な統計情報を表示
    summary_text = f"""Analysis Summary: {name_a} ↔ {name_b}
    
Synchronization:
  Max sync rate (σₛ): {results.sync_profile.max_sync_rate:.3f}
  Optimal lag: {results.sync_profile.optimal_lag} steps

Events:
  {name_a}: {features_a.n_pos_jumps} pos, {features_a.n_neg_jumps} neg jumps
  {name_b}: {features_b.n_pos_jumps} pos, {features_b.n_neg_jumps} neg jumps

Tension (ρT):
  {name_a}: mean = {features_a.mean_tension:.3f}
  {name_b}: mean = {features_b.mean_tension:.3f}
"""
    
    # 相互作用効果の追加
    if results.interaction_effects:
        significant_effects = [(k, v) for k, v in results.interaction_effects.items() 
                              if abs(v) > 0.1]
        if significant_effects:
            summary_text += "\nSignificant Interactions:"
            for key, value in significant_effects[:3]:  # Top 3
                summary_text += f"\n  {key}: β = {value:.3f}"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace',
            fontsize=11, bbox=dict(boxstyle='round,pad=0.5', 
                                 facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Lambda³ Analysis Results: {name_a} ↔ {name_b}', fontsize=16)
    plt.tight_layout()
    
    # 保存処理
    if save_path:
        # 両方の図を保存
        base_path = Path(save_path)
        
        # 予測プロット
        pred_path = base_path.parent / f"{base_path.stem}_predictions{base_path.suffix}"
        plt.figure(1)  # 最初の図を選択
        plt.savefig(pred_path, dpi=config.dpi, bbox_inches='tight')
        
        # サマリープロット
        plt.figure(2)  # 2番目の図を選択
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        
        plt.close('all')
        print(f"Saved plots to {pred_path} and {save_path}")
    else:
        plt.show()

def plot_l3_prediction_dual(
    data_dict: Dict[str, np.ndarray],
    mu_pred_dict: Dict[str, np.ndarray],
    features_dict: Dict[str, Lambda3FeatureSet],
    series_names: Optional[List[str]] = None,
    titles: Optional[List[str]] = None
) -> None:
    """
    Lambda³予測結果のデュアル表示（PyMC/lambda3_abc.py スタイル）
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
        features = features_dict[series]
        
        # データと予測をプロット
        ax.plot(data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
        ax.plot(mu_pred, color='C2', lw=2, label='Model Prediction')
        
        # ジャンプイベントを強調表示
        # 正のジャンプ（青）
        pos_jumps = features.delta_LambdaC_pos
        pos_idx = np.where(pos_jumps > 0)[0]
        if len(pos_idx) > 0:
            ax.plot(pos_idx, data[pos_idx], 'o', color='dodgerblue',
                   markersize=10, label='Positive ΔΛC')
            for idx in pos_idx:
                ax.axvline(x=idx, color='dodgerblue', linestyle='--', alpha=0.5)
        
        # 負のジャンプ（オレンジ）
        neg_jumps = features.delta_LambdaC_neg
        neg_idx = np.where(neg_jumps > 0)[0]
        if len(neg_idx) > 0:
            ax.plot(neg_idx, data[neg_idx], 'o', color='orange',
                   markersize=10, label='Negative ΔΛC')
            for idx in neg_idx:
                ax.axvline(x=idx, color='orange', linestyle='-.', alpha=0.5)
        
        # ローカルジャンプ（マゼンタ）
        local_jumps = features.local_jump
        local_idx = np.where(local_jumps > 0)[0]
        if len(local_idx) > 0:
            # グローバルジャンプと重複しないものだけ表示
            global_jumps = set(pos_idx) | set(neg_idx)
            local_only = [idx for idx in local_idx if idx not in global_jumps]
            if local_only:
                ax.plot(local_only, data[local_only], 'o', color='magenta',
                       markersize=7, alpha=0.7, label='Local Jump')
        
        # フォーマット
        plot_title = titles[i] if titles and i < len(titles) else f"{series}: Fit + Events"
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        
        # 凡例の整理
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
        
        ax.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()

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
    causality_profiles: Dict[str, CausalityProfile]
) -> None:
    """Plot causality profiles with enhanced visualization."""
    colors = plt.cm.tab10(np.arange(len(causality_profiles)))
    
    for idx, (name, profile) in enumerate(causality_profiles.items()):
        if hasattr(profile, 'self_causality'):
            lags = sorted(profile.self_causality.keys())
            probs = [profile.self_causality[lag] for lag in lags]
            
            ax.plot(lags, probs, 'o-', color=colors[idx], 
                   label=f'{name} self', markersize=6)
            
            # Mark maximum causality
            if hasattr(profile, 'max_causality_lag'):
                max_lag = profile.max_causality_lag
                max_strength = profile.max_causality_strength
                ax.scatter(max_lag, max_strength, s=100, color=colors[idx],
                          marker='*', edgecolor='black', linewidth=1)
            
            # Also plot cross-causality if available
            if profile.cross_causality:
                cross_probs = [profile.cross_causality[lag] for lag in lags]
                ax.plot(lags, cross_probs, 's--', color=colors[idx],
                       label=f'{name} cross', markersize=4, alpha=0.7)
    
    # Add significance threshold
    ax.axhline(y=0.3, color='red', linestyle=':', alpha=0.5, 
               label='Significance threshold')
    
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
# Change Point Visualization
# ===============================

def plot_changepoint_analysis(
    features: Lambda3FeatureSet,
    changepoint_results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    Plot time series with detected change points and segments.
    
    Args:
        features: Lambda³ features
        changepoint_results: Results from analyze_with_changepoints
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    plt.style.use(config.style)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Extract data
    data = features.data
    time = np.arange(len(data))
    change_points = changepoint_results['change_points']
    segments = changepoint_results['segments']
    
    # Plot data with change points
    ax1.plot(time, data, 'k-', alpha=0.8, linewidth=1)
    
    # Mark change points
    for cp in change_points:
        ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.7, 
                   label='Change point' if cp == change_points[0] else '')
    
    # Color segments
    colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        start, end = segment['start'], segment['end']
        ax1.axvspan(start, end, alpha=0.1, color=colors[i])
        
        # Add segment trend line
        if segment['length'] > 1:
            x = np.arange(start, end)
            y = segment['mean'] + segment['trend'] * (x - start)
            ax1.plot(x, y, color=colors[i], linewidth=2, alpha=0.8)
    
    ax1.set_ylabel('Value')
    ax1.set_title('Data with Change Points and Segment Trends')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot segment statistics
    segment_starts = [s['start'] for s in segments]
    segment_means = [s['mean'] for s in segments]
    segment_stds = [s['std'] for s in segments]
    
    ax2.bar(segment_starts, segment_means, 
            width=[s['length'] for s in segments],
            alpha=0.6, edgecolor='black', linewidth=1)
    
    # Add error bars for std
    for i, (start, mean, std) in enumerate(zip(segment_starts, segment_means, segment_stds)):
        width = segments[i]['length']
        ax2.errorbar(start + width/2, mean, yerr=std, 
                    color='black', capsize=5, capthick=2)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Segment Mean ± Std')
    ax2.set_title('Segment Statistics')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===============================
# Causality Visualization
# ===============================

def plot_causality_matrix(
    causality_matrix: np.ndarray,
    series_names: List[str],
    lag: int = 1,
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    Plot causality matrix as a directed heatmap.
    
    Args:
        causality_matrix: N×N causality matrix
        series_names: List of series names
        lag: Time lag used for causality
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with arrows to indicate direction
    sns.heatmap(
        causality_matrix,
        xticklabels=series_names,
        yticklabels=series_names,
        annot=True,
        fmt='.3f',
        cmap='Reds',
        square=True,
        linewidths=0.5,
        cbar_kws={'label': f'Causality Probability (lag={lag})'}
    )
    
    plt.title(f'Structural Causality Matrix\n(Column causes Row with lag {lag})', 
              fontsize=16)
    plt.xlabel('Cause', fontsize=12)
    plt.ylabel('Effect', fontsize=12)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add diagonal label
    ax = plt.gca()
    for i in range(len(series_names)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                  edgecolor='blue', linewidth=2))
    
    # Add text annotation
    plt.text(0.02, 0.98, 'Diagonal: Self-causality\nOff-diagonal: Cross-causality',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===============================
# Comprehensive Results
# ===============================

def plot_comprehensive_results(
    comprehensive_results: Dict[str, Any],
    features_dict: Dict[str, Lambda3FeatureSet],
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    Create a comprehensive visualization dashboard for all analysis results.
    
    Args:
        comprehensive_results: Results from run_comprehensive_analysis
        features_dict: Dictionary of features
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    # Create large figure with subplots
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)
    
    # 1. Time series overview (top row)
    ax1 = fig.add_subplot(gs[0, :])
    series_names = comprehensive_results['series_names']
    colors = sns.color_palette(config.color_palette, len(series_names))
    
    for i, name in enumerate(series_names):
        data = features_dict[name].data
        data_norm = (data - np.mean(data)) / np.std(data) + i * 3
        ax1.plot(data_norm, color=colors[i], label=name, alpha=0.8)
    
    ax1.set_ylabel('Normalized Value (offset)')
    ax1.set_title('Time Series Overview', fontsize=16)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Causality matrix (second row, left)
    ax2 = fig.add_subplot(gs[1, 0])
    if 'causality' in comprehensive_results:
        causality_mat = comprehensive_results['causality']['matrix']
        sns.heatmap(causality_mat, 
                    xticklabels=series_names,
                    yticklabels=series_names,
                    annot=True, fmt='.2f',
                    cmap='Reds', square=True,
                    cbar_kws={'label': 'Causality Strength'},
                    ax=ax2)
        ax2.set_title('Causality Matrix (Column → Row)')
    
    # 3. Regime visualization (second row, middle and right)
    if 'regimes' in comprehensive_results:
        ax3 = fig.add_subplot(gs[1, 1:])
        
        # Stack regime labels for all series
        regime_data = []
        for name in series_names:
            if name in comprehensive_results['regimes']:
                regime_info = comprehensive_results['regimes'][name]
                regime_data.append(regime_info.labels)
        
        if regime_data:
            regime_array = np.array(regime_data)
            im = ax3.imshow(regime_array, aspect='auto', cmap='tab10')
            ax3.set_yticks(range(len(series_names)))
            ax3.set_yticklabels(series_names)
            ax3.set_xlabel('Time')
            ax3.set_title('Market Regimes Across Series')
            plt.colorbar(im, ax=ax3, label='Regime ID')
    
    # 4. Synchronization and interaction summary (third row)
    if 'cross_analysis' in comprehensive_results:
        cross = comprehensive_results['cross_analysis']
        
        # Sync matrix
        ax4 = fig.add_subplot(gs[2, 0])
        sns.heatmap(cross.sync_matrix,
                    xticklabels=series_names,
                    yticklabels=series_names,
                    annot=True, fmt='.2f',
                    cmap='Blues', square=True,
                    cbar_kws={'label': 'Sync Rate'},
                    ax=ax4)
        ax4.set_title('Synchronization Matrix')
        
        # Interaction matrix
        ax5 = fig.add_subplot(gs[2, 1])
        mask = np.eye(len(series_names), dtype=bool)
        sns.heatmap(cross.interaction_matrix,
                    mask=mask,
                    xticklabels=series_names,
                    yticklabels=series_names,
                    annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0,
                    square=True,
                    cbar_kws={'label': 'Interaction β'},
                    ax=ax5)
        ax5.set_title('Interaction Effects')
        
        # Network
        ax6 = fig.add_subplot(gs[2, 2])
        if cross.network and cross.network.number_of_edges() > 0:
            import networkx as nx
            pos = nx.spring_layout(cross.network)
            nx.draw(cross.network, pos, ax=ax6,
                   with_labels=True, node_color='lightblue',
                   node_size=1000, font_size=10,
                   edge_color='gray', arrows=True)
            ax6.set_title('Synchronization Network')
    
    # 5. Change points summary (fourth row)
    if 'changepoints' in comprehensive_results:
        ax7 = fig.add_subplot(gs[3, :])
        
        # Create timeline of change points
        all_changepoints = []
        for name in series_names:
            if name in comprehensive_results['changepoints']:
                cps = comprehensive_results['changepoints'][name]['change_points']
                for cp in cps:
                    all_changepoints.append((cp, name))
        
        if all_changepoints:
            all_changepoints.sort(key=lambda x: x[0])
            
            # Plot timeline
            for i, (cp, name) in enumerate(all_changepoints):
                color = colors[series_names.index(name)]
                ax7.scatter(cp, i, s=100, color=color, marker='o')
                ax7.text(cp + 5, i, f'{name}', fontsize=8)
            
            ax7.set_xlabel('Time')
            ax7.set_ylabel('Change Point Event')
            ax7.set_title('Change Points Timeline')
            ax7.grid(True, alpha=0.3)
    
    # 6. Summary statistics (bottom row)
    ax8 = fig.add_subplot(gs[4, :])
    ax8.axis('off')
    
    # Create summary text
    summary = comprehensive_results.get('summary', {})
    summary_text = f"""
    Comprehensive Analysis Summary
    ==============================
    Analysis Date: {comprehensive_results.get('timestamp', 'N/A')}
    Series Analyzed: {summary.get('n_series', 'N/A')}
    
    Key Metrics:
    - Maximum Synchronization: {summary.get('max_sync_rate', 0):.3f}
    - Significant Interactions: {summary.get('n_significant_interactions', 0)}
    - Maximum Causality: {summary.get('max_causality', 0):.3f}
    - Causality Density: {summary.get('causality_density', 0):.1%}
    """
    
    if 'best_bayesian_model' in summary:
        summary_text += f"\n    - Best Bayesian Model: {summary['best_bayesian_model']}"
    
    if 'regime_counts' in summary:
        summary_text += "\n\n    Regime Counts:"
        for name, count in summary['regime_counts'].items():
            summary_text += f"\n    - {name}: {count} regimes"
    
    if 'changepoint_counts' in summary:
        summary_text += "\n\n    Change Point Counts:"
        for name, count in summary['changepoint_counts'].items():
            summary_text += f"\n    - {name}: {count} change points"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('Lambda³ Comprehensive Analysis Dashboard', fontsize=20, y=0.995)
    
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
    
    # Import pandas for DataFrame operations
    try:
        import pandas as pd
        event_df = pd.DataFrame(event_data)
        event_df.set_index('Series').plot(kind='bar', ax=ax5)
    except ImportError:
        # Fallback if pandas not available
        x = np.arange(len(event_data))
        width = 0.25
        ax5.bar(x - width, [d['Pos Jumps'] for d in event_data], width, label='Pos Jumps')
        ax5.bar(x, [d['Neg Jumps'] for d in event_data], width, label='Neg Jumps')
        ax5.bar(x + width, [d['Mean Tension'] for d in event_data], width, label='Mean Tension')
        ax5.set_xticks(x)
        ax5.set_xticklabels([d['Series'] for d in event_data])
        ax5.legend()
    
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
    - Density: {results.network_density:.2%}
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

# ===============================
# Posterior Predictive Check Plots
# ===============================

def plot_posterior_predictive_check(
    ppc_results: Dict[str, Any],
    observed_data: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
    config: Optional[PlottingConfig] = None
) -> None:
    """
    Plot posterior predictive check results.
    
    Args:
        ppc_results: PPC results from posterior_predictive_check
        observed_data: Original observed data
        model_name: Name of the model
        save_path: Path to save figure
        config: Plotting configuration
    """
    _check_plotting()
    
    if config is None:
        config = PlottingConfig()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Data vs PPC samples
    ax = axes[0, 0]
    ppc_samples = ppc_results['ppc_samples']
    
    # Plot a subset of PPC samples
    n_plot = min(100, len(ppc_samples))
    for i in range(n_plot):
        ax.plot(ppc_samples[i], color='skyblue', alpha=0.1)
    
    # Plot observed data
    ax.plot(observed_data, color='black', linewidth=2, label='Observed')
    
    # Plot mean and CI of PPC
    ppc_mean = np.mean(ppc_samples, axis=0)
    ppc_lower = np.percentile(ppc_samples, 2.5, axis=0)
    ppc_upper = np.percentile(ppc_samples, 97.5, axis=0)
    
    ax.plot(ppc_mean, color='red', linewidth=2, label='PPC Mean')
    ax.fill_between(range(len(ppc_mean)), ppc_lower, ppc_upper, 
                    color='red', alpha=0.2, label='95% CI')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Data vs Posterior Predictive Samples')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Test statistics
    test_stats = ppc_results['test_statistics']
    observed_stats = ppc_results['observed_stats']
    
    stat_names = ['mean', 'std', 'min', 'max']
    for i, stat_name in enumerate(stat_names):
        ax = axes.flat[i+1] if i < 3 else axes[1, 1]
        
        if stat_name in test_stats:
            # Plot histogram of test statistic
            ax.hist(test_stats[stat_name], bins=30, alpha=0.7, 
                   color='lightblue', edgecolor='black')
            
            # Mark observed value
            obs_val = observed_stats[stat_name]
            ax.axvline(obs_val, color='red', linewidth=2, 
                      label=f'Observed: {obs_val:.3f}')
            
            # Add p-value
            p_val = ppc_results['bayesian_p_values'][stat_name]
            ax.text(0.05, 0.95, f'p-value: {p_val:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(stat_name.capitalize())
            ax.set_ylabel('Frequency')
            ax.set_title(f'PPC: {stat_name.capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle(f'Posterior Predictive Check - {model_name}', fontsize=16)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ===============================
# Posterior Distribution Plots
# ===============================

def plot_posterior_distributions_lambda3(
    samples: Dict[str, np.ndarray],
    titles: Optional[Dict[str, str]] = None,
    hdi_prob: float = 0.94,
    figsize: Tuple[float, float] = (15, 10),
    color: str = 'skyblue'
) -> None:
    """
    Plot posterior distributions in Lambda³/PyMC style with HDI intervals.
    
    Args:
        samples: Dictionary of parameter samples
        titles: Custom titles for each parameter
        hdi_prob: HDI probability (default: 94%)
        figsize: Figure size
        color: Color for distributions
    """
    # Default parameter names and titles for Lambda³
    default_titles = {
        'beta_time': 'beta_time',
        'beta_dLC_pos': 'beta_dLC_pos', 
        'beta_dLC_neg': 'beta_dLC_neg',
        'beta_interact_stress': 'beta_interact_stress',
        'beta_interact_pos': 'beta_interact_pos',
        'beta_interact_neg': 'beta_interact_neg',
        'sigma': 'sigma'
    }
    
    if titles is None:
        titles = default_titles
    
    # Filter parameters that exist in samples
    param_names = [p for p in default_titles.keys() if p in samples]
    n_params = len(param_names)
    
    if n_params == 0:
        print("No parameters to plot")
        return
    
    # Create subplot grid
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for idx, param in enumerate(param_names):
        ax = axes[idx]
        
        # Get samples
        param_samples = np.array(samples[param]).flatten()
        
        # Calculate statistics
        mean_val = np.mean(param_samples)
        
        # Calculate HDI
        sorted_samples = np.sort(param_samples)
        n_samples = len(sorted_samples)
        hdi_low_idx = int((1 - hdi_prob) / 2 * n_samples)
        hdi_high_idx = int((1 + hdi_prob) / 2 * n_samples)
        hdi_low = sorted_samples[hdi_low_idx]
        hdi_high = sorted_samples[hdi_high_idx]
        
        # Plot distribution
        ax.hist(param_samples, bins=50, density=True, 
                alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
        
        # Add mean line
        ax.axvline(mean_val, color='black', linestyle='-', linewidth=2)
        
        # Add HDI region
        ax.axvspan(hdi_low, hdi_high, alpha=0.3, color='gray', 
                   label=f'{int(hdi_prob*100)}% HDI')
        
        # Add HDI boundaries
        ax.axvline(hdi_low, color='black', linestyle='--', linewidth=1)
        ax.axvline(hdi_high, color='black', linestyle='--', linewidth=1)
        
        # Labels and title
        title = titles.get(param, param)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('')
        
        # Add mean value text
        ax.text(0.05, 0.95, f'mean={mean_val:.1f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add HDI text
        y_pos = ax.get_ylim()[1] * 0.02
        ax.text(hdi_low, y_pos, f'{hdi_low:.0f}', ha='center', fontsize=10)
        ax.text(hdi_high, y_pos, f'{hdi_high:.0f}', ha='center', fontsize=10)
        
        # Grid
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('')
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Posterior Distributions', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def plot_lambda3_posterior_grid(
    results: Dict[str, Any],
    param_subset: Optional[List[str]] = None,
    hdi_prob: float = 0.94
) -> None:
    """
    Plot Lambda³ posterior distributions in grid layout (exactly as shown in image).
    
    Args:
        results: Results dictionary containing 'samples'
        param_subset: Subset of parameters to plot
        hdi_prob: HDI probability
    """
    if 'samples' not in results:
        print("No samples found in results")
        return
    
    samples = results['samples']
    
    # Lambda³ specific parameter mapping
    lambda3_params = {
        'beta_time': 'β_time',
        'beta_dLC_pos': 'β_ΔΛC⁺',
        'beta_dLC_neg': 'β_ΔΛC⁻',
        'beta_interact_stress': 'β_interact_ρT',
        'beta_interact_pos': 'β_interact_ΔΛC⁺',
        'beta_interact_neg': 'β_interact_ΔΛC⁻',
        'sigma': 'σ'
    }
    
    # Use subset if provided
    if param_subset:
        params_to_plot = param_subset
    else:
        params_to_plot = [p for p in lambda3_params.keys() if p in samples]
    
    n_params = len(params_to_plot)
    
    # Create figure with 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, param in enumerate(params_to_plot):
        if idx >= 6:  # Only plot first 6 parameters
            break
            
        ax = axes[idx]
        
        # Get samples
        param_samples = np.array(samples[param]).flatten()
        
        # Calculate statistics
        mean_val = np.mean(param_samples)
        
        # Calculate HDI
        sorted_samples = np.sort(param_samples)
        n_samples = len(sorted_samples)
        hdi_low_idx = int((1 - hdi_prob) / 2 * n_samples)
        hdi_high_idx = int((1 + hdi_prob) / 2 * n_samples)
        hdi_low = sorted_samples[hdi_low_idx]
        hdi_high = sorted_samples[hdi_high_idx]
        
        # Create histogram
        counts, bins, patches = ax.hist(param_samples, bins=30, density=True,
                                       alpha=0.8, color='skyblue', 
                                       edgecolor='darkblue', linewidth=0.5)
        
        # Add smooth density curve (optional)
        try:
            from scipy import stats
            kde = stats.gaussian_kde(param_samples)
            x_smooth = np.linspace(param_samples.min(), param_samples.max(), 200)
            ax.plot(x_smooth, kde(x_smooth), 'b-', linewidth=2, alpha=0.8)
        except ImportError:
            pass  # Skip KDE if scipy not available
        
        # Mark HDI region
        ax.axvspan(hdi_low, hdi_high, alpha=0.2, color='gray')
        
        # Add vertical lines for HDI bounds
        ax.axvline(hdi_low, color='black', linestyle='--', linewidth=1)
        ax.axvline(hdi_high, color='black', linestyle='--', linewidth=1)
        
        # Add HDI labels at bottom
        y_min = ax.get_ylim()[0]
        ax.text(hdi_low, y_min, f'{hdi_low:.0f}', 
                ha='center', va='bottom', fontsize=10)
        ax.text(hdi_high, y_min, f'{hdi_high:.0f}', 
                ha='center', va='bottom', fontsize=10)
        
        # Add mean label
        ax.text(0.95, 0.95, f'mean={mean_val:.1f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8),
                fontsize=10)
        
        # Set title
        title = lambda3_params.get(param, param)
        ax.set_title(title, fontsize=12, pad=10)
        
        # Clean up axes
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim(hdi_low - (hdi_high - hdi_low) * 0.2,
                    hdi_high + (hdi_high - hdi_low) * 0.2)
        
        # Add HDI label
        ax.text(0.5, 0.02, f'{int(hdi_prob*100)}% HDI',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9, style='italic')
        
        # Light grid
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
    
    # Hide unused subplots
    for idx in range(n_params, 6):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_lambda3_trace_and_density(
    samples: Dict[str, np.ndarray],
    param_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 3)
) -> None:
    """
    Plot trace plots and density plots side by side for each parameter.
    
    Args:
        samples: Dictionary of parameter samples
        param_names: Parameters to plot (default: all)
        figsize: Figure size per parameter
    """
    if param_names is None:
        param_names = list(samples.keys())
    
    for param in param_names:
        if param not in samples:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        param_samples = np.array(samples[param]).flatten()
        
        # Trace plot
        ax1.plot(param_samples, alpha=0.7, linewidth=0.5)
        ax1.set_title(f'{param} - Trace Plot')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Density plot with HDI
        ax2.hist(param_samples, bins=50, density=True, 
                alpha=0.7, color='skyblue', edgecolor='darkblue')
        
        # Add KDE
        try:
            from scipy import stats
            kde = stats.gaussian_kde(param_samples)
            x_smooth = np.linspace(param_samples.min(), param_samples.max(), 200)
            ax2.plot(x_smooth, kde(x_smooth), 'b-', linewidth=2)
        except ImportError:
            pass  # Skip KDE if scipy not available
        
        # Calculate and show HDI
        sorted_samples = np.sort(param_samples)
        hdi_low = sorted_samples[int(0.03 * len(sorted_samples))]
        hdi_high = sorted_samples[int(0.97 * len(sorted_samples))]
        
        ax2.axvline(hdi_low, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(hdi_high, color='red', linestyle='--', alpha=0.7)
        ax2.axvspan(hdi_low, hdi_high, alpha=0.2, color='red')
        
        ax2.set_title(f'{param} - Posterior Density')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()


# ===============================
# Integration Functions
# ===============================

def plot_numpyro_lambda3_posteriors(numpyro_results: Dict[str, Any]):
    """
    Plot posteriors from NumPyro Lambda³ analysis results.
    
    Args:
        numpyro_results: Results from Lambda³ NumPyro inference
    """
    if 'inference_results' in numpyro_results:
        # Extract samples from first series
        first_series = list(numpyro_results['inference_results'].keys())[0]
        result = numpyro_results['inference_results'][first_series]
        
        if 'samples' in result:
            # Map NumPyro parameter names to Lambda³ names
            param_mapping = {
                'lambda_intercept': 'beta_time',
                'lambda_flow': 'beta_time',
                'lambda_struct_pos': 'beta_dLC_pos',
                'lambda_struct_neg': 'beta_dLC_neg',
                'rho_tension': 'beta_interact_stress',
                'lambda_interact_pos': 'beta_interact_pos',
                'lambda_interact_neg': 'beta_interact_neg',
                'sigma_obs': 'sigma'
            }
            
            # Remap samples
            mapped_samples = {}
            for numpyro_name, lambda3_name in param_mapping.items():
                if numpyro_name in result['samples']:
                    mapped_samples[lambda3_name] = result['samples'][numpyro_name]
            
            # Create results dict and plot
            plot_results = {'samples': mapped_samples}
            plot_lambda3_posterior_grid(plot_results, hdi_prob=0.94)


def demo_lambda3_posteriors():
    """Demo function showing how to use the posterior plotting functions"""
    
    # Generate sample data (replace with actual MCMC samples)
    np.random.seed(42)
    
    samples = {
        'beta_time': np.random.normal(140, 10, 1000),
        'beta_dLC_pos': np.random.normal(-7.8, 2, 1000),
        'beta_dLC_neg': np.random.normal(-6.9, 1.5, 1000),
        'beta_interact_stress': np.random.normal(23, 0.5, 1000),
        'beta_interact_pos': np.random.normal(-0.36, 0.2, 1000),
        'beta_interact_neg': np.random.normal(-1.6, 0.3, 1000)
    }
    
    # Create results dict
    results = {'samples': samples}
    
    # Plot in Lambda³ style (as shown in the image)
    print("Plotting Lambda³ posterior distributions...")
    plot_lambda3_posterior_grid(results, hdi_prob=0.94)
    
    # Alternative: plot with trace plots
    print("\nPlotting trace and density plots...")
    plot_lambda3_trace_and_density(samples, param_names=['beta_time', 'beta_dLC_pos'])
