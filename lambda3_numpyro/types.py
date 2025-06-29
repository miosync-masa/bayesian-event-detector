"""
Type definitions and data classes for Lambda³ framework.

This module provides type-safe data structures for feature sets,
analysis results, and synchronization profiles.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any, Union
import numpy as np
import networkx as nx
from datetime import datetime


@dataclass
class Lambda3FeatureSet:
    """
    Lambda³ feature set containing all extracted structural features.
    
    Attributes:
        data: Original time series data
        delta_LambdaC_pos: Positive jump events (ΔΛC+)
        delta_LambdaC_neg: Negative jump events (ΔΛC-)
        rho_T: Tension scalar (ρT) - local volatility measure
        time_trend: Linear time trend component
        local_jump: Local jump detection based on normalized score
        metadata: Optional metadata (source, timeframe, etc.)
    """
    data: np.ndarray
    delta_LambdaC_pos: np.ndarray
    delta_LambdaC_neg: np.ndarray
    rho_T: np.ndarray
    time_trend: np.ndarray
    local_jump: np.ndarray
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate array lengths"""
        lengths = [
            len(self.data),
            len(self.delta_LambdaC_pos),
            len(self.delta_LambdaC_neg),
            len(self.rho_T),
            len(self.time_trend),
            len(self.local_jump)
        ]
        if len(set(lengths)) > 1:
            raise ValueError(f"All feature arrays must have the same length. Got: {lengths}")
    
    @property
    def length(self) -> int:
        """Get the length of the time series"""
        return len(self.data)
    
    @property
    def n_pos_jumps(self) -> int:
        """Count of positive jump events"""
        return int(np.sum(self.delta_LambdaC_pos))
    
    @property
    def n_neg_jumps(self) -> int:
        """Count of negative jump events"""
        return int(np.sum(self.delta_LambdaC_neg))
    
    @property
    def mean_tension(self) -> float:
        """Average tension scalar value"""
        return float(np.mean(self.rho_T))
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary format (backward compatibility)"""
        return {
            'data': self.data,
            'delta_LambdaC_pos': self.delta_LambdaC_pos,
            'delta_LambdaC_neg': self.delta_LambdaC_neg,
            'rho_T': self.rho_T,
            'time_trend': self.time_trend,
            'local_jump': self.local_jump
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray], metadata: Optional[Dict] = None) -> 'Lambda3FeatureSet':
        """Create from dictionary format"""
        return cls(
            data=data['data'],
            delta_LambdaC_pos=data['delta_LambdaC_pos'],
            delta_LambdaC_neg=data['delta_LambdaC_neg'],
            rho_T=data['rho_T'],
            time_trend=data['time_trend'],
            local_jump=data['local_jump'],
            metadata=metadata
        )


@dataclass
class SyncProfile:
    """
    Synchronization profile between two time series.
    
    Attributes:
        profile: Sync rate at each lag {lag: sync_rate}
        max_sync_rate: Maximum synchronization rate (σₛ)
        optimal_lag: Lag with maximum synchronization
        series_names: Names of the synchronized series
    """
    profile: Dict[int, float]
    max_sync_rate: float
    optimal_lag: int
    series_names: Optional[Tuple[str, str]] = None
    
    @property
    def is_synchronized(self) -> bool:
        """Check if series are significantly synchronized (σₛ > 0.3)"""
        return self.max_sync_rate > 0.3
    
    def get_sync_at_lag(self, lag: int) -> float:
        """Get synchronization rate at specific lag"""
        return self.profile.get(lag, 0.0)


@dataclass
class CausalityProfile:
    """
    Causality analysis results for a series or pair.
    
    Attributes:
        self_causality: P(negative jump | positive jump) by lag
        cross_causality: Cross-series causality by lag (if applicable)
        series_names: Names of analyzed series
    """
    self_causality: Dict[int, float]
    cross_causality: Optional[Dict[int, float]] = None
    series_names: Optional[Union[str, Tuple[str, str]]] = None
    
    @property
    def max_causality_lag(self) -> int:
        """Get lag with maximum causality probability"""
        if not self.self_causality:
            return 0
        return max(self.self_causality.items(), key=lambda x: x[1])[0]


@dataclass
class BayesianResults:
    """
    Container for Bayesian model results.
    
    Attributes:
        trace: ArviZ InferenceData object
        summary: Summary statistics DataFrame
        predictions: Model predictions
        residuals: Prediction residuals
        diagnostics: MCMC diagnostics
    """
    trace: Any  # arviz.InferenceData
    summary: Any  # pandas.DataFrame
    predictions: np.ndarray
    residuals: Optional[np.ndarray] = None
    diagnostics: Optional[Dict[str, Any]] = None
    
    @property
    def converged(self) -> bool:
        """Check if MCMC chains converged (R-hat < 1.01)"""
        if self.diagnostics and 'r_hat' in self.diagnostics:
            return bool(np.all(self.diagnostics['r_hat'] < 1.01))
        return True  # Assume converged if no diagnostics


@dataclass
class AnalysisResult:
    """
    Complete analysis results for a series pair.
    
    Attributes:
        trace_a: Bayesian results for series A
        trace_b: Bayesian results for series B
        sync_profile: Synchronization profile
        interaction_effects: Interaction coefficients
        causality_profiles: Causality analysis results
        metadata: Analysis metadata (timestamps, config, etc.)
    """
    trace_a: BayesianResults
    trace_b: BayesianResults
    sync_profile: SyncProfile
    interaction_effects: Dict[str, float]
    causality_profiles: Optional[Dict[str, CausalityProfile]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def series_names(self) -> Tuple[str, str]:
        """Get series names from metadata"""
        if self.metadata and 'name_a' in self.metadata and 'name_b' in self.metadata:
            return (self.metadata['name_a'], self.metadata['name_b'])
        return ('Series A', 'Series B')
    
    @property
    def primary_interaction(self) -> Tuple[str, float]:
        """Get the strongest interaction effect"""
        if not self.interaction_effects:
            return ('none', 0.0)
        key = max(self.interaction_effects.items(), key=lambda x: abs(x[1]))[0]
        return (key, self.interaction_effects[key])


@dataclass
class CrossAnalysisResult:
    """
    Results from multi-series cross-analysis.
    
    Attributes:
        pairwise_results: Results for each series pair
        sync_matrix: Full synchronization matrix
        interaction_matrix: Full interaction effect matrix
        network: Synchronization network graph
        clusters: Series clustering results
        metadata: Analysis metadata
    """
    pairwise_results: Dict[Tuple[str, str], AnalysisResult]
    sync_matrix: np.ndarray
    interaction_matrix: np.ndarray
    network: Optional[nx.DiGraph] = None
    clusters: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def n_series(self) -> int:
        """Number of analyzed series"""
        return len(self.sync_matrix)
    
    @property
    def n_pairs(self) -> int:
        """Number of analyzed pairs"""
        return len(self.pairwise_results)
    
    def get_series_names(self) -> List[str]:
        """Extract all series names"""
        if self.metadata and 'series_names' in self.metadata:
            return self.metadata['series_names']
        # Extract from pairwise results
        names = set()
        for (a, b) in self.pairwise_results.keys():
            names.add(a)
            names.add(b)
        return sorted(list(names))


@dataclass
class RegimeInfo:
    """
    Market regime detection results.
    
    Attributes:
        labels: Regime label for each time point
        n_regimes: Number of detected regimes
        regime_stats: Statistics for each regime
        transition_points: Indices where regime changes occur
        regime_names: Descriptive names for regimes
    """
    labels: np.ndarray
    n_regimes: int
    regime_stats: Dict[int, Dict[str, float]]
    transition_points: List[int] = field(default_factory=list)
    regime_names: Optional[Dict[int, str]] = None
    
    def __post_init__(self):
        """Calculate transition points"""
        if len(self.transition_points) == 0 and len(self.labels) > 1:
            # Find points where regime changes
            changes = np.where(np.diff(self.labels) != 0)[0] + 1
            self.transition_points = changes.tolist()
    
    @property
    def regime_durations(self) -> Dict[int, float]:
        """Average duration of each regime"""
        durations = {}
        for regime in range(self.n_regimes):
            mask = self.labels == regime
            if np.any(mask):
                # Count consecutive occurrences
                changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                regime_durations = ends - starts
                durations[regime] = float(np.mean(regime_durations))
            else:
                durations[regime] = 0.0
        return durations
    
    def get_regime_at(self, index: int) -> int:
        """Get regime label at specific time index"""
        return int(self.labels[index])


@dataclass
class L3Summary:
    """
    High-level summary of Lambda³ analysis.
    
    Attributes:
        timestamp: Analysis timestamp
        n_series: Number of analyzed series
        total_length: Length of time series
        key_findings: Important discoveries
        recommendations: Actionable insights
    """
    timestamp: datetime
    n_series: int
    total_length: int
    key_findings: List[str]
    recommendations: List[str]
    config_summary: Optional[Dict[str, Any]] = None
    
    def to_text(self) -> str:
        """Generate text summary"""
        lines = [
            f"Lambda³ Analysis Summary",
            f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"Data: {self.n_series} series, {self.total_length} time points",
            f"",
            f"Key Findings:",
        ]
        for i, finding in enumerate(self.key_findings, 1):
            lines.append(f"  {i}. {finding}")
        
        lines.extend([
            f"",
            f"Recommendations:",
        ])
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        return "\n".join(lines)
