"""
Type definitions and data classes for Lambda³ framework.

This module provides type-safe data structures for feature sets,
analysis results, and synchronization profiles.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any, Union, TYPE_CHECKING
import numpy as np
import networkx as nx
from datetime import datetime

# Type checking imports
if TYPE_CHECKING:
    import arviz
    import pandas


@dataclass
class Lambda3FeatureSet:
    """
    Lambda³ feature set containing all extracted structural features.
    
    構造テンソルΛの成分分解：
    - ΛC: 構造変化テンソル（jumps）
    - ΛF: 進行ベクトル（trend）
    - ρT: テンションスカラー
    
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
        """Validate array lengths and data types"""
        # Length validation
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
        
        # Type validation for binary arrays
        for name, arr in [
            ('delta_LambdaC_pos', self.delta_LambdaC_pos),
            ('delta_LambdaC_neg', self.delta_LambdaC_neg),
            ('local_jump', self.local_jump)
        ]:
            if not np.all(np.isin(arr, [0, 1])):
                raise ValueError(f"{name} must be binary (0 or 1)")
        
        # Ensure data types
        self.data = np.asarray(self.data, dtype=np.float64)
        self.delta_LambdaC_pos = np.asarray(self.delta_LambdaC_pos, dtype=np.int32)
        self.delta_LambdaC_neg = np.asarray(self.delta_LambdaC_neg, dtype=np.int32)
        self.rho_T = np.asarray(self.rho_T, dtype=np.float64)
        self.time_trend = np.asarray(self.time_trend, dtype=np.float64)
        self.local_jump = np.asarray(self.local_jump, dtype=np.int32)
    
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
    def n_local_jumps(self) -> int:
        """Count of local jump events"""
        return int(np.sum(self.local_jump))
    
    @property
    def mean_tension(self) -> float:
        """Average tension scalar value"""
        return float(np.mean(self.rho_T))
    
    @property
    def max_tension(self) -> float:
        """Maximum tension scalar value"""
        return float(np.max(self.rho_T))
    
    @property
    def jump_asymmetry(self) -> float:
        """Asymmetry measure: (pos - neg) / (pos + neg)"""
        total = self.n_pos_jumps + self.n_neg_jumps
        if total == 0:
            return 0.0
        return (self.n_pos_jumps - self.n_neg_jumps) / total
    
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
    
    def get_high_tension_periods(self, percentile: float = 90) -> np.ndarray:
        """Get indices where tension exceeds given percentile"""
        threshold = np.percentile(self.rho_T, percentile)
        return np.where(self.rho_T > threshold)[0]


@dataclass
class SyncProfile:
    """
    Synchronization profile between two time series.
    
    Lambda³理論において、同期は構造テンソルΛの
    相関構造として現れる。
    
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
    
    @property
    def sync_strength(self) -> str:
        """Categorize synchronization strength"""
        if self.max_sync_rate < 0.1:
            return "none"
        elif self.max_sync_rate < 0.3:
            return "weak"
        elif self.max_sync_rate < 0.5:
            return "moderate"
        elif self.max_sync_rate < 0.7:
            return "strong"
        else:
            return "very_strong"
    
    def get_sync_at_lag(self, lag: int) -> float:
        """Get synchronization rate at specific lag"""
        return self.profile.get(lag, 0.0)
    
    def get_significant_lags(self, threshold: float = 0.2) -> List[int]:
        """Get lags with sync rate above threshold"""
        return [lag for lag, sync in self.profile.items() if sync > threshold]


@dataclass
class CausalityProfile:
    """
    Causality analysis results for a series or pair.
    
    構造因果性：ΔΛCの伝播パターンとして
    因果関係を定量化。
    
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
    
    @property
    def max_causality_strength(self) -> float:
        """Get maximum causality probability"""
        if not self.self_causality:
            return 0.0
        return max(self.self_causality.values())
    
    def is_causal(self, threshold: float = 0.3) -> bool:
        """Check if significant causality exists"""
        return self.max_causality_strength > threshold


@dataclass
class BayesianResults:
    """
    Container for Bayesian model results.
    
    構造進化の統計的表現を保持。
    
    Attributes:
        trace: ArviZ InferenceData object
        summary: Summary statistics DataFrame
        predictions: Model predictions
        residuals: Prediction residuals
        diagnostics: MCMC diagnostics
    """
    trace: 'arviz.InferenceData'  # Type hint as string to avoid circular import
    summary: 'pandas.DataFrame'    # Same for pandas
    predictions: np.ndarray
    residuals: Optional[np.ndarray] = None
    diagnostics: Optional[Dict[str, Any]] = None
    
    @property
    def converged(self) -> bool:
        """Check if MCMC chains converged (R-hat < 1.01)"""
        if self.diagnostics and 'max_r_hat' in self.diagnostics:
            return bool(self.diagnostics['max_r_hat'] < 1.01)
        return True  # Assume converged if no diagnostics
    
    @property
    def n_divergences(self) -> int:
        """Get number of divergent transitions"""
        if self.diagnostics and 'n_divergences' in self.diagnostics:
            return self.diagnostics['n_divergences']
        return 0
    
    @property
    def effective_samples(self) -> Optional[int]:
        """Get minimum effective sample size"""
        if self.diagnostics and 'min_ess_bulk' in self.diagnostics:
            return int(self.diagnostics['min_ess_bulk'])
        return None
    
    def get_parameter_summary(self, param_name: str) -> Optional[Dict[str, float]]:
        """Get summary statistics for a specific parameter"""
        if self.summary is not None and param_name in self.summary.index:
            return self.summary.loc[param_name].to_dict()
        return None


@dataclass
class AnalysisResult:
    """
    Complete analysis results for a series pair.
    
    構造相互作用の包括的分析結果。
    
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
        elif self.sync_profile.series_names:
            return self.sync_profile.series_names
        return ('Series A', 'Series B')
    
    @property
    def primary_interaction(self) -> Tuple[str, float]:
        """Get the strongest interaction effect"""
        if not self.interaction_effects:
            return ('none', 0.0)
        key = max(self.interaction_effects.items(), key=lambda x: abs(x[1]))[0]
        return (key, self.interaction_effects[key])
    
    def get_significant_interactions(self, threshold: float = 0.1) -> Dict[str, float]:
        """Get structurally significant interactions"""
        return {
            key: value for key, value in self.interaction_effects.items()
            if abs(value) > threshold
        }
    
    def convergence_summary(self) -> Dict[str, bool]:
        """Get convergence status summary"""
        return {
            'trace_a': self.trace_a.converged,
            'trace_b': self.trace_b.converged,
            'both_converged': self.trace_a.converged and self.trace_b.converged
        }
    
    @property
    def is_bidirectional(self) -> bool:
        """Check if interaction is bidirectional"""
        name_a, name_b = self.series_names
        has_a_to_b = any(f"{name_a}_to_{name_b}" in k for k in self.interaction_effects)
        has_b_to_a = any(f"{name_b}_to_{name_a}" in k for k in self.interaction_effects)
        return has_a_to_b and has_b_to_a


@dataclass
class CrossAnalysisResult:
    """
    Results from multi-series cross-analysis.
    
    複数系列の構造ネットワーク分析。
    
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
    
    @property
    def network_density(self) -> float:
        """Network density (fraction of possible edges)"""
        if self.network:
            n = self.network.number_of_nodes()
            if n > 1:
                return self.network.number_of_edges() / (n * (n - 1))
        return 0.0
    
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
    
    def get_hub_nodes(self, threshold: int = 2) -> List[Tuple[str, int]]:
        """Get hub nodes with high connectivity"""
        if not self.network:
            return []
        in_degrees = dict(self.network.in_degree())
        return [(node, deg) for node, deg in in_degrees.items() if deg >= threshold]
    
    def get_strongest_sync_pair(self) -> Tuple[str, str, float]:
        """Get the pair with strongest synchronization"""
        series_names = self.get_series_names()
        # Set diagonal to -1 to exclude self-sync
        sync_mat = self.sync_matrix.copy()
        np.fill_diagonal(sync_mat, -1)
        max_idx = np.unravel_index(np.argmax(sync_mat), sync_mat.shape)
        return (series_names[max_idx[0]], series_names[max_idx[1]], sync_mat[max_idx])


@dataclass
class RegimeInfo:
    """
    Market regime detection results.
    
    Lambda³理論において、regimeは構造テンソルΛの
    準安定状態を表す。
    
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
        """Calculate transition points as ΔΛC singularities"""
        if len(self.transition_points) == 0 and len(self.labels) > 1:
            # Find points where regime changes
            changes = np.where(np.diff(self.labels) != 0)[0] + 1
            self.transition_points = changes.tolist()
        
        # Ensure labels are integers
        self.labels = np.asarray(self.labels, dtype=np.int32)
    
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
    
    @property
    def transition_matrix(self) -> np.ndarray:
        """Calculate regime transition probability matrix"""
        trans_mat = np.zeros((self.n_regimes, self.n_regimes))
        for i in range(len(self.labels) - 1):
            from_regime = self.labels[i]
            to_regime = self.labels[i + 1]
            trans_mat[from_regime, to_regime] += 1
        
        # Normalize rows
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        trans_mat = trans_mat / row_sums
        
        return trans_mat
    
    def get_regime_at(self, index: int) -> int:
        """Get regime label at specific time index"""
        return int(self.labels[index])
    
    def get_regime_name(self, regime_id: int) -> str:
        """Get descriptive name for regime"""
        if self.regime_names and regime_id in self.regime_names:
            return self.regime_names[regime_id]
        return f"Regime-{regime_id}"
    
    def is_stable_period(self, start: int, end: int) -> bool:
        """Check if period has no regime changes"""
        return len(set(self.labels[start:end])) == 1


@dataclass
class L3Summary:
    """
    High-level summary of Lambda³ analysis.
    
    構造進化分析の総括。
    
    Attributes:
        timestamp: Analysis timestamp
        n_series: Number of analyzed series
        total_length: Length of time series
        key_findings: Important discoveries
        recommendations: Actionable insights
        config_summary: Configuration used
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
        
        if self.config_summary:
            lines.extend([
                f"",
                f"Configuration:",
                f"  Bayesian chains: {self.config_summary.get('num_chains', 'N/A')}",
                f"  Samples per chain: {self.config_summary.get('draws', 'N/A')}",
                f"  Feature window: {self.config_summary.get('window', 'N/A')}",
            ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'n_series': self.n_series,
            'total_length': self.total_length,
            'key_findings': self.key_findings,
            'recommendations': self.recommendations,
            'config_summary': self.config_summary
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'L3Summary':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            n_series=data['n_series'],
            total_length=data['total_length'],
            key_findings=data['key_findings'],
            recommendations=data['recommendations'],
            config_summary=data.get('config_summary')
        )
