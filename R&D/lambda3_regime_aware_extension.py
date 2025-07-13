# ==========================================================
# Λ³ Regime-Aware Bayesian Extension (Modified)
# ----------------------------------------------------
# Pairwise regime detection and regime-specific Bayesian inference
# Based on WeatherAnalysis_ny.py implementation pattern
#
# Author: Extension for lambda3_zeroshot_tensor_field.py
# License: MIT
# Version: 2.0 (Pairwise Regime)
# ==========================================================

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit, njit

# Import from main Lambda³ module
from lambda3_zeroshot_tensor_field import (
    L3Config, calc_lambda3_features, Lambda3BayesianLogger,
    fit_l3_pairwise_bayesian_system, extract_interaction_coefficients,
    fit_l3_bayesian_regression_asymmetric
)

# ===============================
# REGIME CONFIGURATION
# ===============================
@dataclass
class RegimeConfig:
    """Configuration for pairwise regime detection"""
    # Regime detection parameters
    n_regimes: int = 3  # Number of regimes per pair
    regime_names: Optional[List[str]] = None  # Custom regime names
    
    # Detection settings
    min_regime_size: int = 30  # Minimum points for valid regime
    regime_overlap_window: int = 5  # Transition smoothing window
    use_adaptive_detection: bool = True  # Use adaptive detection
    detection_method: str = 'auto'  # 'auto', 'kmeans', 'gmm'
    
    # Temporal hints
    use_temporal_hint: bool = True  # Use time information for initialization
    
    # Quality scoring
    score_method: str = 'entropy'  # Method for evaluating clustering quality
    
    # Bayesian settings
    regime_specific_sampling: bool = True  # Adjust MCMC by regime
    adaptive_priors: bool = False  # Use regime-specific priors (optional)

# ===============================
# ADAPTIVE PAIRWISE REGIME DETECTOR
# ===============================
class AdaptivePairwiseRegimeDetector:
    """
    Adaptive regime detection for pairwise Lambda³ analysis.
    Based on WeatherAnalysis_ny.py implementation.
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.regime_labels = None
        self.regime_features = None
        self.detection_method_used = None
        
    def detect_pairwise_regimes(
        self,
        features_a: Dict[str, np.ndarray],
        features_b: Dict[str, np.ndarray],
        pair_name: Tuple[str, str],
        temporal_hint: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Detect structural regimes for a specific pair.
        
        Lambda³: Pair-specific structural states in semantic space
        """
        print(f"\nDetecting {self.config.n_regimes} structural regimes for {pair_name[0]} ⇄ {pair_name[1]}...")
        
        # Combine features from both series
        combined_features = self._combine_pairwise_features(features_a, features_b)
        
        # Create temporal hint if enabled
        if self.config.use_temporal_hint and temporal_hint is None:
            temporal_hint = features_a.get('time_trend', np.arange(len(features_a['data'])))
        
        # Detect regimes
        if self.config.use_adaptive_detection and self.config.detection_method == 'auto':
            regimes = self._auto_select_method(combined_features, temporal_hint)
        else:
            method = self.config.detection_method if self.config.detection_method != 'auto' else 'kmeans'
            regimes = self._detect_with_method(combined_features, temporal_hint, method)
            self.detection_method_used = method
        
        # Post-process: smooth transitions
        regimes = self._smooth_regime_transitions(regimes)
        
        # Calculate regime statistics
        self._calculate_regime_statistics(regimes, features_a, features_b)
        
        self.regime_labels = regimes
        
        # Print summary
        unique, counts = np.unique(regimes, return_counts=True)
        print(f"  Regime distribution: {dict(zip(unique, counts))}")
        
        return regimes
    
    def _combine_pairwise_features(
        self,
        features_a: Dict[str, np.ndarray],
        features_b: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Combine features from both series for regime detection"""
        
        # Stack relevant features
        feature_list = []
        
        # Individual series features
        for features in [features_a, features_b]:
            if 'delta_LambdaC_pos' in features:
                feature_list.append(features['delta_LambdaC_pos'])
            if 'delta_LambdaC_neg' in features:
                feature_list.append(features['delta_LambdaC_neg'])
            if 'rho_T' in features:
                feature_list.append(features['rho_T'])
        
        # Cross-correlation features
        if len(features_a['rho_T']) > 20:
            window = 20
            correlations = []
            for i in range(len(features_a['rho_T']) - window):
                corr = np.corrcoef(
                    features_a['rho_T'][i:i+window],
                    features_b['rho_T'][i:i+window]
                )[0, 1]
                correlations.append(corr)
            
            # Pad to match length
            correlations = np.pad(correlations, (0, len(features_a['rho_T']) - len(correlations)), 'edge')
            feature_list.append(correlations)
        
        # Stack and normalize
        X = np.column_stack(feature_list)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled
    
    def _auto_select_method(
        self,
        features: np.ndarray,
        temporal_hint: Optional[np.ndarray]
    ) -> np.ndarray:
        """Automatically select best detection method"""
        
        print("  Using adaptive detection with automatic method selection...")
        
        methods_to_try = ['gmm', 'kmeans']
        best_method = None
        best_score = -np.inf
        best_regimes = None
        
        for method in methods_to_try:
            print(f"    Trying {method}...")
            
            regimes = self._detect_with_method(features, temporal_hint, method)
            score = self._evaluate_clustering_quality(regimes, features)
            
            # Display distribution
            unique, counts = np.unique(regimes, return_counts=True)
            print(f"      Score: {score:.3f}, Distribution: {dict(zip(unique, counts))}")
            
            if score > best_score:
                best_score = score
                best_method = method
                best_regimes = regimes
        
        print(f"  Selected method: {best_method} (score: {best_score:.3f})")
        self.detection_method_used = best_method
        
        return best_regimes
    
    def _detect_with_method(
        self,
        features: np.ndarray,
        temporal_hint: Optional[np.ndarray],
        method: str
    ) -> np.ndarray:
        """Detect regimes using specified method"""
        
        if method == 'kmeans' and temporal_hint is not None:
            # Use temporal information for better initialization
            n_points = len(features)
            init_centers = []
            
            # Divide data into temporal segments
            for i in range(self.config.n_regimes):
                start_idx = int(i * n_points / self.config.n_regimes)
                end_idx = int((i + 1) * n_points / self.config.n_regimes)
                segment_mean = np.mean(features[start_idx:end_idx], axis=0)
                init_centers.append(segment_mean)
            
            init_centers = np.array(init_centers)
            
            # K-means with temporal initialization
            km = KMeans(
                n_clusters=self.config.n_regimes,
                init=init_centers,
                n_init=1,
                random_state=42
            )
            labels = km.fit_predict(features)
            
        elif method == 'gmm':
            # Gaussian Mixture Model
            gmm = GaussianMixture(
                n_components=self.config.n_regimes,
                covariance_type='full',
                n_init=10,
                random_state=42
            )
            labels = gmm.fit_predict(features)
            
        else:
            # Default k-means
            km = KMeans(
                n_clusters=self.config.n_regimes,
                init='k-means++',
                n_init=20,
                random_state=42
            )
            labels = km.fit_predict(features)
        
        # Post-process: merge small regimes
        labels = self._merge_small_regimes(labels)
        
        return labels
    
    def _evaluate_clustering_quality(
        self,
        labels: np.ndarray,
        features: np.ndarray
    ) -> float:
        """Evaluate clustering quality"""
        
        unique, counts = np.unique(labels, return_counts=True)
        proportions = counts / len(labels)
        
        if self.config.score_method == 'entropy':
            # Entropy-based score
            entropy = -np.sum(proportions * np.log(proportions + 1e-10))
            imbalance_penalty = np.std(proportions) * 2
            score = entropy - imbalance_penalty
        else:
            # Silhouette score
            from sklearn.metrics import silhouette_score
            try:
                score = silhouette_score(features, labels)
            except:
                score = -1.0
        
        return score
    
    def _merge_small_regimes(self, labels: np.ndarray) -> np.ndarray:
        """Merge regimes smaller than minimum size"""
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            if count < self.config.min_regime_size:
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
        
        return labels
    
    def _smooth_regime_transitions(self, regimes: np.ndarray) -> np.ndarray:
        """Smooth regime transitions to reduce noise"""
        return median_filter(regimes, size=self.config.regime_overlap_window)
    
    def _calculate_regime_statistics(
        self,
        regimes: np.ndarray,
        features_a: Dict[str, np.ndarray],
        features_b: Dict[str, np.ndarray]
    ):
        """Calculate statistics for each regime"""
        
        self.regime_features = {}
        
        for r in range(self.config.n_regimes):
            mask = (regimes == r)
            n_points = np.sum(mask)
            
            if n_points > 0:
                self.regime_features[r] = {
                    'n_points': n_points,
                    'frequency': n_points / len(regimes),
                    'mean_rhoT_a': np.mean(features_a['rho_T'][mask]),
                    'mean_rhoT_b': np.mean(features_b['rho_T'][mask]),
                    'std_rhoT_a': np.std(features_a['rho_T'][mask]),
                    'std_rhoT_b': np.std(features_b['rho_T'][mask]),
                    'jumps_a': np.sum(features_a['delta_LambdaC_pos'][mask] + 
                                     features_a['delta_LambdaC_neg'][mask]),
                    'jumps_b': np.sum(features_b['delta_LambdaC_pos'][mask] + 
                                     features_b['delta_LambdaC_neg'][mask])
                }
            else:
                self.regime_features[r] = {
                    'n_points': 0,
                    'frequency': 0,
                    'mean_rhoT_a': 0,
                    'mean_rhoT_b': 0
                }
    
    def detect_regime_transitions(self, smooth_window: int = 5) -> List[Tuple[int, int, int]]:
        """Detect transition points between regimes"""
        
        if self.regime_labels is None:
            raise ValueError("Must detect regimes first")
        
        # Smooth labels
        smoothed_labels = median_filter(self.regime_labels, size=smooth_window)
        
        # Find transitions
        transitions = []
        for i in range(1, len(smoothed_labels)):
            if smoothed_labels[i] != smoothed_labels[i-1]:
                transitions.append((i, smoothed_labels[i-1], smoothed_labels[i]))
        
        return transitions
    
    def get_regime_specific_config(
        self,
        regime_idx: int,
        base_config: L3Config
    ) -> L3Config:
        """Get regime-specific Bayesian configuration"""
        
        if not self.config.regime_specific_sampling:
            return base_config
        
        # Create a copy
        import copy
        regime_config = copy.deepcopy(base_config)
        
        # Adjust based on regime size
        if hasattr(self, 'regime_features') and regime_idx in self.regime_features:
            regime_size = self.regime_features[regime_idx]['n_points']
            
            if regime_size < 100:
                # Small regime: reduce samples
                regime_config.draws = min(regime_config.draws, 4000)
                regime_config.tune = min(regime_config.tune, 4000)
            elif regime_size > 500:
                # Large regime: can use more samples
                regime_config.draws = int(regime_config.draws * 1.2)
                regime_config.tune = int(regime_config.tune * 1.2)
        
        return regime_config

# ===============================
# REGIME-AWARE PAIRWISE ANALYSIS
# ===============================
class RegimeAwarePairwiseAnalysis:
    """
    Regime-specific pairwise Bayesian analysis for Lambda³ framework.
    Based on WeatherAnalysis_ny.py pattern.
    """
    
    def __init__(
        self,
        regime_config: RegimeConfig,
        bayes_logger: Optional[Lambda3BayesianLogger] = None
    ):
        self.regime_config = regime_config
        self.bayes_logger = bayes_logger
        self.regime_detector = AdaptivePairwiseRegimeDetector(regime_config)
    
    def analyze_pair_with_regimes(
        self,
        name_a: str,
        name_b: str,
        series_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, Dict[str, np.ndarray]],
        base_config: L3Config,
        show_plots: bool = True,
        analyze_regimes_separately: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a pair with regime detection.
        
        Process:
        1. Detect pair-specific regimes
        2. Overall analysis across all data
        3. Regime-specific analysis
        4. Compare across regimes
        """
        
        print(f"\n{'='*60}")
        print(f"ANALYZING PAIR WITH REGIME DETECTION: {name_a} ⇄ {name_b}")
        print(f"{'='*60}")
        
        # Get features
        feats_a = features_dict[name_a]
        feats_b = features_dict[name_b]
        
        # ===============================
        # Step 1: Regime Detection
        # ===============================
        regimes = self.regime_detector.detect_pairwise_regimes(
            feats_a, feats_b, (name_a, name_b)
        )
        
        # Get or generate regime names
        if self.regime_config.regime_names is None:
            regime_names = [f"Regime-{i+1}" for i in range(self.regime_config.n_regimes)]
        else:
            regime_names = self.regime_config.regime_names
        
        # Calculate regime statistics
        regime_stats = self._calculate_regime_statistics(
            regimes, regime_names, feats_a, feats_b, name_a, name_b
        )
        
        self._print_regime_summary(regime_stats)
        
        # ===============================
        # Step 2: Overall Analysis
        # ===============================
        print(f"\n2. Overall analysis across all regimes...")
        
        overall_results = self._analyze_overall(
            name_a, name_b, series_dict, features_dict, base_config, show_plots
        )
        
        # ===============================
        # Step 3: Regime-Specific Analysis
        # ===============================
        regime_results = {}
        
        if analyze_regimes_separately:
            print(f"\n3. Analyzing each structural regime separately...")
            
            regime_results = self._analyze_by_regime(
                name_a, name_b, series_dict, features_dict,
                regimes, regime_names, regime_stats, base_config
            )
        
        # ===============================
        # Step 4: Visualization
        # ===============================
        if show_plots and len(regime_results) > 0:
            self._plot_regime_comparison(
                regime_results, overall_results, regime_names, name_a, name_b
            )
        
        # Detect transitions
        transitions = self.regime_detector.detect_regime_transitions()
        
        return {
            'overall_results': overall_results,
            'regime_results': regime_results,
            'regime_labels': regimes,
            'regime_statistics': regime_stats,
            'regime_detector': self.regime_detector,
            'detection_method_used': self.regime_detector.detection_method_used,
            'transitions': transitions
        }
    
    def _calculate_regime_statistics(
        self,
        regimes: np.ndarray,
        regime_names: List[str],
        feats_a: Dict[str, np.ndarray],
        feats_b: Dict[str, np.ndarray],
        name_a: str,
        name_b: str
    ) -> Dict[str, Dict]:
        """Calculate detailed statistics for each regime"""
        
        regime_stats = {}
        
        for i, regime_name in enumerate(regime_names):
            mask = (regimes == i)
            n_points = np.sum(mask)
            
            if n_points > 0:
                regime_stats[regime_name] = {
                    'count': n_points,
                    'percentage': n_points / len(regimes) * 100,
                    f'mean_{name_a}': np.mean(feats_a['data'][mask]),
                    f'mean_{name_b}': np.mean(feats_b['data'][mask]),
                    f'mean_rhoT_{name_a}': np.mean(feats_a['rho_T'][mask]),
                    f'mean_rhoT_{name_b}': np.mean(feats_b['rho_T'][mask]),
                    f'jumps_{name_a}': np.sum(feats_a['delta_LambdaC_pos'][mask] + 
                                             feats_a['delta_LambdaC_neg'][mask]),
                    f'jumps_{name_b}': np.sum(feats_b['delta_LambdaC_pos'][mask] + 
                                             feats_b['delta_LambdaC_neg'][mask])
                }
        
        return regime_stats
    
    def _print_regime_summary(self, regime_stats: Dict[str, Dict]):
        """Print regime summary statistics"""
        
        print("\nDetected Structural Regimes:")
        for regime_name, stats in regime_stats.items():
            print(f"\n{regime_name}:")
            print(f"  - Points: {stats['count']} ({stats['percentage']:.1f}%)")
            
            # Print mean tensions
            rho_keys = [k for k in stats.keys() if k.startswith('mean_rhoT_')]
            for key in rho_keys:
                series_name = key.replace('mean_rhoT_', '')
                print(f"  - Mean ρT ({series_name}): {stats[key]:.3f}")
            
            # Print jumps
            jump_keys = [k for k in stats.keys() if k.startswith('jumps_')]
            for key in jump_keys:
                series_name = key.replace('jumps_', '')
                print(f"  - Total jumps ({series_name}): {stats[key]}")
    
    def _analyze_overall(
        self,
        name_a: str,
        name_b: str,
        series_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, Dict[str, np.ndarray]],
        config: L3Config,
        show_plots: bool
    ) -> Dict[str, Any]:
        """Perform overall analysis across all data"""
        
        # Fit asymmetric models
        trace_a, trace_b = self._fit_asymmetric_models(
            name_a, name_b, series_dict, features_dict, config
        )
        
        # Extract coefficients
        beta_b_on_a, beta_a_on_b, hdi_results = self._extract_interaction_effects(
            trace_a, trace_b, name_a, name_b, config
        )
        
        # Log with Bayesian logger if available
        if self.bayes_logger:
            model_id = f"overall_{name_a}_vs_{name_b}"
            self.bayes_logger.log_trace(
                trace_a,
                model_id=model_id + "_a",
                model_type="asymmetric_regression",
                series_names=[name_a, name_b],
                verbose=False
            )
        
        return {
            'beta_b_on_a': beta_b_on_a,
            'beta_a_on_b': beta_a_on_b,
            'hdi_results': hdi_results,
            'trace_a': trace_a,
            'trace_b': trace_b
        }
    
    def _analyze_by_regime(
        self,
        name_a: str,
        name_b: str,
        series_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, Dict[str, np.ndarray]],
        regimes: np.ndarray,
        regime_names: List[str],
        regime_stats: Dict[str, Dict],
        base_config: L3Config
    ) -> Dict[str, Dict]:
        """Analyze each regime separately"""
        
        regime_results = {}
        
        for regime_idx, regime_name in enumerate(regime_names):
            mask = (regimes == regime_idx)
            n_regime_points = np.sum(mask)
            
            # Skip if too few points
            if n_regime_points < 20:
                print(f"\nSkipping {regime_name}: insufficient data ({n_regime_points} points)")
                continue
            
            print(f"\n{'='*50}")
            print(f"REGIME: {regime_name} ({n_regime_points} points)")
            print(f"{'='*50}")
            
            # Create regime-specific data and features
            regime_data = {
                name_a: series_dict[name_a][mask],
                name_b: series_dict[name_b][mask]
            }
            
            regime_features = {
                name_a: self._extract_regime_features(features_dict[name_a], mask),
                name_b: self._extract_regime_features(features_dict[name_b], mask)
            }
            
            # Get regime-specific config
            regime_config = self.regime_detector.get_regime_specific_config(
                regime_idx, base_config
            )
            
            try:
                # Fit regime-specific models
                trace_a_regime, trace_b_regime = self._fit_asymmetric_models(
                    name_a, name_b, regime_data, regime_features, regime_config
                )
                
                # Extract results
                beta_b_on_a, beta_a_on_b, hdi_results = self._extract_interaction_effects(
                    trace_a_regime, trace_b_regime, name_a, name_b, regime_config
                )
                
                # Store results
                regime_results[regime_name] = {
                    'beta_b_on_a_pos': beta_b_on_a,
                    'beta_a_on_b_pos': beta_a_on_b,
                    'hdi': hdi_results,
                    'n_points': n_regime_points
                }
                
                # Log with Bayesian logger
                if self.bayes_logger:
                    model_id = f"regime_{regime_name}_{name_a}_vs_{name_b}"
                    self.bayes_logger.log_trace(
                        trace_a_regime,
                        model_id=model_id,
                        model_type="regime_specific",
                        series_names=[name_a, name_b],
                        verbose=False
                    )
                
                # Print results
                print(f"\n{regime_name} - Asymmetric Interaction Effects:")
                print(f"  {name_b} → {name_a}: β = {beta_b_on_a:.3f}")
                print(f"  {name_a} → {name_b}: β = {beta_a_on_b:.3f}")
                
                # Print HDI
                if hdi_results:
                    print(f"\n{regime_name} - HDI ({regime_config.hdi_prob*100:.0f}%):")
                    for key, (low, high) in hdi_results.items():
                        print(f"  {key}: [{low:.3f}, {high:.3f}]")
                
            except Exception as e:
                print(f"Error analyzing regime {regime_name}: {e}")
                continue
        
        return regime_results
    
    def _extract_regime_features(
        self,
        features: Dict[str, np.ndarray],
        mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Extract features for regime-specific analysis"""
        
        regime_features = {}
        
        for key in ['delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T', 'data']:
            if key in features:
                regime_features[key] = features[key][mask]
        
        # Reset time trend
        regime_features['time_trend'] = np.arange(np.sum(mask))
        
        # Include local jump if available
        if 'local_jump' in features:
            regime_features['local_jump'] = features['local_jump'][mask]
        
        return regime_features
    
    def _fit_asymmetric_models(
        self,
        name_a: str,
        name_b: str,
        data_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, Dict[str, np.ndarray]],
        config: L3Config
    ) -> Tuple[Any, Any]:
        """Fit asymmetric interaction models"""
        
        feats_a = features_dict[name_a]
        feats_b = features_dict[name_b]
        
        # Model for A (with B interaction)
        trace_a = fit_l3_bayesian_regression_asymmetric(
            data=data_dict[name_a],
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
        
        # Model for B (with A interaction)
        trace_b = fit_l3_bayesian_regression_asymmetric(
            data=data_dict[name_b],
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
        
        return trace_a, trace_b
    
    def _extract_interaction_effects(
        self,
        trace_a: Any,
        trace_b: Any,
        name_a: str,
        name_b: str,
        config: L3Config
    ) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
        """Extract interaction coefficients and HDI"""
        
        summary_a = az.summary(trace_a, hdi_prob=config.hdi_prob)
        summary_b = az.summary(trace_b, hdi_prob=config.hdi_prob)
        
        # Extract main effects (using positive as primary)
        beta_b_on_a = summary_a.loc['beta_interact_pos', 'mean'] if 'beta_interact_pos' in summary_a.index else 0
        beta_a_on_b = summary_b.loc['beta_interact_pos', 'mean'] if 'beta_interact_pos' in summary_b.index else 0
        
        # Extract HDI
        hdi_results = {}
        
        def get_hdi_bounds(summary, param_name):
            if param_name in summary.index:
                hdi_cols = [col for col in summary.columns if 'hdi' in col.lower()]
                if len(hdi_cols) >= 2:
                    return summary.loc[param_name, hdi_cols[0]], summary.loc[param_name, hdi_cols[1]]
            return None, None
        
        # B → A effects
        low, high = get_hdi_bounds(summary_a, 'beta_interact_pos')
        if low is not None:
            hdi_results[f'{name_b}_to_{name_a}_pos'] = (low, high)
        
        low, high = get_hdi_bounds(summary_a, 'beta_interact_neg')
        if low is not None:
            hdi_results[f'{name_b}_to_{name_a}_neg'] = (low, high)
        
        # A → B effects
        low, high = get_hdi_bounds(summary_b, 'beta_interact_pos')
        if low is not None:
            hdi_results[f'{name_a}_to_{name_b}_pos'] = (low, high)
        
        low, high = get_hdi_bounds(summary_b, 'beta_interact_neg')
        if low is not None:
            hdi_results[f'{name_a}_to_{name_b}_neg'] = (low, high)
        
        return beta_b_on_a, beta_a_on_b, hdi_results
    
    def _plot_regime_comparison(
        self,
        regime_results: Dict,
        overall_results: Dict,
        regime_names: List[str],
        name_a: str,
        name_b: str
    ):
        """Plot comparison across regimes"""
        
        # Prepare data
        regimes = ['Overall'] + list(regime_results.keys())
        beta_b_on_a = [overall_results['beta_b_on_a']]
        beta_a_on_b = [overall_results['beta_a_on_b']]
        
        for regime in regime_results.keys():
            beta_b_on_a.append(regime_results[regime]['beta_b_on_a_pos'])
            beta_a_on_b.append(regime_results[regime]['beta_a_on_b_pos'])
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot of interaction coefficients
        ax1 = axes[0]
        x = np.arange(len(regimes))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, beta_b_on_a, width,
                        label=f'{name_b} → {name_a}', color='royalblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, beta_a_on_b, width,
                        label=f'{name_a} → {name_b}', color='darkorange', alpha=0.8)
        
        ax1.set_xlabel('Structural Regime')
        ax1.set_ylabel('Interaction Coefficient β')
        ax1.set_title('Cross-Series Interactions by Regime')
        ax1.set_xticks(x)
        ax1.set_xticklabels(regimes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -15),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=8)
        
        # HDI width comparison
        ax2 = axes[1]
        hdi_widths_b_on_a = []
        hdi_widths_a_on_b = []
        regime_labels_hdi = []
        
        for regime_name in regime_results.keys():
            if 'hdi' in regime_results[regime_name]:
                hdi_dict = regime_results[regime_name]['hdi']
                
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
            ax2.bar(x_hdi - width/2, hdi_widths_b_on_a, width,
                   label=f'{name_b} → {name_a}', color='royalblue', alpha=0.8)
            ax2.bar(x_hdi + width/2, hdi_widths_a_on_b, width,
                   label=f'{name_a} → {name_b}', color='darkorange', alpha=0.8)
            
            ax2.set_xlabel('Structural Regime')
            ax2.set_ylabel('HDI Width')
            ax2.set_title('Uncertainty by Regime')
            ax2.set_xticks(x_hdi)
            ax2.set_xticklabels(regime_labels_hdi, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Regime Analysis: {name_a} ⇄ {name_b}', fontsize=16)
        plt.tight_layout()
        plt.show()

# ===============================
# MAIN ANALYSIS FUNCTION
# ===============================
def run_lambda3_regime_aware_analysis(
    data_source: Union[str, Dict[str, np.ndarray]],
    base_config: L3Config = None,
    regime_config: RegimeConfig = None,
    target_series: List[str] = None,
    max_pairs: int = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Lambda³ analysis with pairwise regime detection.
    
    Parameters:
    -----------
    data_source : Data source
    base_config : Lambda³ configuration
    regime_config : Regime detection configuration
    target_series : Series to analyze
    max_pairs : Maximum pairs to analyze
    verbose : Verbose output
    
    Returns:
    --------
    Complete analysis results
    """
    
    if base_config is None:
        base_config = L3Config()
    
    if regime_config is None:
        regime_config = RegimeConfig()
    
    # Load data
    if isinstance(data_source, str):
        from lambda3_zeroshot_tensor_field import load_csv_data
        series_dict = load_csv_data(data_source)
    else:
        series_dict = data_source
    
    if target_series:
        series_dict = {k: v for k, v in series_dict.items() if k in target_series}
    
    series_names = list(series_dict.keys())
    
    if verbose:
        print("="*80)
        print("Lambda³ Regime-Aware Analysis (Pairwise)")
        print(f"Series: {len(series_names)}, Length: {len(next(iter(series_dict.values())))}")
        print("="*80)
    
    # Initialize Bayesian logger
    bayes_logger = Lambda3BayesianLogger(hdi_prob=base_config.hdi_prob)
    
    # Extract features
    features_dict = {}
    for name, data in series_dict.items():
        features_dict[name] = calc_lambda3_features(data, base_config)
    
    # Initialize analyzer
    analyzer = RegimeAwarePairwiseAnalysis(regime_config, bayes_logger)
    
    # Analyze pairs
    from itertools import combinations
    all_pairs = list(combinations(series_names, 2))
    
    if max_pairs and len(all_pairs) > max_pairs:
        all_pairs = all_pairs[:max_pairs]
        print(f"\nLimiting to {max_pairs} pairs")
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {len(all_pairs)} PAIRS WITH REGIME DETECTION")
    print(f"{'='*60}")
    
    pairwise_results = {}
    
    for i, (name_a, name_b) in enumerate(all_pairs, 1):
        print(f"\n[{i}/{len(all_pairs)}] Analyzing: {name_a} ⇄ {name_b}")
        
        try:
            results = analyzer.analyze_pair_with_regimes(
                name_a, name_b,
                series_dict, features_dict,
                base_config,
                show_plots=(i <= 3),  # Show plots for first 3 pairs
                analyze_regimes_separately=True
            )
            
            pairwise_results[(name_a, name_b)] = results
            
        except Exception as e:
            print(f"Error analyzing pair {name_a} ⇄ {name_b}: {e}")
            continue
    
    # Generate summary
    if verbose and pairwise_results:
        create_regime_summary(pairwise_results, series_names)
    
    # Compile results
    results = {
        'series_dict': series_dict,
        'series_names': series_names,
        'features_dict': features_dict,
        'pairwise_regime_results': pairwise_results,
        'config': base_config,
        'regime_config': regime_config,
        'bayes_logger': bayes_logger
    }
    
    return results

# ===============================
# SUMMARY FUNCTIONS
# ===============================
def create_regime_summary(pairwise_results: Dict, series_names: List[str]):
    """Create comprehensive summary of regime analysis"""
    
    print("\n" + "="*70)
    print("PAIRWISE REGIME ANALYSIS SUMMARY")
    print("="*70)
    
    # Collect unique regimes
    all_regimes = set()
    for pair_key, results in pairwise_results.items():
        if 'regime_results' in results:
            all_regimes.update(results['regime_results'].keys())
    
    all_regimes = sorted(list(all_regimes))
    
    print(f"\nDetected {len(all_regimes)} unique regimes across all pairs:")
    for regime in all_regimes:
        print(f"  • {regime}")
    
    # Summary table
    print("\n" + "-"*70)
    print("INTERACTION EFFECTS BY REGIME")
    print("-"*70)
    
    for pair_key, results in pairwise_results.items():
        name_a, name_b = pair_key
        print(f"\n{name_a} ⇄ {name_b}:")
        
        # Overall
        overall = results['overall_results']
        print(f"  Overall: {name_b}→{name_a} β={overall['beta_b_on_a']:.3f}, "
              f"{name_a}→{name_b} β={overall['beta_a_on_b']:.3f}")
        
        # By regime
        if 'regime_results' in results:
            for regime_name, regime_data in results['regime_results'].items():
                print(f"  {regime_name}: {name_b}→{name_a} β={regime_data['beta_b_on_a_pos']:.3f}, "
                      f"{name_a}→{name_b} β={regime_data['beta_a_on_b_pos']:.3f}")
    
    # Key insights
    print("\n" + "-"*70)
    print("KEY INSIGHTS")
    print("-"*70)
    
    # Find strongest regime effects
    max_effect = 0
    max_regime = None
    max_pair = None
    
    for pair_key, results in pairwise_results.items():
        if 'regime_results' in results:
            for regime_name, regime_data in results['regime_results'].items():
                effect = abs(regime_data['beta_b_on_a_pos']) + abs(regime_data['beta_a_on_b_pos'])
                if effect > max_effect:
                    max_effect = effect
                    max_regime = regime_name
                    max_pair = pair_key
    
    if max_regime:
        print(f"\n• Strongest coupling in {max_regime} for {max_pair[0]} ⇄ {max_pair[1]}")
    
    # Theoretical interpretation
    print("\n" + "-"*70)
    print("LAMBDA³ THEORETICAL INTERPRETATION")
    print("-"*70)
    print("• Pairwise regimes represent local structural configurations")
    print("• Different β values indicate context-dependent resonance")
    print("• Regime transitions are structural phase changes in pair space")
    print("• Narrower HDI suggests increased structural coherence")
    
    print("\n" + "="*70)

# ===============================
# EXAMPLE USAGE
# ===============================
if __name__ == "__main__":
    print("Lambda³ Regime-Aware Extension (Pairwise)")
    print("="*60)
    
    # Example configuration
    regime_config = RegimeConfig(
        n_regimes=3,
        regime_names=['Stable', 'Transition', 'Volatile'],
        use_adaptive_detection=True,
        detection_method='auto',
        min_regime_size=30
    )
    
    print("Configuration:")
    print(f"  Regimes: {regime_config.n_regimes}")
    print(f"  Detection: {regime_config.detection_method}")
    print(f"  Adaptive: {regime_config.use_adaptive_detection}")
    
    print("\nThis module provides pairwise regime-aware analysis")
    print("for lambda3_zeroshot_tensor_field.py")
