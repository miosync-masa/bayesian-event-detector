"""
Lambda3 Detectors v2.0 - 完全統合版
===========================================
プロジェクトファイル lambda3_zeroshot_tensor_field.py の高度実装を統合

3段階検出器：
1. Lambda3Detector: 基本版（後方互換性）
2. Lambda3DetectorBidirectional: 両方向検出
3. Lambda3DetectorHierarchical: 階層的完全版（NEW！）

Author: 環ちゃん with ご主人さま
Updated: 2024-11-23
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time

# パス設定（lambda3_abc は2階層上のルート、lambda3_zeroshot_tensor_field は R&D/ にある）
import sys
import os
_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
_rd_path = os.path.join(_root_path, 'R&D')

# 両方のパスを追加
for path in [_root_path, _rd_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Advanced functions from project file (R&D/ にある) - 先にインポート
try:
    from lambda3_zeroshot_tensor_field import (
        calc_lambda3_features,  # 階層的特徴抽出
        fit_l3_pairwise_bayesian_system,  # ペアワイズベイズ
        Lambda3BayesianLogger,  # HDIロギング
        detect_basic_structural_causality,  # 因果分析
        analyze_comprehensive_causality,  # 統合因果分析
        complete_hierarchical_analysis,  # 階層的完全分析
        calculate_structural_hierarchy_metrics  # 階層メトリクス
    )
    ADVANCED_AVAILABLE = True
    
    # L3Config も lambda3_zeroshot_tensor_field から取得（hierarchical対応版）
    try:
        from lambda3_zeroshot_tensor_field import L3Config as L3ConfigHierarchical
        L3Config = L3ConfigHierarchical  # 拡張版を使う
        print("✅ Using hierarchical L3Config from lambda3_zeroshot_tensor_field")
    except ImportError:
        # フォールバック：lambda3_abc から取得
        from lambda3_abc import L3Config
        print("⚠️  Using basic L3Config from lambda3_abc (hierarchical parameter not supported)")
        
except ImportError:
    ADVANCED_AVAILABLE = False
    print("Warning: Advanced Lambda3 functions not available. Using basic mode.")
    # 基本版にフォールバック
    from lambda3_abc import L3Config

# Lambda3 core functions from project file (基本機能は lambda3_abc から)
from lambda3_abc import (
    calc_lambda3_features_v2,
    fit_l3_bayesian_regression_asymmetric,
    calculate_sync_profile
)
# L3Config は上で既にインポート済み

import arviz as az


# ===============================
# 共通ユーティリティ
# ===============================
def _estimate_beta_on_raw_data(data_A: np.ndarray, 
                                data_B: np.ndarray, 
                                lag: int) -> float:
    """
    lagが判明した後、生データで線形回帰してβを推定
    
    Parameters:
    -----------
    data_A : np.ndarray
        原因系列（生データ）
    data_B : np.ndarray
        結果系列（生データ）
    lag : int
        遅延（Lambda3で検出済み）
    
    Returns:
    --------
    beta_refined : float
        生データ空間での真の係数
    
    Example:
    --------
    # B(t) = β * A(t-lag) + noise の β を推定
    beta_true = _estimate_beta_on_raw_data(data_A, data_B, lag=5)
    """
    from sklearn.linear_model import LinearRegression
    
    # lag分ずらして線形回帰
    if lag > 0:
        X = data_A[:-lag].reshape(-1, 1)
        Y = data_B[lag:]
    elif lag < 0:
        # 負のlag（B→A の場合）
        X = data_A[-lag:].reshape(-1, 1)
        Y = data_B[:lag]
    else:
        # lag=0（同時）
        X = data_A.reshape(-1, 1)
        Y = data_B
    
    # 線形回帰
    model = LinearRegression()
    model.fit(X, Y)
    
    return model.coef_[0]


# ===============================
# 基本検出器（後方互換性）
# ===============================
class Lambda3Detector:
    """
    Lambda3検出器（基本版）
    
    特徴：
    - 単方向検出（A→B）
    - 基本的なΔΛC + ρT 解析
    - 高速動作
    
    用途：
    - クイックテスト
    - 後方互換性が必要な場合
    """
    def __init__(self, config: L3Config = None, verbose: bool = False):
        self.config = config or L3Config(draws=4000, tune=4000)
        self.name = "Lambda3_Basic"
        self.verbose = verbose
    
    def detect(self, data: np.ndarray) -> dict:
        """
        A→B の単方向検出
        
        Parameters:
        -----------
        data : np.ndarray (T, 2)
            2系列の時系列データ
        
        Returns:
        --------
        dict : 検出結果
            - detected_edges: [(source, target, lag, beta)]
            - beta: 総合結合強度
            - beta_pos/neg/stress: 個別係数
            - lag: 最適ラグ
            - sync_rate: 同期率
            - computation_time: 計算時間
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"[{self.name}] Starting detection...")
        
        # A系列の特徴抽出
        feats_a = calc_lambda3_features_v2(data[:, 0], self.config)
        features_a = {
            'delta_LambdaC_pos': feats_a[0],
            'delta_LambdaC_neg': feats_a[1],
            'rho_T': feats_a[2],
            'time_trend': feats_a[3]
        }
        
        # B系列の特徴抽出
        feats_b = calc_lambda3_features_v2(data[:, 1], self.config)
        features_b = {
            'delta_LambdaC_pos': feats_b[0],
            'delta_LambdaC_neg': feats_b[1],
            'rho_T': feats_b[2],
            'time_trend': feats_b[3]
        }
        
        # ベイズ推論（A→B）
        if self.verbose:
            print(f"  Running Bayesian inference...")
        
        trace = fit_l3_bayesian_regression_asymmetric(
            data=data[:, 1],
            features_dict=features_b,
            config=self.config,
            interaction_pos=features_a['delta_LambdaC_pos'],
            interaction_neg=features_a['delta_LambdaC_neg'],
            interaction_rhoT=features_a['rho_T']
        )
        
        # 結合強度抽出
        summary = az.summary(trace)
        beta_pos = summary.loc['beta_interact_pos', 'mean']
        beta_neg = summary.loc['beta_interact_neg', 'mean']
        beta_stress = summary.loc['beta_interact_stress', 'mean']
        beta_total = abs(beta_pos) + abs(beta_neg) + abs(beta_stress)
        
        # 同期率とラグ
        sync_profile, max_sync, optimal_lag = calculate_sync_profile(
            features_a['delta_LambdaC_pos'].astype(np.float64),
            features_b['delta_LambdaC_pos'].astype(np.float64),
            lag_window=10
        )
        
        computation_time = time.time() - start_time
        
        # NEW: 生データでβ再推定
        beta_refined = _estimate_beta_on_raw_data(
            data[:, 0], 
            data[:, 1], 
            optimal_lag
        )
        
        if self.verbose:
            print(f"  Detection complete: β={beta_total:.3f}, lag={optimal_lag}")
            print(f"  β (Lambda): {beta_total:.3f}")
            print(f"  β (Refined): {beta_refined:.3f}")
            print(f"  Computation time: {computation_time:.2f}s")
        
        return {
            'detected_edges': [(0, 1, optimal_lag, beta_total)],
            'beta': beta_refined,  # メインは refined を使う
            'beta_lambda': beta_total,  # Λ空間の係数
            'beta_refined': beta_refined,  # 生データの係数
            'beta_pos': beta_pos,
            'beta_neg': beta_neg,
            'beta_stress': beta_stress,
            'lag': optimal_lag,
            'sync_rate': max_sync,
            'trace': trace,
            'computation_time': computation_time
        }


# ===============================
# 両方向検出器
# ===============================
class Lambda3DetectorBidirectional:
    """
    Lambda3検出器（両方向版）
    
    特徴：
    - A→B と B→A の両方向検出
    - 非対称性の定量化
    - 主方向の自動判定
    
    用途：
    - Unidirectional Domino 検出
    - 非対称性の評価
    """
    def __init__(self, config: L3Config = None, verbose: bool = False):
        self.config = config or L3Config(draws=4000, tune=4000)
        self.name = "Lambda3_Bidirectional"
        self.verbose = verbose
    
    def _detect_single_direction(self, data: np.ndarray, 
                                 source_idx: int, target_idx: int) -> dict:
        """
        単一方向の因果関係を検出
        
        Parameters:
        -----------
        data : np.ndarray
            時系列データ
        source_idx : int
            原因系列のインデックス
        target_idx : int
            結果系列のインデックス
        
        Returns:
        --------
        dict : 検出結果
        """
        # Source系列の特徴抽出
        feats_source = calc_lambda3_features_v2(data[:, source_idx], self.config)
        features_source = {
            'delta_LambdaC_pos': feats_source[0],
            'delta_LambdaC_neg': feats_source[1],
            'rho_T': feats_source[2],
            'time_trend': feats_source[3]
        }
        
        # Target系列の特徴抽出
        feats_target = calc_lambda3_features_v2(data[:, target_idx], self.config)
        features_target = {
            'delta_LambdaC_pos': feats_target[0],
            'delta_LambdaC_neg': feats_target[1],
            'rho_T': feats_target[2],
            'time_trend': feats_target[3]
        }
        
        # ベイズ推論
        trace = fit_l3_bayesian_regression_asymmetric(
            data=data[:, target_idx],
            features_dict=features_target,
            config=self.config,
            interaction_pos=features_source['delta_LambdaC_pos'],
            interaction_neg=features_source['delta_LambdaC_neg'],
            interaction_rhoT=features_source['rho_T']
        )
        
        # 結合強度
        summary = az.summary(trace)
        beta_pos = summary.loc['beta_interact_pos', 'mean']
        beta_neg = summary.loc['beta_interact_neg', 'mean']
        beta_stress = summary.loc['beta_interact_stress', 'mean']
        beta_total = abs(beta_pos) + abs(beta_neg) + abs(beta_stress)
        
        # 同期率とラグ
        sync_profile, max_sync, optimal_lag = calculate_sync_profile(
            features_source['delta_LambdaC_pos'].astype(np.float64),
            features_target['delta_LambdaC_pos'].astype(np.float64),
            lag_window=10
        )
        
        return {
            'beta': beta_total,
            'beta_pos': beta_pos,
            'beta_neg': beta_neg,
            'beta_stress': beta_stress,
            'lag': optimal_lag,
            'sync_rate': max_sync,
            'trace': trace
        }
    
    def detect(self, data: np.ndarray) -> dict:
        """
        両方向の因果関係を検出
        
        Parameters:
        -----------
        data : np.ndarray (T, 2)
            2系列の時系列データ
        
        Returns:
        --------
        dict : 検出結果
            - forward: A→B の結果
            - backward: B→A の結果
            - asymmetry_ratio: 非対称性比率
            - primary_direction: 主方向
            - primary_beta: 主方向の結合強度
            - detected_edges: 検出されたエッジ
            - computation_time: 計算時間
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"[{self.name}] Detecting A→B...")
        
        result_AB = self._detect_single_direction(data, source_idx=0, target_idx=1)
        
        if self.verbose:
            print(f"  A→B: β={result_AB['beta']:.3f}, lag={result_AB['lag']}")
            print(f"[{self.name}] Detecting B→A...")
        
        result_BA = self._detect_single_direction(data, source_idx=1, target_idx=0)
        
        if self.verbose:
            print(f"  B→A: β={result_BA['beta']:.3f}, lag={result_BA['lag']}")
        
        computation_time = time.time() - start_time
        
        # NEW: 生データでβ再推定（両方向）
        beta_refined_AB = _estimate_beta_on_raw_data(
            data[:, 0], 
            data[:, 1], 
            result_AB['lag']
        )
        beta_refined_BA = _estimate_beta_on_raw_data(
            data[:, 1], 
            data[:, 0], 
            result_BA['lag']
        )
        
        # 結果に追加
        result_AB['beta_refined'] = beta_refined_AB
        result_AB['beta_lambda'] = result_AB['beta']
        
        result_BA['beta_refined'] = beta_refined_BA
        result_BA['beta_lambda'] = result_BA['beta']
        
        # 非対称性（refined版）
        asymmetry_ratio_refined = beta_refined_AB / (beta_refined_BA + 0.01)
        
        # 主方向（refined版で判定）
        if beta_refined_AB > beta_refined_BA:
            primary_direction = 'A→B'
            primary_beta = beta_refined_AB
            primary_beta_lambda = result_AB['beta']
            primary_lag = result_AB['lag']
        else:
            primary_direction = 'B→A'
            primary_beta = beta_refined_BA
            primary_beta_lambda = result_BA['beta']
            primary_lag = result_BA['lag']
        
        computation_time = time.time() - start_time
        
        if self.verbose:
            print(f"  Primary direction: {primary_direction}")
            print(f"  β (Lambda): {primary_beta_lambda:.3f}")
            print(f"  β (Refined): {primary_beta:.3f}")
            print(f"  Asymmetry (Lambda): {asymmetry_ratio:.2f}")
            print(f"  Asymmetry (Refined): {asymmetry_ratio_refined:.2f}")
            print(f"  Computation time: {computation_time:.2f}s")
        
        return {
            'forward': result_AB,
            'backward': result_BA,
            'asymmetry_ratio': asymmetry_ratio_refined,  # refined版を使う
            'asymmetry_ratio_lambda': asymmetry_ratio,
            'primary_direction': primary_direction,
            'primary_beta': primary_beta,  # refined版
            'primary_beta_lambda': primary_beta_lambda,
            'primary_lag': primary_lag,
            'beta': primary_beta,  # メインはrefined
            'beta_lambda': primary_beta_lambda,
            'lag': primary_lag,
            'detected_edges': [(0, 1, primary_lag, primary_beta)],
            'computation_time': computation_time
        }


# ===============================
# 階層的完全検出器（NEW！）
# ===============================
class Lambda3DetectorHierarchical:
    """
    Lambda3検出器（階層的完全版） - プロジェクトファイル統合版
    
    特徴：
    - 階層的構造変化検出（local/global）
    - ペアワイズベイズ推定
    - Bayesian HDI ロギング
    - 統合因果分析
    - 階層メトリクス計算
    
    用途：
    - 完全なLambda3解析
    - 論文用の最高精度検出
    - Hidden Domino 検出
    """
    def __init__(self, 
                 config: L3Config = None, 
                 hierarchical_config: dict = None,
                 verbose: bool = True):
        """
        Parameters:
        -----------
        config : L3Config
            基本設定
        hierarchical_config : dict
            階層的検出の設定
            - local_window: 短期ウィンドウ（デフォルト: 5）
            - global_window: 長期ウィンドウ（デフォルト: 30）
            - local_percentile: 短期閾値（デフォルト: 85.0）
            - global_percentile: 長期閾値（デフォルト: 92.5）
        verbose : bool
            詳細出力フラグ
        """
        if not ADVANCED_AVAILABLE:
            raise ImportError(
                "Advanced Lambda3 functions not available. "
                "Please ensure lambda3_zeroshot_tensor_field.py is in path."
            )
        
        self.config = config or L3Config(
            draws=8000,
            tune=8000,
            hierarchical=True  # hierarchical 対応版 L3Config なら使える
        )
        
        # 階層的設定
        hier_cfg = hierarchical_config or {}
        self.local_window = hier_cfg.get('local_window', 5)
        self.global_window = hier_cfg.get('global_window', 30)
        self.local_percentile = hier_cfg.get('local_percentile', 85.0)
        self.global_percentile = hier_cfg.get('global_percentile', 92.5)
        
        self.name = "Lambda3_Hierarchical"
        self.verbose = verbose
        
        # Bayesian HDI ロガー
        self.bayes_logger = Lambda3BayesianLogger(hdi_prob=self.config.hdi_prob)
    
    def detect(self, data: np.ndarray, series_names: List[str] = None) -> dict:
        """
        完全な階層的因果検出
        
        Parameters:
        -----------
        data : np.ndarray (T, 2) または (T, N)
            時系列データ（2系列以上）
        series_names : List[str], optional
            系列名（デフォルト: ['Series_A', 'Series_B', ...]）
        
        Returns:
        --------
        dict : 完全な検出結果
            - hierarchical_features: 階層的特徴量
            - pairwise_results: ペアワイズベイズ結果
            - causality_results: 因果分析結果
            - hierarchy_metrics: 階層メトリクス
            - hdi_summary: Bayesian HDI サマリー
            - detected_edges: 検出されたエッジ
            - computation_time: 計算時間
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[{self.name}] Hierarchical Detection Started")
            print(f"{'='*60}")
        
        # データ整形
        if data.ndim == 1:
            raise ValueError("Data must be 2D: (T, N)")
        
        T, N = data.shape
        if series_names is None:
            series_names = [f'Series_{chr(65+i)}' for i in range(N)]
        
        if self.verbose:
            print(f"Data shape: {data.shape}")
            print(f"Series: {series_names}")
        
        # ===========================
        # Stage 1: 階層的特徴抽出
        # ===========================
        if self.verbose:
            print(f"\n[Stage 1] Hierarchical Feature Extraction")
        
        features_dict = {}
        hierarchy_metrics_all = {}
        
        for i, name in enumerate(series_names):
            if self.verbose:
                print(f"  {name}: ", end="")
            
            # 階層的特徴抽出（デフォルト設定）
            # calc_lambda3_features は内部でデフォルトの階層的設定を使う
            features = calc_lambda3_features(data[:, i], self.config)
            features_dict[name] = features
            
            # 階層メトリクス
            hierarchy_metrics = calculate_structural_hierarchy_metrics(features)
            hierarchy_metrics_all[name] = hierarchy_metrics
            
            if self.verbose:
                print(f"Local={hierarchy_metrics['local_dominance']:.2f}, "
                      f"Global={hierarchy_metrics['global_dominance']:.2f}, "
                      f"Events={np.sum(features['delta_LambdaC_pos'] + features['delta_LambdaC_neg'])}")
        
        # ===========================
        # Stage 2: ペアワイズベイズ推定
        # ===========================
        if self.verbose:
            print(f"\n[Stage 2] Pairwise Bayesian Estimation")
        
        # データ辞書作成
        data_dict = {name: data[:, i] for i, name in enumerate(series_names)}
        
        # ペアワイズ推定（最初の2系列）
        if N >= 2:
            series_pair = (series_names[0], series_names[1])
            
            if self.verbose:
                print(f"  Analyzing: {series_pair[0]} ⇄ {series_pair[1]}")
            
            trace, model = fit_l3_pairwise_bayesian_system(
                {series_pair[0]: data_dict[series_pair[0]], 
                 series_pair[1]: data_dict[series_pair[1]]},
                {series_pair[0]: features_dict[series_pair[0]], 
                 series_pair[1]: features_dict[series_pair[1]]},
                self.config,
                series_pair=series_pair
            )
            
            # HDI ロギング
            model_id = f"hierarchical_pairwise_{series_pair[0]}_vs_{series_pair[1]}"
            hdi_results = self.bayes_logger.log_trace(
                trace,
                model_id=model_id,
                model_type="hierarchical_pairwise",
                series_names=list(series_pair),
                verbose=self.verbose
            )
            
            # 係数抽出
            summary = az.summary(trace)
            
            # A→B の係数
            beta_ab_pos = summary.loc['beta_interact_ab_pos', 'mean']
            beta_ab_neg = summary.loc['beta_interact_ab_neg', 'mean']
            beta_ab_stress = summary.loc['beta_interact_ab_stress', 'mean']
            beta_ab_total = abs(beta_ab_pos) + abs(beta_ab_neg) + abs(beta_ab_stress)
            
            # B→A の係数
            beta_ba_pos = summary.loc['beta_interact_ba_pos', 'mean']
            beta_ba_neg = summary.loc['beta_interact_ba_neg', 'mean']
            beta_ba_stress = summary.loc['beta_interact_ba_stress', 'mean']
            beta_ba_total = abs(beta_ba_pos) + abs(beta_ba_neg) + abs(beta_ba_stress)
            
            # ラグ係数
            beta_lag_ab = summary.loc['beta_lag_ab', 'mean']
            beta_lag_ba = summary.loc['beta_lag_ba', 'mean']
            
            # 相関
            rho_ab = summary.loc['rho_ab', 'mean']
            
            pairwise_results = {
                'trace': trace,
                'model': model,
                'hdi_results': hdi_results,
                'forward': {
                    'beta': beta_ab_total,
                    'beta_pos': beta_ab_pos,
                    'beta_neg': beta_ab_neg,
                    'beta_stress': beta_ab_stress,
                    'lag_coef': beta_lag_ab
                },
                'backward': {
                    'beta': beta_ba_total,
                    'beta_pos': beta_ba_pos,
                    'beta_neg': beta_ba_neg,
                    'beta_stress': beta_ba_stress,
                    'lag_coef': beta_lag_ba
                },
                'correlation': rho_ab,
                'asymmetry_ratio': beta_ab_total / (beta_ba_total + 0.01)
            }
            
            if self.verbose:
                print(f"  {series_pair[0]}→{series_pair[1]}: β={beta_ab_total:.3f}")
                print(f"  {series_pair[1]}→{series_pair[0]}: β={beta_ba_total:.3f}")
                print(f"  Asymmetry: {pairwise_results['asymmetry_ratio']:.2f}")
                print(f"  Correlation: {rho_ab:.3f}")
        
        else:
            pairwise_results = None
        
        # ===========================
        # Stage 3: 統合因果分析
        # ===========================
        if self.verbose:
            print(f"\n[Stage 3] Comprehensive Causality Analysis")
        
        if N >= 2:
            causality_results = analyze_comprehensive_causality(
                features_dict,
                series_names[:2],
                lag_window=10,
                verbose=self.verbose
            )
        else:
            causality_results = None
        
        # ===========================
        # Stage 4: 結果統合 + β再推定
        # ===========================
        computation_time = time.time() - start_time
        
        # エッジ検出
        detected_edges = []
        beta_main = 0.0
        beta_lambda = 0.0
        lag_main = 0
        
        if pairwise_results is not None:
            # 主方向のエッジ（Lambda空間）
            if pairwise_results['forward']['beta'] > pairwise_results['backward']['beta']:
                primary_direction = 'forward'
                beta_lambda = pairwise_results['forward']['beta']
                lag_main = 0  # ペアワイズではlag係数で表現
            else:
                primary_direction = 'backward'
                beta_lambda = pairwise_results['backward']['beta']
                lag_main = 0
            
            # NEW: 生データでβ再推定
            if N >= 2 and causality_results is not None:
                # causality_results から最適lagを取得
                optimal_lag = causality_results.get('optimal_lag', 0)
                
                if primary_direction == 'forward':
                    beta_refined = _estimate_beta_on_raw_data(
                        data[:, 0], 
                        data[:, 1], 
                        optimal_lag
                    )
                else:
                    beta_refined = _estimate_beta_on_raw_data(
                        data[:, 1], 
                        data[:, 0], 
                        optimal_lag
                    )
                
                beta_main = beta_refined
                lag_main = optimal_lag
                
                # pairwise_results に追加
                pairwise_results['beta_refined'] = beta_refined
                pairwise_results['beta_lambda'] = beta_lambda
                
                if self.verbose:
                    print(f"  β (Lambda): {beta_lambda:.3f}")
                    print(f"  β (Refined): {beta_refined:.3f}")
            else:
                beta_main = beta_lambda
            
            # エッジを追加
            if primary_direction == 'forward':
                detected_edges.append((0, 1, lag_main, beta_main))
            else:
                detected_edges.append((1, 0, lag_main, beta_main))
        
        # HDI サマリー
        hdi_summary_df = self.bayes_logger.generate_summary_report()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[{self.name}] Detection Complete")
            print(f"Computation time: {computation_time:.2f}s")
            print(f"{'='*60}")
            
            # HDI サマリー表示
            if not hdi_summary_df.empty:
                print(f"\nBayesian HDI Summary:")
                print(hdi_summary_df.to_string(index=False))
        
        return {
            'hierarchical_features': features_dict,
            'hierarchy_metrics': hierarchy_metrics_all,
            'pairwise_results': pairwise_results,
            'causality_results': causality_results,
            'hdi_summary': hdi_summary_df,
            'detected_edges': detected_edges,
            'beta': beta_main,  # メインはrefined
            'beta_lambda': beta_lambda,
            'lag': lag_main,
            'computation_time': computation_time,
            'bayes_logger': self.bayes_logger
        }


# ===============================
# 検出器ファクトリー
# ===============================
def create_lambda3_detector(mode: str = 'basic', **kwargs) -> Any:
    """
    Lambda3検出器のファクトリー関数
    
    Parameters:
    -----------
    mode : str
        'basic': 基本版
        'bidirectional': 両方向版
        'hierarchical': 階層的完全版
    **kwargs : dict
        各検出器の初期化パラメータ
    
    Returns:
    --------
    detector : Lambda3Detector インスタンス
    
    Examples:
    ---------
    >>> detector = create_lambda3_detector('basic', verbose=True)
    >>> detector = create_lambda3_detector('bidirectional', config=L3Config(draws=2000))
    >>> detector = create_lambda3_detector('hierarchical', 
    ...                                    hierarchical_config={'local_window': 10})
    """
    if mode == 'basic':
        return Lambda3Detector(**kwargs)
    elif mode == 'bidirectional':
        return Lambda3DetectorBidirectional(**kwargs)
    elif mode == 'hierarchical':
        return Lambda3DetectorHierarchical(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'basic', 'bidirectional', 'hierarchical'")


# ===============================
# テスト用コード
# ===============================
if __name__ == '__main__':
    print("=" * 60)
    print("Lambda3 Detectors v2.0 - Test")
    print("=" * 60)
    
    # テストデータ生成
    np.random.seed(42)
    T = 200
    
    # A→B の因果関係
    A = np.cumsum(np.random.randn(T)) + np.random.randn(T) * 0.3
    B = 1.5 * np.roll(A, 3) + np.random.randn(T) * 0.3
    
    data = np.column_stack([A, B])
    
    print(f"\nTest data shape: {data.shape}")
    
    # ===========================
    # Test 1: 基本版
    # ===========================
    print("\n" + "=" * 60)
    print("Test 1: Basic Detector")
    print("=" * 60)
    
    detector_basic = Lambda3Detector(verbose=True)
    result_basic = detector_basic.detect(data)
    
    print(f"\nResults:")
    print(f"  Detected edge: {result_basic['detected_edges']}")
    print(f"  Beta: {result_basic['beta']:.3f}")
    print(f"  Lag: {result_basic['lag']}")
    print(f"  Time: {result_basic['computation_time']:.2f}s")
    
    # ===========================
    # Test 2: 両方向版
    # ===========================
    print("\n" + "=" * 60)
    print("Test 2: Bidirectional Detector")
    print("=" * 60)
    
    detector_bidir = Lambda3DetectorBidirectional(verbose=True)
    result_bidir = detector_bidir.detect(data)
    
    print(f"\nResults:")
    print(f"  Forward (A→B): β={result_bidir['forward']['beta']:.3f}")
    print(f"  Backward (B→A): β={result_bidir['backward']['beta']:.3f}")
    print(f"  Asymmetry: {result_bidir['asymmetry_ratio']:.2f}")
    print(f"  Primary: {result_bidir['primary_direction']}")
    print(f"  Time: {result_bidir['computation_time']:.2f}s")
    
    # ===========================
    # Test 3: 階層的完全版
    # ===========================
    if ADVANCED_AVAILABLE:
        print("\n" + "=" * 60)
        print("Test 3: Hierarchical Detector")
        print("=" * 60)
        
        detector_hier = Lambda3DetectorHierarchical(verbose=True)
        result_hier = detector_hier.detect(data, series_names=['A', 'B'])
        
        print(f"\nResults:")
        print(f"  Detected edges: {result_hier['detected_edges']}")
        print(f"  Beta: {result_hier['beta']:.3f}")
        if result_hier['pairwise_results']:
            print(f"  Asymmetry: {result_hier['pairwise_results']['asymmetry_ratio']:.2f}")
            print(f"  Correlation: {result_hier['pairwise_results']['correlation']:.3f}")
        print(f"  Time: {result_hier['computation_time']:.2f}s")
    else:
        print("\n[Test 3 skipped: Advanced functions not available]")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
