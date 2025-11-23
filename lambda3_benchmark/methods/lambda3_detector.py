"""
Lambda3 Detector (Improved with V1_AbsSum)
"""
import numpy as np
from lambda3_abc import (
    calc_lambda3_features_v2,
    fit_l3_bayesian_regression_asymmetric,
    calculate_sync_profile,
    L3Config
)
import arviz as az


class Lambda3Detector:
    """
    Lambda3検出器（改善版）
    Beta計算: V1_AbsSum = abs(β_pos) + abs(β_neg) + abs(β_stress)
    """
    def __init__(self, config: L3Config = None):
        self.config = config or L3Config(draws=4000, tune=4000)
        self.name = "Lambda3"
    
    def detect(self, data: np.ndarray) -> dict:
        """
        遅延ドミノを検出
        
        Args:
            data: (T, N) 時系列データ
        
        Returns:
            検出結果 {beta, lag, sync, ...}
        """
        # A系列（系列0）の特徴抽出
        feats_a = calc_lambda3_features_v2(data[:, 0], self.config)
        features_a = {
            'delta_LambdaC_pos': feats_a[0],
            'delta_LambdaC_neg': feats_a[1],
            'rho_T': feats_a[2],
            'time_trend': feats_a[3]
        }
        
        # B系列（系列1）の特徴抽出
        feats_b = calc_lambda3_features_v2(data[:, 1], self.config)
        features_b = {
            'delta_LambdaC_pos': feats_b[0],
            'delta_LambdaC_neg': feats_b[1],
            'rho_T': feats_b[2],
            'time_trend': feats_b[3]
        }
        
        # ベイズ推論：A→B の影響を推定
        trace = fit_l3_bayesian_regression_asymmetric(
            data=data[:, 1],
            features_dict=features_b,
            config=self.config,
            interaction_pos=features_a['delta_LambdaC_pos'],
            interaction_neg=features_a['delta_LambdaC_neg'],
            interaction_rhoT=features_a['rho_T']
        )
        
        # 結合強度 β の推定
        summary = az.summary(trace)
        beta_pos = summary.loc['beta_interact_pos', 'mean']
        beta_neg = summary.loc['beta_interact_neg', 'mean']
        beta_stress = summary.loc['beta_interact_stress', 'mean']
        
        # V1_AbsSum で総合結合強度を計算
        beta_total = abs(beta_pos) + abs(beta_neg) + abs(beta_stress)
        
        # 同期率とラグ検出
        sync_profile, max_sync, optimal_lag = calculate_sync_profile(
            features_a['delta_LambdaC_pos'].astype(np.float64),
            features_b['delta_LambdaC_pos'].astype(np.float64),
            lag_window=10
        )
        
        return {
            'detected_edges': [(0, 1, optimal_lag, beta_total)],
            'beta': beta_total,
            'beta_pos': beta_pos,
            'beta_neg': beta_neg,
            'beta_stress': beta_stress,
            'lag': optimal_lag,
            'sync_rate': max_sync,
            'trace': trace
        }
