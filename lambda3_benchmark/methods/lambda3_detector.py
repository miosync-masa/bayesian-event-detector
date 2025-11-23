# =======================================
# Lambda3Detector（両方向検出版）
# =======================================

import numpy as np
from lambda3_abc import (
    calc_lambda3_features_v2,
    fit_l3_bayesian_regression_asymmetric,
    calculate_sync_profile,
    L3Config
)
import arviz as az


class Lambda3DetectorBidirectional:
    """
    Lambda3検出器（両方向検出版）
    A→B と B→A の両方を検出して非対称性を評価
    """
    def __init__(self, config: L3Config = None):
        self.config = config or L3Config(draws=4000, tune=4000)
        self.name = "Lambda3_Bidirectional"
    
    def _detect_single_direction(self, data: np.ndarray, 
                                 source_idx: int, target_idx: int) -> dict:
        """
        単一方向の因果関係を検出
        
        Args:
            data: 時系列データ
            source_idx: 原因系列のインデックス
            target_idx: 結果系列のインデックス
        
        Returns:
            検出結果
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
        
        # ベイズ推論：Source→Target の影響を推定
        trace = fit_l3_bayesian_regression_asymmetric(
            data=data[:, target_idx],
            features_dict=features_target,
            config=self.config,
            interaction_pos=features_source['delta_LambdaC_pos'],
            interaction_neg=features_source['delta_LambdaC_neg'],
            interaction_rhoT=features_source['rho_T']
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
            'sync_rate': max_sync
        }
    
    def detect(self, data: np.ndarray) -> dict:
        """
        両方向の因果関係を検出
        
        Args:
            data: (T, N) 時系列データ
        
        Returns:
            両方向の検出結果 + 非対称性評価
        """
        print(f"  Detecting A→B...")
        result_AB = self._detect_single_direction(data, source_idx=0, target_idx=1)
        
        print(f"  Detecting B→A...")
        result_BA = self._detect_single_direction(data, source_idx=1, target_idx=0)
        
        # 非対称性の計算
        asymmetry_ratio = result_AB['beta'] / (result_BA['beta'] + 0.01)
        
        # 主方向の決定（betaが大きい方）
        if result_AB['beta'] > result_BA['beta']:
            primary_direction = 'A→B'
            primary_beta = result_AB['beta']
            primary_lag = result_AB['lag']
        else:
            primary_direction = 'B→A'
            primary_beta = result_BA['beta']
            primary_lag = result_BA['lag']
        
        return {
            'forward': result_AB,    # A→B
            'backward': result_BA,   # B→A
            'asymmetry_ratio': asymmetry_ratio,
            'primary_direction': primary_direction,
            'primary_beta': primary_beta,
            'primary_lag': primary_lag,
            # 後方互換性のため
            'beta': primary_beta,
            'lag': primary_lag,
            'detected_edges': [(0, 1, primary_lag, primary_beta)]
        }


print("✅ Lambda3DetectorBidirectional クラス定義完了！")
