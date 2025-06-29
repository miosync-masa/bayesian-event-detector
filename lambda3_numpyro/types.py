from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any
import numpy as np

@dataclass
class Lambda3FeatureSet:
    """Lambda³特徴量セット"""
    data: np.ndarray
    delta_LambdaC_pos: np.ndarray
    delta_LambdaC_neg: np.ndarray
    rho_T: np.ndarray
    time_trend: np.ndarray
    local_jump: np.ndarray
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """辞書形式に変換（後方互換性）"""
        return {
            'data': self.data,
            'delta_LambdaC_pos': self.delta_LambdaC_pos,
            'delta_LambdaC_neg': self.delta_LambdaC_neg,
            'rho_T': self.rho_T,
            'time_trend': self.time_trend,
            'local_jump': self.local_jump
        }

@dataclass
class SyncProfile:
    """同期プロファイル結果"""
    profile: Dict[int, float]
    max_sync_rate: float
    optimal_lag: int

@dataclass
class AnalysisResult:
    """解析結果のコンテナ"""
    trace_a: Any  # ArviZ InferenceData
    trace_b: Any
    sync_profile: SyncProfile
    interaction_effects: Dict[str, float]
    metadata: Dict[str, Any]
