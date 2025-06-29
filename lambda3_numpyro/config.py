from dataclasses import dataclass, field
from typing import List, Optional
import os

# グローバル定数（JIT用）
DELTA_PERCENTILE = 97.0
LOCAL_JUMP_PERCENTILE = 97.0
WINDOW_SIZE = 10
LOCAL_WINDOW_SIZE = 10
LAG_WINDOW_DEFAULT = 10
SYNC_THRESHOLD_DEFAULT = 0.3
NOISE_STD_DEFAULT = 0.5
NOISE_STD_HIGH = 1.5

@dataclass
class L3Config:
    """Lambda³解析設定"""
    T: int = 150
    # 特徴抽出パラメータ
    window: int = WINDOW_SIZE
    local_window: int = LOCAL_WINDOW_SIZE
    delta_percentile: float = DELTA_PERCENTILE
    local_jump_percentile: float = LOCAL_JUMP_PERCENTILE
    
    # NumPyro MCMC設定
    draws: int = 8000
    tune: int = 8000
    target_accept: float = 0.95
    num_chains: int = 4
    
    # 可視化設定
    var_names: List[str] = field(default_factory=lambda: [
        'beta_time_a', 'beta_time_b', 'beta_interact', 
        'beta_rhoT_a', 'beta_rhoT_b'
    ])
    hdi_prob: float = 0.94
    
    # クラウド設定
    use_cloud_storage: bool = False
    cloud_bucket: Optional[str] = None
    cloud_provider: str = 'gcs'  # 'gcs' or 's3'
    
    @classmethod
    def from_env(cls) -> 'L3Config':
        """環境変数から設定を読み込み"""
        config = cls()
        if os.getenv('L3_NUM_CHAINS'):
            config.num_chains = int(os.getenv('L3_NUM_CHAINS'))
        if os.getenv('L3_CLOUD_BUCKET'):
            config.use_cloud_storage = True
            config.cloud_bucket = os.getenv('L3_CLOUD_BUCKET')
        return config
