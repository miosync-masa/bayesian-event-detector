# =======================================
# Type B: 一方向ドミノ データ生成
# =======================================

from dataclasses import dataclass
import numpy as np

@dataclass
class UnidirectionalDominoConfig:
    """一方向ドミノ設定"""
    T: int = 500
    N: int = 5
    beta_forward: float = 2.5   # A→B の強度
    beta_backward: float = 0.1  # B→A の強度（ほぼゼロ）
    noise_level: float = 0.3
    n_datasets: int = 10


def generate_unidirectional_domino(config: UnidirectionalDominoConfig, 
                                   seed: int = None):
    """
    一方向ドミノデータを生成
    
    A → B (強い)
    B → A (ほぼゼロ)
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = np.zeros((config.T, config.N))
    
    # A系列（独立）
    data[:, 0] = np.random.randn(config.T)
    
    # B系列（Aから強く影響を受ける）
    for t in range(config.T):
        data[t, 1] = (
            config.beta_forward * data[t, 0] +  # A→B 強い
            config.noise_level * np.random.randn()
        )
    
    # Aには Bからほぼ影響しない（ノイズレベルの影響のみ）
    # すでに独立に生成済みなので、わずかな影響を追加
    for t in range(1, config.T):
        data[t, 0] += config.beta_backward * data[t-1, 1]
    
    # C, D, E（独立）
    data[:, 2:] = np.random.randn(config.T, config.N - 2)
    
    ground_truth = {
        'edges': [
            (0, 1, 0, config.beta_forward),   # A→B
            (1, 0, 0, config.beta_backward)   # B→A（弱い）
        ],
        'beta_forward': config.beta_forward,
        'beta_backward': config.beta_backward,
        'true_asymmetry': config.beta_forward / config.beta_backward,
        'type': 'unidirectional'
    }
    
    return data, ground_truth


def generate_unidirectional_batch(config: UnidirectionalDominoConfig):
    """複数データセット生成"""
    datasets = []
    for i in range(config.n_datasets):
        data, gt = generate_unidirectional_domino(config, seed=100+i)
        datasets.append({
            'id': f'unidirectional_set_{i:02d}',
            'data': data,
            'ground_truth': gt
        })
    return datasets


print("✅ Type B データ生成関数 完成！")
