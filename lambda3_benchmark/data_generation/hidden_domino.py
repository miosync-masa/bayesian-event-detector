"""
Type C: Hidden Domino Data Generation
σ_s ≈ 0 (同期率ゼロ) だが |β| > 1.5 (強い結合)
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class HiddenDominoConfig:
    """隠れドミノ設定"""
    T: int = 500
    N: int = 5
    beta: float = 2.0         # 真の結合強度
    phase_shift: float = np.pi / 2  # 位相差（90度）
    frequency: float = 2 * np.pi / 50  # 周期
    noise_level: float = 0.3
    n_datasets: int = 10


def generate_hidden_domino(config: HiddenDominoConfig, 
                          seed: int = None) -> Tuple[np.ndarray, Dict]:
    """
    隠れドミノデータを生成
    
    物理的意味:
    - AとBは位相が90度ずれている
    - → 相関≈0（同期率ゼロ）
    - でも因果関係は強い（β=2.0）
    
    例: 露点 → 気温
    - 同時には動かない（位相ずれ）
    - でも強く影響している
    
    Args:
        config: 設定
        seed: 乱数シード
    
    Returns:
        data: (T, N) 時系列データ
        ground_truth: 真の構造情報
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = np.zeros((config.T, config.N))
    t = np.arange(config.T)
    
    # A系列（サイン波）
    data[:, 0] = (
        np.sin(config.frequency * t) + 
        config.noise_level * np.random.randn(config.T)
    )
    
    # B系列（位相が90度ずれたサイン波 = コサイン波）
    # B(t) = β * cos(ωt) = β * sin(ωt + π/2)
    data[:, 1] = (
        config.beta * np.sin(config.frequency * t + config.phase_shift) + 
        config.noise_level * np.random.randn(config.T)
    )
    
    # これにより：
    # - A と B の相関 ≈ 0（直交）
    # - でも B は A の位相シフト版なので因果あり
    
    # C, D, E（独立）
    data[:, 2:] = np.random.randn(config.T, config.N - 2)
    
    # 理論的な同期率（位相が90度ずれてるのでゼロに近い）
    theoretical_sync = abs(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
    
    ground_truth = {
        'edges': [(0, 1, 0, config.beta)],  # A→B、ラグ0
        'true_beta': config.beta,
        'phase_shift': config.phase_shift,
        'theoretical_sync': theoretical_sync,  # ≈0
        'type': 'hidden'
    }
    
    return data, ground_truth


def generate_hidden_batch(config: HiddenDominoConfig) -> list:
    """複数の隠れドミノデータセット生成"""
    datasets = []
    for i in range(config.n_datasets):
        data, gt = generate_hidden_domino(config, seed=200+i)
        datasets.append({
            'id': f'hidden_set_{i:02d}',
            'data': data,
            'ground_truth': gt
        })
    return datasets
