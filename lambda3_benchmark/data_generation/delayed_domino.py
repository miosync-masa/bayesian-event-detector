"""
Type A: 遅延ドミノ合成データ生成
A(t) → B(t+τ), where τ ∈ [3, 10]
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class DelayedDominoConfig:
    T: int = 500  # 時系列長
    N: int = 5    # 系列数
    tau: int = 5  # 真の遅延
    beta: float = 2.0  # 真の結合強度
    noise_level: float = 0.3
    n_datasets: int = 10

def generate_delayed_domino(config: DelayedDominoConfig, 
                           seed: int = None) -> Tuple[np.ndarray, Dict]:
    """
    遅延ドミノデータを生成
    
    Returns:
        data: (T, N) 時系列データ
        ground_truth: 真の構造情報
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = np.zeros((config.T, config.N))
    
    # A系列（独立）
    data[:, 0] = np.random.randn(config.T) + \
                 config.noise_level * np.random.randn(config.T)
    
    # B系列（Aから遅延τで影響）
    for t in range(config.tau, config.T):
        data[t, 1] = config.beta * data[t - config.tau, 0] + \
                     config.noise_level * np.random.randn()
    
    # C, D, E（独立）
    data[:, 2:] = np.random.randn(config.T, config.N - 2)
    
    ground_truth = {
        'edges': [(0, 1, config.tau, config.beta)],  # (source, target, lag, strength)
        'true_lag': config.tau,
        'true_beta': config.beta,
        'type': 'delayed'
    }
    
    return data, ground_truth

def generate_dataset_batch(config: DelayedDominoConfig) -> list:
    """複数データセットを生成"""
    datasets = []
    for i in range(config.n_datasets):
        data, gt = generate_delayed_domino(config, seed=42+i)
        datasets.append({
            'id': f'delayed_set_{i:02d}',
            'data': data,
            'ground_truth': gt
        })
    return datasets
