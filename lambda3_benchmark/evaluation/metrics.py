"""
評価指標の計算
"""
import numpy as np
from typing import Dict, List

def is_correct_delayed(detected: dict, ground_truth: dict, 
                       lag_tolerance: int = 2, 
                       beta_threshold: float = 0.5) -> bool:
    """
    遅延ドミノの検出が正解かチェック
    """
    if not detected['detected_edges']:
        return False
    
    edge = detected['detected_edges'][0]
    source, target, det_lag, det_beta = edge
    
    gt_lag = ground_truth['true_lag']
    gt_beta = ground_truth['true_beta']
    
    # 正解条件
    correct_nodes = (source == 0) and (target == 1)
    correct_lag = abs(det_lag - gt_lag) <= lag_tolerance
    significant_beta = abs(det_beta) > beta_threshold
    
    return correct_nodes and correct_lag and significant_beta

def calculate_metrics(results: List[dict], 
                     ground_truths: List[dict]) -> dict:
    """
    複数データセットでの評価指標を計算
    """
    n = len(results)
    correct = sum(is_correct_delayed(r, gt) 
                  for r, gt in zip(results, ground_truths))
    
    lag_errors = [abs(r['lag'] - gt['true_lag']) 
                  for r, gt in zip(results, ground_truths)]
    beta_errors = [abs(r['beta'] - gt['true_beta']) 
                   for r, gt in zip(results, ground_truths)]
    
    return {
        'accuracy': correct / n,
        'mean_lag_error': np.mean(lag_errors),
        'std_lag_error': np.std(lag_errors),
        'mean_beta_error': np.mean(beta_errors),
        'std_beta_error': np.std(beta_errors)
    }
