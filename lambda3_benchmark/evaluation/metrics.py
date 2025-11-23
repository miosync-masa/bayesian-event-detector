"""
Evaluation Metrics
"""
import numpy as np


def is_correct_delayed(detected: dict, ground_truth: dict, 
                       lag_tolerance: int = 2, 
                       beta_threshold: float = 0.5) -> bool:
    """遅延ドミノの検出が正解かチェック"""
    if not detected['detected_edges']:
        return False
    
    edge = detected['detected_edges'][0]
    source, target, det_lag, det_beta = edge
    
    gt_lag = ground_truth['true_lag']
    
    correct_nodes = (source == 0) and (target == 1)
    correct_lag = abs(det_lag - gt_lag) <= lag_tolerance
    significant_beta = abs(det_beta) > beta_threshold
    
    return correct_nodes and correct_lag and significant_beta


def calculate_metrics(results: list, ground_truths: list) -> dict:
    """複数データセットでの評価指標を計算"""
    n = len(results)
    
    correct_count = sum(
        is_correct_delayed(r, gt) 
        for r, gt in zip(results, ground_truths)
    )
    
    lag_errors = [
        abs(r['lag'] - gt['true_lag']) 
        for r, gt in zip(results, ground_truths)
    ]
    
    beta_errors = [
        abs(r['beta'] - gt['true_beta']) 
        for r, gt in zip(results, ground_truths)
    ]
    
    return {
        'accuracy': correct_count / n,
        'correct_count': correct_count,
        'total': n,
        'mean_lag_error': np.mean(lag_errors),
        'std_lag_error': np.std(lag_errors),
        'mean_beta_error': np.mean(beta_errors),
        'std_beta_error': np.std(beta_errors)
    }

# =======================================
# Type B 専用評価指標
# =======================================

def is_correct_unidirectional(detected: dict, ground_truth: dict,
                              beta_threshold: float = 1.0,
                              asymmetry_threshold: float = 5.0) -> bool:
    """
    一方向ドミノの正解判定
    
    正解条件:
    1. 主方向（A→B）が強い（β > 1.0）
    2. 逆方向（B→A）が弱い（β < 0.5）
    3. 非対称性が十分大きい（> 5.0）
    """
    # 主方向が十分強いか
    forward_strong = detected['forward']['beta'] > beta_threshold
    
    # 逆方向が十分弱いか
    backward_weak = detected['backward']['beta'] < 0.5
    
    # 非対称性が検出されているか
    asymmetry_detected = detected['asymmetry_ratio'] > asymmetry_threshold
    
    return forward_strong and backward_weak and asymmetry_detected


def calculate_metrics_unidirectional(results: list, ground_truths: list) -> dict:
    """Type B 専用評価指標"""
    n = len(results)
    
    correct_count = sum(
        is_correct_unidirectional(r, gt) 
        for r, gt in zip(results, ground_truths)
    )
    
    # 前向き結合強度の誤差
    forward_errors = [
        abs(r['forward']['beta'] - gt['beta_forward']) 
        for r, gt in zip(results, ground_truths)
    ]
    
    # 非対称性の誤差
    asymmetry_errors = [
        abs(r['asymmetry_ratio'] - gt['true_asymmetry']) 
        for r, gt in zip(results, ground_truths)
    ]
    
    return {
        'accuracy': correct_count / n,
        'correct_count': correct_count,
        'total': n,
        'mean_forward_error': np.mean(forward_errors),
        'std_forward_error': np.std(forward_errors),
        'mean_asymmetry_error': np.mean(asymmetry_errors),
        'std_asymmetry_error': np.std(asymmetry_errors)
    }
  
print("✅ Type B 評価指標 完成！")
