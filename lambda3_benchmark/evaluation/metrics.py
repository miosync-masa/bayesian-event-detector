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


def is_correct_hidden(detected: dict, ground_truth: dict,
                     beta_threshold: float = 1.5,
                     sync_threshold: float = 0.2) -> bool:
    """
    隠れドミノの正解判定
    
    正解条件:
    1. 結合強度が強い（|β| > 1.5）
    2. 同期率が低い（σ_s < 0.2）← これが重要！
    
    物理的意味:
    - 「見た目は無関係（同期してない）」
    - 「でも実は強く結びついてる」
    
    Args:
        detected: 検出結果
        ground_truth: 真の構造
        beta_threshold: Beta検出閾値
        sync_threshold: 同期率の上限（これ以下であるべき）
    
    Returns:
        正解ならTrue
    """
    # Lambda3の場合
    if 'forward' in detected:
        # 両方向検出版
        beta_detected = detected['primary_beta']
        sync_rate = detected['forward'].get('sync_rate', 1.0)
    else:
        # 単方向検出版
        beta_detected = detected['beta']
        sync_rate = detected.get('sync_rate', 1.0)
    
    # 条件1: 結合強度が強い
    strong_coupling = abs(beta_detected) > beta_threshold
    
    # 条件2: 同期率が低い（これが「隠れ」の証拠）
    low_sync = sync_rate < sync_threshold
    
    return strong_coupling and low_sync


def calculate_metrics_hidden(results: list, ground_truths: list) -> dict:
    """Type C（隠れドミノ）専用評価指標"""
    n = len(results)
    
    correct_count = sum(
        is_correct_hidden(r, gt) 
        for r, gt in zip(results, ground_truths)
    )
    
    # Beta誤差
    beta_errors = []
    for r, gt in zip(results, ground_truths):
        if 'primary_beta' in r:
            beta_detected = r['primary_beta']
        else:
            beta_detected = r['beta']
        beta_errors.append(abs(beta_detected - gt['true_beta']))
    
    # 同期率の統計
    sync_rates = []
    for r in results:
        if 'forward' in r:
            sync_rates.append(r['forward'].get('sync_rate', 1.0))
        else:
            sync_rates.append(r.get('sync_rate', 1.0))
    
    return {
        'accuracy': correct_count / n,
        'correct_count': correct_count,
        'total': n,
        'mean_beta_error': np.mean(beta_errors),
        'std_beta_error': np.std(beta_errors),
        'mean_sync_rate': np.mean(sync_rates),
        'std_sync_rate': np.std(sync_rates)
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
  
