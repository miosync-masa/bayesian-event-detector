"""
Lambda3 Benchmark - Methods Package
====================================
全検出手法のインポート集約

検出器：
- Lambda3Detector: 基本版（後方互換性）
- Lambda3DetectorBidirectional: 両方向検出
- Lambda3DetectorHierarchical: 階層的完全版（NEW！）
- VARDetector: Vector AutoRegressive
- TransferEntropyDetector: Transfer Entropy
- GraphicalLassoDetector: Graphical Lasso

ユーティリティ：
- create_lambda3_detector: Lambda3検出器ファクトリー関数

Author: 環ちゃん with ご主人さま
Updated: 2024-11-23
"""

# Lambda3 検出器（v2.0 完全統合版）
from .lambda3_detectors import (
    Lambda3Detector,
    Lambda3DetectorBidirectional,
    Lambda3DetectorHierarchical,
    create_lambda3_detector
)

# 比較手法
from .var_detector import VARDetector
from .transfer_entropy_detector import TransferEntropyDetector
from .graphical_lasso_detector import GraphicalLassoDetector


# 公開API
__all__ = [
    # Lambda3 検出器
    'Lambda3Detector',
    'Lambda3DetectorBidirectional',
    'Lambda3DetectorHierarchical',
    'create_lambda3_detector',
    
    # 比較手法
    'VARDetector',
    'TransferEntropyDetector',
    'GraphicalLassoDetector',
]


# バージョン情報
__version__ = '2.0.0'


# 利用可能な検出器一覧を取得
def get_available_detectors():
    """
    利用可能な検出器の一覧を返す
    
    Returns:
    --------
    dict : 検出器名 → クラスのマッピング
    
    Examples:
    ---------
    >>> detectors = get_available_detectors()
    >>> print(list(detectors.keys()))
    ['Lambda3_Basic', 'Lambda3_Bidirectional', 'Lambda3_Hierarchical', 'VAR', 'TransferEntropy', 'GraphicalLasso']
    """
    return {
        'Lambda3_Basic': Lambda3Detector,
        'Lambda3_Bidirectional': Lambda3DetectorBidirectional,
        'Lambda3_Hierarchical': Lambda3DetectorHierarchical,
        'VAR': VARDetector,
        'TransferEntropy': TransferEntropyDetector,
        'GraphicalLasso': GraphicalLassoDetector
    }


# すべての検出器を一括作成
def create_all_detectors(config: dict = None):
    """
    すべての検出器を一括作成
    
    Parameters:
    -----------
    config : dict, optional
        各検出器の設定
        {
            'lambda3': {...},  # Lambda3系の共通設定
            'var': {...},
            'te': {...},
            'glasso': {...}
        }
    
    Returns:
    --------
    dict : 検出器名 → インスタンスのマッピング
    
    Examples:
    ---------
    >>> detectors = create_all_detectors()
    >>> for name, detector in detectors.items():
    ...     result = detector.detect(data)
    """
    cfg = config or {}
    
    return {
        'Lambda3_Basic': Lambda3Detector(**cfg.get('lambda3', {})),
        'Lambda3_Bidirectional': Lambda3DetectorBidirectional(**cfg.get('lambda3', {})),
        'Lambda3_Hierarchical': Lambda3DetectorHierarchical(**cfg.get('lambda3', {})),
        'VAR': VARDetector(**cfg.get('var', {})),
        'TransferEntropy': TransferEntropyDetector(**cfg.get('te', {})),
        'GraphicalLasso': GraphicalLassoDetector(**cfg.get('glasso', {}))
    }


# 簡易テスト
def test_imports():
    """
    すべてのインポートが成功するかテスト
    
    Returns:
    --------
    bool : すべて成功ならTrue
    """
    try:
        detectors = get_available_detectors()
        print(f"✓ Successfully imported {len(detectors)} detectors:")
        for name in detectors.keys():
            print(f"  - {name}")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Lambda3 Benchmark - Methods Package Test")
    print("=" * 60)
    test_imports()
