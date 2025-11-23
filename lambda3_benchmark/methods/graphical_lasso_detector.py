"""
Graphical Lasso Detector
Sparse precision matrix estimation
"""
import numpy as np
from sklearn.covariance import GraphicalLassoCV


class GraphicalLassoDetector:
    """
    Graphical Lasso検出器
    スパース精度行列推定によるグラフ構造検出
    
    注意: 同時刻の相関のみ検出（ラグなし）
    """
    def __init__(self, alphas: int = 10):
        """
        Args:
            alphas: 正則化パラメータの候補数
        """
        self.alphas = alphas
        self.name = "GraphicalLasso"
    
    def detect(self, data: np.ndarray) -> dict:
        """
        Graphical Lassoで因果検出
        
        Args:
            data: (T, N) 時系列データ
        
        Returns:
            検出結果 {beta, lag, precision_matrix, ...}
        """
        try:
            # A, B系列のみ使用
            X = data[:, :2]
            
            # Graphical Lasso CV（交差検証で最適α選択）
            model = GraphicalLassoCV(
                alphas=self.alphas,
                cv=5,
                max_iter=100
            )
            model.fit(X)
            
            # 精度行列（逆共分散行列）
            precision = model.precision_
            
            # A-B 間の結合強度（精度行列の非対角成分）
            # 精度行列の(i,j)成分は、他の変数を固定したときのi,jの条件付き独立性を表す
            beta_glasso = abs(precision[0, 1])
            
            # エッジ検出（閾値：0.1）
            # GLassoはスパース推定なので、小さい値はゼロに近い
            has_edge = beta_glasso > 0.1
            
            return {
                'detected_edges': [(0, 1, 0, beta_glasso)] if has_edge else [],
                'beta': beta_glasso,
                'lag': 0,  # GLassoはラグを検出できない（同時刻のみ）
                'precision_matrix': precision,
                'covariance_matrix': model.covariance_,
                'alpha': model.alpha_
            }
        
        except Exception as e:
            print(f"GraphicalLasso detection failed: {e}")
            return {
                'detected_edges': [],
                'beta': 0.0,
                'lag': 0,
                'precision_matrix': None,
                'covariance_matrix': None,
                'alpha': None
            }
