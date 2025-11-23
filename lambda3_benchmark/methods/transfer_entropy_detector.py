"""
Transfer Entropy Detector
Information-theoretic causality detection
"""
import numpy as np
from scipy.stats import entropy


class TransferEntropyDetector:
    """
    Transfer Entropy検出器
    情報理論ベースの因果検出
    
    TE(X→Y) = I(Y_future; X_past | Y_past)
    """
    def __init__(self, max_lag: int = 10, k: int = 1, bins: int = 10):
        """
        Args:
            max_lag: 最大ラグ
            k: 埋め込み次元（過去何ステップ見るか）
            bins: ヒストグラムのビン数
        """
        self.max_lag = max_lag
        self.k = k
        self.bins = bins
        self.name = "TransferEntropy"
    
    def _discretize(self, data, bins):
        """連続値を離散化"""
        return np.digitize(data, bins=np.linspace(data.min(), data.max(), bins))
    
    def _calculate_te(self, source, target, lag, k):
        """
        Transfer Entropyを計算
        
        TE = H(target_future | target_past) - H(target_future | target_past, source_past)
        
        Args:
            source: 原因系列
            target: 結果系列
            lag: 時間遅れ
            k: 埋め込み次元
        
        Returns:
            Transfer Entropy値
        """
        n = len(source)
        
        # 離散化
        source_disc = self._discretize(source, self.bins)
        target_disc = self._discretize(target, self.bins)
        
        if n < lag + k + 1:
            return 0.0
        
        # データ準備
        target_future = target_disc[lag + k:]
        target_past = np.array([
            target_disc[i:i+k] for i in range(len(target_disc) - lag - k)
        ])
        source_past = np.array([
            source_disc[i:i+k] for i in range(len(source_disc) - lag - k)
        ])
        
        # TE計算（条件付きエントロピーの差）
        try:
            # H(target_future | target_past)
            joint_tt = np.column_stack([target_future, target_past])
            h_tt = self._joint_entropy(joint_tt) - self._joint_entropy(target_past)
            
            # H(target_future | target_past, source_past)
            joint_tts = np.column_stack([target_future, target_past, source_past])
            joint_ts = np.column_stack([target_past, source_past])
            h_tts = self._joint_entropy(joint_tts) - self._joint_entropy(joint_ts)
            
            te = h_tt - h_tts
            return max(0, te)  # TEは非負
        
        except:
            return 0.0
    
    def _joint_entropy(self, data):
        """結合エントロピー"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # 各行をタプルに変換してユニークな組み合わせをカウント
        unique_rows, counts = np.unique(data, axis=0, return_counts=True)
        probabilities = counts / len(data)
        return entropy(probabilities, base=2)
    
    def detect(self, data: np.ndarray) -> dict:
        """
        Transfer Entropyで因果検出
        
        Args:
            data: (T, N) 時系列データ
        
        Returns:
            検出結果 {beta, lag, te_value, ...}
        """
        source = data[:, 0]  # A系列
        target = data[:, 1]  # B系列
        
        # 各ラグでTEを計算
        te_values = []
        for lag in range(1, self.max_lag + 1):
            te = self._calculate_te(source, target, lag, self.k)
            te_values.append(te)
        
        # 最大TEとそのラグを検出
        if len(te_values) > 0 and max(te_values) > 0:
            optimal_lag = np.argmax(te_values) + 1
            max_te = te_values[optimal_lag - 1]
        else:
            optimal_lag = 1
            max_te = 0.0
        
        # Beta相当の値（0-1に正規化）
        # TEの絶対値は小さいので、検出閾値は0.1程度
        beta_equiv = max_te * 2.0  # スケーリング
        
        return {
            'detected_edges': [(0, 1, optimal_lag, beta_equiv)] if max_te > 0.05 else [],
            'beta': beta_equiv,
            'lag': optimal_lag,
            'te_value': max_te,
            'te_profile': te_values
        }
