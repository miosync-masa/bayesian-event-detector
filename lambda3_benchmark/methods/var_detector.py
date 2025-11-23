"""
VAR Detector
"""
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR


class VARDetector:
    """VAR検出器"""
    def __init__(self, max_lag: int = 10):
        self.max_lag = max_lag
        self.name = "VAR"
    
    def detect(self, data: np.ndarray) -> dict:
        """VARモデルで因果検出"""
        try:
            model = VAR(data[:, :2])
            result = model.fit(maxlags=self.max_lag, ic='aic')
            
            gc_result = result.test_causality(1, 0, kind='f')
            
            optimal_lag = result.k_ar
            coef_matrix = result.params
            
            beta_var = 0.0
            for lag in range(1, optimal_lag + 1):
                idx = (lag - 1) * 2
                if idx < len(coef_matrix):
                    beta_var += abs(coef_matrix[idx, 1])
            
            return {
                'detected_edges': [(0, 1, optimal_lag, beta_var)],
                'beta': beta_var,
                'lag': optimal_lag,
                'p_value': gc_result.pvalue,
                'is_significant': gc_result.pvalue < 0.05
            }
        
        except Exception as e:
            print(f"VAR detection failed: {e}")
            return {
                'detected_edges': [],
                'beta': 0.0,
                'lag': 0,
                'p_value': 1.0,
                'is_significant': False
            }
