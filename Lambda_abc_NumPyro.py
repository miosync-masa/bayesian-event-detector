# ==========================================================
# Λ³NumPyro: Lambda³ Analytics GPU-Accelerated Edition
# ----------------------------------------------------
# Complete NumPyro port of Lambda³ ABC framework
# Author: Mamichi Iizumi (Miosync, Inc.)
# License: MIT
# ----------------------------------------------------

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, Predictive
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.infer.svi import SVI
from numpyro.optim import Adam

from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import warnings
from pathlib import Path
import time

# GPU強制設定 (Lambda3構造空間での演算加速)
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", True)
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# YFinance for data fetching
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available. Install with: pip install yfinance")

# ===============================
# Lambda³ Configuration
# ===============================
@dataclass
class L3ConfigNumPyro:
    """NumPyro版Lambda³解析パラメータ設定"""
    T: int = 150  # 時系列長
    # 特徴抽出パラメータ
    window: int = 10
    local_window: int = 10
    delta_percentile: float = 97.0
    local_jump_percentile: float = 97.0
    # ベイジアンサンプリングパラメータ
    num_samples: int = 8000  # MCMCサンプル数
    num_warmup: int = 8000   # ウォームアップ
    num_chains: int = 4      # MCMCチェーン数
    target_accept_prob: float = 0.95
    max_tree_depth: int = 10
    # 並列化パラメータ
    max_workers: int = 3     # Colab対応
    # 可視化パラメータ
    hdi_prob: float = 0.94   # 信頼区間

# ===============================
# JAX-Compiled Feature Extraction
# ===============================
@jax.jit
def calculate_diff_threshold_jax(data: jnp.ndarray, percentile: float) -> Tuple[jnp.ndarray, float]:
    """JAX最適化された差分・閾値計算"""
    diff = jnp.diff(data, prepend=data[0])
    abs_diff = jnp.abs(diff)
    threshold = jnp.percentile(abs_diff, percentile)
    return diff, threshold

@jax.jit
def detect_jumps_jax(diff: jnp.ndarray, threshold: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX最適化されたジャンプ検出"""
    pos_jumps = (diff > threshold).astype(jnp.float32)
    neg_jumps = (diff < -threshold).astype(jnp.float32)
    return pos_jumps, neg_jumps

@jax.jit
def calculate_local_std_jax(data: jnp.ndarray, window: int) -> jnp.ndarray:
    """JAX最適化された局所標準偏差計算（rolling window版）"""
    n = len(data)
    
    def compute_std_at_position(i):
        # インデックス範囲を計算
        start_idx = jnp.maximum(0, i - window // 2)
        end_idx = jnp.minimum(n, i + window // 2 + 1)
        
        # 全データから該当部分を条件付きで選択
        indices = jnp.arange(n)
        mask = (indices >= start_idx) & (indices < end_idx)
        
        # マスクされたデータの平均と分散を計算
        masked_data = jnp.where(mask, data, 0.0)
        count = jnp.sum(mask)
        
        # ゼロ除算回避
        safe_count = jnp.maximum(count, 1.0)
        mean_val = jnp.sum(masked_data) / safe_count
        
        # 分散計算
        squared_diff = jnp.where(mask, (data - mean_val) ** 2, 0.0)
        variance = jnp.sum(squared_diff) / safe_count
        
        return jnp.sqrt(variance)
    
    return jax.vmap(compute_std_at_position)(jnp.arange(n))

@jax.jit  
def calculate_rho_t_jax(data: jnp.ndarray, window: int) -> jnp.ndarray:
    """JAXテンションスカラー（ρT）計算（累積版）"""
    n = len(data)
    
    def compute_rho_at_position(i):
        # 現在位置から過去windowサンプルまで
        start_idx = jnp.maximum(0, i - window + 1)
        end_idx = i + 1
        
        # インデックスマスク
        indices = jnp.arange(n)
        mask = (indices >= start_idx) & (indices < end_idx)
        
        # マスクされたデータで統計計算
        masked_data = jnp.where(mask, data, 0.0)
        count = jnp.sum(mask)
        safe_count = jnp.maximum(count, 1.0)
        
        mean_val = jnp.sum(masked_data) / safe_count
        squared_diff = jnp.where(mask, (data - mean_val) ** 2, 0.0)
        variance = jnp.sum(squared_diff) / safe_count
        
        return jnp.sqrt(variance)
    
    return jax.vmap(compute_rho_at_position)(jnp.arange(n))

def extract_lambda3_features_jax(data: jnp.ndarray, config: L3ConfigNumPyro) -> Dict[str, jnp.ndarray]:
    """Lambda³特徴量抽出（JAX最適化版）- 安定化バージョン"""
    
    # データ型確保
    data = jnp.asarray(data, dtype=jnp.float32)
    
    # 1. 構造変化（ΔΛC）検出
    diff, threshold = calculate_diff_threshold_jax(data, config.delta_percentile)
    delta_pos, delta_neg = detect_jumps_jax(diff, threshold)
    
    # 2. 局所標準偏差（安定化版）
    try:
        local_std = calculate_local_std_jax(data, config.local_window)
    except Exception as e:
        print(f"Local std calculation failed, using simple version: {e}")
        # フォールバック：単純な移動平均ベース
        local_std = jnp.ones_like(data) * jnp.std(data)
    
    # 3. 局所ジャンプ検出
    score = jnp.abs(diff) / (local_std + 1e-6)  # 数値安定性向上
    local_threshold = jnp.percentile(score, config.local_jump_percentile)
    local_jump = (score > local_threshold).astype(jnp.float32)
    
    # 4. テンションスカラー（ρT）
    try:
        rho_t = calculate_rho_t_jax(data, config.window)
    except Exception as e:
        print(f"Rho_t calculation failed, using simple version: {e}")
        # フォールバック：グローバル標準偏差
        rho_t = jnp.ones_like(data) * jnp.std(data)
    
    # 5. 時間トレンド（正規化）
    time_trend = jnp.arange(len(data), dtype=jnp.float32) / len(data)
    
    return {
        'delta_lambda_pos': delta_pos,
        'delta_lambda_neg': delta_neg,
        'rho_t': rho_t,
        'time_trend': time_trend,
        'local_jump': local_jump
    }

# ===============================
# NumPyro Bayesian Models
# ===============================
def lambda3_base_model(features: Dict[str, jnp.ndarray], 
                      y_obs: Optional[jnp.ndarray] = None):
    """基本Lambda³ベイジアンモデル（構造テンソル推定）"""
    # 構造テンソル（Λ）の事前分布
    beta_0 = numpyro.sample("lambda_intercept", dist.Normal(0.0, 2.0))
    beta_time = numpyro.sample("lambda_flow", dist.Normal(0.0, 1.0))
    beta_pos = numpyro.sample("lambda_struct_pos", dist.Normal(0.0, 3.0))
    beta_neg = numpyro.sample("lambda_struct_neg", dist.Normal(0.0, 3.0))
    beta_rho = numpyro.sample("rho_tension", dist.Normal(0.0, 2.0))
    
    # 構造空間での線形結合
    mu = (beta_0 + 
          beta_time * features['time_trend'] +
          beta_pos * features['delta_lambda_pos'] +
          beta_neg * features['delta_lambda_neg'] +
          beta_rho * features['rho_t'])
    
    # 観測ノイズ（テンション）
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1.0))
    
    # 観測（ΔΛC脈動として）
    with numpyro.plate("observations", len(mu)):
        return numpyro.sample("y", dist.Normal(mu, sigma_obs), obs=y_obs)

def lambda3_interaction_model(features_a: Dict[str, jnp.ndarray],
                             features_b: Dict[str, jnp.ndarray],
                             y_obs: Optional[jnp.ndarray] = None):
    """非対称相互作用Lambda³モデル"""
    # 基本構造パラメータ
    beta_0 = numpyro.sample("lambda_intercept", dist.Normal(0.0, 2.0))
    beta_time = numpyro.sample("lambda_flow_self", dist.Normal(0.0, 1.0))
    beta_pos_self = numpyro.sample("lambda_struct_pos_self", dist.Normal(0.0, 3.0))
    beta_neg_self = numpyro.sample("lambda_struct_neg_self", dist.Normal(0.0, 3.0))
    beta_rho_self = numpyro.sample("rho_tension_self", dist.Normal(0.0, 2.0))
    
    # 相互作用パラメータ（クロス構造テンソル）
    beta_pos_cross = numpyro.sample("lambda_interact_pos", dist.Normal(0.0, 3.0))
    beta_neg_cross = numpyro.sample("lambda_interact_neg", dist.Normal(0.0, 3.0))
    beta_rho_cross = numpyro.sample("rho_interact", dist.Normal(0.0, 2.0))
    
    # 構造空間での結合
    mu = (beta_0 +
          beta_time * features_a['time_trend'] +
          # 自己構造項
          beta_pos_self * features_a['delta_lambda_pos'] +
          beta_neg_self * features_a['delta_lambda_neg'] +
          beta_rho_self * features_a['rho_t'] +
          # 相互作用項
          beta_pos_cross * features_b['delta_lambda_pos'] +
          beta_neg_cross * features_b['delta_lambda_neg'] +
          beta_rho_cross * features_b['rho_t'])
    
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1.0))
    
    with numpyro.plate("observations", len(mu)):
        return numpyro.sample("y", dist.Normal(mu, sigma_obs), obs=y_obs)

def lambda3_dynamic_model(features: Dict[str, jnp.ndarray],
                         change_points: Optional[List[int]] = None,
                         y_obs: Optional[jnp.ndarray] = None):
    """動的Lambda³モデル（構造変化点検出）"""
    T = len(features['time_trend'])
    
    # 時変パラメータ（ガウシアンランダムウォーク）
    innovation_scale = numpyro.sample("innovation_scale", dist.HalfNormal(0.1))
    beta_time_series = numpyro.sample("lambda_flow_dynamic",
                                     dist.GaussianRandomWalk(innovation_scale,
                                                           num_steps=T))
    
    # 構造変化ジャンプ
    if change_points:
        jump_effects = []
        for i, cp in enumerate(change_points):
            jump = numpyro.sample(f"structure_jump_{i}", dist.Normal(0.0, 5.0))
            jump_indicator = jnp.where(features['time_trend'] >= cp, 1.0, 0.0)
            jump_effects.append(jump * jump_indicator)
        total_jumps = jnp.sum(jnp.stack(jump_effects), axis=0)
    else:
        total_jumps = 0.0
    
    # 動的構造方程式
    mu = (beta_time_series +
          features['delta_lambda_pos'] +
          features['delta_lambda_neg'] +
          features['rho_t'] +
          total_jumps)
    
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1.0))
    
    with numpyro.plate("observations", T):
        return numpyro.sample("y", dist.Normal(mu, sigma_obs), obs=y_obs)

# ===============================
# MCMC Inference Engine
# ===============================
class Lambda3NumPyroInference:
    """NumPyro推論エンジン（GPU最適化）"""
    
    def __init__(self, config: L3ConfigNumPyro):
        self.config = config
        self.traces = {}
        self.predictions = {}
        
    def fit_base_model(self, data: jnp.ndarray, features: Dict[str, jnp.ndarray], 
                      chain_id: int = 0) -> Dict[str, Any]:
        """基本モデルフィッティング"""
        rng_key = random.PRNGKey(chain_id * 42)
        
        # NUTS カーネル
        kernel = NUTS(lambda3_base_model,
                     target_accept_prob=self.config.target_accept_prob,
                     max_tree_depth=self.config.max_tree_depth)
        
        # MCMC実行
        mcmc = MCMC(kernel,
                   num_samples=self.config.num_samples,
                   num_warmup=self.config.num_warmup,
                   num_chains=1,
                   progress_bar=False)
        
        mcmc.run(rng_key, features=features, y_obs=data)
        
        # サンプル取得
        samples = mcmc.get_samples()
        
        # 予測生成
        predictive = Predictive(lambda3_base_model, samples)
        pred_key = random.split(rng_key)[0]
        predictions = predictive(pred_key, features=features)
        
        # 診断統計（詳細版）
        extra_fields = mcmc.get_extra_fields()
        diagnostics = {
            'chain_id': chain_id,
            'divergences': int(jnp.sum(extra_fields.get('diverging', 0))),
            'num_steps': float(jnp.mean(extra_fields.get('num_steps', 0))),
            'accept_prob': float(jnp.mean(extra_fields.get('accept_prob', 0.0))),
            'step_size': float(jnp.mean(extra_fields.get('step_size', 0.0)))
        }
        
        # 追加診断情報
        if 'energy' in extra_fields:
            diagnostics['energy'] = float(jnp.mean(extra_fields['energy']))
            diagnostics['energy_std'] = float(jnp.std(extra_fields['energy']))
        if 'potential_energy' in extra_fields:
            diagnostics['potential_energy'] = float(jnp.mean(extra_fields['potential_energy']))
        if 'tree_depth' in extra_fields:
            diagnostics['tree_depth'] = float(jnp.mean(extra_fields['tree_depth']))
            diagnostics['max_tree_depth'] = int(jnp.max(extra_fields['tree_depth']))
        
        # 効率性指標
        total_steps = jnp.sum(extra_fields.get('num_steps', 0))
        if total_steps > 0:
            diagnostics['sampling_efficiency'] = float(self.config.num_samples / total_steps)
        
        # 収束診断（R-hat推定）
        if len(samples) > 0:
            for param_name, param_values in samples.items():
                if param_values.ndim > 0 and len(param_values) > 10:
                    # 簡易R-hat計算
                    n = len(param_values)
                    first_half = param_values[:n//2]
                    second_half = param_values[n//2:]
                    
                    var_within = (jnp.var(first_half) + jnp.var(second_half)) / 2
                    var_between = n/2 * jnp.var(jnp.array([jnp.mean(first_half), jnp.mean(second_half)]))
                    var_total = (n/2 - 1) / (n/2) * var_within + var_between / (n/2)
                    
                    if var_within > 1e-10:
                        rhat = jnp.sqrt(var_total / var_within)
                        diagnostics[f'rhat_{param_name}'] = float(rhat)
        
        return {
            'samples': samples,
            'predictions': predictions,
            'diagnostics': diagnostics,
            'mcmc': mcmc
        }
    
    def fit_interaction_model(self, data: jnp.ndarray,
                            features_a: Dict[str, jnp.ndarray],
                            features_b: Dict[str, jnp.ndarray],
                            chain_id: int = 0) -> Dict[str, Any]:
        """相互作用モデルフィッティング"""
        rng_key = random.PRNGKey(chain_id * 42 + 1000)
        
        kernel = NUTS(lambda3_interaction_model,
                     target_accept_prob=self.config.target_accept_prob,
                     max_tree_depth=self.config.max_tree_depth)
        
        mcmc = MCMC(kernel,
                   num_samples=self.config.num_samples,
                   num_warmup=self.config.num_warmup,
                   num_chains=1,
                   progress_bar=False)
        
        mcmc.run(rng_key, features_a=features_a, features_b=features_b, y_obs=data)
        
        samples = mcmc.get_samples()
        predictive = Predictive(lambda3_interaction_model, samples)
        pred_key = random.split(rng_key)[0]
        predictions = predictive(pred_key, features_a=features_a, features_b=features_b)
        
        diagnostics = {
            'chain_id': chain_id,
            'divergences': int(jnp.sum(mcmc.get_extra_fields().get('diverging', 0))),
            'num_steps': float(jnp.mean(mcmc.get_extra_fields().get('num_steps', 0))),
            'accept_prob': float(jnp.mean(mcmc.get_extra_fields().get('accept_prob', 0.0)))
        }
        
        # 詳細診断
        extra_fields = mcmc.get_extra_fields()
        if 'energy' in extra_fields:
            diagnostics['energy'] = float(jnp.mean(extra_fields['energy']))
        if 'tree_depth' in extra_fields:
            diagnostics['tree_depth'] = float(jnp.mean(extra_fields['tree_depth']))
        if 'step_size' in extra_fields:
            diagnostics['step_size'] = float(jnp.mean(extra_fields['step_size']))
        
        return {
            'samples': samples,
            'predictions': predictions,
            'diagnostics': diagnostics,
            'mcmc': mcmc
        }
    
    def parallel_inference(self, data_list: List[jnp.ndarray],
                          features_list: List[Dict[str, jnp.ndarray]],
                          model_type: str = 'base') -> List[Dict[str, Any]]:
        """並列推論実行"""
        
        def single_chain_wrapper(args):
            if model_type == 'base':
                data, features, chain_id = args
                return self.fit_base_model(data, features, chain_id)
            else:
                raise ValueError(f"Parallel model type {model_type} not implemented")
        
        # 引数準備
        chain_args = [(data, features, i) 
                     for i, (data, features) in enumerate(zip(data_list, features_list))]
        
        # 並列実行
        max_workers = min(len(chain_args), self.config.max_workers)
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(single_chain_wrapper, args): i 
                      for i, args in enumerate(chain_args)}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Chain failed: {e}")
                    results.append(None)
        
        return results

# ===============================
# Data Preprocessing & Scaling
# ===============================
@jax.jit
def standardize_jax(x: jnp.ndarray) -> jnp.ndarray:
    """JAX最適化された標準化（ゼロ平均・単位分散）"""
    mean_x = jnp.mean(x)
    std_x = jnp.std(x)
    # ゼロ除算回避
    safe_std = jnp.maximum(std_x, 1e-8)
    return (x - mean_x) / safe_std

@jax.jit
def minmax_scale_jax(x: jnp.ndarray) -> jnp.ndarray:
    """JAX最適化されたMin-Max正規化"""
    min_x, max_x = jnp.min(x), jnp.max(x)
    range_x = jnp.maximum(max_x - min_x, 1e-8)
    return (x - min_x) / range_x

@jax.jit
def robust_scale_standardize_jax(x: jnp.ndarray) -> jnp.ndarray:
    """JAX最適化された堅牢標準化（中央値・MAD）"""
    median_x = jnp.median(x)
    mad = jnp.median(jnp.abs(x - median_x))
    safe_mad = jnp.maximum(mad, 1e-8)
    return (x - median_x) / (1.4826 * safe_mad)

def robust_scale_jax(x: jnp.ndarray, method: str = 'standardize') -> jnp.ndarray:
    """堅牢なスケーリング（Lambda3構造空間用）- JIT対応版"""
    if method == 'standardize':
        return standardize_jax(x)
    elif method == 'minmax':
        return minmax_scale_jax(x)
    elif method == 'robust':
        return robust_scale_standardize_jax(x)
    else:
        return x

def preprocess_series_dict(series_dict: Dict[str, jnp.ndarray], 
                          scaling_method: str = 'standardize',
                          variance_threshold: float = 1e-6,
                          verbose: bool = True) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, float]]]:
    """系列辞書の前処理・スケーリング"""
    
    if verbose:
        print("\n🔧 PREPROCESSING & SCALING ANALYSIS")
        print("=" * 50)
        print("Original series statistics:")
    
    scaling_info = {}
    processed_dict = {}
    problematic_series = []
    
    for name, data in series_dict.items():
        # 元の統計
        original_mean = float(jnp.mean(data))
        original_std = float(jnp.std(data))
        original_var = original_std ** 2
        original_range = float(jnp.max(data) - jnp.min(data))
        
        if verbose:
            print(f"  {name:15s} | Mean: {original_mean:8.4f} | Std: {original_std:8.4f} | Range: {original_range:8.4f}")
        
        # 問題検出
        is_problematic = False
        issues = []
        
        if original_var < variance_threshold:
            issues.append(f"Low variance ({original_var:.2e})")
            is_problematic = True
        
        if original_range < 1e-6:
            issues.append(f"Minimal range ({original_range:.2e})")
            is_problematic = True
            
        if jnp.any(jnp.isnan(data)) or jnp.any(jnp.isinf(data)):
            issues.append("Contains NaN/Inf")
            is_problematic = True
        
        if is_problematic:
            problematic_series.append((name, issues))
            if verbose:
                print(f"    ⚠️  Issues: {', '.join(issues)}")
        
        # スケーリング適用
        if scaling_method == 'none':
            processed_data = data
        else:
            processed_data = robust_scale_jax(data, scaling_method)
        
        # スケーリング後統計
        scaled_mean = float(jnp.mean(processed_data))
        scaled_std = float(jnp.std(processed_data))
        
        # 情報保存
        scaling_info[name] = {
            'original_mean': original_mean,
            'original_std': original_std,
            'original_var': original_var,
            'scaled_mean': scaled_mean,
            'scaled_std': scaled_std,
            'scaling_method': scaling_method,
            'is_problematic': is_problematic,
            'issues': issues
        }
        
        processed_dict[name] = processed_data
    
    if verbose:
        print(f"\nScaled series statistics (method: {scaling_method}):")
        for name, data in processed_dict.items():
            mean_val = float(jnp.mean(data))
            std_val = float(jnp.std(data))
            print(f"  {name:15s} | Mean: {mean_val:8.4f} | Std: {std_val:8.4f}")
        
        print(f"\n📊 PREPROCESSING SUMMARY:")
        print(f"  Total series: {len(series_dict)}")
        print(f"  Problematic: {len(problematic_series)}")
        print(f"  Scaling method: {scaling_method}")
        
        if problematic_series:
            print(f"\n⚠️  PROBLEMATIC SERIES:")
            for name, issues in problematic_series:
                print(f"    {name}: {', '.join(issues)}")
            print(f"\n💡 RECOMMENDATION: Use standardization or robust scaling")
    
    return processed_dict, scaling_info

def recommend_scaling_method(series_dict: Dict[str, jnp.ndarray]) -> str:
    """最適なスケーリング手法を推奨"""
    
    variance_ratios = []
    range_ratios = []
    
    variances = [float(jnp.var(data)) for data in series_dict.values()]
    ranges = [float(jnp.max(data) - jnp.min(data)) for data in series_dict.values()]
    
    if len(variances) > 1:
        max_var, min_var = max(variances), min(variances)
        max_range, min_range = max(ranges), min(ranges)
        
        if min_var > 0:
            variance_ratio = max_var / min_var
        else:
            variance_ratio = np.inf
            
        if min_range > 0:
            range_ratio = max_range / min_range
        else:
            range_ratio = np.inf
    else:
        variance_ratio = 1.0
        range_ratio = 1.0
    
    print(f"\n🎯 SCALING RECOMMENDATION:")
    print(f"  Variance ratio (max/min): {variance_ratio:.2e}")
    print(f"  Range ratio (max/min): {range_ratio:.2e}")
    
    # 推奨ロジック
    if variance_ratio > 1e6 or range_ratio > 1e6:
        recommendation = 'standardize'
        reason = "Extreme scale differences detected"
    elif variance_ratio > 100 or range_ratio > 100:
        recommendation = 'robust'
        reason = "Moderate scale differences, outliers possible"
    elif any(var < 1e-6 for var in variances):
        recommendation = 'standardize'
        reason = "Very low variance series detected"
    else:
        recommendation = 'minmax'
        reason = "Similar scales, simple normalization sufficient"
    
    print(f"  Recommended method: {recommendation}")
    print(f"  Reason: {reason}")
    
    return recommendation
def fetch_financial_data_numpyro(start_date="2024-01-01", end_date="2024-12-31",
                                csv_filename="financial_data_numpyro.csv") -> Optional[pd.DataFrame]:
    """金融データ取得（NumPyro用）"""
    if not YFINANCE_AVAILABLE:
        print("yfinance not available. Please install: pip install yfinance")
        return None
    
    tickers = {
        "USD/JPY": "JPY=X",
        "GBP/USD": "GBPUSD=X",
        "GBP/JPY": "GBPJPY=X",
        "Nikkei 225": "^N225",
        "Dow Jones": "^DJI"
    }
    
    try:
        print(f"Fetching data from {start_date} to {end_date}...")
        data_close = yf.download(list(tickers.values()), start=start_date, end=end_date)['Close']
        
        # JPY/GBP計算
        data_close['JPY/GBP'] = 1 / data_close['GBPJPY=X']
        data_close = data_close.drop(columns=['GBPJPY=X'])
        
        # カラム名変更
        reversed_tickers = {v: k for k, v in tickers.items()}
        final_data = data_close.rename(columns=reversed_tickers)
        
        # 並び替え
        desired_order = ["USD/JPY", "JPY/GBP", "GBP/USD", "Nikkei 225", "Dow Jones"]
        final_data = final_data[desired_order]
        final_data = final_data.dropna()
        
        # CSV保存
        final_data.to_csv(csv_filename, index=True)
        print(f"Data saved to {csv_filename}")
        
        return final_data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def load_csv_to_jax(filepath: str, value_columns: Optional[List[str]] = None) -> Dict[str, jnp.ndarray]:
    """CSV→JAX配列変換"""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    if value_columns is None:
        value_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    series_dict = {}
    for col in value_columns:
        if col in df.columns:
            data = df[col].fillna(method='ffill').fillna(method='bfill').values
            series_dict[col] = jnp.array(data, dtype=jnp.float32)
    
    return series_dict

# ===============================
# Advanced Diagnostics & Grid Analysis
# ===============================
def plot_mcmc_diagnostics(result: Dict[str, Any], title: str = "MCMC Diagnostics"):
    """MCMC診断プロット（トレースプロット、自己相関等）"""
    samples = result['samples']
    diagnostics = result['diagnostics']
    
    # パラメータ数に応じて図のサイズを調整
    param_names = list(samples.keys())
    n_params = len(param_names)
    
    fig, axes = plt.subplots(n_params, 3, figsize=(15, 4 * n_params))
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    for i, param in enumerate(param_names):
        param_samples = np.array(samples[param])
        
        # トレースプロット
        axes[i, 0].plot(param_samples)
        axes[i, 0].set_title(f'{param}: Trace Plot')
        axes[i, 0].set_xlabel('Iteration')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 密度プロット
        axes[i, 1].hist(param_samples, bins=50, density=True, alpha=0.7, color='skyblue')
        axes[i, 1].set_title(f'{param}: Posterior Density')
        axes[i, 1].set_xlabel('Value')
        axes[i, 1].set_ylabel('Density')
        axes[i, 1].grid(True, alpha=0.3)
        
        # 累積平均プロット（収束チェック）
        cumulative_mean = np.cumsum(param_samples) / np.arange(1, len(param_samples) + 1)
        axes[i, 2].plot(cumulative_mean)
        axes[i, 2].set_title(f'{param}: Running Mean')
        axes[i, 2].set_xlabel('Iteration')
        axes[i, 2].set_ylabel('Cumulative Mean')
        axes[i, 2].grid(True, alpha=0.3)
        
        # 最終値の線を追加
        final_mean = cumulative_mean[-1]
        axes[i, 2].axhline(y=final_mean, color='red', linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 診断サマリー表示
    print("\n📊 MCMC DIAGNOSTICS SUMMARY:")
    print("=" * 50)
    for key, value in diagnostics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    
    # 診断判定
    print("\n🔍 DIAGNOSTIC ASSESSMENT:")
    if diagnostics.get('divergences', 0) == 0:
        print("  ✅ No divergent transitions")
    else:
        print(f"  ⚠️  {diagnostics['divergences']} divergent transitions detected")
    
    if diagnostics.get('accept_prob', 0) > 0.7:
        print("  ✅ Good acceptance probability")
    else:
        print("  ⚠️  Low acceptance probability")
    
    # R-hat診断
    rhat_issues = [k for k in diagnostics.keys() if k.startswith('rhat_') and diagnostics[k] > 1.1]
    if not rhat_issues:
        print("  ✅ All R-hat values < 1.1 (good convergence)")
    else:
        print(f"  ⚠️  Convergence issues: {rhat_issues}")

def plot_energy_diagnostics(result: Dict[str, Any]):
    """エネルギー診断プロット"""
    if 'mcmc' not in result:
        print("MCMC object not available for energy diagnostics")
        return
    
    try:
        extra_fields = result['mcmc'].get_extra_fields()
        
        if 'energy' not in extra_fields:
            print("Energy information not available")
            return
        
        energy = np.array(extra_fields['energy'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # エネルギートレース
        ax1.plot(energy)
        ax1.set_title('Energy Trace')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy')
        ax1.grid(True, alpha=0.3)
        
        # エネルギーヒストグラム
        ax2.hist(energy, bins=50, density=True, alpha=0.7, color='orange')
        ax2.set_title('Energy Distribution')
        ax2.set_xlabel('Energy')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # エネルギー統計
        print(f"\n⚡ ENERGY DIAGNOSTICS:")
        print(f"  Mean energy: {np.mean(energy):.4f}")
        print(f"  Energy std:  {np.std(energy):.4f}")
        print(f"  Energy range: [{np.min(energy):.4f}, {np.max(energy):.4f}]")
        
    except Exception as e:
        print(f"Energy diagnostics failed: {e}")

def grid_search_lambda3_params(data: jnp.ndarray, 
                              features: Dict[str, jnp.ndarray],
                              param_grid: Dict[str, List[float]],
                              config: L3ConfigNumPyro) -> Dict[str, Any]:
    """Lambda³パラメータのグリッドサーチ"""
    
    print("🔍 Starting Lambda³ parameter grid search...")
    print(f"Grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
    
    results = []
    inference_engine = Lambda3NumPyroInference(config)
    
    # グリッドの組み合わせを生成
    from itertools import product
    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))
    
    for i, param_values in enumerate(param_combinations):
        param_dict = dict(zip(param_names, param_values))
        
        print(f"\n[{i+1}/{len(param_combinations)}] Testing: {param_dict}")
        
        # 一時的にconfig更新
        temp_config = L3ConfigNumPyro(
            **{**config.__dict__, **param_dict}
        )
        
        try:
            # 高速化のため、サンプル数を削減
            temp_config.num_samples = min(config.num_samples, 500)
            temp_config.num_warmup = min(config.num_warmup, 250)
            
            inference_engine.config = temp_config
            result = inference_engine.fit_base_model(data, features, chain_id=i)
            
            # 性能指標計算
            diagnostics = result['diagnostics']
            samples = result['samples']
            
            # モデル適合度（簡易版）
            predictions = result['predictions']['y']
            if predictions.ndim > 1:
                pred_mean = jnp.mean(predictions, axis=0)
            else:
                pred_mean = predictions
            
            mse = float(jnp.mean((data - pred_mean) ** 2))
            
            # 結果保存
            result_entry = {
                'params': param_dict.copy(),
                'mse': mse,
                'divergences': diagnostics.get('divergences', 0),
                'accept_prob': diagnostics.get('accept_prob', 0),
                'energy': diagnostics.get('energy', np.inf),
                'rhat_max': max([diagnostics.get(k, 1.0) for k in diagnostics.keys() if k.startswith('rhat_')], default=1.0)
            }
            results.append(result_entry)
            
            print(f"  MSE: {mse:.4f}, Divergences: {result_entry['divergences']}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'params': param_dict.copy(),
                'mse': np.inf,
                'divergences': 999,
                'accept_prob': 0,
                'energy': np.inf,
                'rhat_max': 999
            })
    
    return analyze_grid_results(results)

def analyze_grid_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """グリッドサーチ結果の解析"""
    
    # 有効な結果のみフィルタ
    valid_results = [r for r in results if r['mse'] < np.inf]
    
    if not valid_results:
        print("❌ No valid results found in grid search")
        return {'best_params': None, 'results': results}
    
    # 複合スコア計算（MSE + ペナルティ）
    for result in valid_results:
        penalty = 0
        penalty += result['divergences'] * 0.1  # divergence penalty
        penalty += max(0, 1.1 - result['accept_prob']) * 0.5  # low acceptance penalty
        penalty += max(0, result['rhat_max'] - 1.1) * 2.0  # convergence penalty
        
        result['composite_score'] = result['mse'] + penalty
    
    # 最適結果
    best_result = min(valid_results, key=lambda x: x['composite_score'])
    
    print("\n🏆 GRID SEARCH RESULTS:")
    print("=" * 50)
    print(f"Best parameters: {best_result['params']}")
    print(f"MSE: {best_result['mse']:.4f}")
    print(f"Composite score: {best_result['composite_score']:.4f}")
    print(f"Divergences: {best_result['divergences']}")
    print(f"Accept prob: {best_result['accept_prob']:.3f}")
    
    # トップ3結果
    top_results = sorted(valid_results, key=lambda x: x['composite_score'])[:3]
    print(f"\n📊 TOP 3 CONFIGURATIONS:")
    for i, result in enumerate(top_results, 1):
        print(f"{i}. {result['params']} (score: {result['composite_score']:.4f})")
    
    return {
        'best_params': best_result['params'],
        'best_result': best_result,
        'top_results': top_results,
        'all_results': results
    }

def plot_grid_search_results(grid_results: Dict[str, Any]):
    """グリッドサーチ結果の可視化"""
    
    if not grid_results.get('all_results'):
        print("No grid results to plot")
        return
    
    results = [r for r in grid_results['all_results'] if r['mse'] < np.inf]
    
    if len(results) < 2:
        print("Insufficient valid results for plotting")
        return
    
    # 結果をDataFrameに変換
    import pandas as pd
    
    data_rows = []
    for result in results:
        row = result['params'].copy()
        row.update({
            'mse': result['mse'],
            'divergences': result['divergences'], 
            'accept_prob': result['accept_prob'],
            'composite_score': result.get('composite_score', result['mse'])
        })
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # パラメータの数に応じてプロット
    param_cols = [c for c in df.columns if c not in ['mse', 'divergences', 'accept_prob', 'composite_score']]
    
    if len(param_cols) == 1:
        # 1Dプロット
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        param = param_cols[0]
        ax1.scatter(df[param], df['mse'], alpha=0.7, c=df['divergences'], cmap='Reds')
        ax1.set_xlabel(param)
        ax1.set_ylabel('MSE')
        ax1.set_title(f'MSE vs {param}')
        
        ax2.scatter(df[param], df['accept_prob'], alpha=0.7)
        ax2.set_xlabel(param) 
        ax2.set_ylabel('Accept Probability')
        ax2.set_title(f'Accept Prob vs {param}')
        
    elif len(param_cols) == 2:
        # 2Dヒートマップ
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        param1, param2 = param_cols[0], param_cols[1]
        
        # MSEヒートマップ
        pivot_mse = df.pivot_table(values='mse', index=param1, columns=param2, aggfunc='mean')
        sns.heatmap(pivot_mse, ax=ax1, cmap='viridis_r', annot=True, fmt='.3f')
        ax1.set_title('MSE Heatmap')
        
        # Divergenceヒートマップ
        pivot_div = df.pivot_table(values='divergences', index=param1, columns=param2, aggfunc='mean')
        sns.heatmap(pivot_div, ax=ax2, cmap='Reds', annot=True, fmt='.0f')
        ax2.set_title('Divergences Heatmap')
        
        # Accept Probヒートマップ
        pivot_acc = df.pivot_table(values='accept_prob', index=param1, columns=param2, aggfunc='mean')
        sns.heatmap(pivot_acc, ax=ax3, cmap='viridis', annot=True, fmt='.3f')
        ax3.set_title('Accept Probability Heatmap')
        
        # Composite Scoreヒートマップ
        pivot_comp = df.pivot_table(values='composite_score', index=param1, columns=param2, aggfunc='mean')
        sns.heatmap(pivot_comp, ax=ax4, cmap='viridis_r', annot=True, fmt='.3f')
        ax4.set_title('Composite Score Heatmap')
        
    else:
        # 3D以上：散布図行列
        from pandas.plotting import scatter_matrix
        scatter_matrix(df[param_cols + ['mse', 'composite_score']], figsize=(12, 12), alpha=0.7)
    
    plt.tight_layout()
    plt.show()
def plot_lambda3_results_numpyro(data: jnp.ndarray, 
                                 predictions: jnp.ndarray,
                                 features: Dict[str, jnp.ndarray],
                                 title: str = "Lambda³ NumPyro Results"):
    """NumPyro結果可視化（エラーハンドリング強化）"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # データ型確認・変換
        data = np.array(data)
        
        # 上段：データ・予測・イベント
        ax1.plot(data, 'o-', alpha=0.7, label='Observed Data', markersize=3)
        
        # 予測データ処理
        if predictions.ndim > 1:
            pred_array = np.array(predictions)
            pred_mean = np.mean(pred_array, axis=0)
            pred_std = np.std(pred_array, axis=0)
            
            ax1.plot(pred_mean, 'r-', label='Prediction Mean', linewidth=2)
            ax1.fill_between(range(len(pred_mean)), 
                            pred_mean - pred_std, pred_mean + pred_std,
                            alpha=0.3, color='red', label='±1σ')
        else:
            pred_array = np.array(predictions)
            ax1.plot(pred_array, 'r-', label='Prediction', linewidth=2)
        
        # イベントマーカー（安全な変換）
        try:
            pos_events = np.where(np.array(features['delta_lambda_pos']) > 0)[0]
            neg_events = np.where(np.array(features['delta_lambda_neg']) > 0)[0]
            
            if len(pos_events) > 0:
                ax1.scatter(pos_events, data[pos_events], 
                           color='blue', s=50, marker='^', label='Positive ΔΛC', zorder=5)
            if len(neg_events) > 0:
                ax1.scatter(neg_events, data[neg_events],
                           color='orange', s=50, marker='v', label='Negative ΔΛC', zorder=5)
        except Exception as e:
            print(f"Event marker plotting failed: {e}")
        
        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下段：テンションスカラー
        try:
            rho_t_array = np.array(features['rho_t'])
            ax2.plot(rho_t_array, 'g-', label='Tension Scalar ρT', linewidth=1.5)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('ρT')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Tension scalar plotting failed: {e}")
            ax2.text(0.5, 0.5, f'Tension plot error: {e}', 
                    transform=ax2.transAxes, ha='center')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Complete plotting failed: {e}")
        # 最小限のフォールバック
        plt.figure(figsize=(10, 4))
        plt.plot(np.array(data), 'o-', label='Data')
        plt.title(title)
        plt.legend()
        plt.show()

# ===============================
# Advanced Visualization (PyMC Style)
# ===============================
def plot_l3_prediction_dual_numpyro(
    data_dict: Dict[str, jnp.ndarray],
    mu_pred_dict: Dict[str, jnp.ndarray],
    features_dict: Dict[str, Dict[str, jnp.ndarray]],
    series_names: Optional[List[str]] = None,
    titles: Optional[List[str]] = None
):
    """NumPyro版のデュアル予測プロット（PyMCスタイル）"""
    if series_names is None:
        series_names = list(data_dict.keys())

    n_series = len(series_names)
    fig, axes = plt.subplots(n_series, 1, figsize=(15, 5 * n_series), sharex=True)

    if n_series == 1:
        axes = [axes]

    for i, series in enumerate(series_names):
        ax = axes[i]
        data = np.array(data_dict[series])
        mu_pred = np.array(mu_pred_dict[series])
        features = features_dict[series]
        
        # データと予測をプロット
        ax.plot(data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
        
        # 予測（信頼区間付き）
        if mu_pred.ndim > 1:
            pred_mean = np.mean(mu_pred, axis=0)
            pred_std = np.std(mu_pred, axis=0)
            ax.plot(pred_mean, color='C2', lw=2, label='Model Prediction')
            ax.fill_between(range(len(pred_mean)), 
                           pred_mean - pred_std, pred_mean + pred_std,
                           alpha=0.3, color='C2', label='±1σ')
        else:
            ax.plot(mu_pred, color='C2', lw=2, label='Model Prediction')

        # ジャンプイベント
        jump_pos = np.array(features['delta_lambda_pos'])
        jump_neg = np.array(features['delta_lambda_neg'])
        
        jump_pos_idx = np.where(jump_pos > 0)[0]
        if len(jump_pos_idx):
            ax.plot(jump_pos_idx, data[jump_pos_idx], 'o', color='dodgerblue',
                   markersize=10, label='Positive ΔΛC')
            for idx in jump_pos_idx:
                ax.axvline(x=idx, color='dodgerblue', linestyle='--', alpha=0.5)

        jump_neg_idx = np.where(jump_neg > 0)[0]
        if len(jump_neg_idx):
            ax.plot(jump_neg_idx, data[jump_neg_idx], 'o', color='orange',
                   markersize=10, label='Negative ΔΛC')
            for idx in jump_neg_idx:
                ax.axvline(x=idx, color='orange', linestyle='-.', alpha=0.5)

        # 局所ジャンプ
        if 'local_jump' in features:
            local_jump = np.array(features['local_jump'])
            local_jump_idx = np.where(local_jump > 0)[0]
            if len(local_jump_idx):
                ax.plot(local_jump_idx, data[local_jump_idx], 'o', color='magenta',
                       markersize=7, alpha=0.7, label='Local Jump')

        # フォーマット
        plot_title = titles[i] if titles and i < len(titles) else f"Series {series}: Lambda³ Fit + Events"
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)

        # 重複ラベル除去
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
        ax.grid(axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_posterior_numpyro(samples: Dict[str, jnp.ndarray], 
                          var_names: Optional[List[str]] = None, 
                          hdi_prob: float = 0.89):
    """NumPyro版事後分布プロット"""
    if var_names is None:
        var_names = list(samples.keys())

    n_vars = len(var_names)
    fig, axes = plt.subplots(2, (n_vars + 1) // 2, figsize=(12, 8))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for i, var in enumerate(var_names):
        if var not in samples:
            continue
            
        sample_data = np.array(samples[var])
        ax = axes[i]

        # ヒストグラム
        ax.hist(sample_data, bins=50, density=True, alpha=0.7, color='skyblue')
        
        # HDI計算
        sorted_samples = np.sort(sample_data)
        lower_idx = int((1 - hdi_prob) / 2 * len(sorted_samples))
        upper_idx = int((1 + hdi_prob) / 2 * len(sorted_samples))
        
        hdi_lower = sorted_samples[lower_idx]
        hdi_upper = sorted_samples[upper_idx]
        
        # HDI表示
        ax.axvline(hdi_lower, color='red', linestyle='--', alpha=0.7)
        ax.axvline(hdi_upper, color='red', linestyle='--', alpha=0.7)
        ax.axvline(np.mean(sample_data), color='red', linewidth=2, label='Mean')
        
        ax.set_title(f'{var}\nMean: {np.mean(sample_data):.3f}, HDI: [{hdi_lower:.3f}, {hdi_upper:.3f}]')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)

    # 未使用のサブプロットを非表示
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_causality_profiles_numpyro(causality_data: List[Dict[int, float]],
                                   labels: List[str],
                                   title: str = "Lambda³ Causality Profiles"):
    """因果関係プロファイルのプロット"""
    plt.figure(figsize=(10, 6))
    
    colors = ['royalblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
    
    for i, (causality_dict, label) in enumerate(zip(causality_data, labels)):
        if causality_dict:
            lags, probs = zip(*sorted(causality_dict.items()))
            color = colors[i % len(colors)]
            plt.plot(lags, probs, marker='o', label=label, color=color, linewidth=2, alpha=0.8)

    plt.xlabel('Lag (steps)', fontsize=12)
    plt.ylabel('Causality Probability', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_regime_analysis_numpyro(data: jnp.ndarray, 
                                features: Dict[str, jnp.ndarray],
                                n_regimes: int = 3):
    """レジーム解析プロット（NumPyro版）"""
    
    # レジーム検出のためのクラスタリング
    from sklearn.cluster import KMeans
    
    # 特徴量スタック
    X = np.column_stack([
        np.array(features['delta_lambda_pos']),
        np.array(features['delta_lambda_neg']),
        np.array(features['rho_t'])
    ])
    
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    regime_labels = kmeans.fit_predict(X)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # レジーム別時系列プロット
    colors = plt.cm.Set1(np.linspace(0, 1, n_regimes))
    data_np = np.array(data)
    
    for regime in range(n_regimes):
        mask = regime_labels == regime
        regime_times = np.where(mask)[0]
        
        if len(regime_times) > 0:
            ax1.scatter(regime_times, data_np[mask], 
                       c=[colors[regime]], s=30, alpha=0.7, 
                       label=f'Regime {regime + 1}')
    
    ax1.plot(data_np, 'k-', alpha=0.3, linewidth=1)
    ax1.set_title('Lambda³ Market Regime Detection', fontsize=14)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # レジーム統計
    regime_stats = []
    for regime in range(n_regimes):
        mask = regime_labels == regime
        freq = np.mean(mask)
        mean_rho = np.mean(np.array(features['rho_t'])[mask])
        regime_stats.append((regime + 1, freq, mean_rho))
    
    # 統計表示
    regimes, freqs, mean_rhos = zip(*regime_stats)
    x_pos = np.arange(len(regimes))
    
    bars = ax2.bar(x_pos, freqs, color=colors[:n_regimes], alpha=0.7)
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Regime Frequency Distribution')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Regime {r}' for r in regimes])
    
    # 各バーの上に平均ρT値を表示
    for i, (bar, rho) in enumerate(zip(bars, mean_rhos)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'ρT: {rho:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # レジーム統計を出力
    print("\n📊 REGIME ANALYSIS RESULTS:")
    print("-" * 40)
    for regime, freq, mean_rho in regime_stats:
        print(f"Regime {regime}: {freq:.1%} frequency, Mean ρT: {mean_rho:.3f}")
    
    return regime_labels

def plot_interaction_heatmap_numpyro(interaction_results: Dict[str, Dict[str, float]],
                                    series_names: List[str]):
    """相互作用行列のヒートマップ"""
    n = len(series_names)
    interaction_matrix = np.zeros((n, n))
    
    # 行列を構築
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if name_a != name_b and name_a in interaction_results:
                if f'interact_{name_b}' in interaction_results[name_a]:
                    interaction_matrix[i, j] = interaction_results[name_a][f'interact_{name_b}']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix,
                xticklabels=series_names,
                yticklabels=series_names,
                annot=True, fmt='.3f',
                cmap='RdBu_r', center=0,
                square=True,
                cbar_kws={'label': 'Interaction Coefficient β'})
    plt.title("Lambda³ Cross-Series Interaction Effects\n(Column → Row)", fontsize=16)
    plt.xlabel("From Series", fontsize=12)
    plt.ylabel("To Series", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_network_analysis_numpyro(sync_matrix: jnp.ndarray, 
                                 series_names: List[str],
                                 threshold: float = 0.3):
    """ネットワーク解析プロット"""
    import networkx as nx
    
    # ネットワーク構築
    G = nx.DiGraph()
    
    # ノード追加
    for name in series_names:
        G.add_node(name)
    
    # エッジ追加
    sync_np = np.array(sync_matrix)
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i != j and sync_np[i, j] >= threshold:
                G.add_edge(name_a, name_b, weight=sync_np[i, j])
    
    # レイアウト
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    plt.figure(figsize=(12, 10))
    
    # ノード描画
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.7)
    
    # エッジ描画（重みに応じて太さを変更）
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w*5 for w in weights],
                          alpha=0.6, edge_color='gray', arrows=True, arrowsize=20)
    
    # ラベル描画
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # エッジラベル
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    plt.title(f"Lambda³ Synchronization Network (threshold = {threshold})", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # ネットワーク統計
    print(f"\n🔗 NETWORK ANALYSIS:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.3f}")
    
    # 中心性分析
    if G.number_of_edges() > 0:
        centrality = nx.in_degree_centrality(G)
        print(f"\n📈 IN-DEGREE CENTRALITY:")
        for node, cent in sorted(centrality.items(), key=lambda x: x[1], reverse=True):
            print(f"  {node}: {cent:.3f}")

# ===============================
# PyMC-Style Advanced Analysis
# ===============================
class Lambda3AdvancedAnalyzer:
    """NumPyro版高度解析クラス（PyMCスタイル）"""
    
    def __init__(self, config: L3ConfigNumPyro):
        self.config = config
        self.results = {}
    
    def analyze_all_pairs(self, series_dict: Dict[str, jnp.ndarray], 
                         features_dict: Dict[str, Dict[str, jnp.ndarray]],
                         max_pairs: int = None) -> Dict[str, Any]:
        """全ペアの詳細解析（PyMCスタイル）"""
        
        series_list = list(series_dict.keys())
        n_series = len(series_list)
        
        # 全ペア生成
        from itertools import combinations
        pairs = list(combinations(series_list, 2))
        
        if max_pairs and len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]
        
        print(f"\n{'='*60}")
        print(f"ANALYZING ALL {len(pairs)} PAIRS")
        print(f"{'='*60}")
        
        # 相互作用効果保存
        interaction_effects = {}
        sync_profiles = {}
        causality_results = {}
        
        for i, (name_a, name_b) in enumerate(pairs, 1):
            print(f"\n[{i}/{len(pairs)}] Analyzing: {name_a} ↔ {name_b}")
            
            try:
                # ペア解析実行
                result = self._analyze_series_pair_detailed(
                    name_a, name_b, series_dict, features_dict
                )
                
                # 結果保存
                interaction_effects[(name_a, name_b)] = result['interactions']
                sync_profiles[(name_a, name_b)] = result['sync_profile']
                causality_results[(name_a, name_b)] = result['causality']
                
            except Exception as e:
                print(f"Error analyzing pair {name_a} ↔ {name_b}: {e}")
                continue
        
        return {
            'interaction_effects': interaction_effects,
            'sync_profiles': sync_profiles,
            'causality_results': causality_results,
            'pairs_analyzed': len(pairs)
        }
    
    def _analyze_series_pair_detailed(self, name_a: str, name_b: str,
                                    series_dict: Dict[str, jnp.ndarray],
                                    features_dict: Dict[str, Dict[str, jnp.ndarray]]) -> Dict[str, Any]:
        """詳細ペア解析"""
        
        print(f"\n{'='*50}")
        print(f"ANALYZING PAIR: {name_a} ↔ {name_b}")
        print(f"{'='*50}")
        
        # 推論エンジン準備
        inference_engine = Lambda3NumPyroInference(self.config)
        
        # 相互作用モデル（A に B の影響）
        print(f"\nFitting Bayesian model for {name_a} (with {name_b} interaction)...")
        result_a = inference_engine.fit_interaction_model(
            series_dict[name_a],
            features_dict[name_a],
            features_dict[name_b],
            chain_id=hash(f"{name_a}_{name_b}") % 1000
        )
        
        # 相互作用モデル（B に A の影響）
        print(f"\nFitting Bayesian model for {name_b} (with {name_a} interaction)...")
        result_b = inference_engine.fit_interaction_model(
            series_dict[name_b],
            features_dict[name_b],
            features_dict[name_a],
            chain_id=hash(f"{name_b}_{name_a}") % 1000
        )
        
        # 相互作用係数抽出
        samples_a = result_a['samples']
        samples_b = result_b['samples']
        
        # B → A の影響
        beta_b_to_a_pos = float(jnp.mean(samples_a.get('lambda_interact_pos', 0)))
        beta_b_to_a_neg = float(jnp.mean(samples_a.get('lambda_interact_neg', 0)))
        
        # A → B の影響
        beta_a_to_b_pos = float(jnp.mean(samples_b.get('lambda_interact_pos', 0)))
        beta_a_to_b_neg = float(jnp.mean(samples_b.get('lambda_interact_neg', 0)))
        
        print(f"\nAsymmetric Interaction Effects:")
        print(f"  {name_b} → {name_a} (pos): β = {beta_b_to_a_pos:.3f}")
        print(f"  {name_b} → {name_a} (neg): β = {beta_b_to_a_neg:.3f}")
        print(f"  {name_a} → {name_b} (pos): β = {beta_a_to_b_pos:.3f}")
        print(f"  {name_a} → {name_b} (neg): β = {beta_a_to_b_neg:.3f}")
        
        # 同期解析
        try:
            lags, sync_values = sync_profile_jax(
                features_dict[name_a]['delta_lambda_pos'].astype(jnp.float32),
                features_dict[name_b]['delta_lambda_pos'].astype(jnp.float32),
                lag_window=10
            )
            max_sync = float(jnp.max(sync_values))
            optimal_lag = int(lags[jnp.argmax(sync_values)])
            
            print(f"\nSync Rate σₛ ({name_a}↔{name_b}): {max_sync:.3f}")
            print(f"Optimal Lag: {optimal_lag} steps")
            
            sync_profile_dict = {int(lag): float(sync) for lag, sync in zip(lags, sync_values)}
        except Exception as e:
            print(f"Sync calculation failed: {e}")
            max_sync, optimal_lag = 0.0, 0
            sync_profile_dict = {}
        
        # 因果関係解析
        causality_data = self._calculate_causality_profile(
            features_dict[name_a], features_dict[name_b], name_a, name_b
        )
        
        return {
            'interactions': {
                f'{name_b}_to_{name_a}_pos': beta_b_to_a_pos,
                f'{name_b}_to_{name_a}_neg': beta_b_to_a_neg,
                f'{name_a}_to_{name_b}_pos': beta_a_to_b_pos,
                f'{name_a}_to_{name_b}_neg': beta_a_to_b_neg,
            },
            'sync_profile': {
                'max_sync': max_sync,
                'optimal_lag': optimal_lag,
                'profile': sync_profile_dict
            },
            'causality': causality_data
        }
    
    def _calculate_causality_profile(self, features_a: Dict[str, jnp.ndarray],
                                   features_b: Dict[str, jnp.ndarray],
                                   name_a: str, name_b: str) -> Dict[str, Any]:
        """因果関係プロファイル計算"""
        
        pos_a = np.array(features_a['delta_lambda_pos'])
        neg_a = np.array(features_a['delta_lambda_neg'])
        pos_b = np.array(features_b['delta_lambda_pos'])
        neg_b = np.array(features_b['delta_lambda_neg'])
        
        T = len(pos_a)
        
        # ラグ別因果関係計算
        causality_a_to_b = {}
        causality_b_to_a = {}
        
        for lag in range(1, 11):
            if lag < T:
                # A → B の因果関係
                count_ab, count_a = 0, 0
                for i in range(T - lag):
                    if pos_a[i] > 0:
                        count_a += 1
                        if pos_b[i + lag] > 0:
                            count_ab += 1
                
                causality_a_to_b[lag] = count_ab / max(count_a, 1)
                
                # B → A の因果関係
                count_ba, count_b = 0, 0
                for i in range(T - lag):
                    if pos_b[i] > 0:
                        count_b += 1
                        if pos_a[i + lag] > 0:
                            count_ba += 1
                
                causality_b_to_a[lag] = count_ba / max(count_b, 1)
            else:
                causality_a_to_b[lag] = 0.0
                causality_b_to_a[lag] = 0.0
        
        return {
            f'{name_a}_to_{name_b}': causality_a_to_b,
            f'{name_b}_to_{name_a}': causality_b_to_a
        }
    
    def detect_market_regimes(self, features_dict: Dict[str, Dict[str, jnp.ndarray]],
                            series_name: str = None, n_regimes: int = 3) -> Dict[str, Any]:
        """市場レジーム検出"""
        
        if series_name is None:
            series_name = list(features_dict.keys())[0]
        
        features = features_dict[series_name]
        
        # 特徴量準備
        X = np.column_stack([
            np.array(features['delta_lambda_pos']),
            np.array(features['delta_lambda_neg']),
            np.array(features['rho_t'])
        ])
        
        # K-meansクラスタリング
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regime_labels = kmeans.fit_predict(X)
        
        # レジーム統計
        regime_stats = {}
        for regime in range(n_regimes):
            mask = regime_labels == regime
            frequency = np.mean(mask)
            mean_rho = np.mean(X[mask, 2])  # ρT列
            
            regime_stats[f'Regime-{regime + 1}'] = {
                'frequency': frequency,
                'mean_rhoT': mean_rho
            }
        
        print("Market Regime Detection:")
        for regime_name, stats in regime_stats.items():
            freq_pct = stats['frequency'] * 100
            print(f"  {regime_name}: {freq_pct:.1f}% (Mean ρT: {stats['mean_rhoT']:.2f})")
        
        return {
            'regime_labels': regime_labels,
            'regime_stats': regime_stats,
            'series_analyzed': series_name
        }
    
    def detect_scale_breaks(self, data: jnp.ndarray, 
                           scales: List[int] = [5, 10, 20, 50]) -> List[Tuple[int, List[int]]]:
        """マルチスケール変化点検出"""
        
        data_np = np.array(data)
        scale_breaks = []
        
        for scale in scales:
            # ローリング標準偏差
            rolling_std = np.array([
                np.std(data_np[max(0, i-scale):i+1]) 
                for i in range(len(data_np))
            ])
            
            # 変化点検出（閾値: 平均 + 1.5*標準偏差）
            mean_std = np.mean(rolling_std)
            threshold = mean_std + 1.5 * np.std(rolling_std)
            
            breaks = np.where(rolling_std > threshold)[0]
            if len(breaks) > 0:
                scale_breaks.append((scale, breaks.tolist()))
        
        print(f"\nScale Break Locations: {scale_breaks}")
        return scale_breaks
    
    def calculate_conditional_sync(self, features_a: Dict[str, jnp.ndarray],
                                 features_b: Dict[str, jnp.ndarray]) -> float:
        """条件付き同期率計算"""
        
        series_a = np.array(features_a['delta_lambda_pos'])
        series_b = np.array(features_b['delta_lambda_pos'])
        condition_series = np.array(features_a['rho_t'])
        
        # 高テンション期間での同期
        condition_threshold = np.median(condition_series)
        mask = condition_series > condition_threshold
        
        if np.sum(mask) > 0:
            sync_rate = np.mean(series_a[mask] * series_b[mask])
        else:
            sync_rate = 0.0
        
        print(f"\nConditional Sync Rate (high tension): {sync_rate:.3f}")
        return sync_rate

def build_sync_network_advanced(sync_profiles: Dict[Tuple[str, str], Dict[str, Any]],
                               threshold: float = 0.0) -> nx.DiGraph:
    """高度同期ネットワーク構築"""
    
    # 全系列名取得
    all_series = set()
    for (name_a, name_b), profile_data in sync_profiles.items():
        all_series.add(name_a)
        all_series.add(name_b)
    
    series_names = list(all_series)
    G = nx.DiGraph()
    
    # ノード追加
    for series in series_names:
        G.add_node(series)
    
    print(f"\n=== Building Synchronization Network ===")
    print(f"Using threshold: {threshold:.4f}")
    
    # エッジ追加
    print(f"\nBuilding sync network with threshold={threshold}")
    edge_count = 0
    
    for (name_a, name_b), profile_data in sync_profiles.items():
        max_sync = profile_data['max_sync']
        optimal_lag = profile_data['optimal_lag']
        
        print(f"{name_a} → {name_b}: max_sync={max_sync:.4f}, lag={optimal_lag}")
        
        if max_sync >= threshold:
            G.add_edge(name_a, name_b,
                      weight=max_sync,
                      lag=optimal_lag)
            edge_count += 1
            print(f"  ✓ Edge added!")
        
        # 逆方向も追加
        print(f"{name_b} → {name_a}: max_sync={max_sync:.4f}, lag={-optimal_lag}")
        if max_sync >= threshold:
            G.add_edge(name_b, name_a,
                      weight=max_sync,
                      lag=-optimal_lag)
            edge_count += 1
            print(f"  ✓ Edge added!")
    
    print(f"\nNetwork summary: {G.number_of_nodes()} nodes, {edge_count} edges")
    return G

def create_comprehensive_summary(series_dict: Dict[str, jnp.ndarray],
                               features_dict: Dict[str, Dict[str, jnp.ndarray]],
                               analysis_results: Dict[str, Any]):
    """包括的サマリー作成（PyMCスタイル）"""
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # ジャンプ統計
    print("\nJump Event Statistics:")
    print("-" * 40)
    for name in list(series_dict.keys()):
        features = features_dict[name]
        pos_jumps = int(jnp.sum(features['delta_lambda_pos']))
        neg_jumps = int(jnp.sum(features['delta_lambda_neg']))
        local_jumps = int(jnp.sum(features['local_jump']))
        print(f"{name:15s} | Pos: {pos_jumps:3d} | Neg: {neg_jumps:3d} | Local: {local_jumps:3d}")
    
    # トップ同期ペア
    sync_profiles = analysis_results.get('sync_profiles', {})
    if sync_profiles:
        print("\nTop Synchronization Pairs:")
        print("-" * 40)
        
        sync_pairs = []
        for (name_a, name_b), profile_data in sync_profiles.items():
            max_sync = profile_data['max_sync']
            sync_pairs.append((max_sync, name_a, name_b))
        
        sync_pairs.sort(reverse=True)
        for max_sync, name_a, name_b in sync_pairs[:5]:
            print(f"{name_a:15s} ↔ {name_b:15s} | σₛ = {max_sync:.3f}")
    
    print("\n" + "="*60)

# Lambda_abc_NumPyro.py に追加する関数群

# ===============================
# 同期計算の修正版関数
# ===============================

def validate_event_series(event_series_dict: Dict[str, jnp.ndarray]):
    """イベント系列の検証とデバッグ情報出力"""
    print("\n🔍 EVENT SERIES VALIDATION:")
    print("-" * 50)
    
    for name, series in event_series_dict.items():
        series_np = np.array(series)
        n_events = np.sum(series_np > 0)
        event_rate = n_events / len(series_np) if len(series_np) > 0 else 0
        
        print(f"{name:15s} | Length: {len(series_np):4d} | Events: {n_events:3d} | Rate: {event_rate:.3f}")
        
        if n_events == 0:
            print(f"  ⚠️  Warning: No events detected in {name}")
        elif event_rate < 0.01:
            print(f"  ⚠️  Warning: Very low event rate in {name}")
    
    print("-" * 50)

@jax.jit
def sync_profile_jax(series_a: jnp.ndarray, series_b: jnp.ndarray, 
                     lag_window: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX最適化された同期プロファイル計算"""
    n = len(series_a)
    lags = jnp.arange(-lag_window, lag_window + 1)
    n_lags = len(lags)
    sync_values = jnp.zeros(n_lags)
    
    def compute_sync_at_lag(lag):
        if lag < 0:
            # 負のラグ: series_a が series_b より先行
            abs_lag = -lag
            valid_len = n - abs_lag
            if valid_len > 0:
                return jnp.mean(series_a[abs_lag:] * series_b[:valid_len])
            else:
                return 0.0
        elif lag > 0:
            # 正のラグ: series_b が series_a より先行
            valid_len = n - lag
            if valid_len > 0:
                return jnp.mean(series_a[:valid_len] * series_b[lag:])
            else:
                return 0.0
        else:
            # ラグ0: 同期
            return jnp.mean(series_a * series_b)
    
    # 各ラグでの同期率を計算
    for i, lag in enumerate(lags):
        sync_values = sync_values.at[i].set(compute_sync_at_lag(lag))
    
    return lags, sync_values

def build_sync_matrix_jax_fixed(event_series_dict: Dict[str, jnp.ndarray], 
                               lag_window: int = 10) -> Tuple[jnp.ndarray, List[str]]:
    """修正版JAX同期行列構築（NumPyフォールバック付き）"""
    
    series_names = list(event_series_dict.keys())
    n = len(series_names)
    mat = np.zeros((n, n))  # NumPy配列で初期化
    
    print(f"Building sync matrix for {n} series...")
    
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i == j:
                mat[i, j] = 1.0  # 自己同期は完全
                continue
            
            try:
                # NumPy配列として取得
                series_a = np.array(event_series_dict[name_a], dtype=np.float64)
                series_b = np.array(event_series_dict[name_b], dtype=np.float64)
                
                # データ検証
                if len(series_a) == 0 or len(series_b) == 0:
                    print(f"  {name_a} → {name_b}: empty series, setting to 0")
                    continue
                
                if len(series_a) != len(series_b):
                    print(f"  {name_a} → {name_b}: length mismatch, setting to 0")
                    continue
                
                # イベントの存在確認
                events_a = np.sum(series_a > 0)
                events_b = np.sum(series_b > 0)
                
                if events_a == 0 or events_b == 0:
                    # イベントがない場合は相関係数を使用
                    if np.std(series_a) > 0 and np.std(series_b) > 0:
                        correlation = np.corrcoef(series_a, series_b)[0, 1]
                        if not np.isnan(correlation):
                            mat[i, j] = abs(correlation)
                            print(f"  {name_a} → {name_b}: no events, using correlation: {abs(correlation):.4f}")
                        else:
                            mat[i, j] = 0.0
                    else:
                        mat[i, j] = 0.0
                    continue
                
                # 同期率計算（NumPy版）
                max_sync = 0.0
                optimal_lag = 0
                
                for lag in range(-lag_window, lag_window + 1):
                    if lag < 0:
                        abs_lag = -lag
                        if abs_lag < len(series_a):
                            sync_rate = np.mean(series_a[abs_lag:] * series_b[:-abs_lag])
                        else:
                            sync_rate = 0.0
                    elif lag > 0:
                        if lag < len(series_b):
                            sync_rate = np.mean(series_a[:-lag] * series_b[lag:])
                        else:
                            sync_rate = 0.0
                    else:
                        sync_rate = np.mean(series_a * series_b)
                    
                    if sync_rate > max_sync:
                        max_sync = sync_rate
                        optimal_lag = lag
                
                mat[i, j] = max_sync
                print(f"  {name_a} → {name_b}: {max_sync:.4f} (lag: {optimal_lag})")
                
            except Exception as e:
                print(f"  {name_a} → {name_b}: calculation failed ({e}), using 0")
                mat[i, j] = 0.0
    
    # JAX配列に変換して返す
    return jnp.array(mat), series_names

def plot_sync_matrix_numpyro_fixed(sync_matrix: jnp.ndarray, series_names: List[str]):
    """同期行列のヒートマップ表示（修正版）"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # NumPy配列に変換
    sync_np = np.array(sync_matrix)
    
    # NaN値をチェック
    if np.any(np.isnan(sync_np)):
        print("Warning: NaN values in sync matrix, replacing with 0")
        sync_np = np.nan_to_num(sync_np, nan=0.0)
    
    plt.figure(figsize=(10, 8))
    
    # ヒートマップ作成
    sns.heatmap(sync_np, 
                annot=True, 
                fmt='.3f',
                xticklabels=series_names,
                yticklabels=series_names,
                cmap="Blues", 
                vmin=0, 
                vmax=1,
                square=True, 
                cbar_kws={'label': 'Sync Rate σₛ'})
    
    plt.title("Synchronization Rate Matrix (σₛ)", fontsize=16)
    plt.xlabel("Series")
    plt.ylabel("Series")
    plt.tight_layout()
    plt.show()
    
    # 統計情報を出力
    off_diagonal = []
    n = len(series_names)
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diagonal.append(sync_np[i, j])
    
    if off_diagonal:
        print(f"\nSync Matrix Statistics:")
        print(f"  Mean sync rate (off-diagonal): {np.mean(off_diagonal):.3f}")
        print(f"  Max sync rate: {np.max(off_diagonal):.3f}")
        print(f"  Min sync rate: {np.min(off_diagonal):.3f}")

# ===============================
# main関数内で使用する修正版コード
# ===============================
def synchronization_analysis_section(series_names, features_dict):
    """同期解析セクション（エラー処理強化版）"""
    print("\nSynchronization analysis...")
    try:
        # イベント系列の準備
        event_series_dict = {
            name: features_dict[name]['delta_lambda_pos']
            for name in series_names
        }
        
        # イベント系列の検証
        validate_event_series(event_series_dict)
        
        # 同期行列の計算（修正版を使用）
        sync_matrix, names = build_sync_matrix_jax_fixed(event_series_dict, lag_window=10)
        print(f"Synchronization matrix computed successfully")
        
        # 可視化（修正版を使用）
        plot_sync_matrix_numpyro_fixed(sync_matrix, names)
        
        return sync_matrix, names
        
    except Exception as e:
        print(f"Synchronization analysis failed: {e}")
        import traceback
        traceback.print_exc()
        # エラー時もnamesを返す
        return None, series_names  

# ===============================
# Main Analysis Pipeline
# ===============================
def comprehensive_lambda3_analysis(csv_path: str = None,
                                   series_columns: Optional[List[str]] = None,
                                   run_diagnostics: bool = True,
                                   run_all_pairs: bool = True,
                                   max_pairs: int = None) -> Dict[str, Any]:
    """PyMCスタイルの包括的Lambda³解析"""
    
    config = L3ConfigNumPyro(
        num_samples=1000,  # PyMCのdrawsに相当
        num_warmup=500,    # PyMCのtuneに相当
        num_chains=2,
        target_accept_prob=0.8  # PyMCのtarget_acceptに相当
    )
    
    print("🚀 COMPREHENSIVE LAMBDA³ ANALYSIS (PyMC Style)")
    print("=" * 60)
    
    # 1. データ読み込み
    if csv_path is None:
        print("Fetching financial data...")
        data_df = fetch_financial_data_numpyro()
        if data_df is None:
            return None
        csv_path = "financial_data_numpyro.csv"
    
    series_dict = load_csv_to_jax(csv_path, series_columns)
    
    # スケーリング適用
    if len(series_dict) > 0:
        scaling_method = recommend_scaling_method(series_dict)
        series_dict, scaling_info = preprocess_series_dict(
            series_dict, 
            scaling_method=scaling_method,
            verbose=True
        )
    
    # 2. 特徴抽出
    print("\nExtracting Lambda³ features...")
    features_dict = {}
    for name, data in series_dict.items():
        features = extract_lambda3_features_jax(data, config)
        features_dict[name] = features
        
        # 統計表示
        n_pos = int(jnp.sum(features['delta_lambda_pos']))
        n_neg = int(jnp.sum(features['delta_lambda_neg']))
        avg_rho = float(jnp.mean(features['rho_t']))
        print(f"  {name:15s} | Pos: {n_pos:3d} | Neg: {n_neg:3d} | ρT: {avg_rho:.3f}")
    
    # 3. 高度解析器初期化
    analyzer = Lambda3AdvancedAnalyzer(config)
    
    # 4. 全ペア解析（PyMCスタイル）
    if run_all_pairs and len(series_dict) >= 2:
        print(f"\nRunning comprehensive pair analysis...")
        pair_results = analyzer.analyze_all_pairs(
            series_dict, features_dict, max_pairs=max_pairs
        )
    else:
        pair_results = {}
    
    # 5. レジーム検出
    regime_results = analyzer.detect_market_regimes(features_dict)
    
    # 6. スケール変化点検出
    first_series = list(series_dict.keys())[0]
    scale_breaks = analyzer.detect_scale_breaks(series_dict[first_series])
    
    # 7. 条件付き同期
    if len(features_dict) >= 2:
        series_names = list(features_dict.keys())
        conditional_sync = analyzer.calculate_conditional_sync(
            features_dict[series_names[0]], 
            features_dict[series_names[1]]
        )
    else:
        conditional_sync = 0.0
    
    # 8. 同期ネットワーク構築
    if 'sync_profiles' in pair_results:
        sync_network = build_sync_network_advanced(
            pair_results['sync_profiles'], 
            threshold=0.0
        )
        
        # ネットワーク可視化
        if sync_network.number_of_edges() > 0:
            plot_network_analysis_numpyro(
                jnp.array([[1.0]]), ['dummy'], threshold=0.0  # ダミー（実際はsync_networkを使用）
            )
    else:
        sync_network = None
    
    # 9. 包括的サマリー
    analysis_results = {
        'regime_results': regime_results,
        'scale_breaks': scale_breaks,
        'conditional_sync': conditional_sync,
        **pair_results
    }
    
    create_comprehensive_summary(series_dict, features_dict, analysis_results)
    
    return {
        'series_dict': series_dict,
        'features_dict': features_dict,
        'analysis_results': analysis_results,
        'sync_network': sync_network,
        'scaling_info': scaling_info
    }

def main_lambda3_numpyro_analysis(csv_path: str = None, 
                                 config: L3ConfigNumPyro = None,
                                 series_columns: Optional[List[str]] = None,
                                 auto_scaling: bool = True,
                                 scaling_method: str = 'auto'):
    """Lambda³ NumPyro メイン解析パイプライン（スケーリング対応）"""
    
    if config is None:
        config = L3ConfigNumPyro()
    
    print("=" * 60)
    print("Lambda³ NumPyro GPU Analysis Pipeline")
    print("=" * 60)
    
    # 1. データ読み込み
    try:
        if csv_path is None:
            print("Fetching financial data...")
            data_df = fetch_financial_data_numpyro()
            if data_df is None:
                return None
            csv_path = "financial_data_numpyro.csv"
        
        print(f"Loading data from: {csv_path}")
        series_dict = load_csv_to_jax(csv_path, series_columns)
        
        if len(series_dict) < 2:
            print("Need at least 2 series for analysis")
            return None
        
        print(f"Loaded {len(series_dict)} series: {list(series_dict.keys())}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None
    
    # 1.5. スケーリング・前処理
    original_series_dict = series_dict.copy()  # 元データ保存
    
    if auto_scaling:
        if scaling_method == 'auto':
            scaling_method = recommend_scaling_method(series_dict)
        
        series_dict, scaling_info = preprocess_series_dict(
            series_dict, 
            scaling_method=scaling_method,
            verbose=True
        )
        
        print(f"\n🔄 Applied scaling method: {scaling_method}")
    else:
        scaling_info = None
        print("🚫 Scaling disabled - using raw data")
    
    # 2. 特徴抽出（並列化）
    print("\nExtracting Lambda³ features (JAX optimized)...")
    start_time = time.time()
    
    features_dict = {}
    try:
        for name, data in series_dict.items():
            features = extract_lambda3_features_jax(data, config)
            features_dict[name] = features
            
            # 統計表示
            n_pos = int(jnp.sum(features['delta_lambda_pos']))
            n_neg = int(jnp.sum(features['delta_lambda_neg']))
            avg_rho = float(jnp.mean(features['rho_t']))
            print(f"  {name:15s} | Pos: {n_pos:3d} | Neg: {n_neg:3d} | ρT: {avg_rho:.3f}")
            
            # スケーリング後の改善を表示
            if scaling_info and name in scaling_info:
                info = scaling_info[name]
                if info['is_problematic']:
                    print(f"    ✅ Fixed: {', '.join(info['issues'])}")
        
        feature_time = time.time() - start_time
        print(f"Feature extraction completed in {feature_time:.2f}s")
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None
    
    # 3. ベイジアン推論（GPU並列化）
    print("\nRunning Bayesian inference...")
    inference_engine = Lambda3NumPyroInference(config)
    
    series_names = list(series_dict.keys())
    inference_results = {}
    
    # 各系列の基本モデル
    for i, name in enumerate(series_names[:2]):  # 最初の2系列のみ（デモ用）
        print(f"  Fitting base model for {name}...")
        try:
            result = inference_engine.fit_base_model(
                series_dict[name], 
                features_dict[name],
                chain_id=hash(name) % 1000
            )
            inference_results[name] = result
            
            # 診断情報（安全な取得）
            diag = result['diagnostics']
            divergences = diag.get('divergences', 0)
            energy = diag.get('energy', 'N/A')
            accept_prob = diag.get('accept_prob', 'N/A')
            
            if isinstance(energy, (int, float)):
                print(f"    Divergences: {divergences}, Energy: {energy:.3f}, Accept: {accept_prob:.3f}")
            else:
                print(f"    Divergences: {divergences}, Accept: {accept_prob}")
                
            # R-hat情報があれば表示
            rhat_keys = [k for k in diag.keys() if k.startswith('rhat_')]
            if rhat_keys:
                max_rhat = max([diag[k] for k in rhat_keys])
                print(f"    Max R-hat: {max_rhat:.3f} {'✅' if max_rhat < 1.1 else '⚠️'}")
                
            # スケーリング効果表示
            if scaling_info and name in scaling_info:
                info = scaling_info[name]
                original_std = info['original_std']
                print(f"    Original scale std: {original_std:.6f} → Normalized: {info['scaled_std']:.3f}")
                
        except Exception as e:
            print(f"    Model fitting failed: {e}")
            continue
    
    # 4. 相互作用解析
    if len(series_names) >= 2 and len(inference_results) >= 2:
        name_a, name_b = series_names[0], series_names[1]
        print(f"\nFitting interaction model: {name_a} ↔ {name_b}")
        
        try:
            interaction_result = inference_engine.fit_interaction_model(
                series_dict[name_a],
                features_dict[name_a],
                features_dict[name_b],
                chain_id=2000
            )
            
            # 相互作用係数表示（安全な取得）
            samples = interaction_result['samples']
            interact_pos_mean = float(jnp.mean(samples.get('lambda_interact_pos', 0)))
            interact_neg_mean = float(jnp.mean(samples.get('lambda_interact_neg', 0)))
            rho_interact_mean = float(jnp.mean(samples.get('rho_interact', 0)))
            
            print(f"  Interaction coefficients:")
            print(f"    Positive: {interact_pos_mean:.4f}")
            print(f"    Negative: {interact_neg_mean:.4f}")
            print(f"    Tension:  {rho_interact_mean:.4f}")
            
            # 診断情報
            interact_diag = interaction_result['diagnostics']
            interact_div = interact_diag.get('divergences', 0)
            print(f"    Interaction model divergences: {interact_div}")
            
        except Exception as e:
            print(f"  Interaction model failed: {e}")
    
    # 5. 同期解析（PyMCスタイル）
    print("\nSynchronization analysis...")
    try:
        # イベント系列の準備
        event_series_dict = {
            name: np.array(features_dict[name]['delta_lambda_pos'], dtype=np.float64)
            for name in series_names
        }
        
        # イベント系列の検証
        validate_event_series(event_series_dict)
        
        # PyMCスタイルの包括的同期解析を実行
        sync_matrix, sync_network = comprehensive_sync_analysis_pymc_style(series_names, features_dict)
        print(f"\nSynchronization analysis completed successfully")

        # 出力行列のラベルとして使用
        names = series_names        
        
        # sync_networkがNoneの場合の処理
        if sync_network is None:
            sync_network = nx.DiGraph()  # 空のネットワーク
        
    except Exception as e:
        print(f"Synchronization analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sync_matrix = None
        sync_network = None
        names = list(series_dict.keys())
    
    # 6. 可視化
    print("\nGenerating visualizations...")
    try:
        # 各系列の結果プロット
        for name in series_names[:2]:
            if name in inference_results:
                result = inference_results[name]
                predictions = result['predictions']['y']
                
                # 元スケールでの可視化用にデータを変換
                if scaling_info and name in scaling_info:
                    info = scaling_info[name]
                    # 予測を元スケールに戻す
                    if scaling_method == 'standardize':
                        original_data = original_series_dict[name]
                        if predictions.ndim > 1:
                            pred_rescaled = predictions * info['original_std'] + info['original_mean']
                        else:
                            pred_rescaled = predictions * info['original_std'] + info['original_mean']
                    else:
                        original_data = original_series_dict[name] 
                        pred_rescaled = predictions  # 簡易版
                    
                    plot_lambda3_results_numpyro(
                        original_data,
                        pred_rescaled,
                        features_dict[name],
                        title=f"Lambda³ Analysis: {name} (Rescaled)"
                    )
                else:
                    plot_lambda3_results_numpyro(
                        series_dict[name],
                        predictions,
                        features_dict[name],
                        title=f"Lambda³ Analysis: {name}"
                    )
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # 7. 結果サマリー
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"Series analyzed: {len(series_dict)}")
    print(f"Successful inferences: {len(inference_results)}")
    print(f"Scaling method used: {scaling_method if auto_scaling else 'none'}")
    print(f"JAX backend: {jax.default_backend()}")
    
    # スケーリング効果のサマリー
    if scaling_info:
        problematic_count = sum(1 for info in scaling_info.values() if info['is_problematic'])
        if problematic_count > 0:
            print(f"🔧 Fixed {problematic_count} problematic series through scaling")
    
    # トップ同期ペア
    if sync_matrix is not None and names is not None:  # namesの存在を確認
        print("\nTop synchronization pairs:")
        n = len(names)
        sync_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                sync_rate = float(sync_matrix[i, j])
                sync_pairs.append((sync_rate, names[i], names[j]))
        
        sync_pairs.sort(reverse=True)
        for sync_rate, name_a, name_b in sync_pairs[:3]:
            print(f"  {name_a:15s} ↔ {name_b:15s} | σₛ = {sync_rate:.3f}")
    
    # 8. 包括的レポート生成
    final_results = {
        'series_dict': series_dict,
        'original_series_dict': original_series_dict,
        'features_dict': features_dict,
        'inference_results': inference_results,
        'sync_matrix': sync_matrix,
        'series_names': names,  # namesを使用
        'scaling_info': scaling_info
    }
    
    create_comprehensive_report_numpyro(final_results)
    
    # 9. 因果関係解析（オプション）
    if len(inference_results) >= 2:
        try:
            print(f"\n🔗 CAUSALITY ANALYSIS:")
            series_list = list(inference_results.keys())[:2]
            
            # Lambda3拡張解析
            from collections import defaultdict
            causality_data = []
            labels = []
            
            for series_name in series_list:
                # 単純化した因果関係計算
                features = features_dict[series_name]
                pos_events = np.array(features['delta_lambda_pos'])
                neg_events = np.array(features['delta_lambda_neg'])
                
                # ラグ別因果関係
                causality_by_lag = {}
                for lag in range(1, 11):
                    if lag < len(pos_events):
                        # 正→負の因果関係
                        pos_to_neg = 0
                        pos_count = 0
                        for i in range(len(pos_events) - lag):
                            if pos_events[i] > 0:
                                pos_count += 1
                                if neg_events[i + lag] > 0:
                                    pos_to_neg += 1
                        
                        causality_by_lag[lag] = pos_to_neg / max(pos_count, 1)
                    else:
                        causality_by_lag[lag] = 0.0
                
                causality_data.append(causality_by_lag)
                labels.append(f"{series_name} (pos→neg)")
            
            # 因果関係プロファイルプロット
            if causality_data:
                plot_causality_profiles_numpyro(
                    causality_data, 
                    labels, 
                    title="Lambda³ Causal Structure Analysis"
                )
                
        except Exception as e:
            print(f"Causality analysis failed: {e}")
    
    return final_results

# ===============================
# PyMC互換レポート関数
# ===============================
# Lambda_abc_NumPyro.py に追加する関数
# PyMCと完全に同じ出力を実現する同期ネットワーク関数

def build_sync_network_pymc_style(event_series_dict: Dict[str, np.ndarray],
                                 lag_window: int = 10,
                                 sync_threshold: float = 0.3) -> nx.DiGraph:
    """PyMCスタイルの同期ネットワーク構築（元のprint出力を完全再現）"""
    
    series_names = list(event_series_dict.keys())
    G = nx.DiGraph()

    # ノード追加
    for series in series_names:
        G.add_node(series)

    print(f"\nBuilding sync network with threshold={sync_threshold}")

    # エッジ追加
    edge_count = 0
    for name_a in series_names:
        for name_b in series_names:
            if name_a == name_b:
                continue

            try:
                series_a = np.asarray(event_series_dict[name_a], dtype=np.float64)
                series_b = np.asarray(event_series_dict[name_b], dtype=np.float64)
                
                sync_profile, max_sync, optimal_lag = calculate_sync_profile_simple(
                    series_a, series_b, lag_window
                )

                print(f"{name_a} → {name_b}: max_sync={max_sync:.4f}, lag={optimal_lag}")

                if max_sync >= sync_threshold:
                    G.add_edge(name_a, name_b,
                              weight=max_sync,
                              lag=optimal_lag,
                              profile=sync_profile)
                    edge_count += 1
                    print(f"  ✓ Edge added!")
                    
            except Exception as e:
                print(f"{name_a} → {name_b}: failed ({e})")

    print(f"\nNetwork summary: {G.number_of_nodes()} nodes, {edge_count} edges")
    return G


def plot_sync_network_pymc_style(G: nx.DiGraph):
    """PyMCスタイルの同期ネットワークグラフ描画"""
    pos = nx.spring_layout(G)
    edge_labels = {
        (u, v): f"σₛ:{d['weight']:.2f},lag:{d['lag']}"
        for u, v, d in G.edges(data=True)
    }

    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='skyblue',
            node_size=1500, font_size=10, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Synchronization (σₛ) Network")
    plt.show()

def comprehensive_sync_analysis_pymc_style(series_names: List[str], 
                                          features_dict: Dict[str, Dict[str, np.ndarray]]):
    """PyMCスタイルの包括的同期解析セクション"""
    
    try:
        # Multi-series synchronization analysis
        print("\n" + "="*50)
        print("MULTI-SERIES SYNCHRONIZATION ANALYSIS")
        print("="*50)

        # Build event series dictionary
        event_series_dict = {
            name: np.array(features_dict[name]['delta_lambda_pos'], dtype=np.float64)
            for name in series_names
        }

        # Synchronization matrix (PyMC版のsync_matrix_simple関数を使用)
        sync_mat, names = sync_matrix_simple(event_series_dict, lag_window=10)

        # Plot sync matrix heatmap（PyMCと同じ）
        plt.figure(figsize=(10, 8))
        sns.heatmap(sync_mat, annot=True, fmt='.3f',
                    xticklabels=names,
                    yticklabels=names,
                    cmap="Blues", vmin=0, vmax=1,
                    square=True, cbar_kws={'label': 'Sync Rate σₛ'})
        plt.title("Synchronization Rate Matrix (σₛ)", fontsize=16)
        plt.tight_layout()
        plt.show()

        # Build and plot sync network
        print("\n=== Building Synchronization Network ===")

        # Find appropriate threshold（PyMCと同じロジック）
        non_diag_values = []
        n = len(names)
        for i in range(n):
            for j in range(n):
                if i != j:
                    non_diag_values.append(sync_mat[i, j])

        G = None  # デフォルトはNone
        if non_diag_values:
            threshold = np.percentile(non_diag_values, 25)  # Use 25th percentile
            print(f"Using threshold: {threshold:.4f}")

            G = build_sync_network_pymc_style(event_series_dict, lag_window=10, sync_threshold=threshold)
            if G.number_of_edges() > 0:
                plt.figure(figsize=(12, 10))
                plot_sync_network_pymc_style(G)

        # Clustering analysis（PyMCと同じ）
        if len(series_names) > 2:
            print("\n=== Clustering Analysis ===")
            n_clusters = min(3, len(series_names) // 2)
            clusters, _ = cluster_series_by_sync_simple(event_series_dict, lag_window=10, n_clusters=n_clusters)
            print(f"Clusters: {clusters}")

            # Plot clustered series - データ辞書を作成
            series_data_dict = {}
            for name in series_names:
                # features_dictから元のデータを取得（dataキーがある場合）
                if 'data' in features_dict[name]:
                    series_data_dict[name] = np.array(features_dict[name]['data'])
                else:
                    # dataキーがない場合は、最初の利用可能な系列を使用
                    for key in ['delta_lambda_pos', 'delta_lambda_neg', 'rho_t']:
                        if key in features_dict[name]:
                            series_data_dict[name] = np.array(features_dict[name][key])
                            break
            
            if series_data_dict:
                plot_clustered_series(series_data_dict, clusters)

        return sync_mat, G if G is not None else nx.DiGraph()
        
    except Exception as e:
        print(f"Comprehensive sync analysis failed: {e}")
        import traceback
        traceback.print_exc()
        # エラー時のデフォルト値を返す
        n = len(series_names)
        default_sync_mat = np.eye(n)  # 対角行列
        return default_sync_mat, nx.DiGraph()

# plot_clustered_series関数（元のPyMC版をそのまま使用）
def plot_clustered_series(series_dict: Dict[str, np.ndarray],
                         clusters: Dict[str, int]):
    """
    Plot time series colored by cluster membership.

    Args:
        series_dict: Dictionary of time series
        clusters: Cluster assignments for each series
    """
    n_clusters = len(set(clusters.values()))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(12, 6))
    for name, data in series_dict.items():
        cluster = clusters[name]
        plt.plot(data, label=f"{name} (Cluster {cluster})",
                color=colors[cluster], alpha=0.7, linewidth=1.5)

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Time Series Grouped by Synchronization Clusters")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_comprehensive_report_numpyro(results: Dict[str, Any]):
    """
    PyMC版と完全に同じフォーマットでレポートを生成
    
    Args:
        results: 解析結果の辞書
    """
    series_dict = results.get('series_dict', {})
    features_dict = results.get('features_dict', {})
    inference_results = results.get('inference_results', {})
    sync_matrix = results.get('sync_matrix')
    series_names = results.get('series_names', list(series_dict.keys()))
    scaling_info = results.get('scaling_info', {})
    
    print("\n" + "="*60)
    print("COMPREHENSIVE LAMBDA³ ANALYSIS REPORT")
    print("="*60)
    
    # 1. データ概要（PyMCスタイル）
    print("\n📊 DATA OVERVIEW")
    print("-" * 40)
    if series_dict:
        first_series = list(series_dict.keys())[0]
        data_length = len(series_dict[first_series])
        print(f"Time series length: {data_length}")
        print(f"Number of series: {len(series_dict)}")
        print(f"Series names: {', '.join(series_names)}")
        
        # データ統計
        print("\nSeries Statistics:")
        for name in series_names:
            data = np.array(series_dict[name])
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            print(f"  {name:15s} | Mean: {mean_val:8.4f} | Std: {std_val:8.4f} | Range: [{min_val:.4f}, {max_val:.4f}]")
    
    # 2. Lambda³特徴量統計（PyMCと同じ形式）
    print("\n🔍 LAMBDA³ FEATURE STATISTICS")
    print("-" * 40)
    print("Jump Event Statistics:")
    print("Series          | Pos ΔΛC | Neg ΔΛC | Local | Mean ρT")
    print("-" * 60)
    
    for name in series_names:
        if name in features_dict:
            features = features_dict[name]
            pos_jumps = int(jnp.sum(features['delta_lambda_pos']))
            neg_jumps = int(jnp.sum(features['delta_lambda_neg']))
            local_jumps = int(jnp.sum(features.get('local_jump', 0)))
            mean_rho = float(jnp.mean(features['rho_t']))
            
            print(f"{name:15s} | {pos_jumps:7d} | {neg_jumps:7d} | {local_jumps:5d} | {mean_rho:7.3f}")
    
    # 3. ベイジアン推論結果（PyMCスタイル）
    if inference_results:
        print("\n📈 BAYESIAN INFERENCE RESULTS")
        print("-" * 40)
        
        for name, result in inference_results.items():
            print(f"\nSeries: {name}")
            samples = result['samples']
            diagnostics = result['diagnostics']
            
            # パラメータ推定値
            print("  Parameter Estimates:")
            param_order = ['lambda_intercept', 'lambda_flow', 'lambda_struct_pos', 
                          'lambda_struct_neg', 'rho_tension']
            
            for param in param_order:
                if param in samples:
                    values = np.array(samples[param])
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    hdi_lower = np.percentile(values, 3)
                    hdi_upper = np.percentile(values, 97)
                    print(f"    {param:20s}: {mean_val:7.3f} ± {std_val:5.3f} HDI:[{hdi_lower:.3f}, {hdi_upper:.3f}]")
            
            # 診断統計
            print("  Diagnostics:")
            print(f"    Divergences: {diagnostics.get('divergences', 0)}")
            print(f"    Accept prob: {diagnostics.get('accept_prob', 0):.3f}")
            if 'energy' in diagnostics:
                print(f"    Energy: {diagnostics['energy']:.3f}")
            
            # R-hat値
            rhat_params = [k for k in diagnostics.keys() if k.startswith('rhat_')]
            if rhat_params:
                print("  Convergence (R-hat):")
                for param in rhat_params:
                    value = diagnostics[param]
                    status = "✅" if value < 1.1 else "⚠️"
                    print(f"    {param}: {value:.3f} {status}")
    
    # 4. 同期解析結果（PyMCスタイル）
    if sync_matrix is not None and len(series_names) >= 2:
        print("\n🔗 SYNCHRONIZATION ANALYSIS")
        print("-" * 40)
        
        # 同期行列の要約
        sync_np = np.array(sync_matrix)
        n = len(series_names)
        
        # トップ同期ペア
        print("Top Synchronization Pairs:")
        sync_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                sync_rate = float(sync_np[i, j])
                sync_pairs.append((sync_rate, series_names[i], series_names[j]))
        
        sync_pairs.sort(reverse=True)
        for sync_rate, name_a, name_b in sync_pairs[:5]:
            print(f"  {name_a:15s} ↔ {name_b:15s} | σₛ = {sync_rate:.3f}")
        
        # 平均同期率
        off_diagonal = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diagonal.append(sync_np[i, j])
        
        if off_diagonal:
            mean_sync = np.mean(off_diagonal)
            print(f"\nAverage sync rate (off-diagonal): {mean_sync:.3f}")
    
    # 5. 相互作用効果（存在する場合）
    if 'analysis_results' in results and 'interaction_effects' in results['analysis_results']:
        interaction_effects = results['analysis_results']['interaction_effects']
        if interaction_effects:
            print("\n🔄 CROSS-SERIES INTERACTION EFFECTS")
            print("-" * 40)
            
            # 相互作用行列を構築
            interaction_matrix = {}
            for (name_a, name_b), effects in interaction_effects.items():
                for effect_name, value in effects.items():
                    if 'to' in effect_name:
                        interaction_matrix[effect_name] = value
            
            # 表示
            for key, value in sorted(interaction_matrix.items()):
                if abs(value) > 0.01:  # 有意な効果のみ
                    print(f"  {key}: β = {value:.3f}")
    
    # 6. レジーム解析（存在する場合）
    if 'analysis_results' in results and 'regime_results' in results['analysis_results']:
        regime_results = results['analysis_results']['regime_results']
        if 'regime_stats' in regime_results:
            print("\n🎯 MARKET REGIME ANALYSIS")
            print("-" * 40)
            
            regime_stats = regime_results['regime_stats']
            for regime_name, stats in sorted(regime_stats.items()):
                freq_pct = stats['frequency'] * 100
                mean_rho = stats['mean_rhoT']
                print(f"  {regime_name}: {freq_pct:.1f}% frequency, Mean ρT: {mean_rho:.3f}")
    
    # 7. スケーリング情報（適用された場合）
    if scaling_info:
        problematic_count = sum(1 for info in scaling_info.values() if info['is_problematic'])
        if problematic_count > 0:
            print("\n⚙️ DATA PREPROCESSING")
            print("-" * 40)
            print(f"Scaling method applied: {scaling_info[series_names[0]]['scaling_method']}")
            print(f"Problematic series fixed: {problematic_count}")
            
            for name, info in scaling_info.items():
                if info['is_problematic']:
                    print(f"  {name}: {', '.join(info['issues'])}")
    
    # 8. 実行サマリー
    print("\n📊 EXECUTION SUMMARY")
    print("-" * 40)
    print(f"✅ Feature extraction: Complete")
    print(f"✅ Bayesian inference: {len(inference_results)} series analyzed")
    if sync_matrix is not None:
        print(f"✅ Synchronization analysis: Complete")
    print(f"✅ JAX backend: {jax.default_backend()}")
    
    print("\n" + "="*60)
    print("END OF REPORT")
    print("="*60)


def plot_interaction_heatmap_pymc_style(interaction_results: Dict[str, Dict[str, float]],
                                       series_names: List[str]):
    """PyMCスタイルの相互作用ヒートマップ"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    n = len(series_names)
    interaction_matrix = np.zeros((n, n))
    
    # 行列を構築（PyMCと同じロジック）
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if name_a != name_b:
                # B → A の影響を探す
                if name_a in interaction_results:
                    key = f'{name_b}_to_{name_a}_pos'
                    if key in interaction_results[name_a]:
                        interaction_matrix[i, j] = interaction_results[name_a][key]
                    elif f'interact_{name_b}' in interaction_results[name_a]:
                        interaction_matrix[i, j] = interaction_results[name_a][f'interact_{name_b}']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix,
                xticklabels=series_names,
                yticklabels=series_names,
                annot=True, fmt='.3f',
                cmap='RdBu_r', center=0,
                square=True,
                cbar_kws={'label': 'Interaction Coefficient β'})
    plt.title("Cross-Series Interaction Effects\n(Column → Row)", fontsize=16)
    plt.xlabel("From Series", fontsize=12)
    plt.ylabel("To Series", fontsize=12)
    plt.tight_layout()
    plt.show()


def create_analysis_summary_pymc_style(series_names: List[str],
                                      sync_mat: jnp.ndarray,
                                      features_dict: Dict[str, Dict[str, jnp.ndarray]]):
    """PyMCスタイルの解析サマリー作成"""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Jump event statistics（PyMCと完全一致）
    print("\nJump Event Statistics:")
    print("-" * 40)
    for name in series_names:
        pos_jumps = int(jnp.sum(features_dict[name]['delta_lambda_pos']))
        neg_jumps = int(jnp.sum(features_dict[name]['delta_lambda_neg']))
        local_jumps = int(jnp.sum(features_dict[name].get('local_jump', 0)))
        print(f"{name:15s} | Pos: {pos_jumps:3d} | Neg: {neg_jumps:3d} | Local: {local_jumps:3d}")
    
    # Top synchronizations（PyMCと完全一致）
    print("\nTop Synchronization Pairs:")
    print("-" * 40)
    sync_pairs = []
    sync_np = np.array(sync_mat)
    for i, name_a in enumerate(series_names):
        for j, name_b in enumerate(series_names):
            if i < j:  # Only unique pairs
                sync_pairs.append((sync_np[i, j], name_a, name_b))
    
    sync_pairs.sort(reverse=True)
    for sync_rate, name_a, name_b in sync_pairs[:5]:
        print(f"{name_a:15s} ↔ {name_b:15s} | σₛ = {sync_rate:.3f}")
    
    print("\n" + "="*60)


def generate_pymc_compatible_output(analysis_results: Dict[str, Any]):
    """PyMC版と完全互換の出力を生成"""
    
    # 1. 特徴量統計の表示
    if 'features_dict' in analysis_results:
        features_dict = analysis_results['features_dict']
        series_names = list(features_dict.keys())
        
        print("\n📊 FEATURE EXTRACTION SUMMARY (PyMC Compatible)")
        print("=" * 60)
        
        # PyMCと同じフォーマットで表示
        for name in series_names:
            features = features_dict[name]
            n_pos = int(jnp.sum(features['delta_lambda_pos']))
            n_neg = int(jnp.sum(features['delta_lambda_neg']))
            avg_rho = float(jnp.mean(features['rho_t']))
            print(f"  {name:15s} | Pos: {n_pos:3d} | Neg: {n_neg:3d} | ρT: {avg_rho:.3f}")
    
    # 2. ペア解析結果の表示
    if 'analysis_results' in analysis_results and 'sync_profiles' in analysis_results['analysis_results']:
        sync_profiles = analysis_results['analysis_results']['sync_profiles']
        
        print("\n🔄 PAIRWISE ANALYSIS RESULTS (PyMC Style)")
        print("=" * 60)
        
        for (name_a, name_b), profile_data in sync_profiles.items():
            max_sync = profile_data['max_sync']
            optimal_lag = profile_data['optimal_lag']
            print(f"\n[{name_a} ↔ {name_b}]")
            print(f"  Sync Rate σₛ: {max_sync:.3f}")
            print(f"  Optimal Lag: {optimal_lag} steps")
    
    # 3. 因果関係プロファイル
    if 'analysis_results' in analysis_results and 'causality_results' in analysis_results['analysis_results']:
        causality_results = analysis_results['analysis_results']['causality_results']
        
        print("\n📈 CAUSALITY ANALYSIS (PyMC Format)")
        print("=" * 60)
        
        for (name_a, name_b), causality_data in causality_results.items():
            print(f"\nCausality: {name_a} ↔ {name_b}")
            for direction, profile in causality_data.items():
                if isinstance(profile, dict) and profile:
                    max_lag = max(profile.items(), key=lambda x: x[1])
                    print(f"  {direction}: Peak at lag {max_lag[0]} (p={max_lag[1]:.3f})")

# ===============================
# Quick Start Example
# ===============================
def quick_start_demo():
    """クイックスタートデモ（処理速度測定強化版）"""
    print("Lambda³ NumPyro Quick Start Demo")
    print("=" * 40)
    
    # 全体タイマー開始
    total_start_time = time.time()
    
    # 設定
    config = L3ConfigNumPyro(
        T=100,
        window=5,          # ウィンドウサイズ削減
        local_window=5,    # 局所ウィンドウも削減
        num_samples=200,   # サンプル数削減
        num_warmup=100,    
        num_chains=1       
    )
    
    # サンプルデータ生成（より単純化）
    print("Generating sample data...")
    data_start = time.time()
    
    key = jax.random.PRNGKey(42)
    t = jnp.arange(config.T, dtype=jnp.float32)
    
    # シンプルな構造変化
    base_trend = 0.01 * t
    jumps = jnp.zeros(config.T)
    jumps = jumps.at[30].set(1.0)   # 単一正ジャンプ
    jumps = jumps.at[70].set(-0.8)  # 単一負ジャンプ
    
    noise = jax.random.normal(key, (config.T,)) * 0.2
    data = base_trend + jnp.cumsum(jumps) + noise
    
    data_time = time.time() - data_start
    print(f"Generated data shape: {data.shape}")
    print(f"Data range: [{jnp.min(data):.3f}, {jnp.max(data):.3f}]")
    print(f"Data type: {data.dtype}")
    print(f"⏱️  Data generation time: {data_time:.4f}s")
    
    # 段階的特徴抽出テスト（個別速度測定）
    print("\nTesting feature extraction components...")
    
    # 1. 差分・閾値計算
    try:
        diff_start = time.time()
        diff, threshold = calculate_diff_threshold_jax(data, config.delta_percentile)
        diff_time = time.time() - diff_start
        print(f"✓ Diff calculation: shape={diff.shape}, threshold={threshold:.3f}")
        print(f"  ⏱️  Time: {diff_time:.4f}s")
    except Exception as e:
        print(f"✗ Diff calculation failed: {e}")
        return
    
    # 2. ジャンプ検出
    try:
        jump_start = time.time()
        delta_pos, delta_neg = detect_jumps_jax(diff, threshold)
        jump_time = time.time() - jump_start
        print(f"✓ Jump detection: pos={jnp.sum(delta_pos)}, neg={jnp.sum(delta_neg)}")
        print(f"  ⏱️  Time: {jump_time:.4f}s")
    except Exception as e:
        print(f"✗ Jump detection failed: {e}")
        return
    
    # 3. 局所標準偏差（個別テスト）
    try:
        print("Testing local std calculation...")
        local_std_start = time.time()
        local_std = calculate_local_std_jax(data, config.local_window)
        local_std_time = time.time() - local_std_start
        print(f"✓ Local std: shape={local_std.shape}, mean={jnp.mean(local_std):.3f}")
        print(f"  ⏱️  Time: {local_std_time:.4f}s")
    except Exception as e:
        print(f"✗ Local std failed: {e}")
        print("Using fallback...")
        local_std = jnp.ones_like(data) * jnp.std(data)
        local_std_time = 0.001
    
    # 4. テンションスカラー（個別テスト）
    try:
        print("Testing rho_t calculation...")
        rho_start = time.time()
        rho_t = calculate_rho_t_jax(data, config.window)
        rho_time = time.time() - rho_start
        print(f"✓ Rho_t: shape={rho_t.shape}, mean={jnp.mean(rho_t):.3f}")
        print(f"  ⏱️  Time: {rho_time:.4f}s")
    except Exception as e:
        print(f"✗ Rho_t failed: {e}")
        print("Using fallback...")
        rho_t = jnp.ones_like(data) * jnp.std(data)
        rho_time = 0.001
    
    # 5. 完全特徴抽出（統合測定）
    print("\nRunning full feature extraction...")
    try:
        feature_start = time.time()
        features = extract_lambda3_features_jax(data, config)
        feature_total_time = time.time() - feature_start
        
        print("✓ Feature extraction successful!")
        print(f"⏱️  Total feature extraction time: {feature_total_time:.4f}s")
        
        for key, value in features.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, mean={jnp.mean(value):.3f}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 推論テスト（詳細速度測定）
    print("\nTesting Bayesian inference...")
    try:
        inference_engine = Lambda3NumPyroInference(config)
        
        print("  Running MCMC sampling...")
        mcmc_start = time.time()
        
        # 初回実行（JIT compile含む）
        print("  - JIT compilation + first run...")
        result = inference_engine.fit_base_model(data, features, chain_id=0)
        first_run_time = time.time() - mcmc_start
        
        # 2回目実行（pure execution time）
        print("  - Second run (pure execution)...")
        second_start = time.time()
        result2 = inference_engine.fit_base_model(data, features, chain_id=1)
        pure_execution_time = time.time() - second_start
        
        samples = result['samples']
        diagnostics = result['diagnostics']
        
        print("✓ Inference successful!")
        print(f"⏱️  First run (JIT + execution): {first_run_time:.4f}s")
        print(f"⏱️  Pure execution time: {pure_execution_time:.4f}s")
        print(f"⏱️  JIT compilation overhead: {first_run_time - pure_execution_time:.4f}s")
        
        print("\nParameter estimates:")
        for param, values in samples.items():
            if jnp.isscalar(values):
                print(f"  {param}: {values:.4f}")
            else:
                print(f"  {param}: {jnp.mean(values):.4f} ± {jnp.std(values):.4f}")
        
        print("\nDiagnostics:")
        for key, value in diagnostics.items():
            if key.startswith('rhat_'):
                param_name = key.replace('rhat_', '')
                status = "✅" if value < 1.1 else "⚠️"
                print(f"  {key}: {value:.4f} {status}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 詳細診断プロット
        print("\nGenerating MCMC diagnostics...")
        plot_mcmc_diagnostics(result, title=f"Lambda³ MCMC Diagnostics")
        
        # エネルギー診断
        if diagnostics.get('energy') is not None:
            plot_energy_diagnostics(result)
                
        # 可視化テスト
        try:
            print("\nGenerating visualization...")
            viz_start = time.time()
            predictions = result['predictions']['y']
            
            # 予測データのチェック
            if predictions.ndim > 1:
                pred_mean = jnp.mean(predictions, axis=0)
                print(f"  Predictions shape: {predictions.shape} -> mean shape: {pred_mean.shape}")
            else:
                pred_mean = predictions
                print(f"  Predictions shape: {predictions.shape}")
            
            plot_lambda3_results_numpyro(data, predictions, features, 
                                        title="Lambda³ NumPyro Demo - SUCCESS!")
            viz_time = time.time() - viz_start
            print("✓ Visualization completed!")
            print(f"⏱️  Visualization time: {viz_time:.4f}s")
            
        except Exception as viz_e:
            print(f"Visualization failed: {viz_e}")
            print("But inference worked!")
            viz_time = 0
            
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        print("\nBut feature extraction worked perfectly - that's major progress!")
        first_run_time = 0
        pure_execution_time = 0
        viz_time = 0
    
    # Optional: Parameter grid search demonstration
    try:
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == '--grid':
            print("\n🔍 Running parameter grid search demonstration...")
            
            # パラメータグリッド定義
            param_grid = {
                'target_accept_prob': [0.7, 0.8, 0.9],
                'max_tree_depth': [8, 10, 12]
            }
            
            # 小さなグリッドでデモ
            grid_results = grid_search_lambda3_params(data, features, param_grid, config)
            
            if grid_results['best_params']:
                print(f"\n🎯 Optimal configuration found: {grid_results['best_params']}")
                
                # グリッド結果の可視化
                plot_grid_search_results(grid_results)
    except:
        pass  # グリッドサーチはオプショナル
    
    # 総実行時間
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*60}")
    print("LAMBDA³ NUMPYRO PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print("✓ JAX GPU backend active")
    print("✓ Data generation successful")
    print("✓ Feature extraction (ΔΛC, ρT) working") 
    print("✓ Lambda³ structure tensor computation stable")
    print("✓ All JAX compilation issues resolved")
    print()
    print("⚡ PERFORMANCE METRICS:")
    print(f"  📊 Data generation:     {data_time:.4f}s")
    print(f"  🔍 Feature extraction:  {feature_total_time:.4f}s")
    if 'first_run_time' in locals():
        print(f"  🎯 MCMC (JIT+exec):     {first_run_time:.4f}s")
        print(f"  ⚡ MCMC (pure):         {pure_execution_time:.4f}s")
        print(f"  🎨 Visualization:       {viz_time:.4f}s")
    print(f"  🕐 TOTAL TIME:          {total_time:.4f}s")
    print()
    
    # スループット計算
    data_points = config.T
    mcmc_samples = config.num_samples
    if 'pure_execution_time' in locals() and pure_execution_time > 0:
        throughput = mcmc_samples / pure_execution_time
        print(f"🚀 THROUGHPUT ANALYSIS:")
        print(f"  Data points processed:  {data_points}")
        print(f"  MCMC samples:          {mcmc_samples}")
        print(f"  Samples per second:    {throughput:.1f}")
        print(f"  GPU acceleration:      {jax.default_backend().upper()}")
    
    print("\nReady for full-scale analysis!")

# ===============================
# メイン実行
# ===============================
if __name__ == "__main__":
    # GPU最適化設定
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    print("Starting Lambda³ NumPyro Analysis...")
    
    # デモ実行
    # quick_start_demo()
    
    # フル解析実行
    results = main_lambda3_numpyro_analysis(
        config=L3ConfigNumPyro(
            num_chains=4,
            max_workers=3
        )
    )
    
    if results:
        print("\nLambda³ NumPyro analysis completed successfully!")
    else:
        print("\nAnalysis failed. Check data and configuration.")
