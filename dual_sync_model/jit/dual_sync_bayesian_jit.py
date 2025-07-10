import pymc as pm
import numpy as np
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from numba import jit, njit, prange
from typing import Tuple, Dict, List, Optional

# ===============================
# Global Constants for JIT
# ===============================
DELTA_PERCENTILE = 97.0
LOCAL_JUMP_PERCENTILE = 97.0
WINDOW_SIZE = 10
LOCAL_WINDOW_SIZE = 10
LAG_WINDOW_DEFAULT = 10
SYNC_THRESHOLD_DEFAULT = 0.3
NOISE_STD_DEFAULT = 0.5
NOISE_STD_HIGH = 1.5

# ===============================
# Lambda³ Config Class
# ===============================
@dataclass
class L3Config:
    # Data generation
    pattern_a: str = "consecutive_jumps"
    pattern_b: str = "overlapping_events"
    T: int = 150
    seed: int = 42
    # Feature extraction (uses globals for JIT)
    window: int = WINDOW_SIZE
    local_window: int = LOCAL_WINDOW_SIZE
    delta_percentile: float = DELTA_PERCENTILE
    local_jump_percentile: float = LOCAL_JUMP_PERCENTILE
    # Sampling
    draws: int = 4000
    tune: int = 4000
    target_accept: float = 0.95
    # Posterior viz
    var_names: list = ('beta_time_a', 'beta_time_b', 'beta_interact', 'beta_rhoT_a', 'beta_rhoT_b')
    hdi_prob: float = 0.94

# ===============================
# JIT-compiled Core Functions
# ===============================
@njit
def calculate_diff_and_threshold(data: np.ndarray, percentile: float) -> Tuple[np.ndarray, float]:
    """JIT-compiled difference calculation and threshold computation."""
    diff = np.empty(len(data))
    diff[0] = 0
    for i in range(1, len(data)):
        diff[i] = data[i] - data[i-1]

    abs_diff = np.abs(diff)
    threshold = np.percentile(abs_diff, percentile)
    return diff, threshold

@njit
def detect_jumps(diff: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """JIT-compiled jump detection."""
    n = len(diff)
    pos_jumps = np.zeros(n, dtype=np.int32)
    neg_jumps = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if diff[i] > threshold:
            pos_jumps[i] = 1
        elif diff[i] < -threshold:
            neg_jumps[i] = 1

    return pos_jumps, neg_jumps

@njit
def calculate_local_std(data: np.ndarray, window: int) -> np.ndarray:
    """JIT-compiled local standard deviation calculation."""
    n = len(data)
    local_std = np.empty(n)

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)

        # Calculate std manually for JIT
        subset = data[start:end]
        mean = np.mean(subset)
        variance = np.sum((subset - mean) ** 2) / len(subset)
        local_std[i] = np.sqrt(variance)

    return local_std

@njit
def calculate_rho_t(data: np.ndarray, window: int) -> np.ndarray:
    """JIT-compiled tension scalar (ρT) calculation."""
    n = len(data)
    rho_t = np.empty(n)

    for i in range(n):
        start = max(0, i - window)
        end = i + 1

        subset = data[start:end]
        if len(subset) > 1:
            mean = np.mean(subset)
            variance = np.sum((subset - mean) ** 2) / len(subset)
            rho_t[i] = np.sqrt(variance)
        else:
            rho_t[i] = 0.0

    return rho_t

@njit
def sync_rate_at_lag(series_a: np.ndarray, series_b: np.ndarray, lag: int) -> float:
    """JIT-compiled synchronization rate calculation for a specific lag."""
    if lag < 0:
        if -lag < len(series_a):
            return np.mean(series_a[-lag:] * series_b[:lag])
        else:
            return 0.0
    elif lag > 0:
        if lag < len(series_b):
            return np.mean(series_a[:-lag] * series_b[lag:])
        else:
            return 0.0
    else:
        return np.mean(series_a * series_b)

@njit(parallel=True)
def calculate_sync_profile_jit(series_a: np.ndarray, series_b: np.ndarray,
                               lag_window: int) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """JIT-compiled synchronization profile calculation with parallelization."""
    n_lags = 2 * lag_window + 1
    lags = np.arange(-lag_window, lag_window + 1)
    sync_values = np.empty(n_lags)

    # Parallel computation of sync rates
    for i in prange(n_lags):
        lag = lags[i]
        sync_values[i] = sync_rate_at_lag(series_a, series_b, lag)

    # Find maximum
    max_sync = 0.0
    optimal_lag = 0
    for i in range(n_lags):
        if sync_values[i] > max_sync:
            max_sync = sync_values[i]
            optimal_lag = lags[i]

    return lags, sync_values, max_sync, optimal_lag

# ===============================
# Feature Extraction Wrapper
# ===============================
def calc_lambda3_features_v2(data: np.ndarray, config: L3Config) -> Tuple[np.ndarray, ...]:
    """
    Wrapper for Lambda³ feature extraction using JIT-compiled functions.
    """
    # Use JIT functions with global constants
    diff, threshold = calculate_diff_and_threshold(data, DELTA_PERCENTILE)
    delta_pos, delta_neg = detect_jumps(diff, threshold)

    # Local jump detection
    local_std = calculate_local_std(data, LOCAL_WINDOW_SIZE)
    score = np.abs(diff) / (local_std + 1e-8)
    local_threshold = np.percentile(score, LOCAL_JUMP_PERCENTILE)
    local_jump_detect = (score > local_threshold).astype(int)

    # Tension scalar
    rho_t = calculate_rho_t(data, WINDOW_SIZE)

    # Time trend
    time_trend = np.arange(len(data))

    return delta_pos, delta_neg, rho_t, time_trend, local_jump_detect

# ===============================
# Pattern Generation (Non-JIT)
# ===============================
def generate_data_pattern(config: L3Config, pattern: str, seed_offset: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data patterns (not JIT-compiled due to complexity)."""
    np.random.seed(config.seed + seed_offset)
    t = np.arange(config.T)
    trend = 0.05 * t + np.sin(t * 0.2)
    jumps = np.zeros(config.T)
    noise_std = NOISE_STD_DEFAULT

    if pattern == "no_jump":
        pass
    elif pattern == "single_jump":
        jumps[60] = 8.0
    elif pattern == "multi_jump":
        jumps[[30, 80, 110]] = [5.0, -7.0, 3.5]
    elif pattern == "mixed_sign":
        jumps[30] = 8.0
        jumps[70] = -5.0
        jumps[120] = 3.5
    elif pattern == "consecutive_jumps":
        jumps[40:44] = [3, -3, 4, -4]
    elif pattern == "step_trend":
        trend[75:] += 10
    elif pattern == "noisy":
        noise_std = NOISE_STD_HIGH
    elif pattern == "hidden_jump_noise":
        jumps[50] = 1.2
        jumps[95] = -1.5
        noise_std = 1.0
    elif pattern == "periodic_plus_jump":
        trend = 2.0 * np.sin(t * 0.15)
        jumps[40] = 4.0
        jumps[90] = -6.5
        jumps[120] = 2.5
    elif pattern == "causal_chain":
        jumps[20] = 5.0
        jumps[23] = -4.5
        jumps[60] = 3.0
        jumps[65] = -3.2
        jumps[100] = 7.0
        jumps[108] = -7.1
    elif pattern == "overlapping_events":
        jumps[50] = 8.0 + np.random.uniform(-0.5, 0.5)
        jumps[80] = -3.0 + 0.7

    noise = np.random.randn(config.T) * noise_std
    data = trend + jumps + noise
    return data, trend, jumps

# ===============================
# Bayesian Model (PyMC3)
# ===============================
def fit_l3_bayesian_regression(
    data: np.ndarray,
    features_dict: Dict[str, np.ndarray],
    config: L3Config,
    interaction_feature: Optional[np.ndarray] = None,
    interaction_label: str = 'interaction'
):
    """
    Lambda³ Bayesian regression model with optional interaction terms.
    """
    with pm.Model() as model:
        # Priors
        beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
        beta_time = pm.Normal('beta_time', mu=0, sigma=1)
        beta_dLC_pos = pm.Normal('beta_dLC_pos', mu=0, sigma=5)
        beta_dLC_neg = pm.Normal('beta_dLC_neg', mu=0, sigma=5)
        beta_rhoT = pm.Normal('beta_rhoT', mu=0, sigma=3)

        # Linear model
        mu = (
            beta_0
            + beta_time * features_dict['time_trend']
            + beta_dLC_pos * features_dict['delta_LambdaC_pos']
            + beta_dLC_neg * features_dict['delta_LambdaC_neg']
            + beta_rhoT * features_dict['rho_T']
        )

        # Add interaction term if provided
        if interaction_feature is not None:
            beta_interact = pm.Normal(f'beta_{interaction_label}', mu=0, sigma=3)
            mu += beta_interact * interaction_feature

        # Likelihood
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=data)

        # Sample
        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            target_accept=config.target_accept,
            return_inferencedata=True
        )

    return trace

# ===============================
# Lambda³ Extended Analysis
# ===============================
class Lambda3BayesianExtended:
    """Extended Lambda³ analysis with event memory and causality detection."""

    def __init__(self, config: L3Config, series_names: Optional[List[str]] = None):
        self.config = config
        self.series_names = series_names or ['A']
        self.event_memory = []
        self.structure_evolution = []

    def update_event_memory(self, events_dict: Dict[str, Dict[str, int]]):
        """Update event memory with new events."""
        if len(self.event_memory) == 0:
            self.series_names = list(events_dict.keys())
        self.event_memory.append(events_dict)

    def detect_causality_chain(self, series: str = 'A') -> Optional[float]:
        """Detect causality chains in a single series."""
        if len(self.event_memory) < 2:
            return None

        count_pairs = 0
        count_pos = 0

        for i in range(len(self.event_memory) - 1):
            if self.event_memory[i][series]['pos']:
                count_pos += 1
                if self.event_memory[i + 1][series]['neg']:
                    count_pairs += 1

        return count_pairs / max(count_pos, 1)

    def detect_time_dependent_causality(self, series: str = 'A', lag_window: int = LAG_WINDOW_DEFAULT) -> Dict[int, float]:
        """Calculate time-dependent causality probabilities."""
        causality_by_lag = {}

        for lag in range(1, lag_window + 1):
            count_pairs, count_pos = 0, 0

            for i in range(len(self.event_memory) - lag):
                if self.event_memory[i][series]['pos']:
                    count_pos += 1
                    if self.event_memory[i + lag][series]['neg']:
                        count_pairs += 1

            causality_by_lag[lag] = count_pairs / max(count_pos, 1)

        return causality_by_lag

    def detect_cross_causality(self, from_series: str, to_series: str, lag: int = 1) -> float:
        """Detect cross-causality between series."""
        count_pairs, count_from = 0, 0

        for i in range(len(self.event_memory) - lag):
            if self.event_memory[i][from_series]['pos']:
                count_from += 1
                if self.event_memory[i + lag][to_series]['pos']:
                    count_pairs += 1

        return count_pairs / max(count_from, 1)

# ===============================
# Synchronization Analysis
# ===============================
def calculate_sync_profile(series_a: np.ndarray, series_b: np.ndarray,
                          lag_window: int = LAG_WINDOW_DEFAULT) -> Tuple[Dict[int, float], float, int]:
    """
    Calculate synchronization profile using JIT-compiled function.
    """
    lags, sync_values, max_sync, optimal_lag = calculate_sync_profile_jit(
        series_a.astype(np.float64),
        series_b.astype(np.float64),
        lag_window
    )

    # Convert to dictionary
    sync_profile = {int(lag): float(sync) for lag, sync in zip(lags, sync_values)}

    return sync_profile, float(max_sync), int(optimal_lag)

def calculate_sync_rate(series_a_events, series_b_events, lag_window=10):
    """
    Calculates the synchronization rate σₛ between two event series.
    """
    max_sync, optimal_lag = 0, 0
    for lag in range(-lag_window, lag_window+1):
        if lag < 0:
            sync = np.mean(series_a_events[-lag:] * series_b_events[:lag])
        elif lag > 0:
            sync = np.mean(series_a_events[:-lag] * series_b_events[lag:])
        else:
            sync = np.mean(series_a_events * series_b_events)

        if sync > max_sync:
            max_sync, optimal_lag = sync, lag

    return max_sync, optimal_lag

def calculate_dynamic_sync(series_a_events, series_b_events, window=20, lag_window=10):
    """
    Calculates a dynamic synchronization rate over time.
    """
    T = len(series_a_events)
    sync_rates, optimal_lags = [], []

    for t in range(T - window + 1):
        sync, lag = calculate_sync_rate(
            series_a_events[t:t+window],
            series_b_events[t:t+window],
            lag_window
        )
        sync_rates.append(sync)
        optimal_lags.append(lag)

    time_points = np.arange(window//2, T - window//2 + 1)
    return time_points, sync_rates, optimal_lags

def sync_matrix(event_series_dict: Dict[str, np.ndarray], lag_window: int = LAG_WINDOW_DEFAULT) -> Tuple[np.ndarray, List[str]]:
    """
    Create a sync rate matrix (all pairs, max σₛ over all lags).
    """
    series_names = list(event_series_dict.keys())
    n = len(series_names)
    mat = np.zeros((n, n))

    for i, a in enumerate(series_names):
        for j, b in enumerate(series_names):
            if i == j:
                mat[i, j] = 1.0  # Self-sync is perfect
                continue

            # Ensure float64 for JIT function
            series_a = event_series_dict[a].astype(np.float64)
            series_b = event_series_dict[b].astype(np.float64)

            _, _, max_sync, _ = calculate_sync_profile_jit(series_a, series_b, lag_window)
            mat[i, j] = max_sync

    return mat, series_names

def cluster_series_by_sync(event_series_dict, lag_window=10, n_clusters=2):
    """
    Cluster time series based on their maximum pairwise sync rates.
    """
    mat, names = sync_matrix(event_series_dict, lag_window)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(1 - mat)
    clusters = {name: label for name, label in zip(names, labels)}
    return clusters, mat

def build_sync_network(event_series_dict: Dict[str, np.ndarray],
                      lag_window: int = LAG_WINDOW_DEFAULT,
                      sync_threshold: float = SYNC_THRESHOLD_DEFAULT) -> nx.DiGraph:
    """Build synchronization network from event series."""
    series_names = list(event_series_dict.keys())
    G = nx.DiGraph()

    # Add nodes
    for series in series_names:
        G.add_node(series)

    # Debug: print sync calculations
    print(f"\nBuilding sync network with threshold={sync_threshold}")

    # Add edges based on synchronization
    for series_a in series_names:
        for series_b in series_names:
            if series_a == series_b:
                continue

            sync_profile, max_sync, optimal_lag = calculate_sync_profile(
                event_series_dict[series_a].astype(np.float64),
                event_series_dict[series_b].astype(np.float64),
                lag_window
            )

            print(f"{series_a} → {series_b}: max_sync={max_sync:.3f}, lag={optimal_lag}")

            if max_sync >= sync_threshold:
                G.add_edge(series_a, series_b,
                          weight=max_sync,
                          lag=optimal_lag,
                          profile=sync_profile)
                print(f"  Edge added!")

    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

# ===============================
# Visualization Functions
# ===============================
def plot_posterior(trace, var_names: Optional[List[str]] = None, hdi_prob: float = 0.94):
    """Visualize posterior distributions."""
    if var_names is None:
        var_names = list(trace.posterior.data_vars)
    az.plot_posterior(trace, var_names=var_names, hdi_prob=hdi_prob)
    plt.tight_layout()
    plt.show()

def plot_l3_prediction_dual(
    data_dict: Dict[str, np.ndarray],
    mu_pred_dict: Dict[str, np.ndarray],
    jump_pos_dict: Dict[str, np.ndarray],
    jump_neg_dict: Dict[str, np.ndarray],
    local_jump_dict: Optional[Dict[str, np.ndarray]] = None,
    series_names: Optional[List[str]] = None,
    titles: Optional[List[str]] = None
):
    """Plot observed data, model predictions, and jump events for multiple series."""
    if series_names is None:
        series_names = list(data_dict.keys())

    n_series = len(series_names)
    fig, axes = plt.subplots(n_series, 1, figsize=(15, 5 * n_series), sharex=True)

    if n_series == 1:
        axes = [axes]

    for i, series in enumerate(series_names):
        ax = axes[i]
        data = data_dict[series]
        mu_pred = mu_pred_dict[series]
        jump_pos = jump_pos_dict[series]
        jump_neg = jump_neg_dict[series]
        local_jump = local_jump_dict[series] if local_jump_dict else None

        # Plot data and prediction
        ax.plot(data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
        ax.plot(mu_pred, color='C2', lw=2, label='Model Prediction')

        # Plot jump events
        jump_pos_idx = np.where(jump_pos > 0)[0]
        if len(jump_pos_idx):
            ax.plot(jump_pos_idx, data[jump_pos_idx], 'o', color='dodgerblue',
                   markersize=10, label='Positive Jump')
            for idx in jump_pos_idx:
                ax.axvline(x=idx, color='dodgerblue', linestyle='--', alpha=0.5)

        jump_neg_idx = np.where(jump_neg > 0)[0]
        if len(jump_neg_idx):
            ax.plot(jump_neg_idx, data[jump_neg_idx], 'o', color='orange',
                   markersize=10, label='Negative Jump')
            for idx in jump_neg_idx:
                ax.axvline(x=idx, color='orange', linestyle='-.', alpha=0.5)

        if local_jump is not None:
            local_jump_idx = np.where(local_jump > 0)[0]
            if len(local_jump_idx):
                ax.plot(local_jump_idx, data[local_jump_idx], 'o', color='magenta',
                       markersize=7, alpha=0.7, label='Local Jump')

        # Formatting
        plot_title = titles[i] if titles and i < len(titles) else f"Series {series}: Fit + Events"
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
        ax.grid(axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_sync_profile(sync_profile: Dict[int, float], title: str = "Sync Profile (σₛ vs Lag)"):
    """Plot synchronization profile."""
    lags, syncs = zip(*sorted(sync_profile.items()))
    plt.figure(figsize=(8, 4))
    plt.plot(lags, syncs, marker='o')
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Sync Rate σₛ')
    plt.grid(alpha=0.5)
    plt.show()

def plot_dynamic_sync(time_points, sync_rates, optimal_lags):
    """
    Plots dynamic synchronization rate and optimal lag over time.
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(time_points, sync_rates, label='σₛ Sync Rate', color='royalblue')
    ax1.set_ylabel('σₛ Sync Rate')
    ax1.set_xlabel('Time Step')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(time_points, optimal_lags, label='Optimal Lag', color='darkorange', linestyle='--')
    ax2.set_ylabel('Optimal Lag')
    ax2.legend(loc='upper right')

    plt.title("Dynamic Synchronization (σₛ) and Optimal Lag")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_multi_causality_lags(
    causality_dicts,
    labels=None,
    colors=None,
    title="Lagged Causality Profiles",
    xlabel="Lag (steps)",
    ylabel="Causality Probability",
    figsize=(9,5),
    alpha=0.7
):
    """
    Plot multiple lagged causality profiles together.
    """
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(causality_dicts))]
    if colors is None:
        base_colors = ['royalblue', 'darkorange', 'forestgreen', 'crimson']
        colors = base_colors[:len(causality_dicts)]

    plt.figure(figsize=figsize)

    for i, (causality_by_lag, label, color) in enumerate(zip(causality_dicts, labels, colors)):
        lags, probs = zip(*sorted(causality_by_lag.items()))
        plt.plot(lags, probs, marker='o', label=label, color=color, alpha=alpha, lw=2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_sync_network(G: nx.DiGraph):
    """Plot synchronization network."""
    pos = nx.spring_layout(G)
    edge_labels = {
        (u, v): f"σₛ:{d['weight']:.2f},lag:{d['lag']}"
        for u, v, d in G.edges(data=True)
    }

    nx.draw(G, pos, with_labels=True, node_color='skyblue',
            node_size=1500, font_size=10, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Synchronization (σₛ) Network")
    plt.show()

# ===============================
# Main Execution Pipeline
# ===============================
def main_dual_series():
    """Main execution pipeline for dual series analysis."""
    config = L3Config()

    # 1. Generate data
    data_a, _, _ = generate_data_pattern(config, config.pattern_a, seed_offset=0)
    data_b, _, _ = generate_data_pattern(config, config.pattern_b, seed_offset=100)

    # 2. Extract features
    feats_a = calc_lambda3_features_v2(data_a, config)
    feats_b = calc_lambda3_features_v2(data_b, config)

    # 3. Prepare feature dictionaries
    features_dict_a = {
        'delta_LambdaC_pos': feats_a[0],
        'delta_LambdaC_neg': feats_a[1],
        'rho_T': feats_a[2],
        'time_trend': feats_a[3]
    }

    features_dict_b = {
        'delta_LambdaC_pos': feats_b[0],
        'delta_LambdaC_neg': feats_b[1],
        'rho_T': feats_b[2],
        'time_trend': feats_b[3]
    }

    # 4. Fit models with cross-series interactions
    trace_a = fit_l3_bayesian_regression(
        data=data_a,
        features_dict=features_dict_a,
        config=config,
        interaction_feature=features_dict_b['delta_LambdaC_pos'],
        interaction_label='B_pos_jump'
    )

    trace_b = fit_l3_bayesian_regression(
        data=data_b,
        features_dict=features_dict_b,
        config=config,
        interaction_feature=features_dict_a['delta_LambdaC_pos'],
        interaction_label='A_pos_jump'
    )

    # 5. Calculate predictions
    summary_a = az.summary(trace_a)
    mu_pred_a = (
        summary_a.loc['beta_0', 'mean']
        + summary_a.loc['beta_time', 'mean'] * features_dict_a['time_trend']
        + summary_a.loc['beta_dLC_pos', 'mean'] * features_dict_a['delta_LambdaC_pos']
        + summary_a.loc['beta_dLC_neg', 'mean'] * features_dict_a['delta_LambdaC_neg']
        + summary_a.loc['beta_rhoT', 'mean'] * features_dict_a['rho_T']
        + summary_a.loc['beta_B_pos_jump', 'mean'] * features_dict_b['delta_LambdaC_pos']
    )

    summary_b = az.summary(trace_b)
    mu_pred_b = (
        summary_b.loc['beta_0', 'mean']
        + summary_b.loc['beta_time', 'mean'] * features_dict_b['time_trend']
        + summary_b.loc['beta_dLC_pos', 'mean'] * features_dict_b['delta_LambdaC_pos']
        + summary_b.loc['beta_dLC_neg', 'mean'] * features_dict_b['delta_LambdaC_neg']
        + summary_b.loc['beta_rhoT', 'mean'] * features_dict_b['rho_T']
        + summary_b.loc['beta_A_pos_jump', 'mean'] * features_dict_a['delta_LambdaC_pos']
    )

    # 6. Visualization
    plot_l3_prediction_dual(
        data_dict={'A': data_a, 'B': data_b},
        mu_pred_dict={'A': mu_pred_a, 'B': mu_pred_b},
        jump_pos_dict={'A': features_dict_a['delta_LambdaC_pos'],
                       'B': features_dict_b['delta_LambdaC_pos']},
        jump_neg_dict={'A': features_dict_a['delta_LambdaC_neg'],
                       'B': features_dict_b['delta_LambdaC_neg']},
        local_jump_dict={'A': feats_a[4], 'B': feats_b[4]},
        titles=['Series A: Fit + Events', 'Series B: Fit + Events']
    )

    # 6.5 Plot posteriors
    print("\nPosterior for Series A (with B interaction):")
    plot_posterior(
        trace_a,
        var_names=['beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT', 'beta_B_pos_jump'],
        hdi_prob=config.hdi_prob
    )

    print("\nPosterior for Series B (with A interaction):")
    plot_posterior(
        trace_b,
        var_names=['beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT', 'beta_A_pos_jump'],
        hdi_prob=config.hdi_prob
    )

    # 7. Causality analysis
    lambda3_ext = Lambda3BayesianExtended(config, series_names=['A', 'B'])

    for i in range(config.T):
        lambda3_ext.update_event_memory({
            'A': {'pos': int(features_dict_a['delta_LambdaC_pos'][i]),
                  'neg': int(features_dict_a['delta_LambdaC_neg'][i])},
            'B': {'pos': int(features_dict_b['delta_LambdaC_pos'][i]),
                  'neg': int(features_dict_b['delta_LambdaC_neg'][i])}
        })

    # Compute all single/cross-series lagged causality profiles
    causality_by_lag_a = lambda3_ext.detect_time_dependent_causality(series='A', lag_window=10)
    causality_by_lag_b = lambda3_ext.detect_time_dependent_causality(series='B', lag_window=10)
    causality_ab = {lag: lambda3_ext.detect_cross_causality('A', 'B', lag=lag) for lag in range(1, 11)}
    causality_ba = {lag: lambda3_ext.detect_cross_causality('B', 'A', lag=lag) for lag in range(1, 11)}

    # Plot ALL causality profiles at once
    plot_multi_causality_lags(
        [causality_by_lag_a, causality_by_lag_b, causality_ab, causality_ba],
        labels=['A', 'B', 'A→B', 'B→A'],
        title='Lagged Causality Profiles'
    )

    # Print cross-causality values
    print("\nCross Causality Lags (B→A, A→B):")
    for lag in range(1, 11):
        print(f"Lag {lag}: B→A = {causality_ba[lag]:.2f} | A→B = {causality_ab[lag]:.2f}")

    # 8. Synchronization analysis
    sync_profile, sync_rate, optimal_lag = calculate_sync_profile(
        features_dict_a['delta_LambdaC_pos'].astype(np.float64),
        features_dict_b['delta_LambdaC_pos'].astype(np.float64),
        lag_window=10
    )

    print(f"\nSync Rate σₛ (A↔B): {sync_rate:.2f}")
    print(f"Optimal Lag: {optimal_lag} steps")

    plot_sync_profile(sync_profile, title="Sync Profile (σₛ vs Lag, A↔B)")

    # Dynamic synchronization analysis
    time_points, sync_rates, optimal_lags = calculate_dynamic_sync(
        features_dict_a['delta_LambdaC_pos'],
        features_dict_b['delta_LambdaC_pos'],
        window=20, lag_window=10
    )
    plot_dynamic_sync(time_points, sync_rates, optimal_lags)

    # 9. Build sync network
    event_series_dict = {
        'A': features_dict_a['delta_LambdaC_pos'].astype(np.float64),
        'B': features_dict_b['delta_LambdaC_pos'].astype(np.float64)
    }

    # Try different thresholds if needed
    G = build_sync_network(event_series_dict, lag_window=10, sync_threshold=0.1)
    if G.number_of_edges() > 0:
        plot_sync_network(G)
    else:
        print("No edges in sync network. Trying lower threshold...")
        G = build_sync_network(event_series_dict, lag_window=10, sync_threshold=0.05)
        if G.number_of_edges() > 0:
            plot_sync_network(G)
        else:
            print("Still no edges. Sync rates might be too low.")

    # 10. Show sync matrix heatmap
    sync_mat, series_names = sync_matrix(event_series_dict, lag_window=10)
    plt.figure(figsize=(5, 4))
    sns.heatmap(sync_mat, annot=True, fmt='.3f',
                xticklabels=series_names,
                yticklabels=series_names,
                cmap="Blues", vmin=0, vmax=1)
    plt.title("Sync Rate Matrix (σₛ, max over all lags)")
    plt.show()

if __name__ == '__main__':
    main_dual_series()
