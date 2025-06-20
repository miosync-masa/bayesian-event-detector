import pymc as pm
import numpy as np
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering
import networkx as nx


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
    # Feature extraction
    window: int = 10
    local_window: int = 10
    delta_percentile: float = 97.0
    local_jump_percentile: float = 97.0
    # Sampling
    draws: int = 4000
    tune: int = 4000
    target_accept: float = 0.95
    # Posterior viz
    var_names: list = ('beta_time_a', 'beta_time_b', 'beta_interact', 'beta_rhoT_a', 'beta_rhoT_b')
    hdi_prob: float = 0.94

# ===============================
# 1. Synthetic Data Generator
# ===============================
def generate_data_pattern(config: L3Config, pattern: str, seed_offset=0):
    np.random.seed(config.seed + seed_offset) 
    t = np.arange(config.T)
    trend = 0.05 * t + np.sin(t * 0.2)
    jumps = np.zeros(config.T)
    noise_std = 0.5

    if pattern == "no_jump":
        pass
    elif pattern == "single_jump":
        jumps[60] = 8.0
    elif pattern == "multi_jump":
        jumps[[30, 80, 110]] = [5.0, -7.0, 3.5]
    elif pattern == "mixed_sign":
        jumps[30] = 8.0; jumps[70] = -5.0; jumps[120] = 3.5
    elif pattern == "consecutive_jumps":
        jumps[40:44] = [3, -3, 4, -4]
    elif pattern == "step_trend":
        trend[75:] += 10
    elif pattern == "noisy":
        noise_std = 1.5
    elif pattern == "hidden_jump_noise":
        jumps[50] = 1.2; jumps[95] = -1.5
        noise_std = 1.0
    elif pattern == "periodic_plus_jump":
        trend = 2.0 * np.sin(t * 0.15)
        jumps[40] = 4.0; jumps[90] = -6.5; jumps[120] = 2.5
    elif pattern == "causal_chain":
        jumps[20] = 5.0; jumps[23] = -4.5; jumps[60] = 3.0; jumps[65] = -3.2; jumps[100] = 7.0; jumps[108] = -7.1
    elif pattern == "overlapping_events":
        jumps[50] = 8.0; jumps[50] += np.random.uniform(-0.5, 0.5)
        jumps[80] = -3.0; jumps[80] += 0.7

    noise = np.random.randn(config.T) * noise_std
    data = trend + jumps + noise
    return data, trend, jumps

# ===============================
# 2. Lambda³ Feature Extraction
# ===============================
def calc_lambda3_features_v2(data, config: L3Config):
    diff = np.diff(data, prepend=data[0])
    threshold = np.percentile(np.abs(diff), config.delta_percentile)
    delta_LambdaC_pos = (diff > threshold).astype(int)
    delta_LambdaC_neg = (diff < -threshold).astype(int)
    local_std = np.array([
        data[max(0, i-config.local_window):min(len(data), i+config.local_window+1)].std()
        for i in range(len(data))
    ])
    score = np.abs(diff) / (local_std + 1e-8)
    local_threshold = np.percentile(score, config.local_jump_percentile)
    local_jump_detect = (score > local_threshold).astype(int)
    rho_T = np.array([data[max(0, i-config.window):i+1].std() for i in range(len(data))])
    time_trend = np.arange(len(data))
    return delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend, local_jump_detect

def fit_dual_bayesian_regression(
    data_a, data_b,
    delta_LambdaC_pos_a, delta_LambdaC_neg_a, rho_T_a, time_trend_a,
    delta_LambdaC_pos_b, delta_LambdaC_neg_b, rho_T_b, time_trend_b,
    config: L3Config
):
    with pm.Model() as model:
        beta_0_a = pm.Normal('beta_0_a', mu=0, sigma=2)
        beta_time_a = pm.Normal('beta_time_a', mu=0, sigma=1)
        beta_rhoT_a = pm.Normal('beta_rhoT_a', mu=0, sigma=3)
        beta_0_b = pm.Normal('beta_0_b', mu=0, sigma=2)
        beta_time_b = pm.Normal('beta_time_b', mu=0, sigma=1)
        beta_rhoT_b = pm.Normal('beta_rhoT_b', mu=0, sigma=3)
        # --- interaction ---
        beta_interact = pm.Normal('beta_interact', mu=0, sigma=2)
        # Interaction term: e.g., B influences A's jumps
        interact = beta_interact * delta_LambdaC_pos_b
        mu_a = (
            beta_0_a
            + beta_time_a * time_trend_a
            + beta_rhoT_a * rho_T_a
            + interact
        )
        mu_b = (
            beta_0_b
            + beta_time_b * time_trend_b
            + beta_rhoT_b * rho_T_b
        )
        sigma_a = pm.HalfNormal('sigma_a', sigma=1)
        sigma_b = pm.HalfNormal('sigma_b', sigma=1)
        y_a = pm.Normal('y_a', mu=mu_a, sigma=sigma_a, observed=data_a)
        y_b = pm.Normal('y_b', mu=mu_b, sigma=sigma_b, observed=data_b)
        trace = pm.sample(draws=config.draws, tune=config.tune, target_accept=config.target_accept)
    return trace

# ===============================
# 2. Feature Extraction
# ===============================

class Lambda3BayesianExtended:
    def __init__(self, config: L3Config, series_names=None):
        self.config = config
        # List of series names: ['A'] (single series) or ['A','B','C'] (multi-series)
        self.series_names = series_names or ['A']
        # Event memory: [{'A':{'pos':..,'neg':..},'B':...}, ...]
        self.event_memory = []
        self.structure_evolution = []

    def update_event_memory(self, events_dict):
        """
        Update event memory.
        events_dict: {'A': {'pos':..., 'neg':...}, 'B': {...}, ...}
        For single series: {'A': {'pos':..., 'neg':...}}
        """
        # Automatically detect series names on first call
        if len(self.event_memory) == 0:
            self.series_names = list(events_dict.keys())
        self.event_memory.append(events_dict)

    def update_structure(self, Lambda_tensor):
        """
        Append structural tensor (Lambda) history (optional).
        """
        self.structure_evolution.append(Lambda_tensor)

    def detect_causality_chain(self, series='A'):
        """
        Single series: Probability of "positive jump → negative jump" within the specified series.
        """
        if len(self.event_memory) < 2:
            return None
        count_pairs = 0
        count_pos = 0
        for i in range(len(self.event_memory)-1):
            if self.event_memory[i][series]['pos']:
                count_pos += 1
                if self.event_memory[i+1][series]['neg']:
                    count_pairs += 1
        return count_pairs / max(count_pos, 1)

    def predict_next_event(self, series='A'):
        """
        Predict the next event (for single series).
        Returns: 'negative_jump_expected', 'positive_jump_expected', or 'stable'
        """
        recent_events = np.array([
            em[series]['pos'] - em[series]['neg'] for em in self.event_memory[-self.config.window:]
        ])
        trend = np.mean(recent_events)
        if trend > 0.5:
            return 'negative_jump_expected'
        elif trend < -0.5:
            return 'positive_jump_expected'
        else:
            return 'stable'

    def detect_time_dependent_causality(self, series='A', lag_window=10):
        """
        Return time-dependent causality probabilities (lag-wise) for the specified series.
        Returns: dict {lag: probability}
        """
        causality_by_lag = {}
        for lag in range(1, lag_window+1):
            count_pairs, count_pos = 0, 0
            for i in range(len(self.event_memory)-lag):
                if self.event_memory[i][series]['pos']:
                    count_pos += 1
                    if self.event_memory[i+lag][series]['neg']:
                        count_pairs += 1
            causality_by_lag[lag] = count_pairs / max(count_pos, 1)
        return causality_by_lag

    def detect_cross_causality(self, from_series, to_series, lag=1):
        """
        Multi-series: Probability that a positive jump in from_series is followed (with lag) by a positive jump in to_series.
        """
        count_pairs, count_from = 0, 0
        for i in range(len(self.event_memory)-lag):
            if self.event_memory[i][from_series]['pos']:
                count_from += 1
                if self.event_memory[i+lag][to_series]['pos']:
                    count_pairs += 1
        return count_pairs / max(count_from, 1)

    def detect_cross_causality_matrix(self, lag=1):
        """
        Return the matrix (dictionary) of cross-causality between all series pairs.
        Each entry is {(from_series, to_series): probability}
        """
        mat = {}
        for from_s in self.series_names:
            for to_s in self.series_names:
                if from_s != to_s:
                    mat[(from_s, to_s)] = self.detect_cross_causality(from_s, to_s, lag=lag)
        return mat

# ===============================
# 3. Lambda³ Bayesian Dual Model
# ===============================

def fit_l3_bayesian_regression(
    data,
    features_dict,
    config: L3Config,
    interaction_feature=None,
    interaction_label='interaction'
):
    """
    Flexible Lambda³ Bayesian regression model supporting optional interaction terms.
    """
    with pm.Model() as model:
        beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
        beta_time = pm.Normal('beta_time', mu=0, sigma=1)
        beta_dLC_pos = pm.Normal('beta_dLC_pos', mu=0, sigma=5)
        beta_dLC_neg = pm.Normal('beta_dLC_neg', mu=0, sigma=5)
        beta_rhoT = pm.Normal('beta_rhoT', mu=0, sigma=3)

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

        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=data)
        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            target_accept=config.target_accept,
            return_inferencedata=True
        )

    return trace

# ===============================
# 4. Syncro Rate
# ===============================
def calculate_sync_profile(series_a_events, series_b_events, lag_window=10):
    """
    Compute the synchronization profile σₛ across multiple lags between two binary event series.
    Args:
        series_a_events (np.ndarray): Events for A (binary)
        series_b_events (np.ndarray): Events for B (binary)
        lag_window (int): +/- lag window size
    Returns:
        sync_profile (dict): {lag: σₛ}
        max_sync (float): max σₛ found
        optimal_lag (int): lag with max σₛ
    """
    sync_profile = {}
    max_sync, optimal_lag = 0, 0
    for lag in range(-lag_window, lag_window + 1):
        if lag < 0:
            sync = np.mean(series_a_events[-lag:] * series_b_events[:lag])
        elif lag > 0:
            sync = np.mean(series_a_events[:-lag] * series_b_events[lag:])
        else:
            sync = np.mean(series_a_events * series_b_events)
        sync_profile[lag] = sync
        if sync > max_sync:
            max_sync, optimal_lag = sync, lag
    return sync_profile, max_sync, optimal_lag


def calculate_sync_rate(series_a_events, series_b_events, lag_window=10):
    """
    Calculates the synchronization rate σₛ between two event series.
    Args:
        series_a_events: binary event series for A (e.g., jump occurrences)
        series_b_events: binary event series for B
        lag_window: number of time steps to check synchronization lag
    Returns:
        sync_rate: σₛ synchronization score (0 to 1)
        optimal_lag: lag with maximum synchronization
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

def build_sync_network(event_series_dict, lag_window=10, sync_threshold=0.3):
    """
    Constructs a bidirectional synchronization network from multiple event series.
    """
    import networkx as nx
    series_names = list(event_series_dict.keys())
    G = nx.DiGraph()

    for series in series_names:
        G.add_node(series)

    for series_a in series_names:
        for series_b in series_names:
            if series_a == series_b:
                continue
            sync_profile, max_sync, optimal_lag = calculate_sync_profile(
                event_series_dict[series_a], event_series_dict[series_b], lag_window
            )
            if max_sync >= sync_threshold:
                G.add_edge(series_a, series_b, weight=max_sync, lag=optimal_lag, profile=sync_profile)
    return G

def plot_sync_network(G):
    """
    Plots synchronization network with edge labels showing lag and σₛ.
    """
    pos = nx.spring_layout(G)
    edge_labels = {(u, v): f"σₛ:{d['weight']:.2f},lag:{d['lag']}" for u, v, d in G.edges(data=True)}
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Synchronization (σₛ) Network")
    plt.show()

def calculate_dynamic_sync(series_a_events, series_b_events, window=20, lag_window=10):
    """
    Calculates a dynamic synchronization rate over time.
    Returns:
        time_points: list of time points
        sync_rates: synchronization rates at each time window
        optimal_lags: optimal lags at each time window
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

def plot_sync_profile(sync_profile, title="Sync Profile (σₛ vs Lag)"):
    lags, syncs = zip(*sorted(sync_profile.items()))
    plt.figure(figsize=(8,4))
    plt.plot(lags, syncs, marker='o')
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Sync Rate σₛ')
    plt.grid(alpha=0.5)
    plt.show()

def cluster_series_by_sync(event_series_dict, lag_window=10, n_clusters=2):
    series_names = list(event_series_dict.keys())
    n_series = len(series_names)
    sync_matrix = np.zeros((n_series, n_series))

    for i, series_a in enumerate(series_names):
        for j, series_b in enumerate(series_names):
            if i != j:
                sync_rate, _ = calculate_sync_rate(
                    event_series_dict[series_a],
                    event_series_dict[series_b],
                    lag_window
                )
                sync_matrix[i, j] = sync_rate

    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(1 - sync_matrix)

    clusters = {series: label for series, label in zip(series_names, labels)}
    return clusters, sync_matrix

def sync_matrix(event_series_dict, lag_window=10):
    """
    Create a sync rate matrix (all pairs, max σₛ over all lags).
    """
    series_names = list(event_series_dict.keys())
    n = len(series_names)
    mat = np.zeros((n, n))
    for i, a in enumerate(series_names):
        for j, b in enumerate(series_names):
            if i == j:
                continue
            _, max_sync, _ = calculate_sync_profile(event_series_dict[a], event_series_dict[b], lag_window)
            mat[i, j] = max_sync
    return mat, series_names

def cluster_series_by_sync(event_series_dict, lag_window=10, n_clusters=2):
    """
    Cluster time series based on their maximum pairwise sync rates (over all lags).
    Returns clusters and the sync matrix.
    """
    from sklearn.cluster import AgglomerativeClustering
    mat, names = sync_matrix(event_series_dict, lag_window)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(1 - mat)
    clusters = {name: label for name, label in zip(names, labels)}
    return clusters, mat

# ===============================
# 4. Posterior Visualization
# ===============================
def plot_posterior(
    trace,
    var_names=None,
    hdi_prob=0.94
):
    """
    Visualize posterior distributions of selected parameters with HDI.
    """
    if var_names is None:
        var_names = list(trace.posterior.data_vars)
    az.plot_posterior(trace, var_names=var_names, hdi_prob=hdi_prob)
    plt.tight_layout()
    plt.show()

# ===============================
# 5. Prediction and Jump Event Visualization
# ===============================
def plot_l3_prediction_dual(
    data_dict,
    mu_pred_dict,
    jump_pos_dict,
    jump_neg_dict,
    local_jump_dict=None,
    series_names=None,
    titles=None
):
    """
    Plot observed data, model prediction, and global/local jump events for multiple series.
    Args:
        data_dict (dict): {'A': data_a, 'B': data_b, ...}
        mu_pred_dict (dict): {'A': mu_pred_a, 'B': mu_pred_b, ...}
        jump_pos_dict (dict): {'A': jump_pos_a, ...}
        jump_neg_dict (dict): {'A': jump_neg_a, ...}
        local_jump_dict (dict): {'A': local_jump_a, ...} or None
        series_names (list): list of series names to plot
        titles (list): plot titles for each series
    """
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
        local_jump = local_jump_dict[series] if local_jump_dict is not None else None

        ax.plot(data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
        ax.plot(mu_pred, color='C2', lw=2, label='Model Prediction')

        # Positive jumps: blue
        jump_pos_idx = np.where(jump_pos > 0)[0]
        if len(jump_pos_idx):
            ax.plot(jump_pos_idx, data[jump_pos_idx], 'o', color='dodgerblue', markersize=10, label='Positive Jump')
            for idx in jump_pos_idx:
                ax.axvline(x=idx, color='dodgerblue', linestyle='--', alpha=0.5)

        # Negative jumps: orange
        jump_neg_idx = np.where(jump_neg > 0)[0]
        if len(jump_neg_idx):
            ax.plot(jump_neg_idx, data[jump_neg_idx], 'o', color='orange', markersize=10, label='Negative Jump')
            for idx in jump_neg_idx:
                ax.axvline(x=idx, color='orange', linestyle='-.', alpha=0.5)

        # Local jumps: magenta
        if local_jump is not None:
            local_jump_idx = np.where(local_jump > 0)[0]
            if len(local_jump_idx):
                ax.plot(local_jump_idx, data[local_jump_idx], 'o', color='magenta', markersize=7, alpha=0.7, label='Local Jump')
                for idx in local_jump_idx:
                    ax.axvline(x=idx, color='magenta', linestyle=':', alpha=0.3)

        plot_title = titles[i] if titles and i < len(titles) else f"Series {series}: Fit + Events"
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)
        ax.grid(axis='y', linestyle=':', alpha=0.7)
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
    Plot multiple lagged causality profiles (e.g., A, B, A→B, B→A) together.

    Parameters:
        causality_dicts: list of dicts, each like {lag: prob, ...}
        labels: list of label strings (for legend)
        colors: list of colors for bars/lines
        title, xlabel, ylabel: plot labels
        figsize, alpha: figure params
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

# ===============================
# 6. Main Execution Pipeline
# ===============================
def main_dual_series():
    config = L3Config()

    # 1. Generate data for two time series A and B (with different patterns and seeds)
    data_a, trend_a, jumps_a = generate_data_pattern(config, config.pattern_a, seed_offset=0)
    data_b, trend_b, jumps_b = generate_data_pattern(config, config.pattern_b, seed_offset=0)

    # 2. Extract Lambda³ features separately for each series
    feats_a = calc_lambda3_features_v2(data_a, config)
    feats_b = calc_lambda3_features_v2(data_b, config)

    # 3. Prepare dictionaries for each series' features
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

    # 4. Build interaction feature (B's positive jumps influencing A)
    interaction_feature = features_dict_b['delta_LambdaC_pos']

    # 5. Bayesian regression including cross-series interaction
    trace_a = fit_l3_bayesian_regression(
        data=data_a,
        features_dict=features_dict_a,
        config=config,
        interaction_feature=features_dict_b['delta_LambdaC_pos'],
        interaction_label='B_pos_jump'
    )
    summary_a = az.summary(trace_a, var_names=[
        'beta_0', 'beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT', 'beta_B_pos_jump'
    ])
    mu_pred_a = (summary_a.loc['beta_0', 'mean']
                + summary_a.loc['beta_time', 'mean'] * features_dict_a['time_trend']
                + summary_a.loc['beta_dLC_pos', 'mean'] * features_dict_a['delta_LambdaC_pos']
                + summary_a.loc['beta_dLC_neg', 'mean'] * features_dict_a['delta_LambdaC_neg']
                + summary_a.loc['beta_rhoT', 'mean'] * features_dict_a['rho_T']
                + summary_a.loc['beta_B_pos_jump', 'mean'] * features_dict_b['delta_LambdaC_pos'])
    
    trace_b = fit_l3_bayesian_regression(
        data=data_b,
        features_dict=features_dict_b,
        config=config,
        interaction_feature=features_dict_a['delta_LambdaC_pos'],
        interaction_label='A_pos_jump'
    )
    summary_b = az.summary(trace_b, var_names=[
        'beta_0', 'beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT', 'beta_A_pos_jump'
    ])
    mu_pred_b = (summary_b.loc['beta_0', 'mean']
                + summary_b.loc['beta_time', 'mean'] * features_dict_b['time_trend']
                + summary_b.loc['beta_dLC_pos', 'mean'] * features_dict_b['delta_LambdaC_pos']
                + summary_b.loc['beta_dLC_neg', 'mean'] * features_dict_b['delta_LambdaC_neg']
                + summary_b.loc['beta_rhoT', 'mean'] * features_dict_b['rho_T']
                + summary_b.loc['beta_A_pos_jump', 'mean'] * features_dict_a['delta_LambdaC_pos'])

    # 6. Posterior distribution visualization
    # For A
    plot_posterior(
        trace_a,
        var_names=['beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT', 'beta_B_pos_jump'],
        hdi_prob=config.hdi_prob
    )

    # For B
    plot_posterior(
        trace_b,
        var_names=['beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT', 'beta_A_pos_jump'],
        hdi_prob=config.hdi_prob
    )

    # 7. Calculate predictive mean for A
    summary_a = az.summary(trace_a, var_names=[
        'beta_0', 'beta_time', 'beta_dLC_pos', 
        'beta_dLC_neg', 'beta_rhoT', 'beta_B_pos_jump'
    ])
    mu_pred_a = (
        summary_a.loc['beta_0', 'mean']
        + summary_a.loc['beta_time', 'mean'] * features_dict_a['time_trend']
        + summary_a.loc['beta_dLC_pos', 'mean'] * features_dict_a['delta_LambdaC_pos']
        + summary_a.loc['beta_dLC_neg', 'mean'] * features_dict_a['delta_LambdaC_neg']
        + summary_a.loc['beta_rhoT', 'mean'] * features_dict_a['rho_T']
        + summary_a.loc['beta_B_pos_jump', 'mean'] * features_dict_b['delta_LambdaC_pos']
    )

    # 7. Calculate predictive mean for B
    summary_b = az.summary(trace_b, var_names=[
        'beta_0', 'beta_time', 'beta_dLC_pos', 
        'beta_dLC_neg', 'beta_rhoT', 'beta_A_pos_jump'
    ])
    mu_pred_b = (
        summary_b.loc['beta_0', 'mean']
        + summary_b.loc['beta_time', 'mean'] * features_dict_b['time_trend']
        + summary_b.loc['beta_dLC_pos', 'mean'] * features_dict_b['delta_LambdaC_pos']
        + summary_b.loc['beta_dLC_neg', 'mean'] * features_dict_b['delta_LambdaC_neg']
        + summary_b.loc['beta_rhoT', 'mean'] * features_dict_b['rho_T']
        + summary_b.loc['beta_A_pos_jump', 'mean'] * features_dict_a['delta_LambdaC_pos']
    )

    # 8. Plot model fit and jump events for both A and B (DUAL visualization)
    plot_l3_prediction_dual(
        data_dict={'A': data_a, 'B': data_b},
        mu_pred_dict={'A': mu_pred_a, 'B': mu_pred_b},
        jump_pos_dict={'A': features_dict_a['delta_LambdaC_pos'], 'B': features_dict_b['delta_LambdaC_pos']},
        jump_neg_dict={'A': features_dict_a['delta_LambdaC_neg'], 'B': features_dict_b['delta_LambdaC_neg']},
        local_jump_dict={'A': feats_a[4], 'B': feats_b[4]},
        series_names=['A', 'B'],
        titles=['Series A: Fit + Events', 'Series B: Fit + Events']
    )

    # 9. Lambda3BayesianExtended for multi-series causality/sync analysis
    lambda3_ext = Lambda3BayesianExtended(config, series_names=['A', 'B'])
    # Feed both series jump events to the event memory
    for i in range(config.T):
        lambda3_ext.update_event_memory({
            'A': {'pos': features_dict_a['delta_LambdaC_pos'][i], 'neg': features_dict_a['delta_LambdaC_neg'][i]},
            'B': {'pos': features_dict_b['delta_LambdaC_pos'][i], 'neg': features_dict_b['delta_LambdaC_neg'][i]}
        })

    # 10. Compute all single/cross-series lagged causality profiles
    causality_by_lag_a = lambda3_ext.detect_time_dependent_causality(series='A', lag_window=10)
    causality_by_lag_b = lambda3_ext.detect_time_dependent_causality(series='B', lag_window=10)
    causality_ab = {lag: lambda3_ext.detect_cross_causality('A', 'B', lag=lag) for lag in range(1, 11)}
    causality_ba = {lag: lambda3_ext.detect_cross_causality('B', 'A', lag=lag) for lag in range(1, 11)}

    # 11. Plot ALL causality profiles at once (A, B, A→B, B→A)
    plot_multi_causality_lags(
        [causality_by_lag_a, causality_by_lag_b, causality_ab, causality_ba],
        labels=['A', 'B', 'A→B', 'B→A'],
        title='Lagged Causality Profiles'
    )

    # 12. Print cross-causality values by lag
    print("\nCross Causality Lags (B→A, A→B):")
    for lag in range(1, 11):
        print(f"Lag {lag}: B→A = {causality_ba[lag]:.2f} | A→B = {causality_ab[lag]:.2f}")

    # 13. Calculate and plot synchronization profile (σₛ vs lag) and dynamic sync rate
    sync_profile, sync_rate, optimal_lag = calculate_sync_profile(
        features_dict_a['delta_LambdaC_pos'],
        features_dict_b['delta_LambdaC_pos'],
        lag_window=10
    )
    print(f"\nSync Profile (A↔B): {sync_profile}")
    print(f"Sync Rate σₛ (A↔B): {sync_rate:.2f} | Optimal Lag: {optimal_lag} steps")

    # Plot the full synchronization profile (σₛ vs lag)
    plot_sync_profile(sync_profile, title="Sync Profile (σₛ vs Lag, A↔B)")

    # Calculate time-varying (dynamic) synchronization rate and optimal lag profiles
    time_points, sync_rates, optimal_lags = calculate_dynamic_sync(
        features_dict_a['delta_LambdaC_pos'],
        features_dict_b['delta_LambdaC_pos'],
        window=20, lag_window=10
    )
    plot_dynamic_sync(time_points, sync_rates, optimal_lags)

    # 14. Build and plot the full synchronization network for all series
    event_series_dict = {
        'A': features_dict_a['delta_LambdaC_pos'],
        'B': features_dict_b['delta_LambdaC_pos'],
        # Add more series as needed
    }
    G = build_sync_network(event_series_dict, lag_window=10, sync_threshold=0.2)
    plot_sync_network(G)

    # 15. Cluster series by sync (N > 2 series) and show heatmap (maximum sync rate matrix)
    clusters, sync_mat = cluster_series_by_sync(event_series_dict, lag_window=10, n_clusters=2)
    print("Sync clusters:", clusters)

    plt.figure(figsize=(5, 4))
    sns.heatmap(sync_mat, annot=True,
                xticklabels=list(event_series_dict.keys()),
                yticklabels=list(event_series_dict.keys()),
                cmap="Blues")
    plt.title("Sync Rate Matrix (σₛ, max over all lags)")
    plt.show()

if __name__ == '__main__':
    main_dual_series()
