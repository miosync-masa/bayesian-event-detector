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
    pattern_b: str = "mixed_sign"
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
def generate_data_pattern(config: L3Config, pattern: str):
    np.random.seed(config.seed)
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
    def __init__(self, config: L3Config):
        self.config = config
        self.event_memory = []  # ΔΛC event history
        self.structure_evolution = []  # Λ structure history

    def update_event_memory(self, delta_LambdaC_pos, delta_LambdaC_neg):
        event = {
            'pos': delta_LambdaC_pos,
            'neg': delta_LambdaC_neg
        }
        self.event_memory.append(event)

    def update_structure(self, Lambda_tensor):
        self.structure_evolution.append(Lambda_tensor)

    def detect_causality_chain(self):
        # Simple causality chain detection example
        if len(self.event_memory) < 2:
            return None  # Need at least two events for causality

        # Very simple logic: if positive jump often precedes negative jump
        pos_to_neg_transitions = sum(
            self.event_memory[i]['pos'] and self.event_memory[i+1]['neg']
            for i in range(len(self.event_memory)-1)
        )
        total_pos_events = sum(event['pos'] for event in self.event_memory[:-1])

        if total_pos_events == 0:
            return 0
        return pos_to_neg_transitions / total_pos_events

    def predict_next_event(self):
        # Bayesian-based predictive logic (very simplified)
        recent_events = np.array([
            event['pos'] - event['neg'] for event in self.event_memory[-self.config.window:]
        ])

        # If recent trend is positive jumps, predict a negative jump coming soon
        trend = np.mean(recent_events)

        if trend > 0.5:
            prediction = 'negative_jump_expected'
        elif trend < -0.5:
            prediction = 'positive_jump_expected'
        else:
            prediction = 'stable'

        return prediction

    def detect_time_dependent_causality(self, lag_window=10):
        causality_by_lag = {}

        for lag in range(1, lag_window+1):
            count_pairs = 0
            count_pos = 0

            for i in range(len(self.event_memory)-lag):
                if self.event_memory[i]['pos']:
                    count_pos += 1
                    if self.event_memory[i+lag]['neg']:
                        count_pairs += 1

            causality_by_lag[lag] = count_pairs / max(count_pos, 1)

        return causality_by_lag

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
    Constructs a synchronization network from multiple event series.
    Args:
        event_series_dict: dict of {series_name: event_series}
        lag_window: maximum lag considered for synchronization
        sync_threshold: threshold for visualizing synchronization
    Returns:
        G: networkx graph object representing the synchronization network
    """
    series_names = list(event_series_dict.keys())
    G = nx.DiGraph()

    for series in series_names:
        G.add_node(series)

    for i, series_a in enumerate(series_names):
        for series_b in series_names[i+1:]:
            sync_rate, lag = calculate_sync_rate(
                event_series_dict[series_a],
                event_series_dict[series_b],
                lag_window
            )
            if sync_rate >= sync_threshold:
                G.add_edge(series_a, series_b, weight=sync_rate, lag=lag)

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
def plot_l3_prediction(
    data,
    mu_pred,
    delta_LambdaC_pos,
    delta_LambdaC_neg,
    local_jump_detect=None,
    title='L³ Model Fit and Detected Jump Events'
):
    """
    Plot observed data, model prediction, and global/local jump events.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
    ax.plot(mu_pred, color='C2', lw=2, label='L3 Model Prediction (mean)')

    # Global positive jumps: blue
    jump_pos = np.where(delta_LambdaC_pos > 0)[0]
    if len(jump_pos):
        ax.plot(jump_pos, data[jump_pos], 'o', color='dodgerblue', markersize=10, label='Positive Jump')
        for idx in jump_pos:
            ax.axvline(x=idx, color='dodgerblue', linestyle='--', alpha=0.5)

    # Global negative jumps: orange
    jump_neg = np.where(delta_LambdaC_neg > 0)[0]
    if len(jump_neg):
        ax.plot(jump_neg, data[jump_neg], 'o', color='orange', markersize=10, label='Negative Jump')
        for idx in jump_neg:
            ax.axvline(x=idx, color='orange', linestyle='-.', alpha=0.5)

    # Local jumps: magenta
    if local_jump_detect is not None:
        local_jump_idx = np.where(local_jump_detect > 0)[0]
        if len(local_jump_idx):
            ax.plot(local_jump_idx, data[local_jump_idx], 'o', color='magenta', markersize=7, alpha=0.7, label='Local Jump')
            for idx in local_jump_idx:
                ax.axvline(x=idx, color='magenta', linestyle=':', alpha=0.3)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_time_dependent_causality(
    causality_by_lag,
    title="Time-Dependent Causality",
    xlabel="Lag (steps)",
    ylabel="Causality Probability",
    color='royalblue',
    figsize=(8,4),
    alpha=0.7
):
    """
    General-purpose plotting function for lag-dependent causality.

    Parameters:
        causality_by_lag (dict): Lag steps as keys and causality probabilities as values.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        color (str): Bar color.
        figsize (tuple): Figure size.
        alpha (float): Bar transparency.
    """
    lags, probs = zip(*sorted(causality_by_lag.items()))

    plt.figure(figsize=figsize)
    plt.bar(lags, probs, color=color, alpha=alpha)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ===============================
# 6. Main Execution Pipeline
# ===============================
def main_dual_series():
    config = L3Config()

    # Generate data for two series (A and B)
    data_a, trend_a, jumps_a = generate_data_pattern(config, config.pattern_a)
    data_b, trend_b, jumps_b = generate_data_pattern(config, config.pattern_b)

    # Extract features separately for each series
    feats_a = calc_lambda3_features_v2(data_a, config)
    feats_b = calc_lambda3_features_v2(data_b, config)

    # Prepare feature dictionaries
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

    # Interaction feature (series B positive jumps influencing series A)
    interaction_feature = features_dict_b['delta_LambdaC_pos']

    # Bayesian regression with interaction
    trace = fit_l3_bayesian_regression(
        data=data_a,
        features_dict=features_dict_a,
        config=config,
        interaction_feature=interaction_feature,
        interaction_label='B_pos_jump'
    )

    # Posterior visualization
    plot_posterior(
        trace,
        var_names=['beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT', 'beta_B_pos_jump'],
        hdi_prob=config.hdi_prob
    )

    # Predictive mean calculation
    summary = az.summary(trace, var_names=[
        'beta_0', 'beta_time', 'beta_dLC_pos', 
        'beta_dLC_neg', 'beta_rhoT', 'beta_B_pos_jump'
    ])
    mu_pred = (
        summary.loc['beta_0', 'mean']
        + summary.loc['beta_time', 'mean'] * features_dict_a['time_trend']
        + summary.loc['beta_dLC_pos', 'mean'] * features_dict_a['delta_LambdaC_pos']
        + summary.loc['beta_dLC_neg', 'mean'] * features_dict_a['delta_LambdaC_neg']
        + summary.loc['beta_rhoT', 'mean'] * features_dict_a['rho_T']
        + summary.loc['beta_B_pos_jump', 'mean'] * interaction_feature
    )

    # Plot predictions
    plot_l3_prediction(
        data_a, mu_pred,
        features_dict_a['delta_LambdaC_pos'],
        features_dict_a['delta_LambdaC_neg'],
        local_jump_detect=feats_a[4],
        title='Dual Bayesian Model (Series A with B Pos-Jump Interaction)'
    )

    # Initialize Lambda3BayesianExtended for causality analysis
    lambda3_ext = Lambda3BayesianExtended(config)
    for pos, neg in zip(features_dict_a['delta_LambdaC_pos'], features_dict_a['delta_LambdaC_neg']):
        lambda3_ext.update_event_memory(pos, neg)

    causality_prob = lambda3_ext.detect_causality_chain()
    next_event_prediction = lambda3_ext.predict_next_event()

    causality_by_lag = lambda3_ext.detect_time_dependent_causality(lag_window=10)
    plot_time_dependent_causality(causality_by_lag)

    print(f"\nCausality Probability (Positive Jump → Negative Jump): {causality_prob:.2f}")
    print(f"Predicted Next Event: {next_event_prediction}")
    print(f"Time-Dependent Causality (lag steps → P):\n{causality_by_lag}")

    # 1. Sync rate and optimal lag (A→B)
    sync_rate, optimal_lag = calculate_sync_rate(
        features_dict_a['delta_LambdaC_pos'],
        features_dict_b['delta_LambdaC_pos'],
        lag_window=10
    )
    print(f"\nSync Rate σₛ (A→B): {sync_rate:.2f} | Optimal Lag: {optimal_lag} steps")

    # 2. Dynamic (time-varying) sync rate
    time_points, sync_rates, optimal_lags = calculate_dynamic_sync(
        features_dict_a['delta_LambdaC_pos'],
        features_dict_b['delta_LambdaC_pos'],
        window=20, lag_window=10
    )
    plot_dynamic_sync(time_points, sync_rates, optimal_lags)

    # 3. If you have multiple series (say, C, D...): Build sync network
    event_series_dict = {
        'A': features_dict_a['delta_LambdaC_pos'],
        'B': features_dict_b['delta_LambdaC_pos'],
        # 'C': features_dict_c['delta_LambdaC_pos'], ...
    }
    G = build_sync_network(event_series_dict, lag_window=10, sync_threshold=0.2)
    plot_sync_network(G)

    # 4. Clustering by sync (if more than 2 series)
    event_series_dict = {
        'A': features_dict_a['delta_LambdaC_pos'],
        'B': features_dict_b['delta_LambdaC_pos'],
        # 'C': features_dict_c['delta_LambdaC_pos'],  # ←add OK!!
        # ... 
    }

    # n_clusters
    clusters, sync_matrix = cluster_series_by_sync(event_series_dict, lag_window=10, n_clusters=2)
    print("Sync clusters:", clusters)

    # ↓Sync Matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(sync_matrix, annot=True, xticklabels=list(event_series_dict.keys()), yticklabels=list(event_series_dict.keys()), cmap="Blues")
    plt.title("Sync Rate Matrix (σₛ)")
    plt.show()

if __name__ == '__main__':
    main_dual_series()
