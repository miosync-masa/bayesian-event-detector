import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ===============================
# Lambda³ Config Class
# ===============================
@dataclass
class L3Config:
    # --- Data generation parameters ---
    pattern: str = "causal_chain"  # Data scenario: choose from 'single_jump', 'multi_jump', etc.
    T: int = 300                  # Number of time steps
    seed: int = 42                # Random seed for reproducibility

    # --- Lambda³ feature extraction parameters ---
    window: int = 10              # Window size for global std calculation (rho_T)
    local_window: int = 10        # Window size for local std calculation (for local jump detection)
    delta_percentile: float = 97.0      # Percentile for global jump threshold
    local_jump_percentile: float = 97.0 # Percentile for local jump threshold

    # --- Bayesian model sampling parameters ---
    draws: int = 3000
    tune: int = 3000
    target_accept: float = 0.95

    # --- Posterior visualization parameters ---
    var_names: list = ('beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT')
    hdi_prob: float = 0.94

# ===============================
# 1. Synthetic Data Generator
# ===============================
def generate_data_pattern(config: L3Config):
    """
    Generates synthetic time series data with controllable jumps, trend, and noise.
    This is used for demonstrating and testing the Lambda³ jump event detection pipeline.
    """
    np.random.seed(config.seed)
    t = np.arange(config.T)
    trend = 0.05 * t + np.sin(t * 0.2)
    jumps = np.zeros(config.T)
    noise_std = 0.5
    
    if config.pattern == "no_jump":
        pass
    elif config.pattern == "single_jump":
        jumps[60] = 8.0
    elif config.pattern == "multi_jump":
        jumps[[30, 80, 110]] = [5.0, -7.0, 3.5]
    elif config.pattern == "mixed_sign":
        jumps[30] = 8.0; jumps[70] = -5.0; jumps[120] = 3.5
    elif config.pattern == "consecutive_jumps":
        jumps[40:44] = [3, -3, 4, -4]
    elif config.pattern == "step_trend":
        trend[75:] += 10
    elif config.pattern == "noisy":
        noise_std = 1.5
    elif config.pattern == "hidden_jump_noise":
        jumps[50] = 1.2; jumps[95] = -1.5
        noise_std = 1.0
    elif config.pattern == "periodic_plus_jump":
        trend = 2.0 * np.sin(t * 0.15)
        jumps[40] = 4.0; jumps[90] = -3.5
        jumps[120] = 2.5
    elif config.pattern == "causal_chain":
        jumps[20] = 5.0; jumps[23] = -4.5
        jumps[60] = 3.0; jumps[65] = -3.2
        jumps[100] = 7.0; jumps[108] = -7.1
    elif config.pattern == "overlapping_events":
        jumps[50] = 8.0
        jumps[50] += np.random.uniform(-0.5, 0.5)  
        jumps[80] = -3.0; jumps[80] += 0.7

    noise = np.random.randn(config.T) * noise_std
    data = trend + jumps + noise
    return data, trend, jumps

# ===============================
# 2. Lambda³ Feature Extraction
# ===============================
def calc_lambda3_features_v2(data, config: L3Config):
    """
    Extracts Lambda³ features:
    - Global jump events (Delta_LambdaC, based on global percentile)
    - Local jump events (local_jump_detect, based on local z-score style thresholding)
    - Local volatility (rho_T)
    - Linear time trend
    """
    # --- Global (history-wide) jump detection ---
    diff = np.diff(data, prepend=data[0])
    threshold = np.percentile(np.abs(diff), config.delta_percentile)
    delta_LambdaC_pos = (diff > threshold).astype(int)
    delta_LambdaC_neg = (diff < -threshold).astype(int)

    # --- Local jump detection (contextual anomaly) ---
    local_std = np.array([
        data[max(0, i-config.local_window):min(len(data), i+config.local_window+1)].std()
        for i in range(len(data))
    ])
    score = np.abs(diff) / (local_std + 1e-8)
    local_threshold = np.percentile(score, config.local_jump_percentile)
    local_jump_detect = (score > local_threshold).astype(int)

    # --- Local volatility feature (rho_T) ---
    rho_T = np.array([data[max(0, i-config.window):i+1].std() for i in range(len(data))])
    time_trend = np.arange(len(data))

    return delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend, local_jump_detect


# ===============================
#  Feature view
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

# --- ⬇⬇⬇  Expansion Ideas ⬇⬇⬇ ---

        # [Expansion Ideas]
        # 1. Multiclass Forecasting:
        #    - Add more event types: "major jump," "minor jump," "noise fluctuation," etc.
        #    - Could also predict "no event" (quiescence).
        #
        # 2. Time Series Model Integration:
        #    - Use LSTM or Bayesian time series models on event_memory for richer trend prediction.
        #    - Predict not just type, but also *timing* of the next event.
        #
        # 3. Markov/Hidden Markov State Transitions:
        #    - Use N-step history to model transition probabilities (P(next_event | past_events)).
        #    - Could output a probability distribution over possible next events.
        #
        # 4. Feature Expansion:
        #    - Include other features such as "rho_T", local volatility, jump interval statistics, etc.
        #    - This can improve accuracy, especially in complex signals.
        #
        # 5. Periodicity and Anomaly Detection:
        #    - Detect if events follow a periodic or bursty pattern, and predict accordingly.
        #    - "Sudden jumps after long quiet" can be modeled with dedicated logic.
        #
        # 6. Uncertainty Estimation:
        #    - Use Bayesian inference to report not just a forecast, but also the confidence/credible interval.
        #
        # [Implementation Ideas]
        # - Return richer output: {'type': ..., 'prob': ..., 'expected_in': ...}
        # - Use posterior predictive sampling to simulate future event chains and average outcomes.
        # - Allow the user to set thresholds for what counts as "expected" or "significant."

# ===============================
# 3. Lambda³ Bayesian Regression Model
# ===============================
def fit_l3_bayesian_regression_v2(data, delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend, config: L3Config):
    """
    Bayesian regression: fits model to data using Lambda³ features.
    Estimates coefficients for global trend, positive jumps, negative jumps, and local volatility.
    """
    with pm.Model() as model:
        beta_0 = pm.Normal('beta_0', mu=0, sigma=2)
        beta_time = pm.Normal('beta_time', mu=0, sigma=1)
        beta_dLC_pos = pm.Normal('beta_dLC_pos', mu=0, sigma=5)
        beta_dLC_neg = pm.Normal('beta_dLC_neg', mu=0, sigma=5)
        beta_rhoT = pm.Normal('beta_rhoT', mu=0, sigma=3)

        mu = (beta_0
              + beta_time * time_trend
              + beta_dLC_pos * delta_LambdaC_pos
              + beta_dLC_neg * delta_LambdaC_neg
              + beta_rhoT * rho_T)

        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=data)
        trace = pm.sample(draws=config.draws, tune=config.tune, target_accept=config.target_accept)
    return trace

# ===============================
# 4. Posterior Visualization
# ===============================
def plot_posterior(trace, var_names=['beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT'], hdi_prob=0.94):
    """
    Visualizes posterior distributions for selected model parameters.
    Shows the credible interval (HDI) and mean for each parameter.
    """
    az.plot_posterior(trace, var_names=var_names, hdi_prob=hdi_prob)
    plt.show()

# ===============================
# 5. Prediction and Jump Event Visualization
# ===============================
def plot_l3_prediction(
    data, mu_pred, delta_LambdaC_pos, delta_LambdaC_neg, time_trend, local_jump_detect=None
):
    """
    Plots original data, model prediction, and both types of detected jump events.
    - Blue markers: global positive jumps (ΔΛC_pos)
    - Orange markers: global negative jumps (ΔΛC_neg)
    - Magenta markers: local jumps (contextual anomalies)
    
    This "dual detection logic" visualizes both "phase-changing" global events and micro/local anomalies,
    allowing users to see both "forest" (macro events) and "trees" (local oddities) simultaneously.
    
    [Lambda³ Philosophy]: This two-scale anomaly detection is essential for modeling real-world systems,
    where both major regime shifts and small precursors/foreshocks may coexist.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
    ax.plot(mu_pred, color='C2', lw=2, label='L3 Model Prediction (mean)')

    # Positive jumps: blue
    jump_pos = np.where(delta_LambdaC_pos > 0)[0]
    if len(jump_pos) > 0:
        ax.plot(jump_pos, data[jump_pos], 'o', color='dodgerblue', markersize=10, label='Positive Jump')
        for idx in jump_pos:
            ax.axvline(x=idx, color='dodgerblue', linestyle='--', alpha=0.5)

    # Negative jumps: orange
    jump_neg = np.where(delta_LambdaC_neg > 0)[0]
    if len(jump_neg) > 0:
        ax.plot(jump_neg, data[jump_neg], 'o', color='orange', markersize=10, label='Negative Jump')
        for idx in jump_neg:
            ax.axvline(x=idx, color='orange', linestyle='-.', alpha=0.5)

    # Local jumps: magenta
    if local_jump_detect is not None:
        local_jump_idx = np.where(local_jump_detect > 0)[0]
        if len(local_jump_idx) > 0:
            ax.plot(local_jump_idx, data[local_jump_idx], 'o', color='magenta', markersize=7, alpha=0.7, label='Local Jump')
            for idx in local_jump_idx:
                ax.axvline(x=idx, color='magenta', linestyle=':', alpha=0.3)

    ax.set_title('L³ Model Fit and Detected Jump Events (ΔΛC + Local)', fontsize=16)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.show()

def plot_time_dependent_causality(causality_by_lag):
    """
    Plots the lag-dependent causality probability from positive to negative jump events.
    """
    lags = list(causality_by_lag.keys())
    probs = list(causality_by_lag.values())
    plt.figure(figsize=(7,3))
    plt.bar(lags, probs, color='royalblue', alpha=0.7)
    plt.xlabel("Lag (steps)")
    plt.ylabel("Causality P(Pos→Neg)")
    plt.title("Time-Dependent Causality: Positive Jump → Negative Jump")
    plt.tight_layout()
    plt.show()
    
# ===============================
# 6. Main Execution Pipeline
# ===============================
def main():
    config = L3Config()

    # Generate synthetic data
    data, trend, jumps = generate_data_pattern(config)

    # Calculate Lambda³ features
    delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend, local_jump_detect = calc_lambda3_features_v2(data, config)

    # Initialize the extended Lambda3 Bayesian framework
    lambda3_ext = Lambda3BayesianExtended(config)

    # Update event memory for causality detection
    for pos, neg in zip(delta_LambdaC_pos, delta_LambdaC_neg):
        lambda3_ext.update_event_memory(pos, neg)

    # Bayesian regression
    trace = fit_l3_bayesian_regression_v2(
        data, delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend, config
    )

    # Visualize posterior distributions
    plot_posterior(trace, var_names=config.var_names, hdi_prob=config.hdi_prob)

    # Predictive visualization
    summary = az.summary(trace, var_names=['beta_0', 'beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT'])
    mu_pred = (
        summary.loc['beta_0', 'mean']
        + summary.loc['beta_time', 'mean'] * time_trend
        + summary.loc['beta_dLC_pos', 'mean'] * delta_LambdaC_pos
        + summary.loc['beta_dLC_neg', 'mean'] * delta_LambdaC_neg
        + summary.loc['beta_rhoT', 'mean'] * rho_T
    )

    plot_l3_prediction(
        data, mu_pred, delta_LambdaC_pos, delta_LambdaC_neg, time_trend, local_jump_detect=local_jump_detect
    )

    # Detect causality and predict next event
    causality_prob = lambda3_ext.detect_causality_chain()
    next_event_prediction = lambda3_ext.predict_next_event()

    # --- NEW: Time-dependent causality analysis ---
    lag_window = 10
    causality_by_lag = lambda3_ext.detect_time_dependent_causality(lag_window=lag_window)
    plot_time_dependent_causality(causality_by_lag)

    # Output results
    print(f"\nCausality Probability (Positive Jump → Negative Jump): {causality_prob:.2f}")
    print(f"Predicted Next Event: {next_event_prediction}")
    print(f"Time-Dependent Causality (lag steps → P):\n{causality_by_lag}")

if __name__ == '__main__':
    main()
