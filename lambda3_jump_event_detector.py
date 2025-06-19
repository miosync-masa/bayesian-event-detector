import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor

"""
Note:
    The variable 'T' (and np.arange(T)) represents *transaction steps* or *progress indices*,
    not 'time' in the traditional sense. In Lambda³ theory, this axis should be interpreted
    as generic progress (transaction) steps, decoupled from physical time.
"""

# --- [1] Generate Synthetic Data Patterns for Model Testing ---
def generate_data_pattern(pattern="single_jump", T=150, seed=42):
    """
    Generate synthetic time series with customizable jump and trend patterns.

    Args:
        pattern (str): Pattern type. 
            Options: "no_jump", "single_jump", "multi_jump", "mixed_sign", "consecutive_jumps", "step_trend", "noisy".
        T (int): Number of transaction steps (not time points!).
        seed (int): Random seed for reproducibility.
    Returns:
        data (np.ndarray): Synthetic observed data.
        trend (np.ndarray): Underlying smooth trend.
        jumps (np.ndarray): Injected jump event values.
    """
    np.random.seed(seed)
    trend = 0.05 * np.arange(T) + np.sin(np.arange(T) * 0.2)
    jumps = np.zeros(T)
    noise_std = 0.5  # default noise

    if pattern == "no_jump":
        pass  # No jump events
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
        noise_std = 1.5  # Increase noise

    noise = np.random.randn(T) * noise_std
    data = trend + jumps + noise
    return data, trend, jumps

# --- [2] Calculate Lambda³ Features (Directional ΔΛC, ρT, transaction index) ---
def calc_lambda3_features_v2(data, window=10, delta_percentile=97):
    """
    Calculate directionally separated jump features (ΔΛC_pos, ΔΛC_neg),
    local volatility (ρT), and sequential index (transaction_trend).

    Args:
        data (np.ndarray): Observed series.
        window (int): Window size for local volatility.
        delta_percentile (float): Percentile threshold for event detection.
    Returns:
        delta_LambdaC_pos (np.ndarray): Positive jump event indicators (1=jump, 0=otherwise).
        delta_LambdaC_neg (np.ndarray): Negative jump event indicators (1=jump, 0=otherwise).
        rho_T (np.ndarray): Local volatility ("tension density").
        time_trend (np.ndarray): Transaction/progress index (not time).
    """
    diff = np.diff(data, prepend=data[0])
    threshold = np.percentile(np.abs(diff), delta_percentile)
    delta_LambdaC_pos = (diff > threshold).astype(int)
    delta_LambdaC_neg = (diff < -threshold).astype(int)
    rho_T = np.array([data[max(0, i-window):i+1].std() for i in range(len(data))])
    time_trend = np.arange(len(data))
    return delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend

# --- [3] Lambda³ Bayesian Regression Model (Directional Jumps) ---
def fit_l3_bayesian_regression_v2(data, delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend, draws=2000, tune=2000):
    """
    Fit a Bayesian regression using Lambda³ features as explanatory variables.

    Args:
        data (np.ndarray): Observed series.
        delta_LambdaC_pos/neg: Directional jump indicators.
        rho_T (np.ndarray): Local volatility.
        time_trend (np.ndarray): Transaction step indices.
        draws (int): Number of posterior samples.
        tune (int): Tuning steps.
    Returns:
        trace_l3_v2 (arviz.InferenceData): Posterior samples.
    """
    with pm.Model() as l3_test_model_v2:
        beta_0 = pm.Normal('beta_0', mu=0, sigma=2)   # Intercept
        beta_time = pm.Normal('beta_time', mu=0, sigma=1)  # Progress/transaction trend
        beta_dLC_pos = pm.Normal('beta_dLC_pos', mu=0, sigma=5)  # Positive jumps
        beta_dLC_neg = pm.Normal('beta_dLC_neg', mu=0, sigma=5)  # Negative jumps
        beta_rhoT = pm.Normal('beta_rhoT', mu=0, sigma=3)   # Local volatility effect
        mu = (beta_0
              + beta_time * time_trend
              + beta_dLC_pos * delta_LambdaC_pos
              + beta_dLC_neg * delta_LambdaC_neg
              + beta_rhoT * rho_T)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=data)
        trace_l3_v2 = pm.sample(draws=draws, tune=tune, target_accept=0.95)
    return trace_l3_v2

# --- [4] Plot Posterior Distributions of Model Coefficients ---
def plot_posterior(trace):
    """
    Plot posterior distributions of model parameters (for classical models).
    """
    az.plot_posterior(
        trace,
        var_names=['beta_time', 'beta_dLC', 'beta_rhoT'],
        hdi_prob=0.94
    )
    plt.show()

# --- [5] Plot L³ Model Prediction and Detected Jumps ---
def plot_l3_prediction(data, delta_LambdaC, rho_T, time_trend, trace):
    """
    Visualize the model fit and jump event detection.
    """
    summary = az.summary(trace, var_names=['beta_0', 'beta_time', 'beta_dLC', 'beta_rhoT'])
    mu_pred = (summary.loc['beta_0', 'mean']
               + summary.loc['beta_time', 'mean'] * time_trend
               + summary.loc['beta_dLC', 'mean'] * delta_LambdaC
               + summary.loc['beta_rhoT', 'mean'] * rho_T)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
    ax.plot(mu_pred, color='C2', lw=2, label='L3 Model Prediction (mean)')
    jump_indices = np.where(delta_LambdaC > 0)[0]
    for idx in jump_indices:
        ax.axvline(x=idx, color='C1', linestyle='--', alpha=0.6, label='ΔΛC Event' if idx==jump_indices[0] else "")
    ax.set_title('L³ Model Fit and Detected Jump Events (ΔΛC)', fontsize=16)
    ax.set_xlabel('Transaction Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.show()

if __name__ == '__main__':
    # Example: run with "no_jump" pattern (change as needed)
    data, trend, jumps = generate_data_pattern(pattern="no_jump")
    delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend = calc_lambda3_features_v2(data)
    trace_l3_v2 = fit_l3_bayesian_regression_v2(
        data, delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend
    )

    # Plot posterior distributions for the Lambda³ model coefficients
    az.plot_posterior(
        trace_l3_v2,
        var_names=['beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT'],
        hdi_prob=0.94
    )
    plt.show()

    # --- Prediction plot using mean posterior parameters ---
    summary = az.summary(trace_l3_v2, var_names=['beta_0', 'beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT'])
    mu_pred = (
        summary.loc['beta_0', 'mean']
        + summary.loc['beta_time', 'mean'] * time_trend
        + summary.loc['beta_dLC_pos', 'mean'] * delta_LambdaC_pos
        + summary.loc['beta_dLC_neg', 'mean'] * delta_LambdaC_neg
        + summary.loc['beta_rhoT', 'mean'] * rho_T
    )

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
    ax.plot(mu_pred, color='C2', lw=2, label='L3 Model Prediction (mean)')

    # Visualize positive and negative jump detections
    jump_indices_pos = np.where(delta_LambdaC_pos > 0)[0]
    jump_indices_neg = np.where(delta_LambdaC_neg > 0)[0]
    for idx in jump_indices_pos:
        ax.axvline(x=idx, color='C0', linestyle='--', alpha=0.5, label='ΔΛC Event (pos)' if idx==jump_indices_pos[0] else "")
    for idx in jump_indices_neg:
        ax.axvline(x=idx, color='C1', linestyle='-.', alpha=0.5, label='ΔΛC Event (neg)' if idx==jump_indices_neg[0] else "")

    ax.set_title('L³ Model Fit and Detected Jump Events (ΔΛC)', fontsize=16)
    ax.set_xlabel('Transaction Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.show()
