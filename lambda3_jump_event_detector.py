"""
Note:
    The variable 'T' (and arrays like np.arange(T)) represent *transaction steps* or *progress indices*, 
    not 'time' in the traditional sense. 
    In Lambda³ theory, this axis should be interpreted as generic progress (transaction) steps,
    decoupled from physical time.
"""
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as azhttps://github.com/miosync-masa/bayesian-event-detector/tree/main

# --- [1] Generate Dummy Data ---
def generate_data(seed=42, T=150):
    """
    Generate synthetic time series with trend, noise, and jump events.
    """
    np.random.seed(seed)
    trend = 0.05 * np.arange(T) + np.sin(np.arange(T) * 0.2)
    jumps = np.zeros(T)
    jumps[40], jumps[85], jumps[120] = 5.0, -6.0, 4.0
    noise = np.random.randn(T) * 0.5
    data = trend + jumps + noise
    return data, trend, jumps

# --- [2] Calculate Lambda³ Features (Directional ΔΛC, ρT, time_trend) ---
def calc_lambda3_features_v2(data, window=10, delta_percentile=97):
    """
    Calculate directionally separated jump features (ΔΛC_pos, ΔΛC_neg),
    local volatility (ρT), and sequential index (time_trend).
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
    """
    with pm.Model() as l3_test_model_v2:
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
        trace_l3_v2 = pm.sample(draws=draws, tune=tune, target_accept=0.95)
    return trace_l3_v2

# --- [4] Posterior Plot ---
def plot_posterior(trace):
    """
    Plot posterior distributions for main coefficients.
    """
    az.plot_posterior(
        trace,
        var_names=['beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT'],
        hdi_prob=0.94
    )
    plt.show()

# --- [5] L³ Model Prediction and Jump Event Visualization ---
def plot_l3_prediction_v2(data, delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend, trace):
    """
    Plot model mean prediction and detected jump events (positive/negative).
    """
    summary = az.summary(
        trace,
        var_names=['beta_0', 'beta_time', 'beta_dLC_pos', 'beta_dLC_neg', 'beta_rhoT']
    )
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

    # Plot positive and negative jump events
    jump_indices_pos = np.where(delta_LambdaC_pos > 0)[0]
    jump_indices_neg = np.where(delta_LambdaC_neg > 0)[0]
    for idx in jump_indices_pos:
        ax.axvline(x=idx, color='C0', linestyle='--', alpha=0.5, label='ΔΛC Event (pos)' if idx==jump_indices_pos[0] else "")
    for idx in jump_indices_neg:
        ax.axvline(x=idx, color='C1', linestyle='-.', alpha=0.5, label='ΔΛC Event (neg)' if idx==jump_indices_neg[0] else "")

    ax.set_title('L³ Model Fit and Detected Jump Events (ΔΛC)', fontsize=16)
    ax.set_xlabel('Transaction Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.show()

# --- [Main Execution Block] ---
if __name__ == '__main__':
    # Generate synthetic data
    data, trend, jumps = generate_data()
    # Calculate Lambda³ features (directional jumps)
    delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend = calc_lambda3_features_v2(data)
    # Fit Bayesian regression model
    trace_l3_v2 = fit_l3_bayesian_regression_v2(
        data, delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend
    )

    # Plot posterior distributions for all coefficients
    plot_posterior(trace_l3_v2)

    # Plot mean prediction with detected jump events
    plot_l3_prediction_v2(data, delta_LambdaC_pos, delta_LambdaC_neg, rho_T, time_trend, trace_l3_v2)
