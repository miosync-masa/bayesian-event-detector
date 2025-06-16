import pymc as pm
import numpy as np
import arviz as az

# (Optional) Print version info for reproducibility/debugging
import pymc
import pytensor
print("PyMC version:", pymc.__version__)
print("pytensor version:", pytensor.__version__)

# ---- Generate dummy data: smooth trend + 3 spike (jump) events ----
np.random.seed(42)
T = 150
trend_data = 0.05 * np.arange(T) + np.sin(np.arange(T) * 0.2)
jumps = np.zeros(T)
jumps[40] = 5.0    # Spike event (up)
jumps[85] = -6.0   # Spike event (down)
jumps[120] = 4.0   # Spike event (up)
noise = np.random.randn(T) * 0.5
data = trend_data + jumps + noise
# -------------------------------------------------------------------

with pm.Model() as historical_model_v2:
    # 1. Prior for event occurrence probability (expecting ~3 events in T samples)
    p_event = pm.Beta('p_event', alpha=1., beta=(T/3)-1)

    # 2. Event indicator (0/1) for each time point
    event_indicator = pm.Bernoulli('event_indicator', p=p_event, shape=len(data))

    # 3. Magnitude of jump at each time point (zero unless event occurs)
    jump_magnitudes = pm.Normal('jump_magnitudes', mu=0, sigma=5, shape=len(data))
    jump_effect = event_indicator * jump_magnitudes

    # 4. Smooth underlying trend modeled as a random walk (via cumulative sum)
    trend_sigma = pm.HalfNormal('trend_sigma', sigma=0.1)  # Controls smoothness
    # Each time step's difference (increment) follows a normal distribution
    trend_steps = pm.Normal('trend_steps', mu=0, sigma=trend_sigma, shape=len(data))
    # Cumulative sum to get the overall trend trajectory
    trend = pm.Deterministic('trend', trend_steps.cumsum())

    # 5. Combine trend and jump effect to get the mean of the observation model
    mu = trend + jump_effect

    # Observation noise
    sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)

    # Likelihood (data model)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=data)

    # Run MCMC sampling
    trace = pm.sample(2000, tune=2000, target_accept=0.95)

# Visualization example (uncomment for interactive plots)
# az.plot_posterior(trace, var_names=['p_event', 'trend_sigma'])
# az.plot_trace(trace, var_names=['p_event'])
