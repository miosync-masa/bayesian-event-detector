import pymc as pm
import numpy as np
import arviz as az

# ---- Generate Dummy Data ----
# Smooth trend + 3 spike-like jump events
np.random.seed(42)
T = 150
trend_data = 0.05 * np.arange(T) + np.sin(np.arange(T) * 0.2)
jumps = np.zeros(T)
jumps[40] = 5.0     # Spike event 1 (up)
jumps[85] = -6.0    # Spike event 2 (down)
jumps[120] = 4.0    # Spike event 3 (up)
noise = np.random.randn(T) * 0.5
data = trend_data + jumps + noise
# -----------------------------

with pm.Model() as historical_model:
    # 1. Prior for the probability of event occurrence (expected ~3 events in T)
    p_event = pm.Beta('p_event', alpha=1., beta=(T/3)-1)
    
    # 2. Event indicator (0/1) for each time point
    event_indicator = pm.Bernoulli('event_indicator', p=p_event, shape=len(data))
    
    # 3. Jump magnitudes (only matter if event occurs)
    jump_magnitudes = pm.Normal('jump_magnitudes', mu=0, sigma=5, shape=len(data))
    
    # Extract jump effect only when event occurs
    jump_effect = event_indicator * jump_magnitudes
    
    # 4. Smooth underlying trend (Gaussian Random Walk)
    # Making the prior sigma small encourages sudden changes to be explained by jumps
    trend_sigma = pm.HalfNormal('trend_sigma', sigma=0.1)
    trend = pm.GaussianRandomWalk('trend', sigma=trend_sigma, shape=len(data))
    
    # 5. Combine trend and jump effect
    mu = trend + jump_effect
    
    # Observation noise
    sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
    
    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=data)
    
    # Run MCMC sampling
    trace = pm.sample(2000, tune=2000, target_accept=0.95)

# Visualization (optional)
# az.plot_posterior(trace, var_names=['p_event', 'trend_sigma'])
# az.plot_trace(trace, var_names=['p_event'])
