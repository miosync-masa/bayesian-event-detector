"""
Bayesian inference module for Lambda³ framework using NumPyro.

This module implements various Bayesian models for analyzing
structural evolution and interactions in time series.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import arviz as az
import numpy as np
from typing import Dict, Optional, Any, List, Tuple, Union

from .config import L3Config, BayesianConfig
from .types import Lambda3FeatureSet, BayesianResults

# ===============================
# NumPyro Model Definitions
# ===============================

def lambda3_base_model(
    features_dict: Dict[str, jnp.ndarray],
    y_obs: Optional[jnp.ndarray] = None,
    change_points: Optional[List[int]] = None,
    prior_scales: Optional[Dict[str, float]] = None
):
    """
    Base Lambda³ Bayesian model without interactions.
    
    This model captures:
    - Linear time trend (ΛF: progression vector)
    - Jump event effects (ΔΛC±: structural changes)
    - Tension scalar influence (ρT: local volatility)
    - Local jump effects (local ΔΛC pulsations)
    
    Args:
        features_dict: Dictionary of feature arrays
        y_obs: Observed data
        change_points: Known structural change points
        prior_scales: Custom prior scales for parameters
    """
    # Default prior scales
    if prior_scales is None:
        prior_scales = {
            'innovation_scale': 0.1,
            'beta_dLC_pos': 5.0,
            'beta_dLC_neg': 5.0,
            'beta_rhoT': 3.0,
            'beta_local_jump': 2.0,
            'sigma_base': 1.0,
            'sigma_scale': 0.5
        }
    
    # Get data length
    n = len(y_obs) if y_obs is not None else len(features_dict['time_trend'])
    time_idx = jnp.arange(n)
    
    # Time-varying parameter using Gaussian Random Walk
    innovation_scale = numpyro.sample('innovation_scale', 
                                    dist.HalfNormal(prior_scales['innovation_scale']))
    beta_time_series = numpyro.sample(
        'beta_time_series',
        dist.GaussianRandomWalk(innovation_scale, num_steps=n)
    )
    
    # Static coefficients for Lambda³ features
    beta_dLC_pos = numpyro.sample('beta_dLC_pos', 
                                 dist.Normal(0.0, prior_scales['beta_dLC_pos']))
    beta_dLC_neg = numpyro.sample('beta_dLC_neg', 
                                 dist.Normal(0.0, prior_scales['beta_dLC_neg']))
    beta_rhoT = numpyro.sample('beta_rhoT', 
                               dist.Normal(0.0, prior_scales['beta_rhoT']))
    
    # Build mean function with all structural components
    mu = (
        beta_time_series
        + beta_dLC_pos * features_dict['delta_LambdaC_pos']
        + beta_dLC_neg * features_dict['delta_LambdaC_neg']
        + beta_rhoT * features_dict['rho_T']
    )
    
    # Add local jump effects (local ΔΛC pulsations)
    if 'local_jump' in features_dict:
        beta_local_jump = numpyro.sample('beta_local_jump', 
                                       dist.Normal(0.0, prior_scales['beta_local_jump']))
        mu = mu + beta_local_jump * features_dict['local_jump']
    
    # Add structural change jumps
    if change_points:
        for i, cp in enumerate(change_points):
            jump = numpyro.sample(f'jump_{i}', dist.Normal(0.0, 5.0))
            # Step function at change point
            mu = mu + jump * (time_idx >= cp)
    
    # Time-varying volatility dependent on tension scalar (ρT)
    sigma_base = numpyro.sample('sigma_base', 
                               dist.HalfNormal(prior_scales['sigma_base']))
    sigma_scale = numpyro.sample('sigma_scale', 
                                dist.HalfNormal(prior_scales['sigma_scale']))
    sigma_obs = sigma_base + sigma_scale * features_dict['rho_T']
    
    with numpyro.plate('observations', n):
        numpyro.sample('y_obs', dist.Normal(mu, sigma_obs), obs=y_obs)


def lambda3_interaction_model(
    features_dict: Dict[str, jnp.ndarray],
    interaction_pos: Optional[jnp.ndarray] = None,
    interaction_neg: Optional[jnp.ndarray] = None,
    interaction_rhoT: Optional[jnp.ndarray] = None,
    y_obs: Optional[jnp.ndarray] = None,
    prior_scales: Optional[Dict[str, float]] = None
):
    """
    Lambda³ model with asymmetric cross-series interactions.
    
    This model extends the base model by including:
    - Cross-series positive jump interactions
    - Cross-series negative jump interactions
    - Cross-series tension interactions
    
    Args:
        features_dict: Dictionary of feature arrays
        interaction_pos: Positive jumps from interacting series
        interaction_neg: Negative jumps from interacting series
        interaction_rhoT: Tension from interacting series
        y_obs: Observed data
        prior_scales: Custom prior scales
    """
    # Default prior scales
    if prior_scales is None:
        prior_scales = {
            'beta_0': 2.0,
            'beta_time': 1.0,
            'beta_dLC_pos': 5.0,
            'beta_dLC_neg': 5.0,
            'beta_rhoT': 3.0,
            'beta_interact': 3.0,
            'beta_local_jump': 2.0,
            'sigma_obs': 1.0
        }
    
    # Extract features with validation
    required_keys = ['time_trend', 'delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T']
    for key in required_keys:
        if key not in features_dict:
            raise KeyError(f"Required feature '{key}' not found in features_dict")
    
    time_trend = features_dict['time_trend']
    delta_pos = features_dict['delta_LambdaC_pos']
    delta_neg = features_dict['delta_LambdaC_neg']
    rho_t = features_dict['rho_T']
    
    # Optional features
    local_jump = features_dict.get('local_jump', None)
    
    # Base parameters
    beta_0 = numpyro.sample('beta_0', dist.Normal(0.0, prior_scales['beta_0']))
    beta_time = numpyro.sample('beta_time', dist.Normal(0.0, prior_scales['beta_time']))
    beta_dLC_pos = numpyro.sample('beta_dLC_pos', dist.Normal(0.0, prior_scales['beta_dLC_pos']))
    beta_dLC_neg = numpyro.sample('beta_dLC_neg', dist.Normal(0.0, prior_scales['beta_dLC_neg']))
    beta_rhoT = numpyro.sample('beta_rhoT', dist.Normal(0.0, prior_scales['beta_rhoT']))
    
    # Base model
    mu = (
        beta_0
        + beta_time * time_trend
        + beta_dLC_pos * delta_pos
        + beta_dLC_neg * delta_neg
        + beta_rhoT * rho_t
    )
    
    # Add local jump effects if available
    if local_jump is not None:
        beta_local_jump = numpyro.sample(
            'beta_local_jump',
            dist.Normal(0.0, prior_scales.get('beta_local_jump', 2.0))
        )
        mu = mu + beta_local_jump * local_jump
    
    # Add interaction terms
    if interaction_pos is not None:
        beta_interact_pos = numpyro.sample(
            'beta_interact_pos', 
            dist.Normal(0.0, prior_scales['beta_interact'])
        )
        mu = mu + beta_interact_pos * interaction_pos
    
    if interaction_neg is not None:
        beta_interact_neg = numpyro.sample(
            'beta_interact_neg',
            dist.Normal(0.0, prior_scales['beta_interact'])
        )
        mu = mu + beta_interact_neg * interaction_neg
    
    if interaction_rhoT is not None:
        beta_interact_stress = numpyro.sample(
            'beta_interact_stress',
            dist.Normal(0.0, prior_scales['beta_interact'] * 0.67)  # Slightly tighter prior
        )
        mu = mu + beta_interact_stress * interaction_rhoT
    
    # Observation model
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(prior_scales['sigma_obs']))
    
    with numpyro.plate('observations', len(mu)):
        numpyro.sample('y_obs', dist.Normal(mu, sigma_obs), obs=y_obs)


def lambda3_dynamic_model(
    features_dict: Dict[str, jnp.ndarray],
    change_points: Optional[List[int]] = None,
    y_obs: Optional[jnp.ndarray] = None,
    prior_scales: Optional[Dict[str, float]] = None
):
    """
    Dynamic Lambda³ model with time-varying parameters.
    
    This model includes:
    - Time-varying coefficients (Gaussian Random Walk)
    - Structural change points with jump effects
    - Adaptive volatility based on tension scalar
    - All Lambda³ features including local jumps
    
    Args:
        features_dict: Dictionary of feature arrays
        change_points: Known structural change points
        y_obs: Observed data
        prior_scales: Custom prior scales
    """
    # Default prior scales
    if prior_scales is None:
        prior_scales = {
            'innovation_scale': 0.1,
            'beta_dLC_pos': 5.0,
            'beta_dLC_neg': 5.0,
            'beta_rhoT': 3.0,
            'beta_local_jump': 2.0,
            'sigma_base': 1.0,
            'sigma_scale': 0.5
        }
    
    n = len(y_obs) if y_obs is not None else len(features_dict['time_trend'])
    
    # Time-varying baseline (Gaussian Random Walk)
    innovation_scale = numpyro.sample('innovation_scale', 
                                    dist.HalfNormal(prior_scales['innovation_scale']))
    beta_time_series = numpyro.sample(
        'beta_time_series',
        dist.GaussianRandomWalk(innovation_scale, num_steps=n)
    )
    
    # Static coefficients for structural features
    beta_dLC_pos = numpyro.sample('beta_dLC_pos', 
                                 dist.Normal(0.0, prior_scales['beta_dLC_pos']))
    beta_dLC_neg = numpyro.sample('beta_dLC_neg', 
                                 dist.Normal(0.0, prior_scales['beta_dLC_neg']))
    beta_rhoT = numpyro.sample('beta_rhoT', 
                               dist.Normal(0.0, prior_scales['beta_rhoT']))
    
    # Build mean function with time-varying baseline
    mu = (
        beta_time_series
        + beta_dLC_pos * features_dict['delta_LambdaC_pos']
        + beta_dLC_neg * features_dict['delta_LambdaC_neg']
        + beta_rhoT * features_dict['rho_T']
    )
    
    # Add local jump effects
    if 'local_jump' in features_dict:
        beta_local_jump = numpyro.sample('beta_local_jump', 
                                       dist.Normal(0.0, prior_scales['beta_local_jump']))
        mu = mu + beta_local_jump * features_dict['local_jump']
    
    # Add structural change jumps
    if change_points:
        time_idx = jnp.arange(n)
        for i, cp in enumerate(change_points):
            jump = numpyro.sample(f'jump_{i}', dist.Normal(0.0, 5.0))
            # Step function at change point
            mu = mu + jump * (time_idx >= cp)
    
    # Time-varying volatility dependent on tension scalar
    sigma_base = numpyro.sample('sigma_base', 
                               dist.HalfNormal(prior_scales['sigma_base']))
    sigma_scale = numpyro.sample('sigma_scale', 
                                dist.HalfNormal(prior_scales['sigma_scale']))
    sigma_obs = sigma_base + sigma_scale * features_dict['rho_T']
    
    with numpyro.plate('observations', n):
        numpyro.sample('y_obs', dist.Normal(mu, sigma_obs), obs=y_obs)


def lambda3_hierarchical_model(
    features_list: List[Dict[str, jnp.ndarray]],
    y_obs_list: List[jnp.ndarray],
    group_ids: Optional[List[int]] = None
):
    """
    Hierarchical Lambda³ model for multiple related series.
    
    This model captures:
    - Group-level parameters with partial pooling
    - Series-specific deviations
    - Shared structural patterns
    - Full Lambda³ feature integration
    
    Args:
        features_list: List of feature dictionaries for each series
        y_obs_list: List of observed data for each series
        group_ids: Optional group assignments for series
    """
    n_series = len(features_list)
    
    # Validate all feature dictionaries
    required_keys = ['time_trend', 'delta_LambdaC_pos', 'delta_LambdaC_neg', 'rho_T']
    for i, features in enumerate(features_list):
        for key in required_keys:
            if key not in features:
                raise KeyError(f"Required feature '{key}' not found in features_list[{i}]")
    
    # Hyperpriors for population-level parameters
    mu_beta_time = numpyro.sample('mu_beta_time', dist.Normal(0.0, 1.0))
    sigma_beta_time = numpyro.sample('sigma_beta_time', dist.HalfNormal(0.5))
    
    mu_beta_dLC_pos = numpyro.sample('mu_beta_dLC_pos', dist.Normal(0.0, 3.0))
    sigma_beta_dLC_pos = numpyro.sample('sigma_beta_dLC_pos', dist.HalfNormal(1.0))
    
    mu_beta_dLC_neg = numpyro.sample('mu_beta_dLC_neg', dist.Normal(0.0, 3.0))
    sigma_beta_dLC_neg = numpyro.sample('sigma_beta_dLC_neg', dist.HalfNormal(1.0))
    
    mu_beta_rhoT = numpyro.sample('mu_beta_rhoT', dist.Normal(0.0, 2.0))
    sigma_beta_rhoT = numpyro.sample('sigma_beta_rhoT', dist.HalfNormal(0.5))
    
    # Series-level parameters
    with numpyro.plate('series', n_series):
        # Sample series-specific parameters
        beta_0 = numpyro.sample('beta_0', dist.Normal(0.0, 2.0))
        beta_time = numpyro.sample('beta_time', dist.Normal(mu_beta_time, sigma_beta_time))
        beta_dLC_pos = numpyro.sample('beta_dLC_pos', dist.Normal(mu_beta_dLC_pos, sigma_beta_dLC_pos))
        beta_dLC_neg = numpyro.sample('beta_dLC_neg', dist.Normal(mu_beta_dLC_neg, sigma_beta_dLC_neg))
        beta_rhoT = numpyro.sample('beta_rhoT', dist.Normal(mu_beta_rhoT, sigma_beta_rhoT))
        sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(1.0))
    
    # Hyperprior for local jump effects
    mu_beta_local = numpyro.sample('mu_beta_local', dist.Normal(0.0, 2.0))
    sigma_beta_local = numpyro.sample('sigma_beta_local', dist.HalfNormal(0.5))
    
    # Likelihood for each series
    for i in range(n_series):
        features = features_list[i]
        y_obs = y_obs_list[i]
        
        mu = (
            beta_0[i]
            + beta_time[i] * features['time_trend']
            + beta_dLC_pos[i] * features['delta_LambdaC_pos']
            + beta_dLC_neg[i] * features['delta_LambdaC_neg']
            + beta_rhoT[i] * features['rho_T']
        )
        
        # Add local jump effects if available
        if 'local_jump' in features:
            beta_local_i = numpyro.sample(f'beta_local_{i}', 
                                        dist.Normal(mu_beta_local, sigma_beta_local))
            mu = mu + beta_local_i * features['local_jump']
        
        with numpyro.plate(f'obs_{i}', len(mu)):
            numpyro.sample(f'y_obs_{i}', dist.Normal(mu, sigma_obs[i]), obs=y_obs)


# ===============================
# Model Fitting Functions
# ===============================

def fit_bayesian_model(
    features: Lambda3FeatureSet,
    config: Union[L3Config, BayesianConfig],
    interaction_features: Optional[Lambda3FeatureSet] = None,
    model_type: str = 'interaction',
    seed: int = 0,
    additional_params: Optional[Dict[str, Any]] = None
) -> BayesianResults:
    """
    Fit a Bayesian model to Lambda³ features.
    
    Args:
        features: Primary series features
        config: Configuration object
        interaction_features: Features from interacting series
        model_type: 'base', 'interaction', or 'dynamic'
        seed: Random seed for reproducibility
        additional_params: Extra parameters for specific models
        
    Returns:
        BayesianResults: Fitted model results
    """
    # Extract config
    if isinstance(config, L3Config):
        bayes_config = config.bayesian
    else:
        bayes_config = config
    
    # Get prior scales with safe extraction
    prior_scales = getattr(bayes_config, 'prior_scales', {
        'beta_0': 2.0,
        'beta_time': 1.0,
        'beta_dLC_pos': 5.0,
        'beta_dLC_neg': 5.0,
        'beta_rhoT': 3.0,
        'beta_interact': 3.0,
        'sigma_obs': 1.0
    })
    
    # Convert features to JAX arrays (64-bit enabled)
    data_jax = jnp.asarray(features.data)
    
    # Create features dictionary without raw data
    features_jax = {
        'time_trend': jnp.asarray(features.time_trend),
        'delta_LambdaC_pos': jnp.asarray(features.delta_LambdaC_pos),
        'delta_LambdaC_neg': jnp.asarray(features.delta_LambdaC_neg),
        'rho_T': jnp.asarray(features.rho_T),
        'local_jump': jnp.asarray(features.local_jump)
    }
    
    # Prepare interaction terms
    interaction_pos = None
    interaction_neg = None
    interaction_rhoT = None
    
    if interaction_features and model_type == 'interaction':
        interaction_pos = jnp.asarray(interaction_features.delta_LambdaC_pos)
        interaction_neg = jnp.asarray(interaction_features.delta_LambdaC_neg)
        interaction_rhoT = jnp.asarray(interaction_features.rho_T)
    
    # Select model
    if model_type == 'base':
        model = lambda3_base_model
        model_args = (features_jax,)
        model_kwargs = {
            'y_obs': data_jax,
            'change_points': additional_params.get('change_points') if additional_params else None,
            'prior_scales': prior_scales
        }
    elif model_type == 'interaction':
        model = lambda3_interaction_model
        model_args = (features_jax,)
        model_kwargs = {
            'interaction_pos': interaction_pos,
            'interaction_neg': interaction_neg,
            'interaction_rhoT': interaction_rhoT,
            'y_obs': data_jax,
            'prior_scales': prior_scales
        }
    elif model_type == 'dynamic':
        model = lambda3_dynamic_model
        model_args = (features_jax,)
        model_kwargs = {
            'change_points': additional_params.get('change_points') if additional_params else None,
            'y_obs': data_jax,
            'prior_scales': prior_scales
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # MCMC setup with proper key splitting for multiple chains
    rng_key = random.PRNGKey(seed)
    
    kernel = NUTS(
        model,
        target_accept_prob=bayes_config.target_accept,
        max_tree_depth=bayes_config.max_treedepth
    )
    
    mcmc = MCMC(
        kernel,
        num_warmup=bayes_config.tune,
        num_samples=bayes_config.draws,
        num_chains=bayes_config.num_chains,
        chain_method='parallel' if bayes_config.num_chains > 1 else 'sequential',
        progress_bar=True,
        jit_model_args=True
    )
    
    # Run MCMC with proper key handling
    print(f"Running MCMC with {bayes_config.num_chains} chains, "
          f"{bayes_config.draws} samples each...")
    
    if bayes_config.num_chains > 1:
        # Split keys for parallel chains
        rng_keys = random.split(rng_key, bayes_config.num_chains)
        mcmc.run(rng_keys, *model_args, **model_kwargs)
    else:
        mcmc.run(rng_key, *model_args, **model_kwargs)
    
    # Convert to ArviZ
    trace = _convert_to_arviz(mcmc, bayes_config)
    
    # Get predictions
    predictions = predict_with_model(
        trace, features, interaction_features, model_type
    )
    
    # Calculate residuals
    residuals = features.data - predictions
    
    # Extract diagnostics
    diagnostics = _extract_diagnostics(mcmc, trace)
    
    # Create summary
    summary = az.summary(trace)
    
    return BayesianResults(
        trace=trace,
        summary=summary,
        predictions=predictions,
        residuals=residuals,
        diagnostics=diagnostics
    )


def fit_dynamic_model(
    features: Lambda3FeatureSet,
    config: Union[L3Config, BayesianConfig],
    change_points: Optional[List[int]] = None,
    seed: int = 0
) -> BayesianResults:
    """
    Fit dynamic Lambda³ model with time-varying parameters.
    
    Args:
        features: Time series features
        config: Configuration
        change_points: Known change points
        seed: Random seed
        
    Returns:
        BayesianResults: Model results
    """
    # Auto-detect change points if not provided
    if change_points is None:
        change_points = detect_change_points_automatic(features.data)
        print(f"Auto-detected {len(change_points)} change points: {change_points}")
    
    return fit_bayesian_model(
        features=features,
        config=config,
        model_type='dynamic',
        seed=seed,
        additional_params={
            'change_points': change_points
        }
    )


def fit_hierarchical_model(
    features_list: List[Lambda3FeatureSet],
    config: Union[L3Config, BayesianConfig],
    group_ids: Optional[List[int]] = None,
    seed: int = 0
) -> BayesianResults:
    """
    Fit hierarchical Lambda³ model for multiple related series.
    
    Args:
        features_list: List of Lambda³ features for each series
        config: Configuration
        group_ids: Optional group assignments
        seed: Random seed
        
    Returns:
        BayesianResults: Hierarchical model results
    """
    # Extract config
    if isinstance(config, L3Config):
        bayes_config = config.bayesian
    else:
        bayes_config = config
    
    # Convert features to JAX format
    n_series = len(features_list)
    y_obs_list = [jnp.asarray(feat.data) for feat in features_list]
    features_jax_list = []
    
    for feat in features_list:
        features_jax_list.append({
            'time_trend': jnp.asarray(feat.time_trend),
            'delta_LambdaC_pos': jnp.asarray(feat.delta_LambdaC_pos),
            'delta_LambdaC_neg': jnp.asarray(feat.delta_LambdaC_neg),
            'rho_T': jnp.asarray(feat.rho_T),
            'local_jump': jnp.asarray(feat.local_jump)
        })
    
    # Set up group IDs
    if group_ids is None:
        group_ids = list(range(n_series))
    
    # MCMC setup
    rng_key = random.PRNGKey(seed)
    kernel = NUTS(
        lambda3_hierarchical_model,
        target_accept_prob=bayes_config.target_accept,
        max_tree_depth=bayes_config.max_treedepth
    )
    
    mcmc = MCMC(
        kernel,
        num_warmup=bayes_config.tune,
        num_samples=bayes_config.draws,
        num_chains=bayes_config.num_chains,
        chain_method='parallel' if bayes_config.num_chains > 1 else 'sequential',
        progress_bar=True
    )
    
    print(f"Running hierarchical MCMC for {n_series} series...")
    
    # Run with proper key handling
    if bayes_config.num_chains > 1:
        rng_keys = random.split(rng_key, bayes_config.num_chains)
        mcmc.run(rng_keys, features_jax_list, y_obs_list, group_ids)
    else:
        mcmc.run(rng_key, features_jax_list, y_obs_list, group_ids)
    
    # Convert to ArviZ
    trace = _convert_to_arviz(mcmc, bayes_config)
    
    # Generate predictions for each series (vectorized)
    posterior = trace.posterior
    
    # Extract mean parameters efficiently
    beta_0_mean = posterior['beta_0'].mean(dim=['chain', 'draw']).values
    beta_time_mean = posterior['beta_time'].mean(dim=['chain', 'draw']).values
    beta_dLC_pos_mean = posterior['beta_dLC_pos'].mean(dim=['chain', 'draw']).values
    beta_dLC_neg_mean = posterior['beta_dLC_neg'].mean(dim=['chain', 'draw']).values
    beta_rhoT_mean = posterior['beta_rhoT'].mean(dim=['chain', 'draw']).values
    
    predictions_list = []
    for i in range(n_series):
        mu = (
            beta_0_mean[i] +
            beta_time_mean[i] * features_list[i].time_trend +
            beta_dLC_pos_mean[i] * features_list[i].delta_LambdaC_pos +
            beta_dLC_neg_mean[i] * features_list[i].delta_LambdaC_neg +
            beta_rhoT_mean[i] * features_list[i].rho_T
        )
        
        # Add local jump effect if available
        if f'beta_local_{i}' in posterior:
            beta_local_mean = posterior[f'beta_local_{i}'].mean(dim=['chain', 'draw']).item()
            mu = mu + beta_local_mean * features_list[i].local_jump
        
        predictions_list.append(np.asarray(mu))
    
    # Extract diagnostics
    diagnostics = _extract_diagnostics(mcmc, trace)
    diagnostics['n_series'] = n_series
    
    # Create summary
    summary = az.summary(trace)
    
    # Return results (storing list of predictions)
    return BayesianResults(
        trace=trace,
        summary=summary,
        predictions=np.array(predictions_list),  # Shape: (n_series, n_time)
        residuals=None,  # Complex for hierarchical
        diagnostics=diagnostics
    )


# ===============================
# Change Point Detection
# ===============================

def detect_change_points_automatic(
    data: np.ndarray,
    window_size: int = 50,
    threshold_factor: float = 2.0
) -> List[int]:
    """
    Automatic change point detection using PELT-like algorithm.
    
    Args:
        data: Time series data
        window_size: Detection window size
        threshold_factor: Threshold coefficient
        
    Returns:
        List of detected change points
    """
    n = len(data)
    change_points = []
    
    # Calculate rolling statistics
    rolling_mean = np.array([
        np.mean(data[max(0, i-window_size//2):min(n, i+window_size//2+1)])
        for i in range(n)
    ])
    
    rolling_std = np.array([
        np.std(data[max(0, i-window_size//2):min(n, i+window_size//2+1)])
        for i in range(n)
    ])
    
    # Calculate change scores
    mean_diff = np.abs(np.diff(rolling_mean))
    std_diff = np.abs(np.diff(rolling_std))
    
    # Normalize
    mean_score = mean_diff / (np.std(mean_diff) + 1e-8)
    std_score = std_diff / (np.std(std_diff) + 1e-8)
    
    # Combined score
    change_score = mean_score + std_score
    
    # Detect points exceeding threshold
    threshold = threshold_factor * np.std(change_score)
    candidates = np.where(change_score > threshold)[0]
    
    # Merge nearby change points
    if len(candidates) > 0:
        change_points = [int(candidates[0])]
        for cp in candidates[1:]:
            if cp - change_points[-1] > window_size:
                change_points.append(int(cp))
    
    return change_points


# ===============================
# Model Comparison
# ===============================

def compare_models(
    models_dict: Dict[str, BayesianResults],
    features: Lambda3FeatureSet,
    criterion: str = 'loo'
) -> Dict[str, Any]:
    """
    Compare multiple models using LOO-CV or WAIC.
    
    Args:
        models_dict: Dictionary of model names and results
        features: Lambda³ features
        criterion: 'loo' or 'waic'
        
    Returns:
        Comparison results
    """
    comparison_results = {}
    inference_data_dict = {}
    
    for model_name, results in models_dict.items():
        print(f"\nProcessing {model_name}...")
        
        # Get log likelihood
        log_likelihood = calculate_log_likelihood(
            results, features, model_name
        )
        
        # Create InferenceData with log likelihood
        if hasattr(results.trace, 'posterior'):
            # Already InferenceData
            inference_data = results.trace
            # Add log likelihood if not present
            if not hasattr(inference_data, 'log_likelihood'):
                inference_data.log_likelihood = az.from_dict(
                    {'y': log_likelihood}
                ).log_likelihood
        else:
            # Convert to InferenceData
            inference_data = az.from_dict(
                posterior=results.trace,
                log_likelihood={'y': log_likelihood},
                observed_data={'y': features.data}
            )
        
        inference_data_dict[model_name] = inference_data
        
        # Calculate information criterion
        try:
            if criterion == 'loo':
                ic = az.loo(inference_data)
                comparison_results[f'{model_name}_loo'] = {
                    'loo_estimate': float(ic.elpd_loo),
                    'loo_se': float(ic.se),
                    'p_loo': float(ic.p_loo)
                }
            else:  # waic
                ic = az.waic(inference_data)
                comparison_results[f'{model_name}_waic'] = {
                    'waic': float(ic.waic),
                    'waic_se': float(ic.waic_se),
                    'p_waic': float(ic.p_waic)
                }
        except Exception as e:
            print(f"{criterion.upper()} calculation failed for {model_name}: {e}")
    
    # Model comparison table
    if len(inference_data_dict) > 1:
        try:
            compare_df = az.compare(inference_data_dict)
            comparison_results['comparison_table'] = compare_df
            print(f"\nModel Comparison Results ({criterion.upper()}):")
            print(compare_df)
            
            # Best model
            best_model = compare_df.index[0]
            comparison_results['best_model'] = best_model
            print(f"\nBest model: {best_model}")
        except Exception as e:
            print(f"Model comparison failed: {e}")
    
    return comparison_results


def calculate_log_likelihood(
    results: BayesianResults,
    features: Lambda3FeatureSet,
    model_type: str
) -> np.ndarray:
    """
    Calculate log likelihood for model comparison.
    
    Args:
        results: Bayesian results
        features: Lambda³ features
        model_type: Type of model
        
    Returns:
        Log likelihood array
    """
    from scipy.stats import norm
    
    trace = results.trace
    posterior = trace.posterior
    
    # Get dimensions
    n_chains = posterior.dims['chain']
    n_draws = posterior.dims['draw']
    n_obs = len(features.data)
    
    # Initialize log likelihood array
    log_lik = np.zeros((n_chains, n_draws, n_obs))
    
    # Calculate for each sample
    for chain in range(n_chains):
        for draw in range(n_draws):
            # Use unified parameter names
            mu = (
                posterior['beta_0'][chain, draw].item() +
                posterior['beta_time'][chain, draw].item() * features.time_trend +
                posterior['beta_dLC_pos'][chain, draw].item() * features.delta_LambdaC_pos +
                posterior['beta_dLC_neg'][chain, draw].item() * features.delta_LambdaC_neg +
                posterior['beta_rhoT'][chain, draw].item() * features.rho_T
            )
            
            # Add local jump if present
            if 'beta_local_jump' in posterior:
                mu += posterior['beta_local_jump'][chain, draw].item() * features.local_jump
            
            # Handle dynamic model
            if model_type == 'dynamic' and 'beta_time_series' in posterior:
                # For dynamic model, use time-varying component
                mu = posterior['beta_time_series'][chain, draw].values
                # Add other components
                if 'beta_dLC_pos' in posterior:
                    mu = mu + posterior['beta_dLC_pos'][chain, draw].item() * features.delta_LambdaC_pos
                if 'beta_dLC_neg' in posterior:
                    mu = mu + posterior['beta_dLC_neg'][chain, draw].item() * features.delta_LambdaC_neg
                if 'beta_rhoT' in posterior:
                    mu = mu + posterior['beta_rhoT'][chain, draw].item() * features.rho_T
                if 'beta_local_jump' in posterior:
                    mu = mu + posterior['beta_local_jump'][chain, draw].item() * features.local_jump
            
            # Get observation variance
            sigma = posterior['sigma_obs'][chain, draw].item()
            
            # Calculate log likelihood
            log_lik[chain, draw] = norm.logpdf(features.data, mu, sigma)
    
    return log_lik


# ===============================
# Posterior Predictive Checks
# ===============================

def posterior_predictive_check(
    results: BayesianResults,
    features: Lambda3FeatureSet,
    n_samples: int = 500
) -> Dict[str, Any]:
    """
    Perform posterior predictive checks.
    
    Args:
        results: Bayesian results
        features: Lambda³ features
        n_samples: Number of posterior samples to use
        
    Returns:
        PPC results dictionary
    """
    trace = results.trace
    posterior = trace.posterior
    
    # Sample from posterior
    n_chains = posterior.dims['chain']
    n_draws = posterior.dims['draw']
    total_samples = n_chains * n_draws
    
    # Randomly select samples
    if total_samples > n_samples:
        sample_indices = np.random.choice(total_samples, n_samples, replace=False)
    else:
        sample_indices = np.arange(total_samples)
    
    # Generate posterior predictive samples
    n_obs = len(features.data)
    ppc_samples = np.zeros((len(sample_indices), n_obs))
    
    for i, idx in enumerate(sample_indices):
        chain = idx // n_draws
        draw = idx % n_draws
        
        # Reconstruct mu using actual predictions
        mu = results.predictions
        
        # Add noise
        sigma = posterior['sigma_obs'][chain, draw].item()
        ppc_samples[i] = np.random.normal(mu, sigma)
    
    # Calculate test statistics
    test_statistics = {
        'mean': np.mean(ppc_samples, axis=1),
        'std': np.std(ppc_samples, axis=1),
        'min': np.min(ppc_samples, axis=1),
        'max': np.max(ppc_samples, axis=1),
        'median': np.median(ppc_samples, axis=1)
    }
    
    # Observed statistics
    observed_stats = {
        'mean': np.mean(features.data),
        'std': np.std(features.data),
        'min': np.min(features.data),
        'max': np.max(features.data),
        'median': np.median(features.data)
    }
    
    # Bayesian p-values
    bayesian_p_values = {}
    for stat_name in test_statistics:
        ppc_stat = test_statistics[stat_name]
        obs_stat = observed_stats[stat_name]
        p_value = np.mean(ppc_stat >= obs_stat)
        bayesian_p_values[stat_name] = float(p_value)
    
    return {
        'ppc_samples': ppc_samples,
        'test_statistics': test_statistics,
        'observed_stats': observed_stats,
        'bayesian_p_values': bayesian_p_values
    }


# ===============================
# Variational Inference (SVI)
# ===============================

def fit_svi_model(
    features: Lambda3FeatureSet,
    config: Union[L3Config, BayesianConfig],
    model_type: str = 'base',
    n_steps: int = 10000,
    learning_rate: float = 0.01,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Fit model using Stochastic Variational Inference for faster inference.
    
    Args:
        features: Lambda³ features
        config: Configuration
        model_type: Model type
        n_steps: Number of optimization steps
        learning_rate: Learning rate for optimizer
        seed: Random seed
        
    Returns:
        SVI results
    """
    from numpyro.optim import Adam
    
    # Extract config
    if isinstance(config, L3Config):
        bayes_config = config.bayesian
    else:
        bayes_config = config
    
    # Prepare data
    data_jax = jnp.asarray(features.data)
    features_jax = {
        'time_trend': jnp.asarray(features.time_trend),
        'delta_LambdaC_pos': jnp.asarray(features.delta_LambdaC_pos),
        'delta_LambdaC_neg': jnp.asarray(features.delta_LambdaC_neg),
        'rho_T': jnp.asarray(features.rho_T),
        'local_jump': jnp.asarray(features.local_jump)
    }
    
    # Select model
    if model_type == 'base':
        model = lambda3_base_model
        model_kwargs = {
            'prior_scales': bayes_config.prior_scales
        }
    else:
        raise ValueError(f"SVI not implemented for model type: {model_type}")
    
    # Create autoguide
    guide = AutoNormal(model)
    
    # Set up SVI
    optimizer = Adam(learning_rate)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    # Initialize
    rng_key = random.PRNGKey(seed)
    svi_state = svi.init(rng_key, features_jax, y_obs=data_jax, **model_kwargs)
    
    # Run optimization
    print(f"Running SVI for {n_steps} steps...")
    losses = []
    
    for step in range(n_steps):
        svi_state, loss = svi.update(svi_state, features_jax, y_obs=data_jax, **model_kwargs)
        losses.append(loss)
        
        if step % 1000 == 0:
            print(f"Step {step}: loss = {loss:.4f}")
    
    # Get posterior samples
    params = svi.get_params(svi_state)
    posterior_samples = guide.sample_posterior(
        random.PRNGKey(seed + 1),
        params,
        sample_shape=(1000,)
    )
    
    # Calculate predictions
    mu = (
        posterior_samples['beta_0'].mean() +
        posterior_samples['beta_time'].mean() * features.time_trend +
        posterior_samples['beta_dLC_pos'].mean() * features.delta_LambdaC_pos +
        posterior_samples['beta_dLC_neg'].mean() * features.delta_LambdaC_neg +
        posterior_samples['beta_rhoT'].mean() * features.rho_T
    )
    
    if 'beta_local_jump' in posterior_samples:
        mu += posterior_samples['beta_local_jump'].mean() * features.local_jump
    
    predictions = np.asarray(mu)
    
    return {
        'posterior_samples': posterior_samples,
        'predictions': predictions,
        'losses': losses,
        'params': params,
        'guide': guide
    }


# ===============================
# Complete Bayesian Analysis Pipeline
# ===============================

def run_complete_bayesian_analysis(
    features: Lambda3FeatureSet,
    config: Optional[L3Config] = None,
    include_svi: bool = False,
    include_hierarchical: bool = False,
    additional_series: Optional[List[Lambda3FeatureSet]] = None
) -> Dict[str, Any]:
    """
    Run complete Bayesian analysis pipeline.
    
    This includes:
    1. Multiple model fitting
    2. Model comparison
    3. Posterior predictive checks
    4. Diagnostics
    5. Optional hierarchical analysis for multiple series
    
    Args:
        features: Lambda³ features
        config: Configuration
        include_svi: Whether to include SVI analysis
        include_hierarchical: Whether to run hierarchical model
        additional_series: Additional series for hierarchical model
        
    Returns:
        Complete analysis results
    """
    if config is None:
        config = L3Config()
    
    print("\n" + "="*60)
    print("COMPLETE BAYESIAN ANALYSIS PIPELINE")
    print("="*60)
    
    all_results = {}
    
    # 1. Fit base model
    print("\n1. Fitting base model...")
    base_results = fit_bayesian_model(
        features, config, model_type='base', seed=100
    )
    all_results['base'] = base_results
    check_convergence(base_results)
    
    # 2. Fit dynamic model
    print("\n2. Fitting dynamic model...")
    dynamic_results = fit_dynamic_model(
        features, config, seed=200
    )
    all_results['dynamic'] = dynamic_results
    check_convergence(dynamic_results)
    
    # 3. Hierarchical model (if multiple series provided)
    if include_hierarchical and additional_series:
        print("\n3. Fitting hierarchical model...")
        all_series = [features] + additional_series
        hierarchical_results = fit_hierarchical_model(
            all_series, config, seed=300
        )
        all_results['hierarchical'] = hierarchical_results
        check_convergence(hierarchical_results)
    
    # 4. Model comparison
    print("\n4. Comparing models...")
    comparison_results = compare_models(all_results, features)
    
    # 5. Posterior predictive checks for best model
    best_model = comparison_results.get('best_model', 'base')
    print(f"\n5. Running PPC for best model: {best_model}")
    ppc_results = posterior_predictive_check(
        all_results[best_model], features
    )
    
    print("\nBayesian p-values:")
    for stat, p_val in ppc_results['bayesian_p_values'].items():
        print(f"  {stat}: {p_val:.3f}")
    
    # 6. SVI analysis (optional)
    if include_svi:
        print("\n6. Running SVI for fast inference...")
        svi_results = fit_svi_model(features, config)
        all_results['svi'] = svi_results
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"✓ Models fitted: {list(all_results.keys())}")
    print(f"✓ Best model (by LOO): {best_model}")
    print(f"✓ Posterior predictive checks: Complete")
    
    return {
        'models': all_results,
        'comparison': comparison_results,
        'best_model': best_model,
        'ppc': ppc_results
    }


# ===============================
# Extended Inference Class
# ===============================

class Lambda3BayesianInference:
    """
    Extended Bayesian inference engine for Lambda³ framework.
    
    This class provides a high-level interface for running
    comprehensive Bayesian analyses with model comparison,
    diagnostics, and visualization.
    """
    
    def __init__(self, config: Optional[L3Config] = None):
        """Initialize inference engine with configuration."""
        self.config = config or L3Config()
        self.results = {}
        self.comparison_results = None
        self.ppc_results = {}
        
    def fit_model(self, features: Lambda3FeatureSet, 
                  model_type: str = 'base', **kwargs) -> BayesianResults:
        """
        Fit a specific model type.
        
        Args:
            features: Lambda³ features
            model_type: 'base', 'interaction', 'dynamic', or 'hierarchical'
            **kwargs: Additional model-specific arguments
            
        Returns:
            BayesianResults
        """
        if model_type == 'base':
            results = fit_bayesian_model(features, self.config, model_type='base', **kwargs)
        elif model_type == 'interaction':
            results = fit_bayesian_model(features, self.config, model_type='interaction', **kwargs)
        elif model_type == 'dynamic':
            results = fit_dynamic_model(features, self.config, **kwargs)
        elif model_type == 'hierarchical':
            features_list = kwargs.pop('features_list', [features])
            results = fit_hierarchical_model(features_list, self.config, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.results[model_type] = results
        return results
    
    def compare_models(self, features: Lambda3FeatureSet, 
                      criterion: str = 'loo') -> Dict[str, Any]:
        """
        Compare all fitted models.
        
        Args:
            features: Lambda³ features
            criterion: 'loo' or 'waic'
            
        Returns:
            Comparison results
        """
        if not self.results:
            raise ValueError("No models fitted yet")
        
        self.comparison_results = compare_models(
            self.results, features, criterion
        )
        return self.comparison_results
    
    def run_ppc(self, features: Lambda3FeatureSet, 
                model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run posterior predictive checks.
        
        Args:
            features: Lambda³ features
            model_name: Specific model to check (or best model)
            
        Returns:
            PPC results
        """
        if model_name is None:
            if self.comparison_results and 'best_model' in self.comparison_results:
                model_name = self.comparison_results['best_model']
            else:
                model_name = list(self.results.keys())[0]
        
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found")
        
        ppc_results = posterior_predictive_check(
            self.results[model_name], features
        )
        self.ppc_results[model_name] = ppc_results
        
        return ppc_results
    
    def get_best_model(self) -> Tuple[str, BayesianResults]:
        """
        Get the best model based on comparison.
        
        Returns:
            Tuple of (model_name, results)
        """
        if not self.comparison_results or 'best_model' not in self.comparison_results:
            raise ValueError("No model comparison performed yet")
        
        best_name = self.comparison_results['best_model']
        return best_name, self.results[best_name]
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary of all analyses.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'n_models': len(self.results),
            'models': list(self.results.keys()),
            'convergence': {}
        }
        
        # Check convergence for each model
        for name, results in self.results.items():
            summary['convergence'][name] = check_convergence(results, verbose=False)
        
        # Add comparison results
        if self.comparison_results:
            summary['best_model'] = self.comparison_results.get('best_model')
            if 'comparison_table' in self.comparison_results:
                summary['comparison_table'] = self.comparison_results['comparison_table']
        
        # Add PPC results
        if self.ppc_results:
            summary['ppc_p_values'] = {
                model: ppc['bayesian_p_values']
                for model, ppc in self.ppc_results.items()
            }
        
        return summary


def predict_with_model(
    trace: az.InferenceData,
    features: Lambda3FeatureSet,
    interaction_features: Optional[Lambda3FeatureSet] = None,
    model_type: str = 'interaction'
) -> np.ndarray:
    """
    Generate predictions from fitted model.
    
    Args:
        trace: ArviZ InferenceData from MCMC
        features: Features used for prediction
        interaction_features: Interaction features (if applicable)
        model_type: Type of model fitted
        
    Returns:
        predictions: Mean predictions
    """
    # Extract posterior means
    posterior = trace.posterior
    
    if model_type == 'base' or model_type == 'interaction':
        # Linear model predictions
        mu = (
            posterior['beta_0'].mean().item() +
            posterior['beta_time'].mean().item() * features.time_trend +
            posterior['beta_dLC_pos'].mean().item() * features.delta_LambdaC_pos +
            posterior['beta_dLC_neg'].mean().item() * features.delta_LambdaC_neg +
            posterior['beta_rhoT'].mean().item() * features.rho_T
        )
        
        # Add local jump effect if present
        if 'beta_local_jump' in posterior:
            mu += posterior['beta_local_jump'].mean().item() * features.local_jump
        
        # Add interaction terms if present
        if model_type == 'interaction' and interaction_features:
            if 'beta_interact_pos' in posterior:
                mu += posterior['beta_interact_pos'].mean().item() * interaction_features.delta_LambdaC_pos
            if 'beta_interact_neg' in posterior:
                mu += posterior['beta_interact_neg'].mean().item() * interaction_features.delta_LambdaC_neg
            if 'beta_interact_stress' in posterior:
                mu += posterior['beta_interact_stress'].mean().item() * interaction_features.rho_T
    
    elif model_type == 'dynamic':
        # Time-varying predictions
        beta_time_series = posterior['beta_time_series'].mean(dim=['chain', 'draw']).values
        mu = (
            beta_time_series +
            posterior['beta_dLC_pos'].mean().item() * features.delta_LambdaC_pos +
            posterior['beta_dLC_neg'].mean().item() * features.delta_LambdaC_neg +
            posterior['beta_rhoT'].mean().item() * features.rho_T
        )
        
        # Add local jump effect if present
        if 'beta_local_jump' in posterior:
            mu += posterior['beta_local_jump'].mean().item() * features.local_jump
        
        # Add jumps if present
        time_idx = np.arange(len(features.data))
        for var_name in posterior.data_vars:
            if var_name.startswith('jump_'):
                cp_idx = int(var_name.split('_')[1])
                jump_size = posterior[var_name].mean().item()
                # Assuming change_points were provided in order
                mu += jump_size * (time_idx >= cp_idx)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return np.asarray(mu)


# ===============================
# Helper Functions
# ===============================

def _convert_to_arviz(mcmc: MCMC, config: BayesianConfig) -> az.InferenceData:
    """
    Convert MCMC results to ArviZ InferenceData.
    
    Args:
        mcmc: NumPyro MCMC object
        config: Bayesian configuration
        
    Returns:
        trace: ArviZ InferenceData
    """
    # Get samples
    if config.num_chains > 1:
        samples = mcmc.get_samples(group_by_chain=True)
        # Ensure proper shape (chains, draws, ...)
        posterior = {}
        for k, v in samples.items():
            arr = np.asarray(v)
            if arr.ndim == 1:  # Scalar parameter
                arr = arr.reshape(config.num_chains, -1)
            posterior[k] = arr
    else:
        samples = mcmc.get_samples(group_by_chain=False)
        # Add chain dimension
        posterior = {}
        for k, v in samples.items():
            arr = np.asarray(v)
            if arr.ndim == 0:  # Scalar
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            else:
                arr = np.expand_dims(arr, 0)
            posterior[k] = arr
    
    # Prepare sample stats
    sample_stats = {}
    try:
        extra_fields = mcmc.get_extra_fields()
        if extra_fields:
            # Process divergences
            if 'diverging' in extra_fields:
                div = np.asarray(extra_fields['diverging'])
                if config.num_chains == 1 and div.ndim == 1:
                    div = div.reshape(1, -1)
                sample_stats['diverging'] = div
            
            # Process acceptance probability
            if 'accept_prob' in extra_fields:
                acc = np.asarray(extra_fields['accept_prob'])
                if config.num_chains == 1 and acc.ndim == 1:
                    acc = acc.reshape(1, -1)
                sample_stats['acceptance_rate'] = acc
            
            # Process energy
            if 'potential_energy' in extra_fields:
                energy = np.asarray(extra_fields['potential_energy'])
                if config.num_chains == 1 and energy.ndim == 1:
                    energy = energy.reshape(1, -1)
                sample_stats['energy'] = energy
    except Exception as e:
        print(f"Warning: Could not extract sample statistics: {e}")
    
    # Create InferenceData with both posterior and sample_stats
    if sample_stats:
        trace = az.from_dict(
            posterior=posterior,
            sample_stats=sample_stats
        )
    else:
        trace = az.from_dict(posterior=posterior)
    
    return trace


def _extract_diagnostics(mcmc: MCMC, trace: az.InferenceData) -> Dict[str, Any]:
    """
    Extract MCMC diagnostics.
    
    Args:
        mcmc: NumPyro MCMC object
        trace: ArviZ InferenceData
        
    Returns:
        diagnostics: Dictionary of diagnostic values
    """
    diagnostics = {}
    
    try:
        # R-hat values
        r_hat = az.rhat(trace)
        diagnostics['r_hat'] = {var: float(r_hat[var].values) for var in r_hat.data_vars}
        diagnostics['max_r_hat'] = max(diagnostics['r_hat'].values())
        
        # Effective sample size
        ess = az.ess(trace)
        diagnostics['ess_bulk'] = {var: float(ess[var].values) for var in ess.data_vars}
        diagnostics['min_ess_bulk'] = min(diagnostics['ess_bulk'].values())
        
        # Divergences
        if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
            diagnostics['n_divergences'] = int(trace.sample_stats.diverging.sum())
            diagnostics['divergence_rate'] = float(trace.sample_stats.diverging.mean())
        
        # Acceptance rate
        if hasattr(trace, 'sample_stats') and 'acceptance_rate' in trace.sample_stats:
            diagnostics['mean_acceptance_rate'] = float(trace.sample_stats.acceptance_rate.mean())
        
        # MCMC-specific diagnostics
        if hasattr(mcmc, 'print_summary'):
            # Capture summary statistics
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                mcmc.print_summary()
            diagnostics['summary_text'] = f.getvalue()
    
    except Exception as e:
        print(f"Warning: Could not extract all diagnostics: {e}")
    
    return diagnostics


def check_convergence(
    results: BayesianResults, 
    verbose: bool = True,
    r_hat_threshold: float = 1.01,
    ess_threshold: int = 400
) -> bool:
    """
    Check MCMC convergence diagnostics.
    
    Args:
        results: Bayesian results
        verbose: Whether to print diagnostics
        r_hat_threshold: Maximum acceptable R-hat
        ess_threshold: Minimum acceptable ESS
        
    Returns:
        converged: True if all diagnostics pass
    """
    converged = True
    issues = []
    
    if results.diagnostics:
        # Check R-hat
        if 'max_r_hat' in results.diagnostics:
            if results.diagnostics['max_r_hat'] > r_hat_threshold:
                converged = False
                issues.append(f"R-hat too high: {results.diagnostics['max_r_hat']:.3f}")
        
        # Check ESS
        if 'min_ess_bulk' in results.diagnostics:
            if results.diagnostics['min_ess_bulk'] < ess_threshold:
                converged = False
                issues.append(f"ESS too low: {results.diagnostics['min_ess_bulk']:.0f}")
        
        # Check divergences
        if 'n_divergences' in results.diagnostics:
            if results.diagnostics['n_divergences'] > 0:
                converged = False
                issues.append(f"Divergences detected: {results.diagnostics['n_divergences']}")
    
    if verbose:
        if converged:
            print("✓ All convergence diagnostics passed")
        else:
            print("✗ Convergence issues detected:")
            for issue in issues:
                print(f"  - {issue}")
    
    return converged
