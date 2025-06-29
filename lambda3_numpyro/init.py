"""
Lambda³ Analytics for Bayes & Causal Junction - NumPyro Backend
================================================================

A framework for analyzing structural evolution and synchronization
in complex time series using Lambda³ theory.

Key Components:
- Feature extraction with JIT optimization
- Bayesian inference using NumPyro
- Multi-scale synchronization analysis
- Cloud-ready I/O operations

Author: Lambda³ Development Team
License: MIT
"""

__version__ = "1.0.0"

# Core configuration
from .config import L3Config

# Type definitions
from .types import (
    Lambda3FeatureSet,
    SyncProfile,
    AnalysisResult,
    CrossAnalysisResult,
    RegimeInfo
)

# Feature extraction
from .feature import (
    extract_lambda3_features,
    calculate_sync_profile,
    calculate_dynamic_sync,
    build_sync_network,
    sync_matrix
)

# Bayesian analysis
from .bayes import (
    fit_bayesian_model,
    fit_dynamic_model,
    predict_with_model
)

# Analysis functions
from .analysis import (
    analyze_pair,
    analyze_multiple_series,
    detect_regimes,
    calculate_causality_matrix
)

# I/O operations
from .io import (
    load_csv_series,
    load_financial_data,
    save_features,
    load_features,
    save_analysis_results,
    load_analysis_results
)

# Plotting (optional)
try:
    from .plot import (
        plot_features,
        plot_analysis_results,
        plot_sync_network,
        plot_interaction_matrix
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

__all__ = [
    # Version
    "__version__",
    
    # Config
    "L3Config",
    
    # Types
    "Lambda3FeatureSet",
    "SyncProfile", 
    "AnalysisResult",
    "CrossAnalysisResult",
    "RegimeInfo",
    
    # Features
    "extract_lambda3_features",
    "calculate_sync_profile",
    "calculate_dynamic_sync",
    "build_sync_network",
    "sync_matrix",
    
    # Bayes
    "fit_bayesian_model",
    "fit_dynamic_model",
    "predict_with_model",
    
    # Analysis
    "analyze_pair",
    "analyze_multiple_series",
    "detect_regimes",
    "calculate_causality_matrix",
    
    # I/O
    "load_csv_series",
    "load_financial_data",
    "save_features",
    "load_features",
    "save_analysis_results",
    "load_analysis_results",
    
    # Plotting
    "PLOTTING_AVAILABLE",
    "plot_features",
    "plot_analysis_results",
    "plot_sync_network",
    "plot_interaction_matrix"
]
