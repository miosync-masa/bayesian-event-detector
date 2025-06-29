"""
Configuration management for Lambda³ framework.

This module defines global constants and configuration classes
for feature extraction, Bayesian inference, and cloud operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
import json
from pathlib import Path


# ===============================
# Global Constants for JIT Compilation
# ===============================
# These must be module-level constants for Numba JIT
DELTA_PERCENTILE = 97.0              # Percentile threshold for jump detection
LOCAL_JUMP_PERCENTILE = 97.0         # Percentile for local jump detection  
WINDOW_SIZE = 10                     # Window size for tension scalar calculation
LOCAL_WINDOW_SIZE = 10               # Window for local standard deviation
LAG_WINDOW_DEFAULT = 10              # Default lag window for synchronization
SYNC_THRESHOLD_DEFAULT = 0.3         # Default threshold for sync network edges
NOISE_STD_DEFAULT = 0.5              # Default noise standard deviation
NOISE_STD_HIGH = 1.5                 # High noise standard deviation

# Lambda³ theory symbols mapping
LAMBDA3_SYMBOLS = {
    'sigma_s': 'sync_rate',          # σₛ: synchronization rate
    'Lambda': 'structural_tensor',    # Λ: structural tensor
    'Delta_LambdaC': 'jump_events',  # ΔΛC: structural jumps
    'rho_T': 'tension_scalar',       # ρT: tension scalar
    'Lambda_F': 'progression_vector'  # ΛF: progression vector
}


# ===============================
# Configuration Classes
# ===============================

@dataclass
class FeatureConfig:
    """Configuration for Lambda³ feature extraction."""
    window: int = WINDOW_SIZE
    local_window: int = LOCAL_WINDOW_SIZE
    delta_percentile: float = DELTA_PERCENTILE
    local_jump_percentile: float = LOCAL_JUMP_PERCENTILE
    lag_window: int = LAG_WINDOW_DEFAULT
    
    def validate(self):
        """Validate configuration parameters"""
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")
        if self.local_window < 2:
            raise ValueError(f"local_window must be >= 2, got {self.local_window}")
        if not 0 < self.delta_percentile <= 100:
            raise ValueError(f"delta_percentile must be in (0, 100], got {self.delta_percentile}")
        if not 0 < self.local_jump_percentile <= 100:
            raise ValueError(f"local_jump_percentile must be in (0, 100], got {self.local_jump_percentile}")
        if self.lag_window < 1:
            raise ValueError(f"lag_window must be >= 1, got {self.lag_window}")


@dataclass
class BayesianConfig:
    """Configuration for NumPyro Bayesian inference."""
    draws: int = 8000
    tune: int = 8000
    target_accept: float = 0.95
    num_chains: int = 4
    max_treedepth: int = 10
    
    # Prior hyperparameters - unified for all models
    prior_scales: Dict[str, float] = field(default_factory=lambda: {
        'beta_0': 2.0,
        'beta_time': 1.0,
        'beta_dLC_pos': 5.0,
        'beta_dLC_neg': 5.0,
        'beta_rhoT': 3.0,
        'beta_interact': 3.0,
        'beta_local_jump': 2.0,
        'sigma_obs': 1.0,
        'innovation_scale': 0.1,
        'sigma_base': 1.0,
        'sigma_scale': 0.5
    })
    
    def validate(self):
        """Validate Bayesian configuration"""
        if self.draws < 100:
            raise ValueError(f"draws must be >= 100, got {self.draws}")
        if self.tune < 100:
            raise ValueError(f"tune must be >= 100, got {self.tune}")
        if not 0 < self.target_accept < 1:
            raise ValueError(f"target_accept must be in (0, 1), got {self.target_accept}")
        if self.num_chains < 1:
            raise ValueError(f"num_chains must be >= 1, got {self.num_chains}")
        if self.max_treedepth < 1:
            raise ValueError(f"max_treedepth must be >= 1, got {self.max_treedepth}")


@dataclass  
class CloudConfig:
    """Configuration for cloud storage operations."""
    provider: str = 'local'  # 'local', 'gcs', 's3', 'azure'
    bucket: Optional[str] = None
    prefix: str = 'lambda3'
    credentials_path: Optional[str] = None
    
    # Provider-specific settings
    gcs_project: Optional[str] = None
    aws_region: str = 'us-east-1'
    azure_account: Optional[str] = None
    
    def validate(self):
        """Validate cloud configuration"""
        valid_providers = ['local', 'gcs', 's3', 'azure']
        if self.provider not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}, got {self.provider}")
        
        if self.provider != 'local' and not self.bucket:
            raise ValueError(f"bucket required for provider {self.provider}")
        
        # Check provider-specific requirements
        if self.provider == 'gcs' and not self.gcs_project:
            # Try to get from environment
            self.gcs_project = os.getenv('GOOGLE_CLOUD_PROJECT')
            if not self.gcs_project:
                raise ValueError("gcs_project required for GCS provider")


@dataclass
class PlottingConfig:
    """Configuration for visualization."""
    figure_size: tuple = (12, 8)
    dpi: int = 150
    style: str = 'seaborn-v0_8-darkgrid'
    color_palette: str = 'husl'
    save_figures: bool = False
    figure_format: str = 'png'
    figure_dir: str = './figures'
    
    # Specific plot settings
    show_legend: bool = True
    show_grid: bool = True
    alpha: float = 0.7
    line_width: float = 2.0
    marker_size: float = 8.0


@dataclass
class L3Config:
    """
    Main configuration class for Lambda³ analysis.
    
    This aggregates all sub-configurations and provides
    convenience methods for loading/saving configurations.
    """
    # Time series parameters
    T: Optional[int] = None  # Will be set from data length
    
    # Sub-configurations
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    
    # Analysis parameters (removed analyze_all_pairs)
    max_pairs: Optional[int] = None
    parallel_pairs: bool = False
    
    # Output settings
    output_dir: str = './lambda3_output'
    save_intermediate: bool = True
    verbose: bool = True
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # JAX configuration
    enable_x64: bool = True  # Enable 64-bit computation in JAX
    
    def __post_init__(self):
        """Create output directory if needed and configure JAX"""
        if self.output_dir and self.save_intermediate:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure JAX for 64-bit if enabled
        if self.enable_x64:
            import jax
            jax.config.update("jax_enable_x64", True)
    
    def validate(self):
        """Validate all configurations"""
        self.feature.validate()
        self.bayesian.validate()
        self.cloud.validate()
        
        if self.max_pairs is not None and self.max_pairs < 1:
            raise ValueError(f"max_pairs must be >= 1, got {self.max_pairs}")
    
    @classmethod
    def from_file(cls, filepath: str) -> 'L3Config':
        """Load configuration from JSON or YAML file"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.suffix in ['.yml', '.yaml']:
            try:
                import yaml
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported config file format: {filepath.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'L3Config':
        """Create configuration from dictionary"""
        config = cls()
        
        # Handle nested configurations
        if 'feature' in data:
            config.feature = FeatureConfig(**data.pop('feature'))
        if 'bayesian' in data:
            config.bayesian = BayesianConfig(**data.pop('bayesian'))
        if 'cloud' in data:
            config.cloud = CloudConfig(**data.pop('cloud'))
        if 'plotting' in data:
            config.plotting = PlottingConfig(**data.pop('plotting'))
        
        # Handle remaining top-level attributes
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_env(cls) -> 'L3Config':
        """
        Create configuration from environment variables.
        
        Environment variables follow the pattern:
        L3_<SECTION>_<PARAMETER> (e.g., L3_BAYESIAN_NUM_CHAINS)
        """
        config = cls()
        
        # Feature config from env
        if env_val := os.getenv('L3_FEATURE_WINDOW'):
            config.feature.window = int(env_val)
        if env_val := os.getenv('L3_FEATURE_DELTA_PERCENTILE'):
            config.feature.delta_percentile = float(env_val)
        
        # Bayesian config from env
        if env_val := os.getenv('L3_BAYESIAN_NUM_CHAINS'):
            config.bayesian.num_chains = int(env_val)
        if env_val := os.getenv('L3_BAYESIAN_DRAWS'):
            config.bayesian.draws = int(env_val)
        if env_val := os.getenv('L3_BAYESIAN_TUNE'):
            config.bayesian.tune = int(env_val)
        
        # Cloud config from env
        if env_val := os.getenv('L3_CLOUD_PROVIDER'):
            config.cloud.provider = env_val
        if env_val := os.getenv('L3_CLOUD_BUCKET'):
            config.cloud.bucket = env_val
        if env_val := os.getenv('L3_CLOUD_PREFIX'):
            config.cloud.prefix = env_val
        
        # General config from env
        if env_val := os.getenv('L3_OUTPUT_DIR'):
            config.output_dir = env_val
        if env_val := os.getenv('L3_VERBOSE'):
            config.verbose = env_val.lower() in ['true', '1', 'yes']
        if env_val := os.getenv('L3_RANDOM_SEED'):
            config.random_seed = int(env_val)
        if env_val := os.getenv('L3_ENABLE_X64'):
            config.enable_x64 = env_val.lower() in ['true', '1', 'yes']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'T': self.T,
            'feature': {
                'window': self.feature.window,
                'local_window': self.feature.local_window,
                'delta_percentile': self.feature.delta_percentile,
                'local_jump_percentile': self.feature.local_jump_percentile,
                'lag_window': self.feature.lag_window
            },
            'bayesian': {
                'draws': self.bayesian.draws,
                'tune': self.bayesian.tune,
                'target_accept': self.bayesian.target_accept,
                'num_chains': self.bayesian.num_chains,
                'max_treedepth': self.bayesian.max_treedepth,
                'prior_scales': self.bayesian.prior_scales
            },
            'cloud': {
                'provider': self.cloud.provider,
                'bucket': self.cloud.bucket,
                'prefix': self.cloud.prefix,
                'gcs_project': self.cloud.gcs_project,
                'aws_region': self.cloud.aws_region
            },
            'plotting': {
                'figure_size': self.plotting.figure_size,
                'dpi': self.plotting.dpi,
                'style': self.plotting.style,
                'color_palette': self.plotting.color_palette,
                'save_figures': self.plotting.save_figures,
                'figure_format': self.plotting.figure_format,
                'figure_dir': self.plotting.figure_dir
            },
            'max_pairs': self.max_pairs,
            'parallel_pairs': self.parallel_pairs,
            'output_dir': self.output_dir,
            'save_intermediate': self.save_intermediate,
            'verbose': self.verbose,
            'random_seed': self.random_seed,
            'enable_x64': self.enable_x64
        }
    
    def save(self, filepath: str):
        """Save configuration to file"""
        filepath = Path(filepath)
        data = self.to_dict()
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif filepath.suffix in ['.yml', '.yaml']:
            try:
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported config file format: {filepath.suffix}")
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get JAX device configuration based on environment"""
        import jax
        
        backend = jax.default_backend()
        device_count = jax.local_device_count()
        
        return {
            'backend': backend,
            'device_count': device_count,
            'recommended_chains': min(self.bayesian.num_chains, device_count) if backend == 'cpu' else self.bayesian.num_chains,
            'memory_efficient': backend == 'gpu' and self.bayesian.draws > 10000,
            'x64_enabled': self.enable_x64
        }


def get_default_config(preset: str = 'standard') -> L3Config:
    """
    Get pre-configured settings for common use cases.
    
    Args:
        preset: Configuration preset name
            - 'standard': Balanced settings for most analyses
            - 'fast': Quick analysis with reduced samples
            - 'production': High-quality results with more samples
            - 'gpu': Optimized for GPU execution
            - 'cpu': Optimized for CPU execution
            - 'cloud': Cloud-ready configuration
    
    Returns:
        L3Config: Pre-configured settings
    """
    if preset == 'standard':
        return L3Config()
    
    elif preset == 'fast':
        config = L3Config()
        config.bayesian.draws = 2000
        config.bayesian.tune = 1000
        config.bayesian.num_chains = 2
        return config
    
    elif preset == 'production':
        config = L3Config()
        config.bayesian.draws = 20000
        config.bayesian.tune = 10000
        config.bayesian.num_chains = 4
        config.bayesian.target_accept = 0.99
        config.save_intermediate = True
        return config
    
    elif preset == 'gpu':
        config = L3Config()
        config.bayesian.draws = 10000
        config.bayesian.tune = 5000
        config.bayesian.num_chains = 4  # GPU can handle parallel chains efficiently
        config.parallel_pairs = True
        return config
    
    elif preset == 'cpu':
        config = L3Config()
        config.bayesian.draws = 5000
        config.bayesian.tune = 2000
        config.bayesian.num_chains = 2  # Conservative for CPU
        config.parallel_pairs = False
        return config
    
    elif preset == 'cloud':
        config = L3Config()
        config.cloud.provider = 'gcs'  # Default to Google Cloud
        config.cloud.prefix = 'lambda3-analysis'
        config.save_intermediate = True
        config.plotting.save_figures = True
        return config
    
    else:
        raise ValueError(f"Unknown preset: {preset}. "
                       f"Choose from: standard, fast, production, gpu, cpu, cloud")
