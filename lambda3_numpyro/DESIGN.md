# Lambda³ NumPyro Framework Design Document

## Architecture Overview

The Lambda³ NumPyro framework uses a hybrid approach combining Numba and JAX for optimal performance:

### 1. Feature Extraction Layer (Numba)
- **Technology**: Numba JIT compilation
- **Purpose**: Fast, CPU-optimized feature extraction
- **Output**: NumPy arrays (np.ndarray)
- **Files**: `feature.py`

### 2. Bayesian Inference Layer (JAX/NumPyro)
- **Technology**: JAX + NumPyro
- **Purpose**: GPU-accelerated Bayesian inference
- **Input**: JAX arrays (jnp.ndarray)
- **Files**: `bayes.py`

### 3. Analysis Layer
- **Technology**: Mixed (NumPy for general computation)
- **Purpose**: High-level analysis orchestration
- **Files**: `analysis.py`

## Type Conversion Guidelines

### Numba → JAX Conversion
Always convert NumPy arrays to JAX arrays at the boundary:

```python
# Good: Explicit conversion at module boundary
features_jax = {
    'data': jnp.asarray(features.data),
    'rho_T': jnp.asarray(features.rho_T)
}

# Bad: Mixing np and jnp operations
mu = np.mean(data) + jnp.sum(features)  # Don't do this!
```

### JAX → NumPy Conversion
When returning results to users:

```python
# Convert JAX arrays back to NumPy for storage/serialization
predictions = np.asarray(jax_predictions)
```

## Design Rationale

### Why Numba for Feature Extraction?
1. **CPU Optimization**: Feature extraction is inherently sequential
2. **Low Overhead**: Minimal startup time for small datasets
3. **NumPy Compatibility**: Seamless integration with existing NumPy code
4. **Parallel Loops**: Efficient parallel computation with `prange`

### Why JAX/NumPyro for Bayesian Inference?
1. **GPU Acceleration**: Massive speedup for MCMC sampling
2. **Automatic Differentiation**: Required for HMC/NUTS
3. **Vectorization**: Efficient batch operations
4. **Modern PPL**: NumPyro provides state-of-the-art inference algorithms

## Best Practices

### 1. Clear Module Boundaries
- Feature extraction (Numba) → Returns NumPy arrays
- Bayesian models (JAX) → Accepts JAX arrays
- Analysis functions → Handle conversions explicitly

### 2. Type Annotations
```python
def extract_features(data: np.ndarray) -> Lambda3FeatureSet:
    """Numba-accelerated feature extraction."""
    ...

def fit_model(features: Lambda3FeatureSet) -> BayesianResults:
    """JAX/NumPyro model fitting (handles conversion internally)."""
    ...
```

### 3. Error Handling
- Validate array types at module boundaries
- Provide clear error messages for type mismatches
- Document expected input/output types

### 4. Performance Considerations
- Minimize data copies between NumPy/JAX
- Use in-place operations where possible
- Profile to identify bottlenecks

## Future Considerations

### Potential Unification
If performance analysis shows minimal benefit from Numba:
- Consider migrating feature extraction to JAX
- Use `jax.jit` instead of `numba.jit`
- Benefit: Single array type throughout

### Current Approach Benefits
- Flexibility to use best tool for each task
- Easier debugging (NumPy arrays are more familiar)
- Option to run feature extraction without GPU
- Clear separation of concerns
