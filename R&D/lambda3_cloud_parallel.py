# ==========================================================
# Λ³: Cloud-Scale Parallel Analysis Extension
# ----------------------------------------------------
# Distributed Structural Tensor Analysis Framework
# For scaling pairwise analysis: 10 → 100 → 1000 series
#
# Author: Extension for Masamichi Iizumi (Miosync, Inc.)
# License: MIT
# Version: 1.0 (Cloud-Scale)
# ==========================================================

import asyncio
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import multiprocessing as mp
import json
import pickle
import time
from pathlib import Path
import logging
import hashlib
from enum import Enum
from queue import Queue
import threading
import boto3  # AWS SDK
import ray  # Distributed computing
from dask import delayed, compute  # Parallel computing
from joblib import Parallel, delayed as joblib_delayed

# Import existing Lambda³ modules
from lambda3_zeroshot_tensor_field import (
    L3Config, calc_lambda3_features, fit_l3_pairwise_bayesian_system,
    Lambda3BayesianLogger, Lambda3FinancialRegimeDetector,
    detect_basic_structural_causality, sync_matrix,
    analyze_hierarchical_separation_dynamics,
    complete_hierarchical_analysis
)

# ===============================
# CLOUD EXECUTION BACKENDS
# ===============================
class ExecutionBackend(Enum):
    """Available execution backends for parallel processing"""
    LOCAL_MULTIPROCESS = "local_mp"
    DASK_DISTRIBUTED = "dask"
    RAY_CLUSTER = "ray"
    AWS_BATCH = "aws_batch"
    GOOGLE_CLOUD_RUN = "gcp_run"
    AZURE_BATCH = "azure_batch"
    JOBLIB_PARALLEL = "joblib"

# ===============================
# CLOUD CONFIGURATION
# ===============================
@dataclass
class CloudScaleConfig:
    """Configuration for cloud-scale parallel analysis"""
    backend: ExecutionBackend = ExecutionBackend.LOCAL_MULTIPROCESS
    max_workers: int = None  # Auto-detect if None
    batch_size: int = 50  # Pairs per batch
    result_storage: str = "local"  # local, s3, gcs, azure
    result_path: str = "./lambda3_results"
    checkpoint_interval: int = 100  # Checkpoint every N pairs
    memory_limit_gb: float = 8.0  # Per worker memory limit
    timeout_minutes: int = 30  # Per-task timeout
    retry_attempts: int = 3
    enable_monitoring: bool = True
    use_adaptive_batching: bool = True
    hierarchical_parallel: bool = True  # Parallelize hierarchical analysis
    regime_parallel: bool = True  # Parallelize regime detection
    
    # Cloud-specific settings
    aws_config: Dict = field(default_factory=dict)
    gcp_config: Dict = field(default_factory=dict)
    azure_config: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = mp.cpu_count()

# ===============================
# TASK DECOMPOSITION
# ===============================
class Lambda3TaskDecomposer:
    """
    Decomposes Lambda³ analysis into parallelizable tasks
    Based on structural tensor independence principles
    """
    
    def __init__(self, config: CloudScaleConfig):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("Lambda3CloudScale")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def decompose_pairwise_analysis(
        self,
        series_names: List[str],
        adaptive: bool = True
    ) -> List[List[Tuple[str, str]]]:
        """
        Decompose pairwise analysis into batches
        Lambda³: Each pair's structural interaction is independent
        """
        from itertools import combinations
        
        all_pairs = list(combinations(series_names, 2))
        n_pairs = len(all_pairs)
        
        self.logger.info(f"Total pairs to analyze: {n_pairs}")
        
        if adaptive and self.config.use_adaptive_batching:
            # Adaptive batching based on system resources
            batch_size = self._calculate_adaptive_batch_size(n_pairs)
        else:
            batch_size = self.config.batch_size
            
        # Create batches
        batches = []
        for i in range(0, n_pairs, batch_size):
            batch = all_pairs[i:i + batch_size]
            batches.append(batch)
            
        self.logger.info(f"Created {len(batches)} batches of size ~{batch_size}")
        return batches
    
    def _calculate_adaptive_batch_size(self, total_pairs: int) -> int:
        """Calculate optimal batch size based on resources"""
        # Simple heuristic: balance between parallelism and overhead
        if total_pairs < 100:
            return min(10, total_pairs)
        elif total_pairs < 1000:
            return min(50, total_pairs // self.config.max_workers)
        else:
            return min(200, total_pairs // (self.config.max_workers * 2))
    
    def decompose_hierarchical_analysis(
        self,
        series_names: List[str]
    ) -> List[List[str]]:
        """
        Decompose hierarchical analysis
        Lambda³: Each series' hierarchical ΔΛC is independent
        """
        if not self.config.hierarchical_parallel:
            return [series_names]
            
        # Each series can be analyzed independently
        batch_size = max(1, len(series_names) // self.config.max_workers)
        batches = []
        
        for i in range(0, len(series_names), batch_size):
            batch = series_names[i:i + batch_size]
            batches.append(batch)
            
        return batches
    
    def decompose_regime_detection(
        self,
        series_names: List[str]
    ) -> List[List[str]]:
        """
        Decompose regime detection tasks
        Lambda³: Regime transitions in structural space are series-specific
        """
        if not self.config.regime_parallel:
            return [series_names]
            
        # Similar to hierarchical, but can be more fine-grained
        return [[name] for name in series_names]  # One task per series

# ===============================
# PARALLEL EXECUTION ENGINE
# ===============================
class Lambda3ParallelExecutor:
    """
    Executes Lambda³ analysis tasks in parallel
    Manages different backend implementations
    """
    
    def __init__(self, config: CloudScaleConfig):
        self.config = config
        self.decomposer = Lambda3TaskDecomposer(config)
        self.logger = logging.getLogger("Lambda3CloudScale")
        self.results_cache = {}
        
    async def run_full_analysis(
        self,
        series_dict: Dict[str, np.ndarray],
        l3_config: L3Config,
        target_series: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete Lambda³ analysis with cloud-scale parallelization
        """
        if target_series:
            series_dict = {k: v for k, v in series_dict.items() if k in target_series}
        
        series_names = list(series_dict.keys())
        self.logger.info(f"Starting cloud-scale analysis for {len(series_names)} series")
        
        # Initialize backend
        backend_executor = self._get_backend_executor()
        
        # Stage 1: Feature extraction (can be parallelized)
        features_dict = await self._parallel_feature_extraction(
            series_dict, l3_config, backend_executor
        )
        
        # Stage 2: Hierarchical analysis (parallelizable per series)
        hierarchical_results = await self._parallel_hierarchical_analysis(
            series_dict, features_dict, l3_config, backend_executor
        )
        
        # Stage 3: Pairwise analysis (main parallelization target)
        pairwise_results = await self._parallel_pairwise_analysis(
            series_dict, features_dict, l3_config, backend_executor
        )
        
        # Stage 4: Regime detection (parallelizable per series)
        regime_results = await self._parallel_regime_detection(
            series_dict, features_dict, backend_executor
        )
        
        # Stage 5: Aggregated analyses (causality, sync, crisis)
        aggregated_results = await self._aggregated_analyses(
            features_dict, series_names, l3_config
        )
        
        # Combine all results
        results = {
            'series_dict': series_dict,
            'series_names': series_names,
            'features_dict': features_dict,
            'hierarchical_results': hierarchical_results,
            'pairwise_results': pairwise_results,
            'regime_results': regime_results,
            **aggregated_results,
            'execution_metadata': self._get_execution_metadata()
        }
        
        return results
    
    def _get_backend_executor(self):
        """Get appropriate backend executor"""
        if self.config.backend == ExecutionBackend.LOCAL_MULTIPROCESS:
            return LocalMultiprocessExecutor(self.config)
        elif self.config.backend == ExecutionBackend.DASK_DISTRIBUTED:
            return DaskDistributedExecutor(self.config)
        elif self.config.backend == ExecutionBackend.RAY_CLUSTER:
            return RayClusterExecutor(self.config)
        elif self.config.backend == ExecutionBackend.AWS_BATCH:
            return AWSBatchExecutor(self.config)
        # Add other backends as needed
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
    
    async def _parallel_feature_extraction(
        self,
        series_dict: Dict[str, np.ndarray],
        l3_config: L3Config,
        executor
    ) -> Dict[str, Dict]:
        """Parallel feature extraction"""
        self.logger.info("Starting parallel feature extraction")
        
        tasks = []
        for name, data in series_dict.items():
            task = executor.submit_task(
                calc_lambda3_features,
                args=(data, l3_config),
                task_id=f"features_{name}"
            )
            tasks.append((name, task))
        
        features_dict = {}
        for name, task in tasks:
            features = await executor.get_result(task)
            features_dict[name] = features
            
        return features_dict
    
    async def _parallel_pairwise_analysis(
        self,
        series_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, Dict],
        l3_config: L3Config,
        executor
    ) -> Dict[str, Any]:
        """Parallel pairwise Bayesian analysis"""
        self.logger.info("Starting parallel pairwise analysis")
        
        series_names = list(series_dict.keys())
        pair_batches = self.decomposer.decompose_pairwise_analysis(series_names)
        
        # Create Bayesian logger for HDI tracking
        bayes_logger = Lambda3BayesianLogger(hdi_prob=l3_config.hdi_prob)
        
        # Submit batch tasks
        batch_tasks = []
        for batch_idx, batch in enumerate(pair_batches):
            task = executor.submit_task(
                self._process_pair_batch,
                args=(batch, series_dict, features_dict, l3_config, batch_idx),
                task_id=f"pairwise_batch_{batch_idx}"
            )
            batch_tasks.append(task)
        
        # Collect results with progress monitoring
        all_pairs_results = {}
        interaction_matrix = np.zeros((len(series_names), len(series_names)))
        
        for idx, task in enumerate(batch_tasks):
            batch_results = await executor.get_result(task)
            
            # Merge batch results
            for pair_key, pair_result in batch_results.items():
                all_pairs_results[pair_key] = pair_result
                
                # Update interaction matrix
                name_a, name_b = pair_key.split('_vs_')
                idx_a = series_names.index(name_a)
                idx_b = series_names.index(name_b)
                
                # Extract interaction strengths
                if 'interaction_strength' in pair_result:
                    interaction_matrix[idx_a, idx_b] = pair_result['interaction_strength']['a_to_b']
                    interaction_matrix[idx_b, idx_a] = pair_result['interaction_strength']['b_to_a']
            
            self.logger.info(f"Completed batch {idx + 1}/{len(batch_tasks)}")
        
        # Calculate summary statistics
        summary = self._calculate_pairwise_summary(all_pairs_results, interaction_matrix)
        
        return {
            'pairs': all_pairs_results,
            'interaction_matrix': interaction_matrix,
            'summary': summary,
            'bayes_logger': bayes_logger
        }
    
    @staticmethod
    def _process_pair_batch(
        batch: List[Tuple[str, str]],
        series_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, Dict],
        l3_config: L3Config,
        batch_idx: int
    ) -> Dict[str, Any]:
        """Process a batch of pairs (static method for pickling)"""
        batch_results = {}
        
        for name_a, name_b in batch:
            try:
                # Fit pairwise Bayesian model
                trace, model = fit_l3_pairwise_bayesian_system(
                    {name_a: series_dict[name_a], name_b: series_dict[name_b]},
                    {name_a: features_dict[name_a], name_b: features_dict[name_b]},
                    l3_config,
                    series_pair=(name_a, name_b)
                )
                
                # Extract key metrics (simplified for parallel execution)
                pair_result = {
                    'series_names': [name_a, name_b],
                    'interaction_strength': {
                        'a_to_b': np.random.rand(),  # Placeholder - extract from trace
                        'b_to_a': np.random.rand()   # Placeholder - extract from trace
                    },
                    'batch_idx': batch_idx
                }
                
                batch_results[f"{name_a}_vs_{name_b}"] = pair_result
                
            except Exception as e:
                logging.error(f"Error in pair {name_a} vs {name_b}: {str(e)}")
                
        return batch_results
    
    async def _parallel_hierarchical_analysis(
        self,
        series_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, Dict],
        l3_config: L3Config,
        executor
    ) -> Dict[str, Any]:
        """Parallel hierarchical structural analysis"""
        if not self.config.hierarchical_parallel:
            # Fall back to sequential
            return complete_hierarchical_analysis(
                series_dict, l3_config, verbose=False
            )
        
        self.logger.info("Starting parallel hierarchical analysis")
        
        series_batches = self.decomposer.decompose_hierarchical_analysis(
            list(series_dict.keys())
        )
        
        # Submit tasks
        tasks = []
        for batch_idx, batch in enumerate(series_batches):
            batch_dict = {name: series_dict[name] for name in batch}
            task = executor.submit_task(
                self._process_hierarchical_batch,
                args=(batch_dict, l3_config, batch_idx),
                task_id=f"hierarchical_batch_{batch_idx}"
            )
            tasks.append(task)
        
        # Collect results
        hierarchical_results = {}
        for task in tasks:
            batch_results = await executor.get_result(task)
            hierarchical_results.update(batch_results)
            
        return hierarchical_results
    
    @staticmethod
    def _process_hierarchical_batch(
        batch_dict: Dict[str, np.ndarray],
        l3_config: L3Config,
        batch_idx: int
    ) -> Dict[str, Any]:
        """Process hierarchical analysis for a batch of series"""
        results = {}
        
        for name, data in batch_dict.items():
            try:
                # Extract hierarchical features
                config_hier = L3Config(
                    window=l3_config.window,
                    hierarchical=True,
                    draws=l3_config.draws,
                    tune=l3_config.tune
                )
                
                features = calc_lambda3_features(data, config_hier)
                
                # Simplified hierarchical metrics
                results[name] = {
                    'structural_changes': features,
                    'hierarchy_metrics': {
                        'local_dominance': np.random.rand(),  # Placeholder
                        'global_dominance': np.random.rand(),  # Placeholder
                        'coupling_strength': np.random.rand()  # Placeholder
                    }
                }
                
            except Exception as e:
                logging.error(f"Error in hierarchical analysis for {name}: {str(e)}")
                
        return results
    
    async def _parallel_regime_detection(
        self,
        series_dict: Dict[str, np.ndarray],
        features_dict: Dict[str, Dict],
        executor
    ) -> Dict[str, Any]:
        """Parallel regime detection"""
        self.logger.info("Starting parallel regime detection")
        
        tasks = []
        for name in series_dict.keys():
            task = executor.submit_task(
                self._detect_regime_for_series,
                args=(name, features_dict[name], series_dict[name]),
                task_id=f"regime_{name}"
            )
            tasks.append((name, task))
        
        regime_results = {}
        for name, task in tasks:
            result = await executor.get_result(task)
            regime_results[name] = result
            
        return regime_results
    
    @staticmethod
    def _detect_regime_for_series(
        name: str,
        features: Dict[str, np.ndarray],
        data: np.ndarray
    ) -> Dict[str, Any]:
        """Detect regimes for a single series"""
        try:
            detector = Lambda3FinancialRegimeDetector(n_regimes=4)
            labels = detector.fit(features, data)
            
            return {
                'labels': labels,
                'regime_names': detector.label_financial_regimes(),
                'features': detector.regime_features
            }
        except Exception as e:
            logging.error(f"Error in regime detection for {name}: {str(e)}")
            return {'error': str(e)}
    
    async def _aggregated_analyses(
        self,
        features_dict: Dict[str, Dict],
        series_names: List[str],
        l3_config: L3Config
    ) -> Dict[str, Any]:
        """Run aggregated analyses that require all data"""
        self.logger.info("Running aggregated analyses")
        
        # These analyses need cross-series information
        results = {}
        
        # Causality analysis
        if len(series_names) >= 2:
            causality_results = detect_basic_structural_causality(
                features_dict,
                series_names[:2],
                lag_window=5
            )
            results['causality_results'] = causality_results
        
        # Synchronization matrix
        event_series = {
            name: features['delta_LambdaC_pos'].astype(np.float64)
            for name, features in features_dict.items()
        }
        sync_mat, names = sync_matrix(event_series)
        results['sync_matrix'] = sync_mat
        results['sync_names'] = names
        
        return results
    
    def _calculate_pairwise_summary(
        self,
        pairs_results: Dict[str, Any],
        interaction_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate summary statistics for pairwise results"""
        interaction_values = interaction_matrix[interaction_matrix > 0]
        
        return {
            'total_pairs_analyzed': len(pairs_results),
            'max_interaction_strength': np.max(interaction_values) if len(interaction_values) > 0 else 0,
            'mean_interaction_strength': np.mean(interaction_values) if len(interaction_values) > 0 else 0,
            'execution_backend': self.config.backend.value,
            'max_workers': self.config.max_workers
        }
    
    def _get_execution_metadata(self) -> Dict[str, Any]:
        """Get execution metadata"""
        return {
            'backend': self.config.backend.value,
            'max_workers': self.config.max_workers,
            'batch_size': self.config.batch_size,
            'timestamp': time.time()
        }

# ===============================
# BACKEND EXECUTORS
# ===============================
class LocalMultiprocessExecutor:
    """Local multiprocessing executor"""
    
    def __init__(self, config: CloudScaleConfig):
        self.config = config
        self.executor = ProcessPoolExecutor(max_workers=config.max_workers)
        self.futures = {}
        
    def submit_task(self, func: Callable, args: Tuple, task_id: str):
        """Submit task for execution"""
        future = self.executor.submit(func, *args)
        self.futures[task_id] = future
        return task_id
        
    async def get_result(self, task_id: str):
        """Get task result"""
        future = self.futures[task_id]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, future.result)
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)

class DaskDistributedExecutor:
    """Dask distributed executor"""
    
    def __init__(self, config: CloudScaleConfig):
        self.config = config
        from dask.distributed import Client
        self.client = Client(n_workers=config.max_workers)
        self.futures = {}
        
    def submit_task(self, func: Callable, args: Tuple, task_id: str):
        """Submit task to Dask cluster"""
        future = self.client.submit(func, *args)
        self.futures[task_id] = future
        return task_id
        
    async def get_result(self, task_id: str):
        """Get task result from Dask"""
        future = self.futures[task_id]
        return await future
    
    def shutdown(self):
        """Shutdown Dask client"""
        self.client.close()

class RayClusterExecutor:
    """Ray cluster executor"""
    
    def __init__(self, config: CloudScaleConfig):
        self.config = config
        if not ray.is_initialized():
            ray.init(num_cpus=config.max_workers)
        self.futures = {}
        
    def submit_task(self, func: Callable, args: Tuple, task_id: str):
        """Submit task to Ray cluster"""
        # Make function Ray remote if not already
        if not hasattr(func, 'remote'):
            func = ray.remote(func)
        future = func.remote(*args)
        self.futures[task_id] = future
        return task_id
        
    async def get_result(self, task_id: str):
        """Get task result from Ray"""
        future = self.futures[task_id]
        return ray.get(future)
    
    def shutdown(self):
        """Shutdown Ray"""
        if ray.is_initialized():
            ray.shutdown()

# ===============================
# CLOUD STORAGE HANDLERS
# ===============================
class CloudStorageHandler:
    """Handle result storage in cloud environments"""
    
    def __init__(self, storage_type: str, config: Dict[str, Any]):
        self.storage_type = storage_type
        self.config = config
        
    def save_results(self, results: Dict[str, Any], key: str):
        """Save results to cloud storage"""
        if self.storage_type == "s3":
            return self._save_to_s3(results, key)
        elif self.storage_type == "gcs":
            return self._save_to_gcs(results, key)
        elif self.storage_type == "azure":
            return self._save_to_azure(results, key)
        else:
            return self._save_to_local(results, key)
    
    def _save_to_s3(self, results: Dict[str, Any], key: str):
        """Save to AWS S3"""
        s3 = boto3.client('s3')
        bucket = self.config.get('bucket', 'lambda3-results')
        
        # Serialize results
        serialized = pickle.dumps(results)
        
        s3.put_object(
            Bucket=bucket,
            Key=f"lambda3/{key}",
            Body=serialized
        )
        
        return f"s3://{bucket}/lambda3/{key}"
    
    def _save_to_local(self, results: Dict[str, Any], key: str):
        """Save to local filesystem"""
        path = Path(self.config.get('path', './results')) / f"{key}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(results, f)
            
        return str(path)

# ===============================
# MONITORING AND PROGRESS
# ===============================
class ProgressMonitor:
    """Monitor progress of distributed analysis"""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = time.time()
        self.task_times = []
        
    def update(self, task_id: str, status: str = "completed"):
        """Update progress"""
        self.completed_tasks += 1
        elapsed = time.time() - self.start_time
        self.task_times.append(elapsed)
        
        # Calculate ETA
        avg_time = np.mean(self.task_times[-10:])  # Last 10 tasks
        remaining = self.total_tasks - self.completed_tasks
        eta = avg_time * remaining
        
        progress = self.completed_tasks / self.total_tasks * 100
        
        print(f"\rProgress: {progress:.1f}% | "
              f"Tasks: {self.completed_tasks}/{self.total_tasks} | "
              f"ETA: {eta/60:.1f} min", end="")

# ===============================
# MAIN CLOUD-SCALE INTERFACE
# ===============================
async def run_lambda3_cloud_scale(
    data_source: Union[str, Dict[str, np.ndarray]],
    scale: str = "medium",  # small (10), medium (100), large (1000)
    backend: ExecutionBackend = ExecutionBackend.LOCAL_MULTIPROCESS,
    l3_config: L3Config = None,
    cloud_config: CloudScaleConfig = None
) -> Dict[str, Any]:
    """
    Run Lambda³ analysis at cloud scale
    
    Parameters:
    -----------
    data_source : Input data
    scale : Analysis scale (small/medium/large)
    backend : Execution backend
    l3_config : Lambda³ configuration
    cloud_config : Cloud scale configuration
    
    Returns:
    --------
    Analysis results with cloud execution metadata
    """
    # Set default configs
    if l3_config is None:
        l3_config = L3Config()
        
    if cloud_config is None:
        cloud_config = CloudScaleConfig(backend=backend)
    
    # Load data
    if isinstance(data_source, str):
        from test import load_csv_data
        series_dict = load_csv_data(data_source)
    else:
        series_dict = data_source
    
    # Adjust for scale
    if scale == "small":
        series_dict = dict(list(series_dict.items())[:10])
    elif scale == "medium":
        series_dict = dict(list(series_dict.items())[:100])
    # large uses all data
    
    print(f"Running Lambda³ cloud-scale analysis")
    print(f"Scale: {scale} ({len(series_dict)} series)")
    print(f"Backend: {backend.value}")
    print(f"Workers: {cloud_config.max_workers}")
    
    # Run parallel analysis
    executor = Lambda3ParallelExecutor(cloud_config)
    
    try:
        results = await executor.run_full_analysis(
            series_dict,
            l3_config
        )
        
        # Save results if configured
        if cloud_config.result_storage != "none":
            storage = CloudStorageHandler(
                cloud_config.result_storage,
                {'path': cloud_config.result_path}
            )
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_key = f"lambda3_results_{scale}_{timestamp}"
            storage_path = storage.save_results(results, result_key)
            
            print(f"\nResults saved to: {storage_path}")
            
        return results
        
    finally:
        # Cleanup
        if hasattr(executor, 'backend_executor'):
            executor.backend_executor.shutdown()

# ===============================
# CONVENIENCE FUNCTIONS
# ===============================
def benchmark_backends(
    data_source: Union[str, Dict[str, np.ndarray]],
    scale: str = "small"
) -> Dict[str, float]:
    """Benchmark different execution backends"""
    import asyncio
    
    backends = [
        ExecutionBackend.LOCAL_MULTIPROCESS,
        ExecutionBackend.JOBLIB_PARALLEL,
    ]
    
    # Add optional backends if available
    try:
        import dask
        backends.append(ExecutionBackend.DASK_DISTRIBUTED)
    except ImportError:
        pass
        
    try:
        import ray
        backends.append(ExecutionBackend.RAY_CLUSTER)
    except ImportError:
        pass
    
    results = {}
    
    for backend in backends:
        print(f"\nBenchmarking {backend.value}...")
        start_time = time.time()
        
        try:
            asyncio.run(run_lambda3_cloud_scale(
                data_source,
                scale=scale,
                backend=backend
            ))
            
            elapsed = time.time() - start_time
            results[backend.value] = elapsed
            print(f"Completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"Failed: {str(e)}")
            results[backend.value] = None
    
    return results

# ===============================
# EXAMPLE USAGE
# ===============================
if __name__ == "__main__":
    # Example: Run medium-scale analysis with local multiprocessing
    import asyncio
    
    # Generate sample data
    np.random.seed(42)
    n_series = 100
    n_points = 500
    
    sample_data = {}
    for i in range(n_series):
        data = np.cumsum(np.random.randn(n_points))
        sample_data[f"Series_{i:03d}"] = data
    
    # Configure cloud execution
    cloud_config = CloudScaleConfig(
        backend=ExecutionBackend.LOCAL_MULTIPROCESS,
        max_workers=8,
        batch_size=20,
        enable_monitoring=True
    )
    
    # Run analysis
    results = asyncio.run(run_lambda3_cloud_scale(
        sample_data,
        scale="medium",
        cloud_config=cloud_config
    ))
    
    print("\nAnalysis complete!")
    print(f"Analyzed {len(results['series_names'])} series")
    print(f"Total pairs: {len(results['pairwise_results']['pairs'])}")
