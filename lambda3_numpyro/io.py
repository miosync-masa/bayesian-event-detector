"""
I/O module for Lambda³ framework with cloud storage support.

This module handles data loading, saving, and cloud operations
for time series data, features, and analysis results.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Optional, List, Union, Any, Tuple
from datetime import datetime
import warnings

from .types import Lambda3FeatureSet, AnalysisResult, CrossAnalysisResult
from .config import CloudConfig


# ===============================
# CSV Data Loading
# ===============================

def load_csv_series(
    filepath: Union[str, Path],
    time_column: Optional[str] = None,
    value_columns: Optional[List[str]] = None,
    delimiter: str = ',',
    parse_dates: bool = True,
    date_format: Optional[str] = None,
    fillna_method: str = 'ffill',
    validate: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load time series data from CSV file.
    
    Args:
        filepath: Path to CSV file
        time_column: Column to use as time index
        value_columns: Columns to extract as series
        delimiter: CSV delimiter
        parse_dates: Whether to parse date columns
        date_format: Date parsing format
        fillna_method: Method to fill missing values
        validate: Whether to validate series lengths
        
    Returns:
        Dictionary mapping column names to numpy arrays
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    print(f"Loading data from: {filepath}")
    
    # Load data
    df = pd.read_csv(
        filepath,
        delimiter=delimiter,
        parse_dates=[time_column] if time_column and parse_dates else False,
        date_format=date_format
    )
    
    print(f"Loaded CSV with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Sort by time if specified
    if time_column and time_column in df.columns:
        df = df.sort_values(by=time_column)
        print(f"Data sorted by {time_column}")
    
    # Select value columns
    if value_columns is None:
        # Use all numeric columns except time
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_column and time_column in numeric_cols:
            numeric_cols.remove(time_column)
        value_columns = numeric_cols
    
    print(f"Using columns: {value_columns}")
    
    # Extract series
    series_dict = {}
    for col in value_columns:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in CSV")
            continue
        
        # Get data
        data = df[col].values
        
        # Handle missing values
        if pd.isna(data).any():
            n_missing = pd.isna(data).sum()
            print(f"  Warning: '{col}' has {n_missing} missing values")
            
            if fillna_method == 'ffill':
                data = pd.Series(data).ffill().bfill().values
            elif fillna_method == 'interpolate':
                data = pd.Series(data).interpolate().ffill().bfill().values
            elif fillna_method == 'drop':
                data = data[~pd.isna(data)]
            else:
                raise ValueError(f"Unknown fillna method: {fillna_method}")
        
        series_dict[col] = data.astype(np.float64)
    
    # Validate lengths
    if validate:
        series_dict = _validate_series_lengths(series_dict)
    
    print(f"Loaded {len(series_dict)} series")
    return series_dict


def load_multiple_csv_files(
    filepaths: List[Union[str, Path]],
    series_names: Optional[List[str]] = None,
    column_index: Union[int, str] = 0,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Load multiple CSV files, each containing a single series.
    
    Args:
        filepaths: List of CSV file paths
        series_names: Names for each series
        column_index: Column to extract from each file
        **kwargs: Additional arguments for load_csv_series
        
    Returns:
        Dictionary mapping series names to data
    """
    if series_names is None:
        series_names = [Path(fp).stem for fp in filepaths]
    
    if len(series_names) != len(filepaths):
        raise ValueError("Number of series names must match number of files")
    
    series_dict = {}
    
    for filepath, name in zip(filepaths, series_names):
        print(f"\nLoading {name} from {filepath}")
        
        try:
            # Load the CSV
            data_dict = load_csv_series(filepath, **kwargs)
            
            # Extract the specified column
            if isinstance(column_index, int):
                col_name = list(data_dict.keys())[column_index]
            else:
                col_name = column_index
            
            if col_name in data_dict:
                series_dict[name] = data_dict[col_name]
                print(f"  ✓ Loaded {len(series_dict[name])} points")
            else:
                print(f"  ✗ Column '{col_name}' not found")
        
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")
            continue
    
    return series_dict


# ===============================
# Financial Data Loading
# ===============================

def load_financial_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    tickers: Optional[Dict[str, str]] = None,
    save_csv: bool = True,
    csv_filename: str = "financial_data.csv"
) -> Dict[str, np.ndarray]:
    """
    Fetch financial data using yfinance.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        tickers: Dict mapping display names to ticker symbols
        save_csv: Whether to save data to CSV
        csv_filename: Output CSV filename
        
    Returns:
        Dictionary of time series data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: pip install yfinance")
    
    # Default tickers
    if tickers is None:
        tickers = {
            "USD/JPY": "JPY=X",
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X",
            "Nikkei 225": "^N225",
            "S&P 500": "^GSPC"
        }
    
    print(f"Fetching financial data from {start_date} to {end_date}...")
    
    # Download data
    ticker_list = list(tickers.values())
    data = yf.download(ticker_list, start=start_date, end=end_date)
    
    # Extract closing prices
    if len(ticker_list) == 1:
        # Single ticker returns Series
        close_data = data['Close'].to_frame()
        close_data.columns = [list(tickers.keys())[0]]
    else:
        # Multiple tickers
        close_data = data['Close']
        # Rename columns
        reverse_tickers = {v: k for k, v in tickers.items()}
        close_data = close_data.rename(columns=reverse_tickers)
    
    # Drop missing values
    close_data = close_data.dropna()
    
    print(f"Downloaded {len(close_data)} data points")
    print(f"Series: {list(close_data.columns)}")
    
    # Save to CSV if requested
    if save_csv:
        close_data.to_csv(csv_filename)
        print(f"Saved to {csv_filename}")
    
    # Convert to dictionary
    series_dict = {
        col: close_data[col].values.astype(np.float64)
        for col in close_data.columns
    }
    
    return series_dict


# ===============================
# Feature I/O
# ===============================

def save_features(
    features: Union[Lambda3FeatureSet, Dict[str, Lambda3FeatureSet]],
    filepath: Union[str, Path],
    cloud_config: Optional[CloudConfig] = None,
    compress: bool = True
) -> None:
    """
    Save Lambda³ features to file.
    
    Args:
        features: Single feature set or dictionary of features
        filepath: Save path
        cloud_config: Cloud storage configuration
        compress: Whether to compress the pickle file
    """
    filepath = Path(filepath)
    
    # Ensure parent directory exists for local saves
    if cloud_config is None or cloud_config.provider == 'local':
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    if isinstance(features, Lambda3FeatureSet):
        data = {
            'single': True,
            'features': features.to_dict(),
            'metadata': features.metadata,
            'lambda3_summary': features.get_lambda3_summary()
        }
    else:
        data = {
            'single': False,
            'features': {
                name: {
                    'data': feat.to_dict(),
                    'metadata': feat.metadata,
                    'lambda3_summary': feat.get_lambda3_summary()
                }
                for name, feat in features.items()
            }
        }
    
    # Add timestamp and version
    data['saved_at'] = datetime.now().isoformat()
    data['version'] = '1.0.0'
    data['lambda3_framework'] = True
    
    # Save
    if cloud_config and cloud_config.provider != 'local':
        _save_to_cloud(data, filepath, cloud_config, compress)
    else:
        # Local save
        if compress:
            import gzip
            with gzip.open(str(filepath) + '.gz', 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved compressed features to {filepath}.gz")
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved features to {filepath}")


def load_features(
    filepath: Union[str, Path],
    cloud_config: Optional[CloudConfig] = None
) -> Union[Lambda3FeatureSet, Dict[str, Lambda3FeatureSet]]:
    """
    Load Lambda³ features from file.
    
    Args:
        filepath: File path
        cloud_config: Cloud storage configuration
        
    Returns:
        Single feature set or dictionary of features
    """
    filepath = Path(filepath)
    
    # Check for compressed file
    compressed = False
    if filepath.suffix == '.gz':
        compressed = True
    elif not filepath.exists() and filepath.with_suffix('.gz').exists():
        filepath = filepath.with_suffix('.gz')
        compressed = True
    
    # Load data
    if cloud_config and cloud_config.provider != 'local':
        data = _load_from_cloud(filepath, cloud_config)
    else:
        if compressed:
            import gzip
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
    
    print(f"Loaded features saved at {data.get('saved_at', 'unknown')}")
    if data.get('lambda3_framework'):
        print("✓ Lambda³ framework features detected")
    
    # Convert back to Lambda3FeatureSet
    if data['single']:
        # Single feature set
        return Lambda3FeatureSet.from_dict(
            data['features'],
            metadata=data.get('metadata')
        )
    else:
        # Multiple feature sets
        return {
            name: Lambda3FeatureSet.from_dict(
                feat_data['data'],
                metadata=feat_data.get('metadata')
            )
            for name, feat_data in data['features'].items()
        }


# ===============================
# Analysis Results I/O
# ===============================

def save_analysis_results(
    results: Union[AnalysisResult, CrossAnalysisResult],
    filepath: Union[str, Path],
    cloud_config: Optional[CloudConfig] = None,
    include_traces: bool = True
) -> None:
    """
    Save analysis results.
    
    Args:
        results: Analysis results to save
        filepath: Save path
        cloud_config: Cloud configuration
        include_traces: Whether to include full MCMC traces
    """
    filepath = Path(filepath)
    
    # Prepare data with Lambda³ annotations
    data = {
        'type': type(results).__name__,
        'saved_at': datetime.now().isoformat(),
        'version': '1.0.0',
        'lambda3_framework': True
    }
    
    if isinstance(results, AnalysisResult):
        # Single pair analysis
        data['results'] = _serialize_analysis_result(results, include_traces)
        data['lambda3_summary'] = results.get_lambda3_summary()
    else:
        # Cross analysis
        data['results'] = _serialize_cross_analysis_result(results, include_traces)
        data['lambda3_summary'] = results.get_lambda3_network_summary()
    
    # Save
    if cloud_config and cloud_config.provider != 'local':
        _save_to_cloud(data, filepath, cloud_config)
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved analysis results to {filepath}")


def load_analysis_results(
    filepath: Union[str, Path],
    cloud_config: Optional[CloudConfig] = None
) -> Union[AnalysisResult, CrossAnalysisResult]:
    """
    Load analysis results.
    
    Args:
        filepath: File path
        cloud_config: Cloud configuration
        
    Returns:
        Analysis results
    """
    filepath = Path(filepath)
    
    # Load data
    if cloud_config and cloud_config.provider != 'local':
        data = _load_from_cloud(filepath, cloud_config)
    else:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    
    print(f"Loaded {data['type']} results saved at {data.get('saved_at', 'unknown')}")
    if data.get('lambda3_framework'):
        print("✓ Lambda³ framework analysis detected")
    
    # Deserialize based on type
    if data['type'] == 'AnalysisResult':
        return _deserialize_analysis_result(data['results'])
    elif data['type'] == 'CrossAnalysisResult':
        return _deserialize_cross_analysis_result(data['results'])
    else:
        raise ValueError(f"Unknown result type: {data['type']}")


# ===============================
# Cloud Storage Functions
# ===============================

def _save_to_cloud(
    data: Any,
    filepath: Path,
    cloud_config: CloudConfig,
    compress: bool = True
) -> None:
    """
    Save data to cloud storage.
    
    Args:
        data: Data to save
        filepath: Target path
        cloud_config: Cloud configuration
        compress: Whether to compress
    """
    # Serialize data
    if compress:
        import gzip
        import io
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb') as gz:
            pickle.dump(data, gz, protocol=pickle.HIGHEST_PROTOCOL)
        serialized_data = buffer.getvalue()
        filepath = Path(str(filepath) + '.gz')
    else:
        serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Construct full path
    if cloud_config.prefix:
        full_path = f"{cloud_config.prefix}/{filepath}"
    else:
        full_path = str(filepath)
    
    # Save based on provider
    if cloud_config.provider == 'gcs':
        _save_to_gcs(serialized_data, full_path, cloud_config)
    elif cloud_config.provider == 's3':
        _save_to_s3(serialized_data, full_path, cloud_config)
    elif cloud_config.provider == 'azure':
        _save_to_azure(serialized_data, full_path, cloud_config)
    else:
        raise ValueError(f"Unknown cloud provider: {cloud_config.provider}")


def _load_from_cloud(filepath: Path, cloud_config: CloudConfig) -> Any:
    """
    Load data from cloud storage.
    
    Args:
        filepath: Source path
        cloud_config: Cloud configuration
        
    Returns:
        Loaded data
    """
    # Check if compressed
    compressed = filepath.suffix == '.gz'
    
    # Construct full path
    if cloud_config.prefix:
        full_path = f"{cloud_config.prefix}/{filepath}"
    else:
        full_path = str(filepath)
    
    # Load based on provider
    if cloud_config.provider == 'gcs':
        serialized_data = _load_from_gcs(full_path, cloud_config)
    elif cloud_config.provider == 's3':
        serialized_data = _load_from_s3(full_path, cloud_config)
    elif cloud_config.provider == 'azure':
        serialized_data = _load_from_azure(full_path, cloud_config)
    else:
        raise ValueError(f"Unknown cloud provider: {cloud_config.provider}")
    
    # Deserialize
    if compressed:
        import gzip
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(serialized_data)) as gz:
            return pickle.load(gz)
    else:
        return pickle.loads(serialized_data)


def _save_to_gcs(data: bytes, path: str, config: CloudConfig) -> None:
    """Save to Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage required: pip install google-cloud-storage")
    
    # Set up credentials if provided
    if config.credentials_path:
        import os
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.credentials_path
    
    client = storage.Client(project=config.gcs_project)
    bucket = client.bucket(config.bucket)
    blob = bucket.blob(path)
    
    blob.upload_from_string(data)
    print(f"Saved to GCS: gs://{config.bucket}/{path}")


def _load_from_gcs(path: str, config: CloudConfig) -> bytes:
    """Load from Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage required: pip install google-cloud-storage")
    
    if config.credentials_path:
        import os
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.credentials_path
    
    client = storage.Client(project=config.gcs_project)
    bucket = client.bucket(config.bucket)
    blob = bucket.blob(path)
    
    return blob.download_as_bytes()


def _save_to_s3(data: bytes, path: str, config: CloudConfig) -> None:
    """Save to Amazon S3."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 required: pip install boto3")
    
    s3 = boto3.client('s3', region_name=config.aws_region)
    s3.put_object(Bucket=config.bucket, Key=path, Body=data)
    print(f"Saved to S3: s3://{config.bucket}/{path}")


def _load_from_s3(path: str, config: CloudConfig) -> bytes:
    """Load from Amazon S3."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 required: pip install boto3")
    
    s3 = boto3.client('s3', region_name=config.aws_region)
    response = s3.get_object(Bucket=config.bucket, Key=path)
    return response['Body'].read()


def _save_to_azure(data: bytes, path: str, config: CloudConfig) -> None:
    """Save to Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ImportError("azure-storage-blob required: pip install azure-storage-blob")
    
    if config.credentials_path:
        # Load connection string from file
        with open(config.credentials_path, 'r') as f:
            connection_string = f.read().strip()
    else:
        # Use environment variable
        import os
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=config.bucket,
        blob=path
    )
    
    blob_client.upload_blob(data, overwrite=True)
    print(f"Saved to Azure: {config.bucket}/{path}")


def _load_from_azure(path: str, config: CloudConfig) -> bytes:
    """Load from Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ImportError("azure-storage-blob required: pip install azure-storage-blob")
    
    if config.credentials_path:
        with open(config.credentials_path, 'r') as f:
            connection_string = f.read().strip()
    else:
        import os
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=config.bucket,
        blob=path
    )
    
    return blob_client.download_blob().readall()


# ===============================
# Serialization Helpers
# ===============================

def _validate_series_lengths(series_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Validate and align series lengths."""
    lengths = {name: len(data) for name, data in series_dict.items()}
    unique_lengths = set(lengths.values())
    
    if len(unique_lengths) == 1:
        return series_dict
    
    print(f"\nWarning: Series have different lengths: {lengths}")
    min_length = min(lengths.values())
    print(f"Truncating all series to minimum length: {min_length}")
    
    aligned_dict = {}
    for name, data in series_dict.items():
        aligned_dict[name] = data[:min_length]
    
    return aligned_dict


def _serialize_analysis_result(result: AnalysisResult, include_traces: bool) -> Dict:
    """Serialize AnalysisResult for saving."""
    data = {
        'sync_profile': {
            'profile': result.sync_profile.profile,
            'max_sync_rate': result.sync_profile.max_sync_rate,
            'optimal_lag': result.sync_profile.optimal_lag,
            'series_names': result.sync_profile.series_names
        },
        'interaction_effects': result.interaction_effects,
        'metadata': result.metadata
    }
    
    # Optionally include traces
    if include_traces:
        # Save trace data (excluding large arrays)
        data['trace_a_summary'] = result.trace_a.summary.to_dict()
        data['trace_b_summary'] = result.trace_b.summary.to_dict()
        data['predictions_a'] = result.trace_a.predictions.tolist()
        data['predictions_b'] = result.trace_b.predictions.tolist()
        
        # Include Lambda³ parameters
        data['lambda3_params_a'] = result.trace_a.get_lambda3_parameters()
        data['lambda3_params_b'] = result.trace_b.get_lambda3_parameters()
    
    # Causality profiles
    if result.causality_profiles:
        data['causality_profiles'] = {
            name: {
                'self_causality': prof.self_causality,
                'cross_causality': prof.cross_causality,
                'series_names': prof.series_names
            }
            for name, prof in result.causality_profiles.items()
        }
    
    return data


def _deserialize_analysis_result(data: Dict) -> AnalysisResult:
    """Deserialize AnalysisResult from saved data."""
    # Note: This creates a partial AnalysisResult without full traces
    # For full functionality, traces would need to be regenerated
    from .types import SyncProfile, CausalityProfile, BayesianResults
    import pandas as pd
    
    # Reconstruct sync profile
    sync_profile = SyncProfile(
        profile=data['sync_profile']['profile'],
        max_sync_rate=data['sync_profile']['max_sync_rate'],
        optimal_lag=data['sync_profile']['optimal_lag'],
        series_names=tuple(data['sync_profile']['series_names']) if data['sync_profile']['series_names'] else None
    )
    
    # Reconstruct causality profiles
    causality_profiles = None
    if 'causality_profiles' in data:
        causality_profiles = {}
        for name, prof_data in data['causality_profiles'].items():
            causality_profiles[name] = CausalityProfile(
                self_causality=prof_data['self_causality'],
                cross_causality=prof_data['cross_causality'],
                series_names=prof_data['series_names']
            )
    
    # Create placeholder Bayesian results
    if 'trace_a_summary' in data:
        trace_a = BayesianResults(
            trace=None,  # Would need to be regenerated
            summary=pd.DataFrame(data['trace_a_summary']),
            predictions=np.array(data['predictions_a']),
            residuals=None,
            diagnostics=None
        )
        trace_b = BayesianResults(
            trace=None,
            summary=pd.DataFrame(data['trace_b_summary']),
            predictions=np.array(data['predictions_b']),
            residuals=None,
            diagnostics=None
        )
    else:
        # Create minimal placeholders
        trace_a = BayesianResults(
            trace=None,
            summary=pd.DataFrame(),
            predictions=np.array([]),
            residuals=None,
            diagnostics=None
        )
        trace_b = trace_a
    
    return AnalysisResult(
        trace_a=trace_a,
        trace_b=trace_b,
        sync_profile=sync_profile,
        interaction_effects=data['interaction_effects'],
        causality_profiles=causality_profiles,
        metadata=data['metadata']
    )


def _serialize_cross_analysis_result(result: CrossAnalysisResult, include_traces: bool) -> Dict:
    """Serialize CrossAnalysisResult for saving."""
    data = {
        'sync_matrix': result.sync_matrix.tolist(),
        'interaction_matrix': result.interaction_matrix.tolist(),
        'metadata': result.metadata
    }
    
    # Include interaction tensor if available
    if result.get_interaction_tensor() is not None:
        data['interaction_tensor'] = result.get_interaction_tensor().tolist()
    
    # Serialize pairwise results
    data['pairwise_results'] = {}
    for pair_key, pair_result in result.pairwise_results.items():
        serialized_key = f"{pair_key[0]}___{pair_key[1]}"
        data['pairwise_results'][serialized_key] = _serialize_analysis_result(
            pair_result, include_traces
        )
    
    # Network (save basic properties)
    if result.network:
        data['network'] = {
            'nodes': list(result.network.nodes()),
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'weight': d.get('weight', 1.0),
                    'lag': d.get('lag', 0),
                    'sigma_s': d.get('sigma_s', d.get('weight', 1.0))
                }
                for u, v, d in result.network.edges(data=True)
            ],
            'graph_attrs': result.network.graph
        }
    
    # Clusters
    if result.clusters:
        data['clusters'] = result.clusters
    
    return data


def _deserialize_cross_analysis_result(data: Dict) -> CrossAnalysisResult:
    """Deserialize CrossAnalysisResult from saved data."""
    import networkx as nx
    
    # Deserialize pairwise results
    pairwise_results = {}
    for serialized_key, pair_data in data['pairwise_results'].items():
        # Reconstruct tuple key
        parts = serialized_key.split('___')
        pair_key = (parts[0], parts[1])
        pairwise_results[pair_key] = _deserialize_analysis_result(pair_data)
    
    # Reconstruct network
    network = None
    if 'network' in data:
        network = nx.DiGraph()
        network.add_nodes_from(data['network']['nodes'])
        for edge in data['network']['edges']:
            network.add_edge(
                edge['source'],
                edge['target'],
                weight=edge['weight'],
                lag=edge['lag'],
                sigma_s=edge.get('sigma_s', edge['weight'])
            )
        # Restore graph attributes
        if 'graph_attrs' in data['network']:
            network.graph.update(data['network']['graph_attrs'])
    
    # Reconstruct metadata with tensor if available
    metadata = data['metadata'].copy()
    if 'interaction_tensor' in data:
        metadata['interaction_tensor'] = np.array(data['interaction_tensor'])
    
    return CrossAnalysisResult(
        pairwise_results=pairwise_results,
        sync_matrix=np.array(data['sync_matrix']),
        interaction_matrix=np.array(data['interaction_matrix']),
        network=network,
        clusters=data.get('clusters'),
        metadata=metadata
    )


# ===============================
# Export Functions
# ===============================

def export_results_to_excel(
    results: Union[AnalysisResult, CrossAnalysisResult],
    filepath: Union[str, Path],
    include_plots: bool = True
) -> None:
    """
    Export analysis results to Excel workbook with Lambda³ annotations.
    
    Args:
        results: Analysis results
        filepath: Output Excel file path
        include_plots: Whether to include plots (requires matplotlib)
    """
    filepath = Path(filepath)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Add Lambda³ theory sheet
        theory_data = {
            'Symbol': ['Λ', 'ΔΛC⁺', 'ΔΛC⁻', 'ρT', 'ΛF', 'σₛ', 'β'],
            'Component': [
                'Structural tensor',
                'Positive jump events',
                'Negative jump events',
                'Tension scalar',
                'Progression vector',
                'Synchronization rate',
                'Interaction coefficient'
            ],
            'Description': [
                'Complete structural state',
                'Upward structural changes',
                'Downward structural changes',
                'Local volatility/stress',
                'Time evolution component',
                'Cross-series synchronization',
                'Structural coupling strength'
            ]
        }
        pd.DataFrame(theory_data).to_excel(writer, sheet_name='Lambda³ Theory', index=False)
        
        if isinstance(results, AnalysisResult):
            _export_single_result_to_excel(results, writer, include_plots)
        else:
            _export_cross_result_to_excel(results, writer, include_plots)
    
    print(f"Exported results to {filepath}")


def _export_single_result_to_excel(
    result: AnalysisResult,
    writer: pd.ExcelWriter,
    include_plots: bool
) -> None:
    """Export single analysis result to Excel."""
    # Summary sheet with Lambda³ notation
    summary_data = {
        'Metric': ['Max Sync Rate (σₛ)', 'Optimal Lag', 'Primary Interaction', 'Convergence'],
        'Value': [
            f"{result.sync_profile.max_sync_rate:.3f}",
            str(result.sync_profile.optimal_lag),
            f"{result.primary_interaction[0]}: {result.primary_interaction[1]:.3f}",
            'Both converged' if result.trace_a.converged and result.trace_b.converged else 'Not converged'
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    # Sync profile sheet
    sync_df = pd.DataFrame(
        list(result.sync_profile.profile.items()),
        columns=['Lag', 'Sync Rate (σₛ)']
    )
    sync_df.to_excel(writer, sheet_name='Sync Profile', index=False)
    
    # Interaction effects with Lambda³ notation
    if result.interaction_effects:
        effects_data = []
        for effect, coefficient in result.interaction_effects.items():
            # Parse effect type
            if '_pos' in effect:
                lambda_type = 'ΔΛC⁺'
            elif '_neg' in effect:
                lambda_type = 'ΔΛC⁻'
            elif '_stress' in effect:
                lambda_type = 'ρT'
            else:
                lambda_type = 'Unknown'
            
            effects_data.append({
                'Effect': effect,
                'Type': lambda_type,
                'Coefficient (β)': coefficient,
                'Significant': '✓' if abs(coefficient) > 0.1 else ''
            })
        
        effects_df = pd.DataFrame(effects_data)
        effects_df.to_excel(writer, sheet_name='Interactions', index=False)
    
    # Model summaries with Lambda³ parameters
    if result.trace_a.summary is not None and not result.trace_a.summary.empty:
        # Add Lambda³ parameter mapping
        param_mapping = {
            'beta_dLC_pos': 'β_ΔΛC⁺',
            'beta_dLC_neg': 'β_ΔΛC⁻',
            'beta_rhoT': 'β_ρT',
            'beta_time': 'β_ΛF',
            'beta_local_jump': 'β_local_ΔΛC'
        }
        
        summary_a = result.trace_a.summary.copy()
        summary_a['Lambda³ Parameter'] = summary_a.index.map(lambda x: param_mapping.get(x, x))
        summary_a.to_excel(writer, sheet_name='Model A Summary')
        
        summary_b = result.trace_b.summary.copy()
        summary_b['Lambda³ Parameter'] = summary_b.index.map(lambda x: param_mapping.get(x, x))
        summary_b.to_excel(writer, sheet_name='Model B Summary')


def _export_cross_result_to_excel(
    result: CrossAnalysisResult,
    writer: pd.ExcelWriter,
    include_plots: bool
) -> None:
    """Export cross analysis result to Excel with Lambda³ annotations."""
    series_names = result.get_series_names()
    
    # Summary sheet
    summary_data = {
        'Metric': [
            'Number of Series', 
            'Number of Pairs', 
            'Network Density',
            'Max Sync Rate (σₛ)',
            'Mean Sync Rate (σₛ)'
        ],
        'Value': [
            result.n_series,
            result.n_pairs,
            f"{result.network_density:.1%}",
            f"{result.get_strongest_sync_pair()[2]:.3f}",
            f"{np.mean(result.sync_matrix[np.triu_indices_from(result.sync_matrix, k=1)]):.3f}"
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    # Sync matrix (σₛ matrix)
    sync_df = pd.DataFrame(
        result.sync_matrix,
        index=series_names,
        columns=series_names
    )
    sync_df.to_excel(writer, sheet_name='Sync Matrix (σₛ)')
    
    # Interaction matrix (β matrix)
    int_df = pd.DataFrame(
        result.interaction_matrix,
        index=series_names,
        columns=series_names
    )
    int_df.to_excel(writer, sheet_name='Interaction Matrix (β)')
    
    # Interaction tensor if available
    if result.get_interaction_tensor() is not None:
        tensor = result.get_interaction_tensor()
        # Export each slice
        effect_types = ['ΔΛC⁺', 'ΔΛC⁻', 'ρT']
        for i, effect_type in enumerate(effect_types):
            tensor_df = pd.DataFrame(
                tensor[:, :, i],
                index=series_names,
                columns=series_names
            )
            tensor_df.to_excel(writer, sheet_name=f'β Tensor - {effect_type}')
    
    # Network edges
    if result.network:
        edges_data = []
        for u, v, d in result.network.edges(data=True):
            edges_data.append({
                'From': u,
                'To': v,
                'Weight (σₛ)': d.get('weight', 1.0),
                'Lag': d.get('lag', 0),
                'Strength': 'Strong' if d.get('weight', 0) > 0.5 else 'Moderate'
            })
        if edges_data:
            pd.DataFrame(edges_data).to_excel(writer, sheet_name='Network Edges', index=False)
    
    # Clusters
    if result.clusters:
        cluster_df = pd.DataFrame(
            list(result.clusters.items()),
            columns=['Series', 'Cluster']
        )
        cluster_df.to_excel(writer, sheet_name='Clusters', index=False)
        
