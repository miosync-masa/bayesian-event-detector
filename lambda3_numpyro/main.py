#!/usr/bin/env python3
"""
Lambda³ Analytics Command Line Interface

Main entry point for Lambda³ analysis framework.
Provides commands for feature extraction, analysis, visualization, and testing.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import numpy as np
from datetime import datetime

# Lambda³ imports
from lambda3_numpyro import (
    __version__,
    L3Config,
    extract_lambda3_features,
    analyze_pair,
    analyze_multiple_series,
    detect_regimes,
    load_csv_series,
    load_financial_data,
    save_features,
    load_features,
    save_analysis_results,
    load_analysis_results,
    PLOTTING_AVAILABLE
)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Lambda³ Analytics - Structural Evolution Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from CSV
  %(prog)s extract data.csv --output features.pkl
  
  # Analyze two specific series
  %(prog)s analyze data.csv --series "USD/JPY" "EUR/USD" --output results/
  
  # Analyze all pairs in dataset
  %(prog)s analyze-all data.csv --max-pairs 10 --config config.json
  
  # Detect market regimes
  %(prog)s regimes features.pkl --n-regimes 3 --output regimes.pkl
  
  # Download and analyze financial data
  %(prog)s finance --tickers SPY AAPL --start 2024-01-01 --analyze
  
  # Create analysis dashboard
  %(prog)s dashboard results.pkl --features features.pkl --output dashboard.png
  
  # Run integration tests
  %(prog)s test --type integration --verbose
  
  # Run unit tests
  %(prog)s test --type unit --modules feature bayes

For more information, visit: https://github.com/lambda3/lambda3-numpyro
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'Lambda³ Analytics v{__version__}'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file (JSON or YAML)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )
    
    # Extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract Lambda³ features from time series'
    )
    extract_parser.add_argument(
        'input',
        help='Input CSV file or directory'
    )
    extract_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output features file (.pkl)'
    )
    extract_parser.add_argument(
        '--columns', '-c',
        nargs='+',
        help='Specific columns to extract'
    )
    extract_parser.add_argument(
        '--time-column',
        help='Time/date column name'
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze specific series pairs'
    )
    analyze_parser.add_argument(
        'input',
        help='Input CSV file or features file'
    )
    analyze_parser.add_argument(
        '--series', '-s',
        nargs=2,
        required=True,
        help='Two series names to analyze'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        default='./results',
        help='Output directory'
    )
    analyze_parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots'
    )
    
    # Analyze-all command
    analyze_all_parser = subparsers.add_parser(
        'analyze-all',
        help='Analyze all series pairs'
    )
    analyze_all_parser.add_argument(
        'input',
        help='Input CSV file or features file'
    )
    analyze_all_parser.add_argument(
        '--output', '-o',
        default='./results',
        help='Output directory'
    )
    analyze_all_parser.add_argument(
        '--max-pairs',
        type=int,
        help='Maximum number of pairs to analyze'
    )
    analyze_all_parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots'
    )
    
    # Regimes command
    regimes_parser = subparsers.add_parser(
        'regimes',
        help='Detect market regimes'
    )
    regimes_parser.add_argument(
        'input',
        help='Features file (.pkl)'
    )
    regimes_parser.add_argument(
        '--n-regimes',
        type=int,
        default=3,
        help='Number of regimes to detect'
    )
    regimes_parser.add_argument(
        '--output', '-o',
        help='Output file for regime results'
    )
    regimes_parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot regime visualization'
    )
    
    # Finance command
    finance_parser = subparsers.add_parser(
        'finance',
        help='Download and analyze financial data'
    )
    finance_parser.add_argument(
        '--tickers',
        nargs='+',
        help='Ticker symbols to download'
    )
    finance_parser.add_argument(
        '--start',
        default='2024-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    finance_parser.add_argument(
        '--end',
        default='2024-12-31',
        help='End date (YYYY-MM-DD)'
    )
    finance_parser.add_argument(
        '--output', '-o',
        default='financial_data.csv',
        help='Output CSV file'
    )
    finance_parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run analysis after download'
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        'dashboard',
        help='Create analysis dashboard'
    )
    dashboard_parser.add_argument(
        'results',
        help='Analysis results file (.pkl)'
    )
    dashboard_parser.add_argument(
        '--features',
        required=True,
        help='Features file (.pkl)'
    )
    dashboard_parser.add_argument(
        '--output', '-o',
        default='dashboard.png',
        help='Output image file'
    )
    
    # Bayesian command
    bayesian_parser = subparsers.add_parser(
        'bayesian',
        help='Run complete Bayesian analysis pipeline'
    )
    bayesian_parser.add_argument(
        'input',
        help='Features file (.pkl) or CSV file'
    )
    bayesian_parser.add_argument(
        '--output', '-o',
        default='./bayesian_results',
        help='Output directory'
    )
    bayesian_parser.add_argument(
        '--include-svi',
        action='store_true',
        help='Include Stochastic Variational Inference'
    )
    bayesian_parser.add_argument(
        '--criterion',
        choices=['loo', 'waic'],
        default='loo',
        help='Model comparison criterion'
    )
    
    # Hierarchical command
    hierarchical_parser = subparsers.add_parser(
        'hierarchical',
        help='Run hierarchical Bayesian analysis for multiple series'
    )
    hierarchical_parser.add_argument(
        'input',
        help='Features file (.pkl) with multiple series'
    )
    hierarchical_parser.add_argument(
        '--output', '-o',
        default='./hierarchical_results',
        help='Output directory'
    )
    hierarchical_parser.add_argument(
        '--group-ids',
        nargs='+',
        type=int,
        help='Group IDs for each series'
    )
    
    # Model-compare command
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare different model types'
    )
    compare_parser.add_argument(
        'input',
        help='Features file (.pkl)'
    )
    compare_parser.add_argument(
        '--models',
        nargs='+',
        choices=['base', 'dynamic', 'interaction', 'svi'], 
        default=['base', 'dynamic'],
        help='Models to compare'
    )
    compare_parser.add_argument(
        '--criterion',
        choices=['loo', 'waic'],
        default='loo',
        help='Model comparison criterion'
    )
    compare_parser.add_argument(
        '--output', '-o',
        default='./comparison_results',
        help='Output directory'
    )
    
    # PPC command
    ppc_parser = subparsers.add_parser(
        'ppc',
        help='Run posterior predictive checks'
    )
    ppc_parser.add_argument(
        'results',
        help='Bayesian results file (.pkl)'
    )
    ppc_parser.add_argument(
        '--features',
        required=True,
        help='Features file (.pkl)'
    )
    ppc_parser.add_argument(
        '--output', '-o',
        help='Output plot file'
    )
    
    # Change-point command
    changepoint_parser = subparsers.add_parser(
        'changepoints',
        help='Detect structural change points'
    )
    changepoint_parser.add_argument(
        'input',
        help='CSV file or features file'
    )
    changepoint_parser.add_argument(
        '--window-size',
        type=int,
        default=50,
        help='Window size for detection'
    )
    changepoint_parser.add_argument(
        '--threshold',
        type=float,
        default=2.0,
        help='Threshold factor for detection'
    )
    changepoint_parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot detected change points'
    )
    
    # Cloud command
    cloud_parser = subparsers.add_parser(
        'cloud',
        help='Cloud storage operations'
    )
    cloud_parser.add_argument(
        'action',
        choices=['upload', 'download', 'list'],
        help='Cloud action to perform'
    )
    cloud_parser.add_argument(
        '--file',
        help='Local file path'
    )
    cloud_parser.add_argument(
        '--remote',
        help='Remote path in cloud storage'
    )
    cloud_parser.add_argument(
        '--provider',
        choices=['gcs', 's3', 'azure'],
        default='gcs',
        help='Cloud provider'
    )
    cloud_parser.add_argument(
        '--bucket',
        help='Cloud storage bucket'
    )
    
    # Test command
    test_parser = subparsers.add_parser(
        'test',
        help='Run tests for Lambda³ framework'
    )
    test_parser.add_argument(
        '--type',
        choices=['unit', 'integration', 'all'],
        default='all',
        help='Type of tests to run'
    )
    test_parser.add_argument(
        '--modules',
        nargs='+',
        choices=['feature', 'bayes', 'analysis', 'io', 'config', 'types'],
        help='Specific modules to test (for unit tests)'
    )
    test_parser.add_argument(
        '--output', '-o',
        help='Output file for test results'
    )
    
    return parser


def load_config(args) -> L3Config:
    """Load configuration from file or create default."""
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            config = L3Config.from_file(config_path)
            if args.verbose:
                print(f"Loaded config from {config_path}")
        else:
            print(f"Warning: Config file {config_path} not found, using defaults")
            config = L3Config()
    else:
        # Try environment variables
        config = L3Config.from_env()
    
    # Override with command line arguments
    if hasattr(args, 'max_pairs') and args.max_pairs:
        config.max_pairs = args.max_pairs
    
    config.verbose = args.verbose
    
    return config


# ===============================
# Command Implementations
# ===============================

def cmd_extract(args, config: L3Config):
    """Execute feature extraction command."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Load data
    if input_path.suffix == '.csv':
        print(f"Loading data from {input_path}")
        series_dict = load_csv_series(
            input_path,
            time_column=args.time_column,
            value_columns=args.columns
        )
    else:
        print(f"Error: Input must be a CSV file")
        return 1
    
    # Extract features
    print(f"\nExtracting Lambda³ features for {len(series_dict)} series...")
    features_dict = {}
    
    for i, (name, data) in enumerate(series_dict.items(), 1):
        print(f"[{i}/{len(series_dict)}] Processing {name}...")
        try:
            features = extract_lambda3_features(data, config, series_name=name)
            features_dict[name] = features
            
            print(f"  ✓ Length: {features.length}")
            print(f"  ✓ Positive jumps: {features.n_pos_jumps}")
            print(f"  ✓ Negative jumps: {features.n_neg_jumps}")
            print(f"  ✓ Mean tension: {features.mean_tension:.3f}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    if not features_dict:
        print("Error: No features extracted")
        return 1
    
    # Save features
    print(f"\nSaving features to {output_path}")
    save_features(features_dict, output_path, config.cloud)
    
    print(f"✓ Successfully extracted features for {len(features_dict)} series")
    return 0


def cmd_analyze(args, config: L3Config):
    """Execute pairwise analysis command."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data or features
    if input_path.suffix == '.pkl':
        print(f"Loading features from {input_path}")
        features_dict = load_features(input_path, config.cloud)
    elif input_path.suffix == '.csv':
        print(f"Loading data from {input_path}")
        series_dict = load_csv_series(input_path)
        
        # Extract features
        print("Extracting features...")
        from lambda3_numpyro.feature import extract_features_dict
        features_dict = extract_features_dict(series_dict, config)
    else:
        print(f"Error: Unknown input format {input_path.suffix}")
        return 1
    
    # Check if requested series exist
    series_a, series_b = args.series
    if series_a not in features_dict:
        print(f"Error: Series '{series_a}' not found in data")
        return 1
    if series_b not in features_dict:
        print(f"Error: Series '{series_b}' not found in data")
        return 1
    
    # Run analysis
    print(f"\nAnalyzing pair: {series_a} ↔ {series_b}")
    results = analyze_pair(
        series_a, series_b,
        features_dict[series_a],
        features_dict[series_b],
        config
    )
    
    # Display results
    print(f"\n{'='*50}")
    print("ANALYSIS RESULTS")
    print(f"{'='*50}")
    print(f"Synchronization rate: {results.sync_profile.max_sync_rate:.3f}")
    print(f"Optimal lag: {results.sync_profile.optimal_lag}")
    print(f"\nInteraction effects:")
    for effect, value in results.interaction_effects.items():
        if abs(value) > 0.01:
            print(f"  {effect}: {value:.3f}")
    
    # Save results
    results_file = output_dir / f"analysis_{series_a}_{series_b}.pkl"
    save_analysis_results(results, results_file, config.cloud)
    print(f"\nResults saved to {results_file}")
    
    # Generate plots if requested
    if args.plot and PLOTTING_AVAILABLE:
        from lambda3_numpyro.plot import plot_analysis_results
        plot_file = output_dir / f"analysis_{series_a}_{series_b}.png"
        plot_analysis_results(
            results,
            features_dict[series_a],
            features_dict[series_b],
            save_path=plot_file,
            config=config.plotting
        )
        print(f"Plot saved to {plot_file}")
    elif args.plot and not PLOTTING_AVAILABLE:
        print("Warning: Plotting requested but matplotlib not available")
    
    return 0

def cmd_analyze_all(args, config: L3Config):
    """Execute multi-series analysis command."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data or features
    if input_path.suffix == '.pkl':
        print(f"Loading features from {input_path}")
        features_dict = load_features(input_path, config.cloud)
    elif input_path.suffix == '.csv':
        print(f"Loading data from {input_path}")
        series_dict = load_csv_series(input_path)
        
        # Extract features
        print("Extracting features...")
        from lambda3_numpyro.feature import extract_features_dict
        features_dict = extract_features_dict(series_dict, config)
    else:
        print(f"Error: Unknown input format {input_path.suffix}")
        return 1
    
    print(f"\nFound {len(features_dict)} series: {', '.join(features_dict.keys())}")
    
    # Run cross-analysis
    results = analyze_multiple_series(
        features_dict,
        config,
        show_progress=True
    )
    
    # Display summary
    print(f"\n{'='*50}")
    print("CROSS-ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"Series analyzed: {results.n_series}")
    print(f"Pairs analyzed: {results.n_pairs}")
    
    # Top synchronizations
    series_names = results.get_series_names()
    sync_pairs = []
    for i in range(results.n_series):
        for j in range(i+1, results.n_series):
            sync_pairs.append((
                results.sync_matrix[i, j],
                series_names[i],
                series_names[j]
            ))
    
    sync_pairs.sort(reverse=True)
    print("\nTop synchronization pairs:")
    for sync, name_a, name_b in sync_pairs[:5]:
        print(f"  {name_a} ↔ {name_b}: σₛ = {sync:.3f}")
    
    # Save results
    results_file = output_dir / "cross_analysis_results.pkl"
    save_analysis_results(results, results_file, config.cloud)
    print(f"\nResults saved to {results_file}")
    
    # Save summary report
    from lambda3_numpyro.analysis import generate_analysis_summary
    findings = generate_analysis_summary(results, features_dict)
    
    report_file = output_dir / "analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write("Lambda³ Cross-Analysis Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Series analyzed: {', '.join(series_names)}\n")
        f.write(f"Total pairs: {results.n_pairs}\n\n")
        f.write("Key Findings:\n")
        for finding in findings:
            f.write(f"• {finding}\n")
    
    print(f"Report saved to {report_file}")
    
    # Generate plots if requested
    if args.plot and PLOTTING_AVAILABLE:
        from lambda3_numpyro.plot import (
            plot_interaction_matrix,
            plot_sync_network,
            create_analysis_dashboard
        )
        
        # Interaction matrix
        plot_file = output_dir / "interaction_matrix.png"
        plot_interaction_matrix(
            results.interaction_matrix,
            series_names,
            save_path=plot_file,
            config=config.plotting
        )
        
        # Sync network
        if results.network and results.network.number_of_edges() > 0:
            plot_file = output_dir / "sync_network.png"
            plot_sync_network(
                results.network,
                save_path=plot_file,
                config=config.plotting
            )
        
        # Dashboard
        plot_file = output_dir / "analysis_dashboard.png"
        create_analysis_dashboard(
            results,
            features_dict,
            save_path=plot_file,
            config=config.plotting
        )
        
        print(f"Plots saved to {output_dir}")
    
    return 0


def cmd_regimes(args, config: L3Config):
    """Execute regime detection command."""
    input_path = Path(args.input)
    
    # Load features
    print(f"Loading features from {input_path}")
    features = load_features(input_path, config.cloud)
    
    # Handle single or multiple series
    if isinstance(features, dict):
        # Multiple series - analyze each
        regime_results = {}
        for name, feat in features.items():
            print(f"\nDetecting regimes for {name}...")
            regime_info = detect_regimes(feat, n_regimes=args.n_regimes)
            regime_results[name] = regime_info
            
            # Display summary
            print(f"  Found {regime_info.n_regimes} regimes:")
            for regime_id, stats in regime_info.regime_stats.items():
                regime_name = regime_info.get_regime_name(regime_id)
                print(f"    {regime_name}: {stats['frequency']:.1%} of time")
    else:
        # Single series
        print(f"\nDetecting regimes...")
        regime_info = detect_regimes(features, n_regimes=args.n_regimes)
        regime_results = regime_info
        
        # Display summary
        print(f"Found {regime_info.n_regimes} regimes:")
        for regime_id, stats in regime_info.regime_stats.items():
            regime_name = regime_info.get_regime_name(regime_id)
            print(f"  {regime_name}: {stats['frequency']:.1%} of time")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(regime_results, f)
        print(f"\nRegime results saved to {output_path}")
    
    # Generate plot if requested
    if args.plot and PLOTTING_AVAILABLE:
        from lambda3_numpyro.plot import plot_regimes
        
        if isinstance(features, dict):
            # Plot first series as example
            first_name = list(features.keys())[0]
            plot_file = output_path.parent / f"regimes_{first_name}.png" if args.output else "regimes.png"
            plot_regimes(
                features[first_name],
                regime_results[first_name],
                save_path=plot_file,
                config=config.plotting
            )
        else:
            plot_file = output_path.parent / "regimes.png" if args.output else "regimes.png"
            plot_regimes(
                features,
                regime_results,
                save_path=plot_file,
                config=config.plotting
            )
        print(f"Plot saved to {plot_file}")
    
    return 0

def cmd_finance(args, config: L3Config):
    """Execute financial data download command."""
    # Build tickers dictionary
    if args.tickers:
        tickers = {ticker: ticker for ticker in args.tickers}
    else:
        # Default tickers
        tickers = None
    
    # Download data
    print(f"Downloading financial data from {args.start} to {args.end}")
    series_dict = load_financial_data(
        start_date=args.start,
        end_date=args.end,
        tickers=tickers,
        save_csv=True,
        csv_filename=args.output
    )
    
    print(f"\nDownloaded {len(series_dict)} series")
    
    # Run analysis if requested
    if args.analyze:
        print("\nRunning Lambda³ analysis on downloaded data...")
        
        # Extract features
        from lambda3_numpyro.feature import extract_features_dict
        features_dict = extract_features_dict(series_dict, config)
        
        # Save features
        features_path = Path(args.output).with_suffix('.pkl')
        save_features(features_dict, features_path)
        print(f"Features saved to {features_path}")
        
        # Run cross-analysis if multiple series
        if len(features_dict) > 1:
            results = analyze_multiple_series(features_dict, config)
            results_path = Path(args.output).parent / "financial_analysis_results.pkl"
            save_analysis_results(results, results_path)
            print(f"Analysis results saved to {results_path}")
    
    return 0


def cmd_dashboard(args, config: L3Config):
    """Execute dashboard creation command."""
    # Load results and features
    print(f"Loading analysis results from {args.results}")
    results = load_analysis_results(Path(args.results), config.cloud)
    
    print(f"Loading features from {args.features}")
    features_dict = load_features(Path(args.features), config.cloud)
    
    # Create dashboard
    if PLOTTING_AVAILABLE:
        from lambda3_numpyro.plot import create_analysis_dashboard
        
        print("Creating analysis dashboard...")
        create_analysis_dashboard(
            results,
            features_dict,
            save_path=Path(args.output),
            config=config.plotting
        )
        print(f"Dashboard saved to {args.output}")
    else:
        print("Error: Plotting libraries not available")
        return 1
    
    return 0


def cmd_bayesian(args, config: L3Config):
    """Execute Bayesian analysis command."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    if input_path.suffix == '.pkl':
        print(f"Loading features from {input_path}")
        features = load_features(input_path, config.cloud)
        
        # If multiple series, use first one
        if isinstance(features, dict):
            first_name = list(features.keys())[0]
            print(f"Using first series: {first_name}")
            features = features[first_name]
    else:
        print(f"Loading data from {input_path}")
        series_dict = load_csv_series(input_path)
        # Use first series
        first_name = list(series_dict.keys())[0]
        print(f"Extracting features for {first_name}...")
        features = extract_lambda3_features(series_dict[first_name], config)
    
    # Run complete Bayesian analysis
    from lambda3_numpyro.bayes import run_complete_bayesian_analysis
    
    results = run_complete_bayesian_analysis(
        features,
        config,
        include_svi=args.include_svi
    )
    
    # Save results
    results_file = output_dir / "bayesian_results.pkl"
    import pickle
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_file}")
    
    # Save model comparison
    if 'comparison' in results:
        comparison_file = output_dir / "model_comparison.json"
        comparison_data = {
            'best_model': results['best_model'],
            'criterion': args.criterion
        }
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"Model comparison saved to {comparison_file}")
    
    return 0


def cmd_hierarchical(args, config: L3Config):
    """Execute hierarchical Bayesian analysis command."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    print(f"Loading features from {input_path}")
    features_dict = load_features(input_path, config.cloud)
    
    if not isinstance(features_dict, dict):
        print("Error: Hierarchical analysis requires multiple series")
        return 1
    
    # Convert to list
    features_list = list(features_dict.values())
    series_names = list(features_dict.keys())
    
    print(f"Found {len(features_list)} series: {', '.join(series_names)}")
    
    # Run hierarchical model
    from lambda3_numpyro.bayes import fit_hierarchical_model
    
    results = fit_hierarchical_model(
        features_list,
        config,
        group_ids=args.group_ids,
        seed=42
    )
    
    # Save results
    results_file = output_dir / "hierarchical_results.pkl"
    import pickle
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results': results,
            'series_names': series_names,
            'group_ids': args.group_ids
        }, f)
    print(f"\nResults saved to {results_file}")
    
    return 0

def cmd_compare(args, config: L3Config):
    """Execute model comparison command."""
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    print(f"Loading features from {input_path}")
    features = load_features(input_path, config.cloud)
    
    # If multiple series, use first one
    if isinstance(features, dict):
        first_name = list(features.keys())[0]
        print(f"Using first series: {first_name}")
        features = features[first_name]
    
    # Fit requested models
    from lambda3_numpyro.bayes import (
        fit_bayesian_model, fit_dynamic_model, fit_svi_model, 
        compare_models, Lambda3BayesianInference
    )
    
    models_dict = {}
    
    for model_type in args.models:
        print(f"\nFitting {model_type} model...")
        
        if model_type == 'base':
            results = fit_bayesian_model(features, config, model_type='base')
        elif model_type == 'dynamic':
            results = fit_dynamic_model(features, config)
        elif model_type == 'interaction':
            # For interaction model, we need another series
            print("Warning: Interaction model requires two series, using self-interaction")
            results = fit_bayesian_model(
                features, config, 
                interaction_features=features,
                model_type='interaction'
            )
        elif model_type == 'svi':  # 🆕 SVIサポート追加
            results = fit_svi_model(features, config, n_steps=10000)
            print(f"SVI completed. Final loss: {results['losses'][-1]:.4f}")
        
        models_dict[model_type] = results
    
    # Compare models (SVIを除くMCMCモデルのみ)
    mcmc_models = {k: v for k, v in models_dict.items() 
                   if k != 'svi' and hasattr(v, 'trace')}
    
    if mcmc_models:
        print("\nComparing models...")
        comparison = compare_models(mcmc_models, features, 
                                  criterion=getattr(args, 'criterion', 'loo'))
        
        # 🆕 モデル重みを計算（Lambda3BayesianInferenceを一時的に使用）
        inference = Lambda3BayesianInference(config)
        inference.results = mcmc_models
        inference.comparison_results = comparison
        
        weights = inference.get_model_weights(features)
        print("\nModel weights:")
        for model, weight in weights.items():
            print(f"  {model}: {weight:.3f}")
    else:
        comparison = None
        weights = None
    
    # SVIの情報を追加
    if 'svi' in models_dict:
        svi_info = {
            'final_loss': models_dict['svi']['losses'][-1],
            'n_steps': len(models_dict['svi']['losses'])
        }
        print(f"\nSVI: Final ELBO = {svi_info['final_loss']:.4f}")
    else:
        svi_info = None
    
    # Save results
    results_file = output_dir / "comparison_results.pkl"
    import pickle
    with open(results_file, 'wb') as f:
        pickle.dump({
            'models': models_dict,
            'comparison': comparison,
            'weights': weights,  # 🆕 追加
            'svi_info': svi_info  # 🆕 追加
        }, f)
    print(f"\nResults saved to {results_file}")
    
    return 0

def cmd_ppc(args, config: L3Config):
    """Execute posterior predictive check command."""
    # Load results and features
    print(f"Loading Bayesian results from {args.results}")
    import pickle
    with open(args.results, 'rb') as f:
        results_data = pickle.load(f)
    
    print(f"Loading features from {args.features}")
    features = load_features(Path(args.features), config.cloud)
    
    # Extract results
    if isinstance(results_data, dict) and 'results' in results_data:
        results = results_data['results']
    else:
        results = results_data
    
    # If multiple series in features, use first
    if isinstance(features, dict):
        features = list(features.values())[0]
    
    # Run PPC
    from lambda3_numpyro.bayes import posterior_predictive_check
    
    print("Running posterior predictive checks...")
    ppc_results = posterior_predictive_check(results, features)
    
    # Display results
    print("\nBayesian p-values:")
    for stat, p_val in ppc_results['bayesian_p_values'].items():
        print(f"  {stat}: {p_val:.3f}")
    
    # Plot if requested
    if args.output and PLOTTING_AVAILABLE:
        from lambda3_numpyro.plot import plot_posterior_predictive_check
        
        plot_posterior_predictive_check(
            ppc_results,
            features.data,
            save_path=Path(args.output),
            config=config.plotting
        )
        print(f"\nPPC plot saved to {args.output}")
    
    return 0


def cmd_changepoints(args, config: L3Config):
    """Execute change point detection command."""
    input_path = Path(args.input)
    
    # Load data
    if input_path.suffix == '.csv':
        print(f"Loading data from {input_path}")
        series_dict = load_csv_series(input_path)
        # Extract features
        from lambda3_numpyro.feature import extract_features_dict
        features_dict = extract_features_dict(series_dict, config)
    else:
        print(f"Loading features from {input_path}")
        features_dict = load_features(input_path, config.cloud)
    
    # Ensure dict format
    if not isinstance(features_dict, dict):
        features_dict = {'Series': features_dict}
    
    # Detect change points for each series
    from lambda3_numpyro.bayes import detect_change_points_automatic
    
    changepoint_results = {}
    
    for name, features in features_dict.items():
        print(f"\nDetecting change points for {name}...")
        
        change_points = detect_change_points_automatic(
            features.data,
            window_size=args.window_size,
            threshold_factor=args.threshold
        )
        
        changepoint_results[name] = {
            'change_points': change_points,
            'n_changepoints': len(change_points)
        }
        
        print(f"  Found {len(change_points)} change points: {change_points}")
    
    # Plot if requested
    if args.plot and PLOTTING_AVAILABLE:
        from lambda3_numpyro.plot import plot_changepoint_analysis
        
        for name, features in features_dict.items():
            if name in changepoint_results:
                plot_file = f"changepoints_{name}.png"
                
                # Create segments for plotting
                change_points = changepoint_results[name]['change_points']
                segments = []
                
                # Add initial segment
                if change_points:
                    segments.append({
                        'start': 0,
                        'end': change_points[0],
                        'length': change_points[0]
                    })
                
                # Add intermediate segments
                for i in range(len(change_points) - 1):
                    segments.append({
                        'start': change_points[i],
                        'end': change_points[i + 1],
                        'length': change_points[i + 1] - change_points[i]
                    })
                
                # Add final segment
                if change_points:
                    segments.append({
                        'start': change_points[-1],
                        'end': len(features.data),
                        'length': len(features.data) - change_points[-1]
                    })
                else:
                    # No change points - single segment
                    segments.append({
                        'start': 0,
                        'end': len(features.data),
                        'length': len(features.data)
                    })
                
                # Add statistics for each segment
                for seg in segments:
                    data_segment = features.data[seg['start']:seg['end']]
                    if len(data_segment) > 0:
                        seg['mean'] = np.mean(data_segment)
                        seg['std'] = np.std(data_segment)
                        seg['trend'] = np.polyfit(range(len(data_segment)), data_segment, 1)[0] if len(data_segment) > 1 else 0
                    else:
                        seg['mean'] = 0
                        seg['std'] = 0
                        seg['trend'] = 0
                
                changepoint_results[name]['segments'] = segments
                
                plot_changepoint_analysis(
                    features,
                    changepoint_results[name],
                    save_path=plot_file,
                    config=config.plotting
                )
                print(f"  Plot saved to {plot_file}")
    
    return 0


def cmd_cloud(args, config: L3Config):
    """Execute cloud storage operations."""
    # Update config with command line arguments
    if args.provider:
        config.cloud.provider = args.provider
    if args.bucket:
        config.cloud.bucket = args.bucket
    
    # Validate cloud configuration
    try:
        config.cloud.validate()
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    if args.action == 'upload':
        if not args.file or not args.remote:
            print("Error: --file and --remote required for upload")
            return 1
        
        print(f"Uploading {args.file} to {config.cloud.provider}://{config.cloud.bucket}/{args.remote}")
        
        # Load the file and save to cloud
        file_path = Path(args.file)
        if file_path.suffix == '.pkl':
            # Handle pickle files
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            from lambda3_numpyro.io import _save_to_cloud
            _save_to_cloud(data, Path(args.remote), config.cloud)
        else:
            print("Error: Only .pkl files are currently supported for cloud upload")
            return 1
        
        print("✓ Upload complete")
    
    elif args.action == 'download':
        if not args.remote or not args.file:
            print("Error: --remote and --file required for download")
            return 1
        
        print(f"Downloading {config.cloud.provider}://{config.cloud.bucket}/{args.remote} to {args.file}")
        
        from lambda3_numpyro.io import _load_from_cloud
        data = _load_from_cloud(Path(args.remote), config.cloud)
        
        # Save locally
        import pickle
        with open(args.file, 'wb') as f:
            pickle.dump(data, f)
        
        print("✓ Download complete")
    
    elif args.action == 'list':
        print(f"Listing files in {config.cloud.provider}://{config.cloud.bucket}/{config.cloud.prefix}")
        print("Note: List functionality not yet implemented")
        # TODO: Implement list functionality for each provider
    
    return 0


def cmd_test(args, config: L3Config):
    """Execute test command."""
    print(f"\n{'='*60}")
    print(f"Lambda³ Framework Tests")
    print(f"{'='*60}")
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'config': config.to_dict(),
        'results': {}
    }
    
    if args.type in ['unit', 'all']:
        print("\n[UNIT TESTS]")
        unit_results = run_unit_tests(args.modules, config, args.verbose)
        test_results['results']['unit'] = unit_results
    
    if args.type in ['integration', 'all']:
        print("\n[INTEGRATION TESTS]")
        integration_results = run_integration_tests(config, args.verbose)
        test_results['results']['integration'] = integration_results
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    total_passed = 0
    total_failed = 0
    
    for test_type, results in test_results['results'].items():
        passed = results['passed']
        failed = results['failed']
        total = results['total']
        
        total_passed += passed
        total_failed += failed
        
        print(f"{test_type.upper()}: {passed}/{total} passed", end="")
        if failed > 0:
            print(f" ({failed} FAILED)")
        else:
            print(" ✓")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\nTest results saved to {output_path}")
    
    # Return appropriate exit code
    return 0 if total_failed == 0 else 1


# ===============================
# Test Functions
# ===============================

def run_unit_tests(modules: Optional[List[str]], config: L3Config, verbose: bool) -> Dict:
    """Run unit tests for specified modules."""
    if modules is None:
        modules = ['feature', 'bayes', 'analysis', 'io', 'config', 'types']
    
    results = {
        'passed': 0,
        'failed': 0,
        'total': 0,
        'details': {}
    }
    
    for module in modules:
        print(f"\nTesting {module} module...")
        module_results = []
        
        if module == 'feature':
            module_results.extend(test_feature_extraction(config, verbose))
        elif module == 'bayes':
            module_results.extend(test_bayesian_models(config, verbose))
        elif module == 'analysis':
            module_results.extend(test_analysis_functions(config, verbose))
        elif module == 'io':
            module_results.extend(test_io_operations(config, verbose))
        elif module == 'config':
            module_results.extend(test_configuration(verbose))
        elif module == 'types':
            module_results.extend(test_type_definitions(verbose))
        
        # Count results
        passed = sum(1 for r in module_results if r['passed'])
        failed = len(module_results) - passed
        
        results['details'][module] = module_results
        results['passed'] += passed
        results['failed'] += failed
        results['total'] += len(module_results)
        
        print(f"  {module}: {passed}/{len(module_results)} passed")
    
    return results


def run_integration_tests(config: L3Config, verbose: bool) -> Dict:
    """Run integration tests."""
    results = {
        'passed': 0,
        'failed': 0,
        'total': 0,
        'details': []
    }
    
    # Test 1: Complete pipeline with synthetic data
    test_name = "Complete analysis pipeline"
    print(f"\n{test_name}...")
    try:
        result = test_complete_pipeline(config, verbose)
        results['details'].append({
            'name': test_name,
            'passed': result['success'],
            'message': result['message']
        })
        if result['success']:
            results['passed'] += 1
            print("  ✓ PASSED")
        else:
            results['failed'] += 1
            print(f"  ✗ FAILED: {result['message']}")
    except Exception as e:
        results['failed'] += 1
        results['details'].append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        print(f"  ✗ FAILED: {e}")
    results['total'] += 1
    
    # Test 2: Multi-series cross analysis
    test_name = "Multi-series cross analysis"
    print(f"\n{test_name}...")
    try:
        result = test_cross_analysis(config, verbose)
        results['details'].append({
            'name': test_name,
            'passed': result['success'],
            'message': result['message']
        })
        if result['success']:
            results['passed'] += 1
            print("  ✓ PASSED")
        else:
            results['failed'] += 1
            print(f"  ✗ FAILED: {result['message']}")
    except Exception as e:
        results['failed'] += 1
        results['details'].append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        print(f"  ✗ FAILED: {e}")
    results['total'] += 1
    
    # Test 3: Bayesian model comparison
    test_name = "Bayesian model comparison"
    print(f"\n{test_name}...")
    try:
        result = test_model_comparison(config, verbose)
        results['details'].append({
            'name': test_name,
            'passed': result['success'],
            'message': result['message']
        })
        if result['success']:
            results['passed'] += 1
            print("  ✓ PASSED")
        else:
            results['failed'] += 1
            print(f"  ✗ FAILED: {result['message']}")
    except Exception as e:
        results['failed'] += 1
        results['details'].append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        print(f"  ✗ FAILED: {e}")
    results['total'] += 1
    
    return results


# ===============================
# Unit Test Functions
# ===============================

def test_feature_extraction(config: L3Config, verbose: bool) -> List[Dict]:
    """Test feature extraction functions."""
    from lambda3_numpyro import extract_lambda3_features
    results = []
    
    # Test 1: Basic feature extraction
    test_name = "Basic feature extraction"
    try:
        # Create synthetic data with known properties
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100))
        
        features = extract_lambda3_features(data, config)
        
        # Verify structure
        assert features.length == 100
        assert features.n_pos_jumps >= 0
        assert features.n_neg_jumps >= 0
        assert len(features.rho_T) == 100
        assert features.metadata is not None
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    # Test 2: Jump detection
    test_name = "Jump detection"
    try:
        # Create data with known jumps
        data = np.zeros(100)
        data[30] = 10  # Positive jump
        data[60] = -10  # Negative jump
        data = np.cumsum(data)
        
        features = extract_lambda3_features(data, config)
        
        # Should detect jumps near indices 30 and 60
        assert features.n_pos_jumps > 0
        assert features.n_neg_jumps > 0
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    # Test 3: Lambda³ components
    test_name = "Lambda³ component validation"
    try:
        data = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.randn(200) * 0.1
        features = extract_lambda3_features(data, config)
        
        # Verify Lambda³ components
        lambda3_summary = features.get_lambda3_summary()
        assert 'structural_tensor_Λ' in lambda3_summary
        assert 'tension_scalar_ρT' in lambda3_summary
        assert 'progression_vector_ΛF' in lambda3_summary
        
        # Check properties
        assert features.jump_asymmetry >= -1 and features.jump_asymmetry <= 1
        assert features.mean_tension >= 0
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    return results


def test_bayesian_models(config: L3Config, verbose: bool) -> List[Dict]:
    """Test Bayesian model functions."""
    from lambda3_numpyro import extract_lambda3_features, fit_bayesian_model
    from lambda3_numpyro.bayes import check_convergence
    
    results = []
    
    # Test 1: Base model fitting
    test_name = "Base model fitting"
    try:
        # Create test data
        np.random.seed(42)
        data = np.cumsum(np.random.randn(50))
        features = extract_lambda3_features(data, config)
        
        # Fit model with reduced samples for speed
        test_config = L3Config()
        test_config.bayesian.draws = 100
        test_config.bayesian.tune = 100
        test_config.bayesian.num_chains = 2
        
        bayes_results = fit_bayesian_model(
            features, test_config, model_type='base', seed=42
        )
        
        # Check results
        assert bayes_results.predictions is not None
        assert len(bayes_results.predictions) == len(data)
        assert bayes_results.summary is not None
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    # Test 2: Prior scales consistency
    test_name = "Prior scales consistency"
    try:
        # Verify all models accept prior_scales
        data = np.cumsum(np.random.randn(50))
        features = extract_lambda3_features(data, config)
        
        custom_priors = {
            'beta_dLC_pos': 10.0,
            'beta_dLC_neg': 10.0,
            'beta_rhoT': 5.0
        }
        
        test_config = L3Config()
        test_config.bayesian.draws = 50
        test_config.bayesian.tune = 50
        test_config.bayesian.num_chains = 1
        test_config.bayesian.prior_scales.update(custom_priors)
        
        # Should not raise error
        bayes_results = fit_bayesian_model(
            features, test_config, model_type='base', seed=42
        )
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    return results


def test_analysis_functions(config: L3Config, verbose: bool) -> List[Dict]:
    """Test analysis functions."""
    from lambda3_numpyro import extract_lambda3_features, calculate_sync_profile
    from lambda3_numpyro.analysis import analyze_pair
    
    results = []
    
    # Test 1: Synchronization calculation
    test_name = "Synchronization calculation"
    try:
        # Create perfectly synchronized events
        events_a = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        events_b = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        
        sync_profile = calculate_sync_profile(events_a, events_b, lag_window=2)
        
        assert sync_profile.max_sync_rate == 1.0  # Perfect sync
        assert sync_profile.optimal_lag == 0  # No lag
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    # Test 2: analyze_pair function
    test_name = "Pairwise analysis function"
    try:
        # Create test data
        np.random.seed(42)
        data_a = np.cumsum(np.random.randn(100))
        data_b = np.cumsum(np.random.randn(100))
        
        features_a = extract_lambda3_features(data_a, config, series_name='A')
        features_b = extract_lambda3_features(data_b, config, series_name='B')
        
        # Run analysis with reduced config
        test_config = L3Config()
        test_config.bayesian.draws = 50
        test_config.bayesian.tune = 50
        test_config.bayesian.num_chains = 1
        
        result = analyze_pair('A', 'B', features_a, features_b, test_config, seed=42)
        
        # Verify result structure
        assert result.sync_profile is not None
        assert result.interaction_effects is not None
        assert len(result.interaction_effects) > 0
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    return results
    
def test_io_operations(config: L3Config, verbose: bool) -> List[Dict]:
    """Test I/O operations."""
    from lambda3_numpyro import save_features, load_features, extract_lambda3_features
    import tempfile
    
    results = []
    
    # Test 1: Feature save/load
    test_name = "Feature save/load"
    try:
        # Create test features
        data = np.random.randn(100)
        features = extract_lambda3_features(data, config)
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = Path(f.name)
        
        save_features(features, temp_path)
        loaded_features = load_features(temp_path)
        
        # Verify
        assert np.array_equal(loaded_features.data, features.data)
        assert loaded_features.n_pos_jumps == features.n_pos_jumps
        
        # Clean up
        temp_path.unlink()
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    return results


def test_configuration(verbose: bool) -> List[Dict]:
    """Test configuration handling."""
    from lambda3_numpyro import L3Config
    
    results = []
    
    # Test 1: JAX 64-bit configuration
    test_name = "JAX 64-bit configuration"
    try:
        config = L3Config()
        assert config.enable_x64 == True
        
        import jax
        # Should be enabled after config creation
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    # Test 2: Lambda³ symbol mapping
    test_name = "Lambda³ symbol mapping"
    try:
        from lambda3_numpyro.config import LAMBDA3_SYMBOLS
        
        assert 'sigma_s' in LAMBDA3_SYMBOLS
        assert LAMBDA3_SYMBOLS['sigma_s'] == 'sync_rate'
        assert 'rho_T' in LAMBDA3_SYMBOLS
        assert 'Q_Lambda' in LAMBDA3_SYMBOLS  # QΛを追加済み
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    return results


def test_type_definitions(verbose: bool) -> List[Dict]:
    """Test type definitions."""
    from lambda3_numpyro.types import Lambda3FeatureSet, SyncProfile
    
    results = []
    
    # Test 1: Lambda3FeatureSet validation
    test_name = "Lambda3FeatureSet validation"
    try:
        # Should validate array lengths
        data = np.random.randn(100)
        
        # This should raise error (mismatched lengths)
        try:
            features = Lambda3FeatureSet(
                data=data,
                delta_LambdaC_pos=np.zeros(50, dtype=np.int32),  # Wrong length
                delta_LambdaC_neg=np.zeros(100, dtype=np.int32),
                rho_T=np.zeros(100),
                time_trend=np.arange(100),
                local_jump=np.zeros(100, dtype=np.int32)
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        results.append({
            'name': test_name,
            'passed': True,
            'message': 'OK'
        })
        if verbose:
            print(f"    ✓ {test_name}")
    except Exception as e:
        results.append({
            'name': test_name,
            'passed': False,
            'message': str(e)
        })
        if verbose:
            print(f"    ✗ {test_name}: {e}")
    
    return results


# ===============================
# Integration Test Functions
# ===============================

def test_complete_pipeline(config: L3Config, verbose: bool) -> Dict:
    """Test complete analysis pipeline."""
    import tempfile
    from lambda3_numpyro import (
        extract_lambda3_features, save_features, load_features,
        save_analysis_results, load_analysis_results
    )
    from lambda3_numpyro.analysis import analyze_pair
    
    try:
        # 1. Generate synthetic data
        np.random.seed(42)
        n_points = 200
        t = np.linspace(0, 4*np.pi, n_points)
        
        # Two correlated series with some lag
        series_a = np.sin(t) + np.random.randn(n_points) * 0.1
        series_b = np.sin(t - 0.5) + np.random.randn(n_points) * 0.1  # Lagged
        
        # Add some jumps
        series_a[50] += 2
        series_a[150] -= 2
        series_b[55] += 2
        series_b[155] -= 2
        
        # 2. Extract features
        features_a = extract_lambda3_features(series_a, config, series_name='Series_A')
        features_b = extract_lambda3_features(series_b, config, series_name='Series_B')
        
        if verbose:
            print(f"    Features extracted: A={features_a.n_pos_jumps} pos jumps, "
                  f"B={features_b.n_pos_jumps} pos jumps")
        
        # 3. Save and load features
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            features_dict = {'Series_A': features_a, 'Series_B': features_b}
            save_features(features_dict, temp_path / 'features.pkl')
            loaded_features = load_features(temp_path / 'features.pkl')
            
            assert len(loaded_features) == 2
            
            # 4. Run analysis
            test_config = L3Config()
            test_config.bayesian.draws = 100
            test_config.bayesian.tune = 100
            test_config.bayesian.num_chains = 2
            
            results = analyze_pair(
                'Series_A', 'Series_B',
                loaded_features['Series_A'],
                loaded_features['Series_B'],
                test_config,
                seed=42
            )
            
            # 5. Verify results
            assert results.sync_profile.max_sync_rate > 0
            assert results.sync_profile.optimal_lag != 0  # Should detect lag
            assert len(results.interaction_effects) > 0
            
            if verbose:
                print(f"    Analysis complete: σₛ={results.sync_profile.max_sync_rate:.3f}, "
                      f"lag={results.sync_profile.optimal_lag}")
            
            # 6. Save and load results
            save_analysis_results(results, temp_path / 'results.pkl')
            loaded_results = load_analysis_results(temp_path / 'results.pkl')
            
            assert loaded_results.sync_profile.max_sync_rate == results.sync_profile.max_sync_rate
        
        return {'success': True, 'message': 'Pipeline completed successfully'}
        
    except Exception as e:
        return {'success': False, 'message': str(e)}


def test_cross_analysis(config: L3Config, verbose: bool) -> Dict:
    """Test multi-series cross analysis."""
    from lambda3_numpyro import extract_lambda3_features, analyze_multiple_series
    
    try:
        # Generate multiple correlated series
        np.random.seed(42)
        n_points = 100
        n_series = 4
        
        # Create base signal
        t = np.linspace(0, 4*np.pi, n_points)
        base_signal = np.sin(t)
        
        # Generate correlated series with different lags and noise
        series_dict = {}
        features_dict = {}
        
        for i in range(n_series):
            # Add lag and noise
            lag = i * 0.3
            noise_level = 0.1 + i * 0.05
            
            series = np.sin(t - lag) + np.random.randn(n_points) * noise_level
            name = f'Series_{chr(65+i)}'  # A, B, C, D
            
            series_dict[name] = series
            features_dict[name] = extract_lambda3_features(series, config, series_name=name)
        
        # Run cross-analysis
        test_config = L3Config()
        test_config.bayesian.draws = 50
        test_config.bayesian.tune = 50
        test_config.bayesian.num_chains = 1
        test_config.max_pairs = 6  # All pairs for 4 series
        
        cross_results = analyze_multiple_series(
            features_dict,
            test_config,
            show_progress=verbose
        )
        
        # Verify results
        assert cross_results.n_series == n_series
        assert cross_results.n_pairs == 6  # C(4,2) = 6
        assert cross_results.sync_matrix.shape == (n_series, n_series)
        assert cross_results.interaction_matrix.shape == (n_series, n_series)
        
        # Check interaction tensor
        tensor = cross_results.get_interaction_tensor()
        if tensor is not None:
            assert tensor.shape == (n_series, n_series, 3)
        
        # Network should have some edges
        if cross_results.network:
            assert cross_results.network.number_of_nodes() == n_series
            if verbose:
                print(f"    Network: {cross_results.network.number_of_edges()} edges detected")
        
        return {'success': True, 'message': 'Cross-analysis completed successfully'}
        
    except Exception as e:
        return {'success': False, 'message': str(e)}


def test_model_comparison(config: L3Config, verbose: bool) -> Dict:
    """Test Bayesian model comparison."""
    from lambda3_numpyro import extract_lambda3_features
    from lambda3_numpyro.bayes import Lambda3BayesianInference
    
    try:
        # Generate data with change point
        np.random.seed(42)
        n_points = 150
        data = np.zeros(n_points)
        
        # First regime
        data[:75] = np.cumsum(np.random.randn(75) * 0.5)
        
        # Second regime (different volatility)
        data[75:] = data[74] + np.cumsum(np.random.randn(75) * 1.5)
        
        # Extract features
        features = extract_lambda3_features(data, config)
        
        # Create inference engine
        test_config = L3Config()
        test_config.bayesian.draws = 100
        test_config.bayesian.tune = 100
        test_config.bayesian.num_chains = 2
        
        inference = Lambda3BayesianInference(test_config)
        
        # Fit different models
        inference.fit_model(features, 'base')
        inference.fit_model(features, 'dynamic')
        inference.fit_model(features, 'svi')  
        
        # Compare models
        comparison = inference.compare_models(features, criterion='loo')
        
        # Verify comparison
        assert 'best_model' in comparison
        assert comparison['best_model'] in ['base', 'dynamic']
        
        if verbose:
            print(f"    Best model: {comparison['best_model']}")
        
        # Run PPC on best model
        ppc_results = inference.run_ppc(features)
        
        # Check p-values
        assert 'bayesian_p_values' in ppc_results
        assert all(0 <= p <= 1 for p in ppc_results['bayesian_p_values'].values())
        
        return {'success': True, 'message': 'Model comparison completed successfully'}
        
    except Exception as e:
        return {'success': False, 'message': str(e)}


# ===============================
# Main Entry Point
# ===============================

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Load configuration
    try:
        config = load_config(args)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Set up JAX if needed
    import os
    if 'XLA_FLAGS' not in os.environ:
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
    
    # Suppress warnings unless verbose
    if not args.verbose:
        warnings.filterwarnings('ignore')
    
    # Execute command
    try:
        if args.command == 'extract':
            return cmd_extract(args, config)
        elif args.command == 'analyze':
            return cmd_analyze(args, config)
        elif args.command == 'analyze-all':
            return cmd_analyze_all(args, config)
        elif args.command == 'regimes':
            return cmd_regimes(args, config)
        elif args.command == 'finance':
            return cmd_finance(args, config)
        elif args.command == 'dashboard':
            return cmd_dashboard(args, config)
        elif args.command == 'bayesian':
            return cmd_bayesian(args, config)
        elif args.command == 'hierarchical':
            return cmd_hierarchical(args, config)
        elif args.command == 'compare':
            return cmd_compare(args, config)
        elif args.command == 'ppc':
            return cmd_ppc(args, config)
        elif args.command == 'changepoints':
            return cmd_changepoints(args, config)
        elif args.command == 'cloud':
            return cmd_cloud(args, config)
        elif args.command == 'test':
            return cmd_test(args, config)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
