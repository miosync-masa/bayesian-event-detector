#!/usr/bin/env python3
"""
Lambda³ Analytics Command Line Interface

Main entry point for Lambda³ analysis framework.
Provides commands for feature extraction, analysis, and visualization.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Lambda³ imports
from lambda3 import (
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
        from lambda3.feature import extract_features_dict
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
        from lambda3.plot import plot_analysis_results
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
        from lambda3.feature import extract_features_dict
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
    from lambda3.analysis import generate_analysis_summary
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
        from lambda3.plot import (
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
    
    # Handle single or multiple features
    if isinstance(features, dict):
        print(f"Found {len(features)} series")
        results = {}
        
        for name, feat in features.items():
            print(f"\nDetecting regimes for {name}...")
            regime_info = detect_regimes(
                feat,
                n_regimes=args.n_regimes
            )
            results[name] = regime_info
            
            # Display regime statistics
            print(f"Regime statistics:")
            for regime_id, stats in regime_info.regime_stats.items():
                name = regime_info.regime_names.get(regime_id, f"Regime {regime_id}")
                print(f"  {name}: {stats['frequency']:.1%} "
                      f"(tension: {stats['mean_tension']:.3f})")
    else:
        # Single feature set
        print("Detecting regimes...")
        regime_info = detect_regimes(
            features,
            n_regimes=args.n_regimes
        )
        results = regime_info
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nRegime results saved to {output_path}")
    
    # Plot if requested
    if args.plot and PLOTTING_AVAILABLE:
        from lambda3.plot import plot_regimes
        
        if isinstance(results, dict):
            for name, regime_info in results.items():
                plot_file = output_path.parent / f"regimes_{name}.png"
                plot_regimes(
                    features[name],
                    regime_info,
                    save_path=plot_file,
                    config=config.plotting
                )
        else:
            plot_file = output_path.parent / "regimes.png"
            plot_regimes(
                features,
                results,
                save_path=plot_file,
                config=config.plotting
            )
        print("Regime plots saved")
    
    return 0


def cmd_finance(args, config: L3Config):
    """Execute financial data download and analysis."""
    # Default tickers
    if not args.tickers:
        tickers = {
            "SPY": "SPY",
            "QQQ": "QQQ", 
            "TLT": "TLT",
            "GLD": "GLD",
            "VIX": "^VIX"
        }
    else:
        # Create ticker dict
        tickers = {ticker: ticker for ticker in args.tickers}
    
    # Download data
    print(f"Downloading financial data from {args.start} to {args.end}")
    print(f"Tickers: {', '.join(tickers.keys())}")
    
    try:
        series_dict = load_financial_data(
            start_date=args.start,
            end_date=args.end,
            tickers=tickers,
            save_csv=True,
            csv_filename=args.output
        )
        print(f"\n✓ Data saved to {args.output}")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return 1
    
    # Run analysis if requested
    if args.analyze:
        print("\nRunning Lambda³ analysis on downloaded data...")
        
        # Extract features
        from lambda3.feature import extract_features_dict
        features_dict = extract_features_dict(series_dict, config)
        
        # Save features
        features_file = Path(args.output).with_suffix('.features.pkl')
        save_features(features_dict, features_file)
        print(f"Features saved to {features_file}")
        
        # Run cross-analysis if multiple series
        if len(features_dict) > 1:
            results = analyze_multiple_series(features_dict, config)
            
            results_file = Path(args.output).with_suffix('.results.pkl')
            save_analysis_results(results, results_file)
            print(f"Analysis results saved to {results_file}")
            
            # Generate dashboard if plotting available
            if PLOTTING_AVAILABLE:
                from lambda3.plot import create_analysis_dashboard
                dashboard_file = Path(args.output).with_suffix('.dashboard.png')
                create_analysis_dashboard(
                    results,
                    features_dict,
                    save_path=dashboard_file,
                    config=config.plotting
                )
                print(f"Dashboard saved to {dashboard_file}")
    
    return 0


def cmd_dashboard(args, config: L3Config):
    """Create analysis dashboard from results."""
    if not PLOTTING_AVAILABLE:
        print("Error: Plotting libraries not available")
        print("Install with: pip install matplotlib seaborn")
        return 1
    
    # Load results
    print(f"Loading results from {args.results}")
    results = load_analysis_results(Path(args.results), config.cloud)
    
    # Load features
    print(f"Loading features from {args.features}")
    features_dict = load_features(Path(args.features), config.cloud)
    
    # Create dashboard
    from lambda3.plot import create_analysis_dashboard
    print("Creating analysis dashboard...")
    
    create_analysis_dashboard(
        results,
        features_dict,
        save_path=Path(args.output),
        config=config.plotting
    )
    
    print(f"✓ Dashboard saved to {args.output}")
    return 0


def cmd_cloud(args, config: L3Config):
    """Execute cloud storage operations."""
    from lambda3.config import CloudConfig
    
    # Create cloud config
    cloud_config = CloudConfig(
        provider=args.provider,
        bucket=args.bucket or config.cloud.bucket
    )
    
    if args.action == 'upload':
        if not args.file or not args.remote:
            print("Error: --file and --remote required for upload")
            return 1
        
        print(f"Uploading {args.file} to {args.provider}://{cloud_config.bucket}/{args.remote}")
        
        # Determine file type and use appropriate save function
        file_path = Path(args.file)
        if file_path.suffix == '.pkl':
            # Load and re-save with cloud config
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            from lambda3.io import _save_to_cloud
            _save_to_cloud(data, Path(args.remote), cloud_config)
        else:
            print("Error: Only .pkl files supported for cloud upload")
            return 1
        
        print("✓ Upload complete")
    
    elif args.action == 'download':
        if not args.remote or not args.file:
            print("Error: --remote and --file required for download")
            return 1
        
        print(f"Downloading {args.provider}://{cloud_config.bucket}/{args.remote} to {args.file}")
        
        from lambda3.io import _load_from_cloud
        data = _load_from_cloud(Path(args.remote), cloud_config)
        
        # Save locally
        import pickle
        with open(args.file, 'wb') as f:
            pickle.dump(data, f)
        
        print("✓ Download complete")
    
    elif args.action == 'list':
        print(f"Listing contents of {args.provider}://{cloud_config.bucket}")
        # Implementation depends on provider
        print("Note: List functionality not yet implemented")
    
    return 0


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
        elif args.command == 'cloud':
            return cmd_cloud(args, config)
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
