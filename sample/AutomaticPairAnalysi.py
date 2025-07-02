# ===============================
# Lambda³ Automatic Pair Analysis System
# ===============================
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class PairAnalysisConfig:
    """Configuration for automatic pair analysis."""
    # Analysis parameters
    analyze_all_pairs: bool = True
    max_pairs: Optional[int] = None  # None means analyze all
    min_series_length: int = 100
    
    # Pair filtering criteria
    min_correlation: Optional[float] = None  # Filter pairs by minimum correlation
    exclude_patterns: List[str] = field(default_factory=list)  # Patterns to exclude
    include_only_patterns: List[str] = field(default_factory=list)  # If set, only these
    
    # Analysis depth
    detailed_analysis_limit: int = 5  # Number of pairs for detailed plots
    summary_only_after: int = 10  # Switch to summary mode after this many pairs
    
    # Output configuration
    save_results: bool = True
    output_dir: str = "lambda3_results"
    generate_report: bool = True

class Lambda3AutoPairAnalyzer:
    """
    Automatic pair analysis system for Lambda³ framework.
    Intelligently generates and analyzes all relevant pairs based on data structure.
    """
    
    def __init__(self, config: PairAnalysisConfig = None):
        self.config = config or PairAnalysisConfig()
        self.analysis_results = {}
        self.pair_metadata = {}
        
    def detect_data_structure(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> Dict[str, List[str]]:
        """
        Automatically detect data structure and categorize columns.
        
        Returns:
            Dictionary mapping categories to column lists
        """
        if isinstance(data, pd.DataFrame):
            columns = data.columns.tolist()
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            columns = list(data.keys())
            numeric_cols = columns  # Assume all dict entries are numeric
        
        # Categorize columns based on patterns
        categories = {
            'weather': [],
            'temperature': [],
            'humidity': [],
            'pressure': [],
            'wind': [],
            'precipitation': [],
            'other': []
        }
        
        # Pattern matching for categorization
        patterns = {
            'temperature': ['temp', 'temperature', 'dewpoint', 'dew_point'],
            'humidity': ['humid', 'rh', 'relative_humidity'],
            'pressure': ['pressure', 'press', 'hpa', 'mbar'],
            'wind': ['wind', 'gust', 'speed'],
            'precipitation': ['precip', 'rain', 'snow', 'precipitation']
        }
        
        for col in numeric_cols:
            col_lower = col.lower()
            categorized = False
            
            for category, keywords in patterns.items():
                if any(keyword in col_lower for keyword in keywords):
                    categories[category].append(col)
                    categories['weather'].append(col)  # All are weather
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(col)
        
        return categories
    
    def generate_pair_list(self, 
                          series_dict: Dict[str, np.ndarray],
                          categories: Optional[Dict[str, List[str]]] = None) -> List[Tuple[str, str]]:
        """
        Generate intelligent pair list based on data structure and configuration.
        
        Returns:
            List of (series_a, series_b) tuples to analyze
        """
        all_series = list(series_dict.keys())
        
        # Apply include/exclude patterns
        if self.config.include_only_patterns:
            all_series = [s for s in all_series 
                         if any(pattern in s.lower() for pattern in self.config.include_only_patterns)]
        
        if self.config.exclude_patterns:
            all_series = [s for s in all_series 
                         if not any(pattern in s.lower() for pattern in self.config.exclude_patterns)]
        
        # Generate all unique pairs
        all_pairs = list(combinations(all_series, 2))
        
        # If categories provided, prioritize cross-category pairs
        if categories:
            priority_pairs = []
            regular_pairs = []
            
            for pair in all_pairs:
                a, b = pair
                # Check if pair crosses categories
                cross_category = False
                for cat, members in categories.items():
                    if (a in members and b not in members) or (a not in members and b in members):
                        cross_category = True
                        break
                
                if cross_category:
                    priority_pairs.append(pair)
                else:
                    regular_pairs.append(pair)
            
            # Combine with priority pairs first
            all_pairs = priority_pairs + regular_pairs
        
        # Apply correlation filter if specified
        if self.config.min_correlation is not None:
            filtered_pairs = []
            for a, b in all_pairs:
                corr = np.corrcoef(series_dict[a], series_dict[b])[0, 1]
                if abs(corr) >= self.config.min_correlation:
                    filtered_pairs.append((a, b))
                    self.pair_metadata[(a, b)] = {'correlation': corr}
            all_pairs = filtered_pairs
        
        # Apply max pairs limit
        if self.config.max_pairs and len(all_pairs) > self.config.max_pairs:
            print(f"Limiting analysis to {self.config.max_pairs} pairs out of {len(all_pairs)} total")
            all_pairs = all_pairs[:self.config.max_pairs]
        
        return all_pairs
    
    def analyze_pairs(self,
                     series_dict: Dict[str, np.ndarray],
                     features_dict: Dict[str, Dict[str, np.ndarray]],
                     l3_config: 'L3Config',
                     pair_list: Optional[List[Tuple[str, str]]] = None) -> Dict:
        """
        Analyze all pairs with intelligent batching and summarization.
        
        Returns:
            Dictionary containing all analysis results
        """
        # Auto-detect structure if needed
        categories = self.detect_data_structure(series_dict)
        
        # Generate pairs if not provided
        if pair_list is None:
            pair_list = self.generate_pair_list(series_dict, categories)
        
        n_pairs = len(pair_list)
        print(f"\n{'='*60}")
        print(f"LAMBDA³ AUTOMATIC PAIR ANALYSIS")
        print(f"Analyzing {n_pairs} pairs from {len(series_dict)} series")
        print(f"{'='*60}")
        
        # Display data structure
        print("\nDetected Data Structure:")
        for cat, members in categories.items():
            if members:
                print(f"  {cat.capitalize()}: {', '.join(members[:5])}" + 
                      (f" (+{len(members)-5} more)" if len(members) > 5 else ""))
        
        # Initialize results storage
        results = {
            'pair_results': {},
            'interaction_matrix': np.zeros((len(series_dict), len(series_dict))),
            'sync_results': {},
            'categories': categories,
            'summary_stats': {}
        }
        
        # Create index mapping
        series_names = list(series_dict.keys())
        name_to_idx = {name: i for i, name in enumerate(series_names)}
        
        # Analyze pairs in batches
        for batch_idx, (a, b) in enumerate(pair_list):
            pair_key = (a, b)
            
            # Determine analysis depth
            if batch_idx < self.config.detailed_analysis_limit:
                analysis_mode = 'detailed'
                show_plots = True
            elif batch_idx < self.config.summary_only_after:
                analysis_mode = 'standard'
                show_plots = False
            else:
                analysis_mode = 'summary'
                show_plots = False
            
            print(f"\n[{batch_idx+1}/{n_pairs}] Analyzing: {a} ↔ {b} (mode: {analysis_mode})")
            
            try:
                # Run pair analysis
                result = self._analyze_single_pair(
                    a, b, features_dict, l3_config, 
                    show_plots=show_plots,
                    analysis_mode=analysis_mode
                )
                
                results['pair_results'][pair_key] = result
                
                # Update interaction matrix
                i, j = name_to_idx[a], name_to_idx[b]
                results['interaction_matrix'][i, j] = result['interaction_b_to_a']
                results['interaction_matrix'][j, i] = result['interaction_a_to_b']
                
                # Store sync results
                results['sync_results'][pair_key] = {
                    'sync_rate': result['sync_rate'],
                    'optimal_lag': result['optimal_lag']
                }
                
            except Exception as e:
                print(f"  Error analyzing pair: {e}")
                results['pair_results'][pair_key] = {'error': str(e)}
        
        # Generate summary statistics
        results['summary_stats'] = self._generate_summary_stats(results)
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(results, series_names)
        
        # Generate report if configured
        if self.config.generate_report:
            self._generate_report(results, series_names)
        
        return results
    
    def _analyze_single_pair(self, 
                           name_a: str, 
                           name_b: str,
                           features_dict: Dict,
                           config: 'L3Config',
                           show_plots: bool = False,
                           analysis_mode: str = 'standard') -> Dict:
        """
        Analyze a single pair with specified depth.
        
        Returns:
            Dictionary with pair analysis results
        """
        from WeatherAnalysis import (
            fit_l3_bayesian_regression_asymmetric,
            calculate_sync_profile,
            Lambda3BayesianExtended
        )
        import arviz as az
        
        feats_a = features_dict[name_a]
        feats_b = features_dict[name_b]
        
        result = {
            'pair': (name_a, name_b),
            'analysis_mode': analysis_mode
        }
        
        if analysis_mode == 'summary':
            # Quick sync calculation only
            sync_profile, sync_rate, optimal_lag = calculate_sync_profile(
                feats_a['delta_LambdaC_pos'].astype(np.float64),
                feats_b['delta_LambdaC_pos'].astype(np.float64),
                lag_window=10
            )
            
            result.update({
                'sync_rate': sync_rate,
                'optimal_lag': optimal_lag,
                'interaction_b_to_a': 0,  # Placeholder
                'interaction_a_to_b': 0   # Placeholder
            })
            
        else:
            # Full Bayesian analysis
            trace_a = fit_l3_bayesian_regression_asymmetric(
                data=feats_a['data'],
                features_dict={
                    'delta_LambdaC_pos': feats_a['delta_LambdaC_pos'],
                    'delta_LambdaC_neg': feats_a['delta_LambdaC_neg'],
                    'rho_T': feats_a['rho_T'],
                    'time_trend': feats_a['time_trend']
                },
                config=config,
                interaction_pos=feats_b['delta_LambdaC_pos'],
                interaction_neg=feats_b['delta_LambdaC_neg'],
                interaction_rhoT=feats_b['rho_T']
            )
            
            trace_b = fit_l3_bayesian_regression_asymmetric(
                data=feats_b['data'],
                features_dict={
                    'delta_LambdaC_pos': feats_b['delta_LambdaC_pos'],
                    'delta_LambdaC_neg': feats_b['delta_LambdaC_neg'],
                    'rho_T': feats_b['rho_T'],
                    'time_trend': feats_b['time_trend']
                },
                config=config,
                interaction_pos=feats_a['delta_LambdaC_pos'],
                interaction_neg=feats_a['delta_LambdaC_neg'],
                interaction_rhoT=feats_a['rho_T']
            )
            
            # Extract results
            summary_a = az.summary(trace_a)
            summary_b = az.summary(trace_b)
            
            beta_b_on_a = summary_a.loc['beta_interact_pos', 'mean'] if 'beta_interact_pos' in summary_a.index else 0
            beta_a_on_b = summary_b.loc['beta_interact_pos', 'mean'] if 'beta_interact_pos' in summary_b.index else 0
            
            # Sync analysis
            sync_profile, sync_rate, optimal_lag = calculate_sync_profile(
                feats_a['delta_LambdaC_pos'].astype(np.float64),
                feats_b['delta_LambdaC_pos'].astype(np.float64),
                lag_window=10
            )
            
            result.update({
                'interaction_b_to_a': beta_b_on_a,
                'interaction_a_to_b': beta_a_on_b,
                'sync_rate': sync_rate,
                'optimal_lag': optimal_lag,
                'trace_a': trace_a if analysis_mode == 'detailed' else None,
                'trace_b': trace_b if analysis_mode == 'detailed' else None,
                'sync_profile': sync_profile if analysis_mode == 'detailed' else None
            })
            
            # Add causality analysis for detailed mode
            if analysis_mode == 'detailed':
                lambda3_ext = Lambda3BayesianExtended(config, series_names=[name_a, name_b])
                
                # Build event memory
                for i in range(len(feats_a['data'])):
                    lambda3_ext.update_event_memory({
                        name_a: {
                            'pos': int(feats_a['delta_LambdaC_pos'][i]),
                            'neg': int(feats_a['delta_LambdaC_neg'][i])
                        },
                        name_b: {
                            'pos': int(feats_b['delta_LambdaC_pos'][i]),
                            'neg': int(feats_b['delta_LambdaC_neg'][i])
                        }
                    })
                
                causality_ab = {
                    lag: lambda3_ext.detect_cross_causality(name_a, name_b, lag=lag) 
                    for lag in range(1, 11)
                }
                causality_ba = {
                    lag: lambda3_ext.detect_cross_causality(name_b, name_a, lag=lag) 
                    for lag in range(1, 11)
                }
                
                result['causality'] = {
                    'a_to_b': causality_ab,
                    'b_to_a': causality_ba
                }
        
        # Print summary
        print(f"  β({name_b}→{name_a}): {result.get('interaction_b_to_a', 0):.3f}")
        print(f"  β({name_a}→{name_b}): {result.get('interaction_a_to_b', 0):.3f}")
        print(f"  σₛ: {result['sync_rate']:.3f} (lag: {result['optimal_lag']})")
        
        return result
    
    def _generate_summary_stats(self, results: Dict) -> Dict:
        """Generate summary statistics from all pair analyses."""
        stats = {
            'total_pairs': len(results['pair_results']),
            'successful_analyses': sum(1 for r in results['pair_results'].values() if 'error' not in r),
            'top_interactions': [],
            'top_synchronizations': [],
            'network_metrics': {}
        }
        
        # Find top interactions
        interactions = []
        for pair_key, result in results['pair_results'].items():
            if 'error' not in result:
                interactions.append({
                    'pair': pair_key,
                    'strength': abs(result.get('interaction_b_to_a', 0))
                })
        
        interactions.sort(key=lambda x: x['strength'], reverse=True)
        stats['top_interactions'] = interactions[:10]
        
        # Find top synchronizations
        syncs = []
        for pair_key, sync_data in results['sync_results'].items():
            syncs.append({
                'pair': pair_key,
                'sync_rate': sync_data['sync_rate'],
                'optimal_lag': sync_data['optimal_lag']
            })
        
        syncs.sort(key=lambda x: x['sync_rate'], reverse=True)
        stats['top_synchronizations'] = syncs[:10]
        
        return stats
    
    def _save_results(self, results: Dict, series_names: List[str]):
        """Save analysis results to files."""
        import json
        from pathlib import Path
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save interaction matrix
        np.save(output_dir / 'interaction_matrix.npy', results['interaction_matrix'])
        
        # Save sync results
        sync_df = pd.DataFrame([
            {
                'series_a': pair[0],
                'series_b': pair[1],
                'sync_rate': data['sync_rate'],
                'optimal_lag': data['optimal_lag']
            }
            for pair, data in results['sync_results'].items()
        ])
        sync_df.to_csv(output_dir / 'synchronization_results.csv', index=False)
        
        # Save summary
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump({
                'summary_stats': results['summary_stats'],
                'categories': results['categories']
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_dir}/")
    
    def _generate_report(self, results: Dict, series_names: List[str]):
        """Generate markdown report of analysis results."""
        from datetime import datetime
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        report = [
            f"# Lambda³ Automatic Pair Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Overview",
            f"- Total series analyzed: {len(series_names)}",
            f"- Total pairs analyzed: {results['summary_stats']['total_pairs']}",
            f"- Successful analyses: {results['summary_stats']['successful_analyses']}",
            f"",
            f"## Data Structure",
            f"```"
        ]
        
        for cat, members in results['categories'].items():
            if members:
                report.append(f"{cat}: {', '.join(members)}")
        
        report.extend([
            f"```",
            f"",
            f"## Top Interactions",
            f"| Pair | Interaction Strength |",
            f"|------|---------------------|"
        ])
        
        for item in results['summary_stats']['top_interactions'][:10]:
            pair = item['pair']
            report.append(f"| {pair[0]} → {pair[1]} | {item['strength']:.4f} |")
        
        report.extend([
            f"",
            f"## Top Synchronizations",
            f"| Pair | Sync Rate | Optimal Lag |",
            f"|------|-----------|-------------|"
        ])
        
        for item in results['summary_stats']['top_synchronizations'][:10]:
            pair = item['pair']
            report.append(f"| {pair[0]} ↔ {pair[1]} | {item['sync_rate']:.3f} | {item['optimal_lag']} |")
        
        # Save report
        with open(output_dir / 'analysis_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report generated: {output_dir}/analysis_report.md")


# ===============================
# Integration with existing code
# ===============================
def lambda3_auto_analyze(
    data_source: Union[str, pd.DataFrame, Dict[str, np.ndarray]],
    config: Optional[PairAnalysisConfig] = None,
    l3_config: Optional['L3Config'] = None,
    **kwargs
) -> Dict:
    """
    Main entry point for automatic Lambda³ pair analysis.
    
    Args:
        data_source: CSV path, DataFrame, or series dictionary
        config: Pair analysis configuration
        l3_config: Lambda³ configuration
        **kwargs: Additional arguments for data loading
        
    Returns:
        Complete analysis results
    """
    from WeatherAnalysis import L3Config, calc_lambda3_features_v2, load_csv_data
    
    # Initialize configurations
    if config is None:
        config = PairAnalysisConfig()
    if l3_config is None:
        l3_config = L3Config()
    
    # Load data based on source type
    if isinstance(data_source, str):
        # CSV file path
        series_dict = load_csv_data(
            data_source,
            time_column=kwargs.get('time_column'),
            value_columns=kwargs.get('value_columns')
        )
    elif isinstance(data_source, pd.DataFrame):
        # DataFrame
        numeric_cols = data_source.select_dtypes(include=[np.number]).columns
        series_dict = {
            col: data_source[col].values.astype(np.float64)
            for col in numeric_cols
        }
    else:
        # Already a dictionary
        series_dict = data_source
    
    # Validate series lengths
    lengths = {name: len(data) for name, data in series_dict.items()}
    min_length = min(lengths.values())
    
    if min_length < config.min_series_length:
        raise ValueError(f"Series too short. Minimum length: {config.min_series_length}, found: {min_length}")
    
    # Align series if needed
    if len(set(lengths.values())) > 1:
        print(f"Aligning series to common length: {min_length}")
        series_dict = {name: data[:min_length] for name, data in series_dict.items()}
    
    # Update L3 config with actual data length
    l3_config.T = min_length
    
    # Extract Lambda³ features for all series
    print("\nExtracting Lambda³ features for all series...")
    features_dict = {}
    
    for name, data in series_dict.items():
        feats = calc_lambda3_features_v2(data, l3_config)
        features_dict[name] = {
            'data': data,
            'delta_LambdaC_pos': feats[0],
            'delta_LambdaC_neg': feats[1],
            'rho_T': feats[2],
            'time_trend': feats[3],
            'local_jump': feats[4]
        }
    
    # Initialize analyzer and run
    analyzer = Lambda3AutoPairAnalyzer(config)
    results = analyzer.analyze_pairs(series_dict, features_dict, l3_config)
    
    return results


# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    # Example 1: Analyze all weather parameters with smart defaults
    results = lambda3_auto_analyze(
        "tokyo_weather_days.csv",
        config=PairAnalysisConfig(
            analyze_all_pairs=True,
            detailed_analysis_limit=3,  # Detailed plots for first 3 pairs
            summary_only_after=10,      # Summary mode after 10 pairs
            save_results=True,
            generate_report=True
        ),
        time_column="date"
    )
    
    # Example 2: Focus on specific patterns
    results = lambda3_auto_analyze(
        "tokyo_weather_days.csv",
        config=PairAnalysisConfig(
            include_only_patterns=['temperature', 'humidity', 'pressure'],
            min_correlation=0.3,  # Only analyze pairs with |r| > 0.3
            max_pairs=15
        )
    )
    
    # Example 3: From DataFrame
    df = pd.read_csv("tokyo_weather_days.csv")
    results = lambda3_auto_analyze(
        df,
        config=PairAnalysisConfig(
            exclude_patterns=['wind_direction'],  # Exclude certain columns
            detailed_analysis_limit=5
        )
    )
