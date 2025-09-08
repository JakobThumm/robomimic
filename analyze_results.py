#!/usr/bin/env python3
"""
Analyze experimental results from robomimic experiments.
Creates a summary CSV with aggregated statistics across all environments and methods.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple

def parse_method_from_filename(filename: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse method name and parameters from filename.
    Examples: PFL_Ta8_cf20.csv -> method=PFL, params={Ta: 8, cf: 20}
    """
    base_name = filename.replace('.csv', '')
    
    # Split by underscores
    parts = base_name.split('_')
    method = parts[0]
    
    # Parse parameters
    params = {}
    for part in parts[1:]:
        # Extract parameter name and value (e.g., Ta8 -> Ta: 8)
        match = re.match(r'([A-Za-z]+)(\d+)', part)
        if match:
            param_name, param_value = match.groups()
            params[param_name] = param_value
    
    return method, params

def analyze_csv_file(csv_path: str) -> Dict:
    """Analyze a single CSV file and return summary statistics."""
    try:
        df = pd.read_csv(csv_path)
        
        # Basic statistics
        n_runs = len(df)
        success_rate = df['success'].mean()
        mean_steps = df['n_steps'].mean()
        total_critical_collisions = df['critical_collisions'].sum()
        mean_critical_collisions = df['critical_collisions'].mean()
        
        return {
            'n_runs': n_runs,
            'success_rate': success_rate,
            'mean_steps': mean_steps,
            'total_critical_collisions': total_critical_collisions,
            'mean_critical_collisions': mean_critical_collisions
        }
    
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def main():
    """Main function to analyze all results and create summary."""
    
    # Base results directory
    results_dir = Path("/home/jakob/Promotion/code/robomimic/results")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Collect all individual experiment results
    individual_results = []
    
    # Walk through all CSV files in results directory
    for csv_file in results_dir.rglob("*.csv"):
        # Parse path to extract environment and method info
        path_parts = csv_file.relative_to(results_dir).parts
        
        if len(path_parts) < 4:  # Should be: env/ph/method_dir/file.csv
            continue
            
        environment = path_parts[0]  # e.g., 'lift'
        ph_dir = path_parts[1]       # Should be 'ph'
        method_dir = path_parts[2]   # e.g., 'failsafe_single'
        filename = path_parts[3]     # e.g., 'PFL_Ta8_cf20.csv'
        
        # Skip video directories
        if 'video' in str(csv_file):
            continue
            
        # Parse method and parameters from filename
        method_name, params = parse_method_from_filename(filename)
        
        # Analyze the CSV file
        stats = analyze_csv_file(str(csv_file))
        if stats is None:
            continue
        
        # Create result entry
        result = {
            'Environment': environment.capitalize(),
            'Method_Dir': method_dir,
            'Method': method_name,
            'Horizon': stats['mean_steps'],
            'Success_Rate': stats['success_rate'],
            'N_Critical_Collisions': stats['total_critical_collisions'],
            'Mean_Critical_Collisions': stats['mean_critical_collisions'],
            'N_Runs': stats['n_runs'],
        }
        
        # Add parameter information
        for param_name, param_value in params.items():
            result[f'Param_{param_name}'] = param_value
            
        individual_results.append(result)
        print(f"Processed: {environment}/{method_dir}/{filename}")
    
    if not individual_results:
        print("No CSV files found to process!")
        return
    
    # Convert to DataFrame - each CSV file is one method
    summary_df = pd.DataFrame(individual_results)
    
    # Create a more descriptive method name that includes parameters
    summary_df['Method_Full'] = summary_df.apply(lambda row: 
        row['Method'] + ('_' + '_'.join([f"{k.replace('Param_', '')}{v}" 
                                       for k, v in row.items() 
                                       if k.startswith('Param_')]) if any(row.index.str.startswith('Param_')) else ''), 
        axis=1)
    
    # Sort by Environment, Method, then Method_Full for better readability
    summary_df = summary_df.sort_values(['Environment', 'Method', 'Method_Full'])
    
    # Reorder columns for the requested format
    base_columns = ['Environment', 'Method', 'Method_Full', 'Horizon', 'Success_Rate', 'N_Critical_Collisions']
    additional_columns = ['Mean_Critical_Collisions', 'Method_Dir', 'N_Runs']
    param_columns = [col for col in summary_df.columns if col.startswith('Param_')]
    
    final_columns = base_columns + additional_columns + param_columns
    summary_df = summary_df[final_columns]
    
    # Round final values appropriately
    summary_df['Horizon'] = summary_df['Horizon'].round(0).astype(int)
    summary_df['Success_Rate'] = summary_df['Success_Rate'].round(4)
    summary_df['N_Critical_Collisions'] = summary_df['N_Critical_Collisions'].round(0).astype(int)
    summary_df['Mean_Critical_Collisions'] = summary_df['Mean_Critical_Collisions'].round(2)
    summary_df['N_Runs'] = summary_df['N_Runs'].astype(int)
    
    # Save summary
    output_file = results_dir / "experiment_summary.csv"
    summary_df.to_csv(output_file, index=False)
    
    print(f"\nSummary saved to: {output_file}")
    print(f"Total methods analyzed: {len(summary_df)}")
    
    # Display summary statistics
    print("\nSummary by Environment:")
    env_summary = summary_df.groupby('Environment').agg({
        'Success_Rate': ['count', 'mean', 'std'],
        'Horizon': ['mean', 'std'],
        'N_Critical_Collisions': ['mean', 'std']
    }).round(4)
    print(env_summary)
    
    print("\nSummary by Method:")
    method_summary = summary_df.groupby('Method').agg({
        'Success_Rate': ['count', 'mean', 'std'], 
        'Horizon': ['mean', 'std'],
        'N_Critical_Collisions': ['mean', 'std']
    }).round(4)
    print(method_summary)
    
    # Show all rows of the summary
    print("\nComplete Summary (all methods):")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()