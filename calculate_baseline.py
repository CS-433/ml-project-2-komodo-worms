import json
import numpy as np
from pathlib import Path
import os
import pandas as pd

def get_lifespans_from_directory(directory):
    """Get lifespans from all subdirectories in the Lifespan folder."""
    lifespans = []
    
    # Walk through all subdirectories
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            # Get all CSV files
            files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            for file in files:
                file_path = os.path.join(subdir_path, file)
                # Count rows in CSV file
                df = pd.read_csv(file_path)
                lifespans.append(len(df))
    
    return lifespans

def calculate_baseline_metrics():
    # Get lifespans from data directory
    lifespan_dir = 'data/Lifespan'
    all_lifespans = get_lifespans_from_directory(lifespan_dir)
    
    if not all_lifespans:
        print("No lifespan data found in data/Lifespan directory!")
        return
    
    # Calculate mean
    mean_lifespan = np.mean(all_lifespans)
    
    # Calculate errors
    abs_errors = [abs(mean_lifespan - actual) for actual in all_lifespans]
    
    # Calculate MAPE: mean(|actual - predicted| / |actual| * 100)
    pct_errors = [abs(actual - mean_lifespan) / abs(actual) * 100 for actual in all_lifespans]
    
    # Calculate metrics
    mae = float(np.mean(abs_errors))
    mape = float(np.mean(pct_errors))
    std_ae = float(np.std(abs_errors))
    std_pe = float(np.std(pct_errors))
    
    # Create metrics dictionary
    metrics = {
        "mean_lifespan": float(mean_lifespan),
        "mae": mae,
        "mape": mape,
        "std_ae": std_ae,
        "std_pe": std_pe,
        "num_animals": len(all_lifespans)
    }
    
    # Save to JSON
    with open('baseline_stats.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nBaseline Metrics (using mean lifespan = {mean_lifespan:.2f}):")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Std AE: {std_ae:.2f}")
    print(f"Std PE: {std_pe:.2f}%")
    print(f"\nNumber of animals: {len(all_lifespans)}")
    print("\nMetrics saved to baseline_stats.json")

if __name__ == "__main__":
    calculate_baseline_metrics() 