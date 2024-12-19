import subprocess
import argparse
from pathlib import Path
import sys
import time
import json
from tqdm import tqdm

# Default configuration
DEFAULT_CONFIG = {
    "# Section 1: Directory Structure": None,
    "directories": {
        "raw": "raw",
        "processed": "processed",
        "clustered": "clustered",
        "cleaned": "cleaned",
        "reclustered": "reclustered",
        "features": "features",
        "plots": "plots"
    },
    
    "# Section 2: Initial Data Processing": None,
    "processing": {
        "chunk_size": 10799,
        "session_length": 900,
        "gap_hours": 5.5
    },
    
    "# Section 3: Initial Clustering Parameters": None,
    "initial_clustering": {
        "distance_threshold": 30,
        "max_neighbors": 5,
        "time_threshold": 100
    },
    
    "# Section 4: Cluster Cleaning Parameters": None,
    "cleaning": {
        "min_points": 10,
        "span": 5.0,
        "proximity": 2.0,
        "max_size": 1000
    },
    
    "# Section 5: Reclustering Parameters": None,
    "reclustering": {
        "distance_threshold": 25,  # Slightly stricter for reclustering
        "max_neighbors": 4,        # Slightly fewer neighbors for reclustering
        "time_threshold": 90       # Slightly stricter time threshold
    }
}

def load_config(config_path=None):
    """Load configuration from file or return default config."""
    if config_path is None:
        return DEFAULT_CONFIG
    
    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        # Merge user config with default config
        config = DEFAULT_CONFIG.copy()
        for section in user_config:
            if section in config:
                config[section].update(user_config[section])
            else:
                config[section] = user_config[section]
        
        return config
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        print("Using default configuration")
        return DEFAULT_CONFIG

def run_command(command, description):
    """Run a command silently and only show step information."""
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"Running command: {' '.join(command)}")
    print('='*80)
    
    start_time = time.time()
    
    try:
        # Run process with suppressed output
        process = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        duration = time.time() - start_time
        print(f"Step completed successfully in {duration:.1f} seconds\n")
        return True
            
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}\n")
        return False
    except Exception as e:
        print(f"Error running command: {str(e)}\n")
        return False

def create_directory(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def should_skip_step(output_dir, step_name):
    """Check if a step should be skipped because output already exists."""
    if output_dir.exists() and any(output_dir.iterdir()):
        tqdm.write(f"\nSkipping {step_name} - Output directory already exists: {output_dir}")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Run the complete worm analysis pipeline.')
    parser.add_argument('--base-dir', type=str, required=True, 
                       help='Base directory for all input/output')
    parser.add_argument('--config', type=str,
                       help='Path to configuration JSON file (optional)')
    parser.add_argument('--force', action='store_true',
                       help='Force all steps to run even if output exists')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Print configuration
    print("\nUsing Configuration:")
    print(json.dumps(config, indent=2))
    
    # Set up directory structure
    base_dir = Path(args.base_dir)
    dirs = {name: base_dir / path for name, path in config['directories'].items()}
    
    # Create plots directory as it's used by multiple steps
    create_directory(dirs['plots'])
    
    # Print directory structure
    print("\nDirectory Structure:")
    print(f"Base Directory: {base_dir}")
    for name, path in dirs.items():
        print(f"  {name}: {path}")
    
    # Step 1: Process Lifespan Data
    if args.force or not should_skip_step(dirs['processed'], "Processing"):
        create_directory(dirs['processed'])
        if not run_command([
            sys.executable, "process_lifespan_data.py",
            "--input", str(dirs['raw']),
            "--output", str(dirs['processed']),
            "--chunk-size", str(config['processing']['chunk_size']),
            "--session-length", str(config['processing']['session_length']),
            "--gap-hours", str(config['processing']['gap_hours'])
        ], "Processing Lifespan Data"):
            return
    
    # Step 2: Initial Clustering
    if args.force or not should_skip_step(dirs['clustered'], "Initial Clustering"):
        create_directory(dirs['clustered'])
        create_directory(dirs['plots'] / 'initial_clustering')
        if not run_command([
            sys.executable, "temporal_clustering_analysis.py",
            "--input", str(dirs['processed']),
            "--output", str(dirs['clustered']),
            "--plots", str(dirs['plots'] / 'initial_clustering'),
            "--distance-threshold", str(config['initial_clustering']['distance_threshold']),
            "--max-neighbors", str(config['initial_clustering']['max_neighbors']),
            "--time-threshold", str(config['initial_clustering']['time_threshold'])
        ], "Initial Temporal Clustering"):
            return
    
    # Step 3: Clean Clusters
    if args.force or not should_skip_step(dirs['cleaned'], "Cleaning"):
        create_directory(dirs['cleaned'])
        create_directory(dirs['plots'] / 'cleaning')
        if not run_command([
            sys.executable, "clean_clusters.py",
            str(config['cleaning']['min_points']),
            "--input", str(dirs['clustered']),
            "--output", str(dirs['cleaned']),
            "--plots", str(dirs['plots'] / 'cleaning'),
            "--span", str(config['cleaning']['span']),
            "--proximity", str(config['cleaning']['proximity']),
            "--max-size", str(config['cleaning']['max_size'])
        ], "Cleaning Clusters"):
            return
    
    # Step 4: Extract Features
    if args.force or not should_skip_step(dirs['features'], "Feature Extraction"):
        create_directory(dirs['features'])
        if not run_command([
            sys.executable, "extract_features.py",
            "--input", str(dirs['cleaned']),
            "--output", str(dirs['features']),
            "--treatments"] + [d.name for d in Path(dirs['cleaned']).iterdir() if d.is_dir()
        ], "Extracting Features"):
            return
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)
    print("\nOutput Locations:")
    print(f"1. Processed Data: {dirs['processed']}")
    print(f"2. Initial Clustering: {dirs['clustered']}")
    print(f"3. Cleaned Data: {dirs['cleaned']}")
    print(f"4. Extracted Features: {dirs['features']}")
    print(f"\nPlots can be found in: {dirs['plots']}")

if __name__ == "__main__":
    main() 