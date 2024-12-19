import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualization import plot_movement, plot_clustered_movement
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_cluster_metrics(data, cluster_id):
    """
    Calculate span and center for a given cluster.

    Args:
        data (pd.DataFrame): DataFrame containing cluster data
        cluster_id: ID of the cluster to analyze

    Returns:
        tuple: (span, center_x, center_y)
    """
    cluster_data = data[data['Cluster'] == cluster_id]
    points = cluster_data[['X', 'Y']].values
    
    # Calculate span using pairwise distances
    if len(points) >= 2:
        distances = pdist(points)
        span = np.max(distances)
    else:
        span = 0
    
    # Calculate center
    center = np.mean(points, axis=0)
    return span, center[0], center[1]

def filter_clusters_by_span_and_proximity(data, span_threshold, proximity_threshold, max_size_threshold=50):
    """
    Filter out clusters that have small spans and are close to other small-span clusters.
    Only considers clusters smaller than max_size_threshold as potential anomalies.

    Args:
        data (pd.DataFrame): Input DataFrame with clustered points
        span_threshold (float): Maximum span for a cluster to be considered small
        proximity_threshold (float): Distance threshold for considering clusters as close
        max_size_threshold (int): Maximum size (number of points) for a cluster to be considered as a potential anomaly

    Returns:
        pd.DataFrame: Filtered DataFrame with problematic clusters removed
    """
    # Get unique clusters and their sizes
    cluster_sizes = data['Cluster'].value_counts()
    clusters = data['Cluster'].unique()
    
    # Calculate metrics for each cluster
    cluster_metrics = {}
    print("Calculating cluster metrics...")
    with tqdm(total=len(clusters), desc="Processing clusters", leave=False) as pbar:
        for cluster_id in clusters:
            span, center_x, center_y = calculate_cluster_metrics(data, cluster_id)
            cluster_metrics[cluster_id] = {
                'span': span,
                'center': np.array([center_x, center_y]),
                'size': cluster_sizes[cluster_id]
            }
            pbar.update(1)
    
    # Identify small-span clusters that are also below the size threshold
    small_span_clusters = {
        cid: metrics for cid, metrics in cluster_metrics.items()
        if metrics['span'] < span_threshold and metrics['size'] < max_size_threshold
    }
    
    print(f"\nFound {len(small_span_clusters)} clusters with small spans and size < {max_size_threshold}")
    
    # Find clusters to remove (small span + close to other small span clusters)
    clusters_to_remove = set()
    
    print("Analyzing cluster proximity...")
    small_span_ids = list(small_span_clusters.keys())
    
    for i, cluster1_id in enumerate(small_span_ids):
        for cluster2_id in small_span_ids[i+1:]:
            center1 = small_span_clusters[cluster1_id]['center']
            center2 = small_span_clusters[cluster2_id]['center']
            
            # Calculate distance between centers
            distance = np.linalg.norm(center1 - center2)
            
            if distance < proximity_threshold:
                clusters_to_remove.add(cluster1_id)
                clusters_to_remove.add(cluster2_id)
    
    # Filter out the identified clusters
    print(f"Removing {len(clusters_to_remove)} clusters that have small spans and are close to each other")
    filtered_data = data[~data['Cluster'].isin(clusters_to_remove)].copy()
    
    # Print detailed statistics
    points_removed = len(data) - len(filtered_data)
    print(f"\nDetailed filtering statistics:")
    print(f"Total clusters considered: {len(clusters)}")
    print(f"Clusters with small spans and size < {max_size_threshold}: {len(small_span_clusters)}")
    print(f"Clusters removed: {len(clusters_to_remove)}")
    print(f"Points removed: {points_removed:,} ({points_removed/len(data)*100:.1f}%)")
    
    return filtered_data

def clean_small_clusters(file_path, min_points, span_threshold=None, proximity_threshold=None, max_size_threshold=50):
    """
    Remove points belonging to clusters smaller than the threshold and optionally filter by span and proximity.

    Args:
        file_path (Path): Path to the clustered CSV file.
        min_points (int): Minimum number of points required to keep a cluster.
        span_threshold (float, optional): Maximum span for a cluster to be considered small
        proximity_threshold (float, optional): Distance threshold for considering clusters as close
        max_size_threshold (int, optional): Maximum size for a cluster to be considered as a potential anomaly

    Returns:
        pd.DataFrame: Cleaned DataFrame with filtered clusters.
    """
    # Load the data
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    
    if 'Cluster' not in data.columns:
        raise ValueError("No 'Cluster' column found in the data")
    
    original_points = len(data)
    
    # First filter by cluster size
    print("\nStep 1: Filtering by cluster size...")
    cluster_sizes = data['Cluster'].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= min_points].index
    
    print(f"Total clusters before size filtering: {len(cluster_sizes)}")
    print(f"Clusters meeting size threshold ({min_points} points): {len(valid_clusters)}")
    
    filtered_chunks = []
    with tqdm(total=len(valid_clusters), desc="Processing valid clusters", leave=False) as pbar:
        for cluster_id in valid_clusters:
            cluster_data = data[data['Cluster'] == cluster_id].copy()
            filtered_chunks.append(cluster_data)
            pbar.update(1)
    
    data_filtered = pd.concat(filtered_chunks, ignore_index=True)
    data_filtered = data_filtered.sort_values('Frame').reset_index(drop=True)
    
    points_after_size_filter = len(data_filtered)
    print(f"\nPoints after size filtering:")
    print(f"Original points: {original_points:,}")
    print(f"Remaining points: {points_after_size_filter:,}")
    print(f"Removed points: {original_points - points_after_size_filter:,}")
    print(f"Percentage kept: {(points_after_size_filter/original_points*100):.2f}%")
    
    # Then filter by span and proximity if thresholds are provided
    if span_threshold is not None and proximity_threshold is not None:
        print("\nStep 2: Filtering by span and proximity...")
        data_filtered = filter_clusters_by_span_and_proximity(
            data_filtered, 
            span_threshold, 
            proximity_threshold,
            max_size_threshold
        )
        
        final_points = len(data_filtered)
        print(f"\nPoints after span and proximity filtering:")
        print(f"Points before filtering: {points_after_size_filter:,}")
        print(f"Points after filtering: {final_points:,}")
        print(f"Additional points removed: {points_after_size_filter - final_points:,}")
        print(f"Final percentage of original: {(final_points/original_points*100):.2f}%")
    
    return data_filtered

def plot_sample_files(base_path):
    """
    Plot the 4 files that had the most points removed during cleaning, showing before, after, and removed clusters.

    Args:
        base_path (Path): Path to the cleaned data directory.
    """
    # Create output directory for plots
    output_dir = Path("data/cleaned_movement_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input base path for original files
    input_base = Path("data/Lifespan_clustered")
    
    # Track point reduction for all files
    file_stats = []
    
    # Process each treatment group
    treatment_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print("\nAnalyzing point reduction in all files...")
    for treatment_dir in tqdm(treatment_dirs, desc="Processing treatments", leave=True):
        treatment = treatment_dir.name
        
        # Get all CSV files
        csv_files = list(treatment_dir.glob("*.csv"))
        if not csv_files:
            continue
            
        # Process each file
        for csv_file in tqdm(csv_files, desc=f"Processing {treatment} files", leave=False):
            try:
                # Load cleaned and original data
                cleaned_data = pd.read_csv(csv_file)
                original_file = input_base / treatment / csv_file.name
                original_data = pd.read_csv(original_file)
                
                # Calculate point reduction
                points_removed = len(original_data) - len(cleaned_data)
                points_removed_pct = (points_removed / len(original_data)) * 100
                
                file_stats.append({
                    'treatment': treatment,
                    'file_name': csv_file.name,
                    'cleaned_path': csv_file,
                    'original_path': original_file,
                    'points_removed': points_removed,
                    'points_removed_pct': points_removed_pct
                })
                
            except Exception as e:
                print(f"Error processing file {csv_file.name}: {str(e)}")
    
    # Sort files by points removed (descending) and take top 4
    file_stats.sort(key=lambda x: x['points_removed'], reverse=True)
    top_files = file_stats[:4]
    
    print("\nPlotting the 4 files with most points removed:")
    for stats in top_files:
        print(f"\n{stats['treatment']} - {stats['file_name']}")
        print(f"Points removed: {stats['points_removed']:,} ({stats['points_removed_pct']:.1f}%)")
        
        try:
            # Load data
            cleaned_data = pd.read_csv(stats['cleaned_path'])
            original_data = pd.read_csv(stats['original_path'])
            
            # Create removed data by finding points that are in original but not in cleaned
            original_data['Frame_Cluster'] = original_data['Frame'].astype(str) + '_' + original_data['Cluster'].astype(str)
            cleaned_data['Frame_Cluster'] = cleaned_data['Frame'].astype(str) + '_' + cleaned_data['Cluster'].astype(str)
            removed_mask = ~original_data['Frame_Cluster'].isin(cleaned_data['Frame_Cluster'])
            removed_data = original_data[removed_mask].copy()
            
            # Create figure with 3 subplots side by side
            fig, axes = plt.subplots(1, 3, figsize=(30, 12))
            fig.suptitle(
                f"{stats['treatment']} - {Path(stats['file_name']).stem}\n"
                f"Points removed: {stats['points_removed']:,} ({stats['points_removed_pct']:.1f}%)", 
                fontsize=36, y=1.05
            )
            
            # Plot clustered movement before, after, and removed
            plot_clustered_movement(
                axes[0], original_data, original_data['Cluster'], 
                "Before Cleaning"
            )
            plot_clustered_movement(
                axes[1], cleaned_data, cleaned_data['Cluster'], 
                "After Cleaning"
            )
            plot_clustered_movement(
                axes[2], removed_data, removed_data['Cluster'], 
                "Removed Clusters"
            )
            
            plt.tight_layout()
            
            # Save plot
            output_path = output_dir / f"top_removed_{stats['treatment']}_{Path(stats['file_name']).stem}_comparison.png"
            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
        except Exception as e:
            print(f"Error plotting file {stats['file_name']}: {str(e)}")
            plt.close()

def process_directory(input_dir, output_dir, min_points, span_threshold=None, proximity_threshold=None, max_size_threshold=50):
    """
    Process all CSV files in a directory and its subdirectories.

    Args:
        input_dir (Path): Input directory containing clustered CSV files.
        output_dir (Path): Output directory for cleaned files.
        min_points (int): Minimum number of points required to keep a cluster.
        span_threshold (float, optional): Maximum span threshold for considering a cluster as small
        proximity_threshold (float, optional): Distance threshold for considering clusters as close
        max_size_threshold (int, optional): Maximum size for a cluster to be considered as a potential anomaly
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all treatment directories
    treatment_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    # Process each treatment group
    for treatment_dir in tqdm(treatment_dirs, desc="Processing treatment groups", leave=True):
        treatment = treatment_dir.name
        print(f"\nProcessing {treatment} group...")
        
        # Create output treatment directory
        treatment_output_dir = output_dir / treatment
        treatment_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all CSV files in the treatment directory
        csv_files = list(treatment_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} files to process")
        
        # Process each file
        for csv_file in tqdm(csv_files, desc=f"Processing {treatment} files", leave=True):
            try:
                print(f"\nProcessing {csv_file.name}")
                
                # Clean clusters
                data_filtered = clean_small_clusters(
                    csv_file, min_points, span_threshold, proximity_threshold, max_size_threshold
                )
                
                # Save cleaned data with progress tracking
                output_file = treatment_output_dir / csv_file.name
                print(f"Saving cleaned data to: {output_file}")
                with tqdm(total=1, desc="Saving file", leave=False) as pbar:
                    data_filtered.to_csv(output_file, index=False)
                    pbar.update(1)
                
            except Exception as e:
                print(f"Error processing file {csv_file.name}: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Remove points belonging to small clusters for all files.')
    parser.add_argument('min_points', type=int, help='Minimum number of points required to keep a cluster')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing clustered data')
    parser.add_argument('--output', type=str, required=True, help='Output directory for cleaned data')
    parser.add_argument('--plots', type=str, help='Directory for saving plots (optional)', default='data/cleaned_movement_plots')
    parser.add_argument('--span', type=float, help='Maximum span threshold for considering a cluster as small')
    parser.add_argument('--proximity', type=float, help='Distance threshold for considering clusters as close')
    parser.add_argument('--max-size', type=int, default=50, help='Maximum size for a cluster to be considered as a potential anomaly (default: 50)')
    parser.add_argument('--noplot', action='store_true', help='Disable plotting')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up input and output directories
    input_base = Path(args.input)
    output_base = Path(args.output)
    
    print(f"Input directory: {input_base}")
    print(f"Output directory: {output_base}")
    print(f"Minimum points threshold: {args.min_points}")
    if args.span is not None and args.proximity is not None:
        print(f"Span threshold: {args.span}")
        print(f"Proximity threshold: {args.proximity}")
        print(f"Max size threshold: {args.max_size}")
    
    # Process all files
    process_directory(
        input_base, output_base, args.min_points, 
        args.span, args.proximity, args.max_size
    )
    
    # Plot samples from each treatment group if plotting is enabled
    if not args.noplot:
        print("\nGenerating sample plots...")
        plot_sample_files(output_base)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 