import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json
from scipy.spatial import ConvexHull
from scipy.fft import rfft, rfftfreq
from scipy.stats import entropy
import argparse
from pathlib import Path

def calculate_speed(df):
    """Calculate speed between consecutive frames."""
    # First row speed is 0
    speeds = [0]
    
    # Calculate speeds for remaining rows
    for i in range(1, len(df)):
        dx = df.iloc[i]['X'] - df.iloc[i-1]['X']
        dy = df.iloc[i]['Y'] - df.iloc[i-1]['Y']
        dt = df.iloc[i]['Timestamp'] - df.iloc[i-1]['Timestamp']
        
        # Calculate Euclidean distance
        distance = np.sqrt(dx**2 + dy**2)
        speed = distance / dt if dt != 0 else 0
        speeds.append(speed)
    
    return speeds

def classify_roaming_dwelling(df, slope=2.5, intercept=0):
    """Classifies each window as roaming or dwelling based on a threshold."""
    df['State'] = 'D'  # Default to dwelling
    df.loc[df['Speed (window)'] > (slope * df['Angular Velocity (window)'] + intercept), 'State'] = 'R'
    return df

def calculate_curvature(x, y):
    """Calculates curvature using a simple approximation."""
    if len(x) < 2:  # Need at least 2 points for gradient
        return np.zeros(len(x))
        
    try:
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denominator = (dx**2 + dy**2)**1.5
        # Handle division by zero
        curvature = np.zeros_like(denominator)
        mask = denominator > 1e-10  # Small threshold to avoid division by zero
        curvature[mask] = np.abs(ddx[mask] * dy[mask] - dx[mask] * ddy[mask]) / denominator[mask]
        return curvature
    except ValueError:  # If gradient calculation fails
        return np.zeros(len(x))

def calculate_autocorrelation(series, lag=1):
    """Calculates the autocorrelation of a series with a given lag."""
    if len(series) < lag + 1:
        return 0
    return np.corrcoef(series[:-lag], series[lag:])[0, 1]

def calculate_speed_and_angular_velocity(df):
    """Calculate speed and angular velocity for each frame."""
    # Calculate speed
    df['dx'] = df['X'].diff()
    df['dy'] = df['Y'].diff()
    df['dt'] = df['Timestamp'].diff()
    
    # Handle zero time differences
    mask = df['dt'] > 0
    df['Speed'] = 0.0  # Initialize with zeros
    df.loc[mask, 'Speed'] = np.sqrt(df.loc[mask, 'dx']**2 + df.loc[mask, 'dy']**2) / df.loc[mask, 'dt']
    
    # Calculate angles and angular velocity (using every other frame)
    df['dx_2'] = df['X'].diff(2)
    df['dy_2'] = df['Y'].diff(2)
    df['angle_rad'] = np.arctan2(df['dy_2'], df['dx_2'])
    df['angle_deg'] = np.degrees(df['angle_rad'])
    
    # Handle zero time differences for angular velocity
    dt_2 = df['Timestamp'].diff(2)
    df['Angular Velocity'] = 0.0  # Initialize with zeros
    mask_2 = dt_2 > 0
    df.loc[mask_2, 'Angular Velocity'] = df.loc[mask_2, 'angle_deg'].diff() / dt_2[mask_2]
    
    # Calculate acceleration
    df['Acceleration'] = 0.0  # Initialize with zeros
    df.loc[mask, 'Acceleration'] = df.loc[mask, 'Speed'].diff() / df.loc[mask, 'dt']
    
    # Smooth the speed and angular velocity for roaming/dwelling classification
    df['Speed (window)'] = df['Speed'].rolling(5, center=True, min_periods=1).mean()
    df['Angular Velocity (window)'] = df['Angular Velocity'].rolling(5, center=True, min_periods=1).mean()
    
    # Fill any remaining NaN values with 0
    df = df.fillna(0)
    
    # Classify roaming and dwelling states
    df = classify_roaming_dwelling(df)
    
    return df

def extract_cluster_features(cluster_data):
    """Extract features from a single cluster."""
    features = {}
    
    # Basic temporal features
    features['duration'] = cluster_data['Timestamp'].max() - cluster_data['Timestamp'].min()
    features['num_frames'] = len(cluster_data)
    
    # Speed statistics
    features['mean_speed'] = cluster_data['Speed'].mean()
    features['max_speed'] = cluster_data['Speed'].max()
    features['min_speed'] = cluster_data['Speed'].min()
    features['std_speed'] = cluster_data['Speed'].std()
    
    # Angular velocity statistics
    features['mean_angular_velocity'] = cluster_data['Angular Velocity'].mean()
    features['max_angular_velocity'] = cluster_data['Angular Velocity'].max()
    features['min_angular_velocity'] = cluster_data['Angular Velocity'].min()
    features['std_angular_velocity'] = cluster_data['Angular Velocity'].std()
    
    # Acceleration statistics
    features['max_acceleration'] = cluster_data['Acceleration'].max()
    features['min_acceleration'] = cluster_data['Acceleration'].min()
    features['mean_acceleration'] = cluster_data['Acceleration'].mean()
    features['std_acceleration'] = cluster_data['Acceleration'].std()
    
    # Path shape features
    try:
        curvature = calculate_curvature(cluster_data['X'].values, cluster_data['Y'].values)
        features['mean_curvature'] = np.nanmean(curvature)
        features['max_curvature'] = np.nanmax(curvature)
    except:
        features['mean_curvature'] = 0
        features['max_curvature'] = 0
    
    # Convex hull area (if enough points)
    if len(cluster_data) >= 3:
        try:
            hull = ConvexHull(cluster_data[['X', 'Y']].values)
            features['convex_hull_area'] = hull.volume  # In 2D, volume is area
        except:
            features['convex_hull_area'] = 0
    else:
        features['convex_hull_area'] = 0
    
    # Radius of gyration
    centroid = cluster_data[['X', 'Y']].mean()
    distances = np.sqrt((cluster_data['X'] - centroid['X'])**2 + 
                       (cluster_data['Y'] - centroid['Y'])**2)
    features['radius_of_gyration'] = np.sqrt(np.sum(distances**2) / len(cluster_data))
    
    # Roaming/Dwelling features
    roaming_frames = cluster_data['State'] == 'R'
    dwelling_frames = cluster_data['State'] == 'D'
    
    features['fraction_roaming'] = roaming_frames.sum() / len(cluster_data)
    features['fraction_dwelling'] = dwelling_frames.sum() / len(cluster_data)
    
    # Calculate bout durations
    bouts = cluster_data['State'].ne(cluster_data['State'].shift()).cumsum()
    roaming_bouts = cluster_data[roaming_frames].groupby(bouts)['Timestamp'].agg(lambda x: x.max() - x.min())
    dwelling_bouts = cluster_data[dwelling_frames].groupby(bouts)['Timestamp'].agg(lambda x: x.max() - x.min())
    
    features['mean_roaming_bout_duration'] = roaming_bouts.mean() if len(roaming_bouts) > 0 else 0
    features['mean_dwelling_bout_duration'] = dwelling_bouts.mean() if len(dwelling_bouts) > 0 else 0
    
    features['roaming_frequency'] = roaming_frames.sum() / features['duration'] if features['duration'] > 0 else 0
    features['dwelling_frequency'] = dwelling_frames.sum() / features['duration'] if features['duration'] > 0 else 0
    
    # State transitions
    state_changes = cluster_data['State'].ne(cluster_data['State'].shift()).sum() - 1
    features['state_transitions'] = max(state_changes, 0)
    
    # Frequency domain features
    if len(cluster_data) > 1:
        try:
            yf = rfft(cluster_data['Speed'].values)
            xf = rfftfreq(len(cluster_data), 1 / 2)  # Assuming 2 seconds between frames
            dominant_frequency_index = np.argmax(np.abs(yf[1:])) + 1
            features['dominant_frequency'] = xf[dominant_frequency_index]
        except:
            features['dominant_frequency'] = 0
    else:
        features['dominant_frequency'] = 0
    
    # Entropy features
    try:
        speed_counts = np.histogram(cluster_data['Speed'], bins=10)[0]
        features['speed_entropy'] = entropy(speed_counts) if np.any(speed_counts > 0) else 0
    except:
        features['speed_entropy'] = 0
        
    try:
        angular_velocity_counts = np.histogram(cluster_data['Angular Velocity'], bins=10)[0]
        features['angular_velocity_entropy'] = entropy(angular_velocity_counts) if np.any(angular_velocity_counts > 0) else 0
    except:
        features['angular_velocity_entropy'] = 0
    
    # Autocorrelation features
    try:
        features['speed_autocorrelation_1'] = calculate_autocorrelation(cluster_data['Speed'], lag=1)
        features['speed_autocorrelation_5'] = calculate_autocorrelation(cluster_data['Speed'], lag=5)
        features['angular_velocity_autocorrelation_1'] = calculate_autocorrelation(cluster_data['Angular Velocity'], lag=1)
        features['angular_velocity_autocorrelation_5'] = calculate_autocorrelation(cluster_data['Angular Velocity'], lag=5)
    except:
        features['speed_autocorrelation_1'] = 0
        features['speed_autocorrelation_5'] = 0
        features['angular_velocity_autocorrelation_1'] = 0
        features['angular_velocity_autocorrelation_5'] = 0
    
    return features

def process_directory(input_dir, output_dir):
    """Process all CSV files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"\nProcessing directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(csv_files)} CSV files to process")
    
    for filename in tqdm(csv_files, desc="Processing files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.csv', '_features.npz'))
        
        # Skip if output file already exists
        if os.path.exists(output_path):
            print(f"\nSkipping {filename} - features already exist at {output_path}")
            continue
        
        try:
            print(f"\nProcessing {filename}...")
            # Read and preprocess data
            print(f"Loading data from {input_path}")
            df = pd.read_csv(input_path)
            print(f"Loaded {len(df)} rows with {len(df['Cluster'].unique())} clusters")
            
            # Calculate speed, angular velocity, and state features
            print("Calculating speed and angular velocity features...")
            df = calculate_speed_and_angular_velocity(df)
            
            # Extract features for each cluster
            all_features = []
            feature_names = None
            
            print("\nExtracting features for each cluster:")
            for cluster_id in tqdm(df['Cluster'].unique(), desc="Processing clusters"):
                cluster_data = df[df['Cluster'] == cluster_id]
                print(f"\nCluster {cluster_id}:")
                print(f"  Points: {len(cluster_data)}")
                print(f"  Frame range: {cluster_data['Frame'].min()} to {cluster_data['Frame'].max()}")
                
                features = extract_cluster_features(cluster_data)
                print(f"  Extracted {len(features)} features")
                
                if feature_names is None:
                    feature_names = list(features.keys())
                    print(f"\nFeature names ({len(feature_names)}):")
                    for name in feature_names:
                        print(f"  - {name}")
                
                feature_values = [features[name] for name in feature_names]
                print(f"  Feature values: min={min(feature_values):.2f}, max={max(feature_values):.2f}")
                all_features.append(feature_values)
            
            # Convert to numpy array
            features_array = np.array(all_features)
            print(f"\nFinal features array shape: {features_array.shape}")
            
            # Save features
            print(f"Saving features to {output_path}")
            np.savez(output_path,
                    features=features_array,
                    feature_names=feature_names,
                    num_frames=df['Frame'].max(),
                    source_file=filename)
            print(f"Successfully saved features for {filename}")
            
            # Verify saved file
            print("Verifying saved file...")
            loaded = np.load(output_path, allow_pickle=True)
            print(f"Verification results:")
            print(f"  Features shape: {loaded['features'].shape}")
            print(f"  Number of feature names: {len(loaded['feature_names'])}")
            print(f"  Max frame number: {loaded['num_frames']}")
            print(f"  Source file: {loaded['source_file']}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract features from cleaned worm movement data.')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing cleaned data')
    parser.add_argument('--output', type=str, required=True, help='Output directory for calculated features')
    parser.add_argument('--treatments', nargs='+', default=['control', 'Terbinafin', 'controlTerbinafin', 'companyDrug'],
                        help='List of treatment subdirectories to process')
    args = parser.parse_args()
    
    # Process each subdirectory
    base_dir = Path(args.input)
    output_base_dir = Path(args.output)
    output_base_dir.mkdir(parents=True, exist_ok=True)  # Create base output directory
    
    print(f"Input directory: {base_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Processing treatments: {', '.join(args.treatments)}")
    
    for subdir in args.treatments:
        print(f"\nProcessing {subdir}...")
        input_dir = base_dir / subdir
        output_dir = output_base_dir / subdir
        
        if not input_dir.exists():
            print(f"Warning: Input directory {input_dir} does not exist, skipping...")
            continue
            
        # Create output directory for this treatment
        output_dir.mkdir(parents=True, exist_ok=True)
        process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main() 