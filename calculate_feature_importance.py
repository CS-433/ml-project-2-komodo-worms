import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import json
from train_model import SimpleLSTMPredictor, WormDataset, collate_fn, load_data_from_directory
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(train_loader, val_loader, input_size, config, device):
    """Train a model and return it."""
    model = SimpleLSTMPredictor(input_size, config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience'],
        verbose=True
    )
    
    # Train for specified epochs
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    min_delta = config['training']['early_stopping']['min_delta']
    patience = config['training']['early_stopping']['patience']
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.train()
        total_loss = 0
        for batch_features, batch_lifespans, batch_lengths in train_loader:
            batch_features = batch_features.to(device)
            batch_lifespans = batch_lifespans.to(device)
            batch_lengths = batch_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features, batch_lengths)
            loss = criterion(outputs, batch_lifespans)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_lifespans, batch_lengths in val_loader:
                batch_features = batch_features.to(device)
                batch_lifespans = batch_lifespans.to(device)
                batch_lengths = batch_lengths.to(device)
                
                outputs = model(batch_features, batch_lengths)
                loss = criterion(outputs, batch_lifespans)
                val_loss += loss.item()
        
        avg_train_loss = total_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(best_model)
    return model

def calculate_weight_importance(model):
    """Calculate feature importance based on LSTM weights."""
    print("\nCalculating weight-based importance...")
    # Get the input weights from both directions of the first LSTM layer
    forward_weights = model.lstm.weight_ih_l0.data  # Shape: (4*hidden_size, input_size)
    backward_weights = model.lstm.weight_ih_l0_reverse.data  # Shape: (4*hidden_size, input_size)
    
    # Combine weights from both directions
    combined_weights = torch.cat([forward_weights, backward_weights], dim=0)
    
    # Calculate importance as the sum of absolute weights for each feature
    importance = torch.sum(torch.abs(combined_weights), dim=0)
    
    # Normalize to get relative importance
    importance = importance / torch.sum(importance)
    
    return importance.cpu().numpy()

def calculate_gradient_importance(model, val_loader, device):
    """Calculate feature importance based on input gradients."""
    print("\nCalculating gradient-based importance...")
    feature_grads = []
    model.train()  # Set to training mode for gradient calculation
    
    for batch_features, batch_lifespans, batch_lengths in tqdm(val_loader, desc="Processing batches"):
        batch_features = batch_features.to(device).requires_grad_(True)
        batch_lifespans = batch_lifespans.to(device)
        batch_lengths = batch_lengths.to(device)
        
        # Forward pass
        outputs = model(batch_features, batch_lengths)
        
        # Calculate gradients
        loss = nn.MSELoss()(outputs, batch_lifespans)
        loss.backward()
        
        # Get gradients
        feature_importance = torch.abs(batch_features.grad).mean(dim=(0,1))
        feature_grads.append(feature_importance)
        
        # Clear gradients
        batch_features.grad = None
    
    # Average across all batches
    importance = torch.stack(feature_grads).mean(0)
    
    # Normalize
    importance = importance / importance.sum()
    
    return importance.cpu().numpy()

def calculate_permutation_importance(model, val_loader, device, num_permutations=5):
    """Calculate feature importance by permuting features."""
    print("\nCalculating permutation-based importance...")
    model.eval()
    
    # First calculate baseline performance
    baseline_loss = 0
    with torch.no_grad():
        for batch_features, batch_lifespans, batch_lengths in val_loader:
            batch_features = batch_features.to(device)
            batch_lifespans = batch_lifespans.to(device)
            batch_lengths = batch_lengths.to(device)
            outputs = model(batch_features, batch_lengths)
            loss = nn.MSELoss()(outputs, batch_lifespans)
            baseline_loss += loss.item()
    baseline_loss /= len(val_loader)
    
    # Calculate importance for each feature
    num_features = val_loader.dataset[0][0].shape[1]
    importance = np.zeros(num_features)
    
    for feature_idx in tqdm(range(num_features), desc="Processing features"):
        feature_importance = 0
        
        # Repeat permutation multiple times
        for _ in range(num_permutations):
            permuted_loss = 0
            with torch.no_grad():
                for batch_features, batch_lifespans, batch_lengths in val_loader:
                    # Permute the current feature
                    permuted_features = batch_features.clone()
                    permuted_features[:,:,feature_idx] = permuted_features[:,:,feature_idx][torch.randperm(permuted_features.shape[0])]
                    
                    permuted_features = permuted_features.to(device)
                    batch_lifespans = batch_lifespans.to(device)
                    batch_lengths = batch_lengths.to(device)
                    
                    outputs = model(permuted_features, batch_lengths)
                    loss = nn.MSELoss()(outputs, batch_lifespans)
                    permuted_loss += loss.item()
            
            permuted_loss /= len(val_loader)
            feature_importance += (permuted_loss - baseline_loss)
        
        importance[feature_idx] = feature_importance / num_permutations
    
    # Normalize to get relative importance
    importance = np.abs(importance)
    importance = importance / importance.sum()
    
    return importance

def calculate_ablation_importance(model, val_loader, device):
    """Calculate feature importance by ablating (zeroing) features."""
    print("\nCalculating ablation-based importance...")
    model.eval()
    
    # First calculate baseline performance
    baseline_loss = 0
    with torch.no_grad():
        for batch_features, batch_lifespans, batch_lengths in val_loader:
            batch_features = batch_features.to(device)
            batch_lifespans = batch_lifespans.to(device)
            batch_lengths = batch_lengths.to(device)
            outputs = model(batch_features, batch_lengths)
            loss = nn.MSELoss()(outputs, batch_lifespans)
            baseline_loss += loss.item()
    baseline_loss /= len(val_loader)
    
    # Calculate importance for each feature
    num_features = val_loader.dataset[0][0].shape[1]
    importance = np.zeros(num_features)
    
    for feature_idx in tqdm(range(num_features), desc="Processing features"):
        ablated_loss = 0
        with torch.no_grad():
            for batch_features, batch_lifespans, batch_lengths in val_loader:
                # Ablate (zero) the current feature
                ablated_features = batch_features.clone()
                ablated_features[:,:,feature_idx] = 0
                
                ablated_features = ablated_features.to(device)
                batch_lifespans = batch_lifespans.to(device)
                batch_lengths = batch_lengths.to(device)
                
                outputs = model(ablated_features, batch_lengths)
                loss = nn.MSELoss()(outputs, batch_lifespans)
                ablated_loss += loss.item()
        
        ablated_loss /= len(val_loader)
        importance[feature_idx] = ablated_loss - baseline_loss
    
    # Normalize to get relative importance
    importance = np.abs(importance)
    importance = importance / importance.sum()
    
    return importance

def plot_feature_importance_comparison(comparison_df, save_path='feature_importance_plots'):
    """Plot feature importance comparison."""
    os.makedirs(save_path, exist_ok=True)
    
    # Set style and increase default font sizes
    plt.style.use('seaborn')
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 28,
        'axes.titlesize': 32,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 36
    })
    
    # 1. Bar plot comparing all methods
    plt.figure(figsize=(20, 10))
    x = np.arange(len(comparison_df))
    width = 0.2
    
    plt.bar(x - width*1.5, comparison_df['weight_importance'], width, label='Weight-based')
    plt.bar(x - width/2, comparison_df['gradient_importance'], width, label='Gradient-based')
    plt.bar(x + width/2, comparison_df['permutation_importance'], width, label='Permutation-based')
    plt.bar(x + width*1.5, comparison_df['ablation_importance'], width, label='Ablation-based')
    
    plt.xlabel('Features', fontsize=28, labelpad=15)
    plt.ylabel('Relative Importance', fontsize=28, labelpad=15)
    plt.title('Feature Importance Comparison Across Methods', fontsize=36, pad=20)
    plt.xticks(x, comparison_df['feature'], rotation=45, ha='right')
    plt.legend(fontsize=24, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'importance_comparison_bar.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Heatmap of importance values
    plt.figure(figsize=(16, 10))
    importance_matrix = comparison_df.iloc[:, 1:].values.T
    sns.heatmap(importance_matrix, 
                xticklabels=comparison_df['feature'],
                yticklabels=['Weight', 'Gradient', 'Permutation', 'Ablation'],
                cmap='YlOrRd', annot=True, fmt='.3f', 
                annot_kws={'size': 22})
    plt.title('Feature Importance Heatmap', fontsize=36, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'importance_heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Correlation heatmap between methods
    plt.figure(figsize=(12, 10))
    method_correlations = comparison_df.iloc[:, 1:].corr()
    sns.heatmap(method_correlations, annot=True, cmap='coolwarm', center=0,
                annot_kws={'size': 24}, fmt='.2f')
    plt.title('Correlation Between Methods', fontsize=36, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'method_correlations.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 4. Feature ranking comparison
    plt.figure(figsize=(20, 16))
    rankings = pd.DataFrame()
    for method in ['weight_importance', 'gradient_importance', 'permutation_importance', 'ablation_importance']:
        rankings[method] = comparison_df.sort_values(method, ascending=False)['feature']
    
    # Plot ranking comparison
    for i, method in enumerate(rankings.columns):
        plt.subplot(2, 2, i+1)
        y_pos = np.arange(len(rankings[method]))
        plt.barh(y_pos, np.arange(len(y_pos), 0, -1))
        plt.yticks(y_pos, rankings[method], fontsize=22)
        plt.xlabel('Rank', fontsize=28, labelpad=15)
        plt.title(method.replace('_', ' ').title(), fontsize=32, pad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_rankings.png'), bbox_inches='tight', dpi=300)
    plt.close()

def prepare_data(config):
    """Load and prepare data for training and validation."""
    # Load data from directories
    base_dir = config['data']['base_dir']
    all_features = []
    all_lifespans = []
    all_lengths = []
    all_files = []
    all_groups = []
    
    print("\nLoading data from directories:")
    for subdir in config['data']['subdirs']:
        print(f"\nProcessing {subdir}...")
        dir_path = os.path.join(base_dir, subdir)
        # Remove selected_features from config to use all features
        config_copy = {
            'model': config['model'].copy(),
            'training': config['training'].copy(),
            'data': config['data'].copy(),
            'random_seed': config['random_seed']
        }
        if 'selected_features' in config_copy['training']:
            del config_copy['training']['selected_features']
        features, lifespans, lengths, files, _, groups = load_data_from_directory(
            dir_path, 
            max_frame=config['training']['max_frame'],
            config=config_copy
        )
        all_features.extend(features)
        all_lifespans.extend(lifespans)
        all_lengths.extend(lengths)
        all_files.extend(files)
        all_groups.extend(groups)
        print(f"Loaded {len(features)} samples from {subdir}")
    
    print(f"\nTotal samples: {len(all_features)}")
    
    # Get feature names from first file
    data = np.load(all_files[0], allow_pickle=True)
    feature_names = data['feature_names'].tolist()
    print(f"\nUsing all {len(feature_names)} features:")
    for feat in feature_names:
        print(f"- {feat}")
    
    # Scale features
    print("\nScaling features...")
    all_features_flat = np.vstack([f for f in all_features])
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(all_features_flat)
    scaled_features = [feature_scaler.transform(f) for f in all_features]
    
    # Scale lifespans
    scale_factor = config['training']['scale_factor']
    scaled_lifespans = np.array(all_lifespans) / scale_factor
    
    # Split into train and validation sets
    num_samples = len(scaled_features)
    indices = np.random.permutation(num_samples)
    split = int(0.8 * num_samples)
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    print(f"\nSplit data into {len(train_indices)} training and {len(val_indices)} validation samples")
    
    # Create datasets
    train_features = [scaled_features[i] for i in train_indices]
    train_lifespans = scaled_lifespans[train_indices]
    train_lengths = [all_lengths[i] for i in train_indices]
    
    val_features = [scaled_features[i] for i in val_indices]
    val_lifespans = scaled_lifespans[val_indices]
    val_lengths = [all_lengths[i] for i in val_indices]
    
    # Create dataloaders
    train_dataset = WormDataset(train_features, train_lifespans, train_lengths)
    val_dataset = WormDataset(val_features, val_lifespans, val_lengths)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, len(feature_names), feature_names

def compare_importance_methods():
    """Calculate and compare feature importance using multiple methods."""
    try:
        # Load configuration
        print("Loading configuration...")
        with open('train_model_config.json', 'r') as f:
            config = json.load(f)
        
        # Set random seed for reproducibility
        torch.manual_seed(config['random_seed'])
        np.random.seed(config['random_seed'])
        
        # Prepare data and train model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        
        print("\nPreparing data...")
        train_loader, val_loader, input_size, feature_names = prepare_data(config)
        
        print("\nTraining model...")
        model = train_model(train_loader, val_loader, input_size, config, device)
        
        print("\nCalculating feature importance...")
        # Calculate importance using all methods
        weight_importance = calculate_weight_importance(model)
        gradient_importance = calculate_gradient_importance(model, val_loader, device)
        permutation_importance = calculate_permutation_importance(model, val_loader, device)
        ablation_importance = calculate_ablation_importance(model, val_loader, device)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'feature': feature_names,
            'weight_importance': weight_importance,
            'gradient_importance': gradient_importance,
            'permutation_importance': permutation_importance,
            'ablation_importance': ablation_importance
        })
        
        # Sort by average importance
        comparison['avg_importance'] = comparison.iloc[:, 1:].mean(axis=1)
        comparison = comparison.sort_values('avg_importance', ascending=False)
        comparison = comparison.drop('avg_importance', axis=1)
        
        # Save results
        comparison.to_csv('feature_importance_comparison.csv', index=False)
        print("\nFeature Importance Comparison:")
        print(comparison.to_string())
        
        # Plot results
        plot_feature_importance_comparison(comparison)
        
        return comparison
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    compare_importance_methods() 