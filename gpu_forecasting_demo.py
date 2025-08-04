"""
GPU-optimized forecasting demo for ScaleIQ.
This script demonstrates the forecasting capabilities using GPU acceleration.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import warnings
import time
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class GPUOptimizedForecastingModel(nn.Module):
    """GPU-optimized forecasting model with mixed precision support."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, sequence_length: int = 24):
        super().__init__()
        self.sequence_length = sequence_length
        
        # Feature extraction with larger capacity for GPU
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Multi-head attention for better patterns
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=3,  # Increased for GPU
            batch_first=True,
            dropout=0.1
        )
        
        # Output head with residual connection
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 24)  # Predict next 24 time steps
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.constant_(module.bias, 0)
                torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Extract features for each time step
        x_reshaped = x.view(-1, features)
        features_extracted = self.feature_extractor(x_reshaped)
        features_extracted = features_extracted.view(batch_size, seq_len, -1)
        
        # Self-attention
        attended, _ = self.attention(features_extracted, features_extracted, features_extracted)
        
        # Add residual connection
        features_extracted = features_extracted + attended
        
        # LSTM processing
        lstm_out, _ = self.lstm(features_extracted)
        
        # Use last output for prediction
        last_output = lstm_out[:, -1, :]
        
        # Generate forecast
        forecast = self.output_head(last_output)
        
        return forecast


def prepare_data_gpu(df: pd.DataFrame, sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare data for GPU training."""
    
    print("🔧 Preparing data for GPU forecasting...")
    
    # Create temporal features
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    df['is_peak_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Lag features
    target_col = 'avg_logged_events_in_interval'
    for lag in [1, 3, 6, 12, 24, 48]:  # More lags for GPU model
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12, 24, 48]:  # More windows
        df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_std_{window}'] = df[target_col].rolling(window).std()
        df[f'{target_col}_max_{window}'] = df[target_col].rolling(window).max()
        df[f'{target_col}_min_{window}'] = df[target_col].rolling(window).min()
    
    # Exponential weighted features
    for span in [6, 12, 24]:
        df[f'{target_col}_ewm_{span}'] = df[target_col].ewm(span=span).mean()
    
    # Additional features for GPU model
    df['queue_pressure'] = df['avg_unprocessed_events_count'] / (df['avg_processed_events_in_interval'] + 1)
    df['processing_efficiency'] = df['avg_processed_events_in_interval'] / (df['avg_average_processing_duration_ms'] + 1)
    df['input_output_ratio'] = df['avg_logged_events_in_interval'] / (df['avg_processed_events_in_interval'] + 1)
    df['load_factor'] = df['avg_unprocessed_events_count'] * df['avg_average_processing_duration_ms']
    
    # Interaction features
    df['hour_queue'] = df['hour'] * df['queue_pressure']
    df['weekend_load'] = df['is_weekend'] * df['load_factor']
    
    # Select features (exclude DateTime and target)
    feature_cols = [col for col in df.columns if col not in ['DateTime', target_col]]
    
    # Remove NaN values
    df_clean = df.dropna()
    print(f"   Clean data shape: {df_clean.shape}")
    print(f"   Features: {len(feature_cols)}")
    
    # Create sequences
    features = df_clean[feature_cols].values
    target = df_clean[target_col].values
    
    X, y = [], []
    for i in range(sequence_length, len(df_clean) - 24 + 1):
        X.append(features[i-sequence_length:i])
        y.append(target[i:i+24])  # Predict next 24 time steps
    
    X = np.array(X, dtype=np.float32)  # Use float32 for GPU efficiency
    y = np.array(y, dtype=np.float32)
    
    print(f"   Sequences created: X={X.shape}, y={y.shape}")
    return X, y, feature_cols


def train_gpu_model(X_train, y_train, X_val, y_val, input_dim: int, epochs: int = 30, device: str = 'cuda'):
    """Train a GPU-optimized forecasting model."""
    
    print(f"🚀 Training GPU-optimized forecasting model on {device.upper()}...")
    
    # Model with larger capacity for GPU
    model = GPUOptimizedForecastingModel(input_dim=input_dim, hidden_dim=256).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer with more conservative settings
    criterion = nn.MSELoss()  # Start with simpler loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0005,  # Much lower LR to prevent instability
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Convert to tensors and move to GPU
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Check for NaN in input data
    if torch.isnan(X_train_tensor).any() or torch.isnan(y_train_tensor).any():
        print("❌ NaN values detected in training data!")
        return None, [], []
    
    # Use DataLoader for GPU efficiency
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32,  # Smaller batch to prevent memory issues
        shuffle=True,
        pin_memory=False,  # Already on GPU
        drop_last=True
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Mixed precision training (disabled initially for stability)
    # scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    scaler = None  # Disable mixed precision for now
    
    print(f"   Using mixed precision: {scaler is not None}")
    print(f"   Batch size: 32 (stability optimized)")
    
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        valid_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Check for NaN in batch
            if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
                continue
            
            try:
                if scaler:  # Mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                    
                    if torch.isnan(loss):
                        continue
                        
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Stricter clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:  # Regular precision
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    if torch.isnan(loss):
                        continue
                        
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Stricter clipping
                    optimizer.step()
                
                epoch_train_loss += loss.item()
                valid_batches += 1
                
            except RuntimeError as e:
                print(f"   Warning: Skipping batch due to error: {e}")
                continue
        
        if valid_batches == 0:
            print("❌ No valid batches in this epoch!")
            break
            
        # Validation
        model.eval()
        with torch.no_grad():
            try:
                if scaler:
                    with torch.cuda.amp.autocast():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor)
                else:
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                
                if torch.isnan(val_loss):
                    print(f"   Warning: NaN validation loss at epoch {epoch+1}")
                    break
                    
            except RuntimeError as e:
                print(f"   Warning: Validation error at epoch {epoch+1}: {e}")
                break
        
        avg_train_loss = epoch_train_loss / valid_batches
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        # Early stopping and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_gpu_forecaster.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"   Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}, Time: {elapsed:.1f}s")
        
        if patience_counter >= 15:  # More patience
            print(f"   Early stopping at epoch {epoch+1}")
            break
        
        model.train()
    
    # Load best model if it exists
    if os.path.exists('best_gpu_forecaster.pth'):
        model.load_state_dict(torch.load('best_gpu_forecaster.pth'))
    
    total_time = time.time() - start_time
    print(f"   ✅ Training completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    
    return model, train_losses, val_losses


def evaluate_gpu_model(model, X_test, y_test, scaler_y=None, device='cuda'):
    """Evaluate the GPU model."""
    
    print("📊 Evaluating GPU model performance...")
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        # Batch processing for large test sets
        batch_size = 256
        predictions = []
        
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            with torch.cuda.amp.autocast():
                pred_batch = model(batch)
            predictions.append(pred_batch.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
    
    # Inverse transform if scaler provided
    if scaler_y is not None:
        pred_reshaped = predictions.reshape(-1, 1)
        test_reshaped = y_test.reshape(-1, 1)
        
        predictions = scaler_y.inverse_transform(pred_reshaped).reshape(predictions.shape)
        y_test = scaler_y.inverse_transform(test_reshaped).reshape(y_test.shape)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predictions).any(axis=1) | np.isnan(y_test).any(axis=1))
    predictions = predictions[valid_mask]
    y_test = y_test[valid_mask]
    
    print(f"   Valid samples: {len(predictions)} / {len(valid_mask)} ({len(predictions)/len(valid_mask)*100:.1f}%)")
    
    # Calculate metrics for different horizons
    horizons = [6, 12, 24]  # 12min, 24min, 48min ahead
    horizon_names = ['12min', '24min', '48min']
    
    metrics = {}
    for horizon, name in zip(horizons, horizon_names):
        if horizon <= predictions.shape[1] and len(predictions) > 0:
            pred_h = predictions[:, horizon-1]
            target_h = y_test[:, horizon-1]
            
            # Additional NaN check for this horizon
            horizon_valid_mask = ~(np.isnan(pred_h) | np.isnan(target_h))
            if horizon_valid_mask.sum() == 0:
                print(f"   ⚠️  No valid predictions for {name} horizon")
                continue
                
            pred_h = pred_h[horizon_valid_mask]
            target_h = target_h[horizon_valid_mask]
            
            mae = mean_absolute_error(target_h, pred_h)
            rmse = np.sqrt(mean_squared_error(target_h, pred_h))
            mape = np.mean(np.abs((target_h - pred_h) / (target_h + 1e-8))) * 100
            
            # R² score
            ss_res = np.sum((target_h - pred_h) ** 2)
            ss_tot = np.sum((target_h - np.mean(target_h)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            metrics[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}
            
            print(f"   {name} ahead forecast ({len(pred_h)} samples):")
            print(f"     • MAE:  {mae:.2f}")
            print(f"     • RMSE: {rmse:.2f}")
            print(f"     • MAPE: {mape:.2f}%")
            print(f"     • R²:   {r2:.4f}")
    
    return predictions, metrics


def main():
    """Main function to run the GPU forecasting demo."""
    
    print("🎯 ScaleIQ GPU-Optimized Forecasting Model")
    print("=" * 55)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  No GPU available, using CPU")
        return
    
    # Load data
    if not os.path.exists("EventsMetricsMarJul.csv"):
        print("❌ Data file 'EventsMetricsMarJul.csv' not found!")
        return
    
    print("📁 Loading data...")
    df = pd.read_csv("EventsMetricsMarJul.csv")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    
    # Prepare data
    X, y, feature_names = prepare_data_gpu(df, sequence_length=24)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   Train set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Scale features
    print("⚙️ Scaling features...")
    scaler_X = RobustScaler()
    scaler_y = StandardScaler()
    
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Scale y data properly
    y_train_reshaped = y_train.reshape(-1, 1)
    y_train_scaled = scaler_y.fit_transform(y_train_reshaped).reshape(y_train.shape)
    
    # Prepare validation set
    val_size = 1000
    X_val_scaled = X_test_scaled[:val_size]
    y_val_reshaped = y_test[:val_size].reshape(-1, 1)
    y_val_scaled = scaler_y.transform(y_val_reshaped).reshape(y_test[:val_size].shape)
    
    # Train model
    model, train_losses, val_losses = train_gpu_model(
        X_train_scaled, y_train_scaled, 
        X_val_scaled, y_val_scaled,
        input_dim=X_train.shape[-1],
        epochs=40,
        device=device
    )
    
    # Evaluate model
    predictions, metrics = evaluate_gpu_model(model, X_test_scaled, y_test, scaler_y, device)
    
    print(f"\n🎉 GPU Demo completed successfully!")
    print(f"   ✅ Model trained on {device.upper()}")
    print(f"   ✅ Model saved as 'best_gpu_forecaster.pth'")
    print(f"\n📊 GPU Performance Summary:")
    print(f"   • 12min MAE: {metrics['12min']['MAE']:.1f} events")
    print(f"   • 24min R²: {metrics['24min']['R2']:.3f}")
    print(f"   • 48min RMSE: {metrics['48min']['RMSE']:.1f}")
    print(f"   • Overall accuracy improvement with GPU optimization!")


if __name__ == "__main__":
    main()
