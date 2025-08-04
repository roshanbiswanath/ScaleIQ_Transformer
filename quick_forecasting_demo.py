"""
Quick evaluation and demo for the ScaleIQ Forecasting Model.
This script demonstrates the forecasting capabilities on a subset of data.
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
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


class SimpleForecastingModel(nn.Module):
    """Simplified forecasting model for quick demonstration."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, sequence_length: int = 24):
        super().__init__()
        self.sequence_length = sequence_length
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 24)  # Predict next 24 time steps
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Extract features for each time step
        x_reshaped = x.view(-1, features)
        features_extracted = self.feature_extractor(x_reshaped)
        features_extracted = features_extracted.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features_extracted)
        
        # Use last output for prediction
        last_output = lstm_out[:, -1, :]
        
        # Generate forecast
        forecast = self.output_head(last_output)
        
        return forecast


def prepare_data(df: pd.DataFrame, sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare data for forecasting."""
    
    print("🔧 Preparing data for forecasting...")
    
    # Create temporal features
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Lag features
    target_col = 'avg_logged_events_in_interval'
    for lag in [1, 3, 6, 12, 24]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [6, 12, 24]:
        df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_std_{window}'] = df[target_col].rolling(window).std()
    
    # Additional features
    df['queue_pressure'] = df['avg_unprocessed_events_count'] / (df['avg_processed_events_in_interval'] + 1)
    df['processing_efficiency'] = df['avg_processed_events_in_interval'] / (df['avg_average_processing_duration_ms'] + 1)
    
    # Select features (exclude DateTime and target)
    feature_cols = [col for col in df.columns if col not in ['DateTime', target_col]]
    
    # Remove NaN values
    df_clean = df.dropna()
    print(f"   Clean data shape: {df_clean.shape}")
    
    # Create sequences
    features = df_clean[feature_cols].values
    target = df_clean[target_col].values
    
    X, y = [], []
    for i in range(sequence_length, len(df_clean) - 24 + 1):
        X.append(features[i-sequence_length:i])
        y.append(target[i:i+24])  # Predict next 24 time steps
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   Sequences created: X={X.shape}, y={y.shape}")
    return X, y, feature_cols


def train_simple_model(X_train, y_train, X_val, y_val, input_dim: int, epochs: int = 20):
    """Train a simple forecasting model."""
    
    print("🚀 Training simple forecasting model...")
    
    model = SimpleForecastingModel(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(epochs):
        # Training
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        train_loss = criterion(outputs, y_train_tensor)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        model.train()
    
    print(f"   ✅ Training completed!")
    return model, train_losses, val_losses


def evaluate_model(model, X_test, y_test, scaler_y=None):
    """Evaluate the model and generate predictions."""
    
    print("📊 Evaluating model performance...")
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        predictions = model(X_test_tensor).numpy()
    
    # Inverse transform if scaler provided
    if scaler_y is not None:
        # Properly reshape for inverse transform
        pred_reshaped = predictions.reshape(-1, 1)
        test_reshaped = y_test.reshape(-1, 1)
        
        predictions = scaler_y.inverse_transform(pred_reshaped).reshape(predictions.shape)
        y_test = scaler_y.inverse_transform(test_reshaped).reshape(y_test.shape)
    
    # Calculate metrics for different horizons
    horizons = [6, 12, 24]  # 12min, 24min, 48min ahead
    horizon_names = ['12min', '24min', '48min']
    
    metrics = {}
    for horizon, name in zip(horizons, horizon_names):
        if horizon <= predictions.shape[1]:
            pred_h = predictions[:, horizon-1]
            target_h = y_test[:, horizon-1]
            
            mae = mean_absolute_error(target_h, pred_h)
            rmse = np.sqrt(mean_squared_error(target_h, pred_h))
            mape = np.mean(np.abs((target_h - pred_h) / (target_h + 1e-8))) * 100
            
            # R² score
            ss_res = np.sum((target_h - pred_h) ** 2)
            ss_tot = np.sum((target_h - np.mean(target_h)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            metrics[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}
            
            print(f"   {name} ahead forecast:")
            print(f"     • MAE:  {mae:.2f}")
            print(f"     • RMSE: {rmse:.2f}")
            print(f"     • MAPE: {mape:.2f}%")
            print(f"     • R²:   {r2:.4f}")
    
    return predictions, metrics


def create_visualizations(predictions, targets, train_losses, val_losses, metrics):
    """Create comprehensive visualizations."""
    
    print("📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ScaleIQ Forecasting Model - Quick Demo Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Training curves
    ax = axes[0, 0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2-4: Forecasting results for different horizons
    horizons = [6, 12, 24]
    horizon_names = ['12min', '24min', '48min']
    
    for i, (horizon, name) in enumerate(zip(horizons, horizon_names)):
        if horizon <= predictions.shape[1]:
            ax = axes[0, i+1] if i < 2 else axes[1, 0]
            
            # Show last 200 points for clarity
            n_plot = min(200, len(predictions))
            pred_h = predictions[-n_plot:, horizon-1]
            target_h = targets[-n_plot:, horizon-1]
            
            ax.plot(range(n_plot), target_h, label='Actual', linewidth=2, alpha=0.8)
            ax.plot(range(n_plot), pred_h, label='Predicted', linewidth=2, alpha=0.8)
            ax.set_title(f'{name} Ahead Forecast')
            ax.set_ylabel('Events')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics
            mae = metrics[name]['MAE']
            r2 = metrics[name]['R2']
            ax.text(0.02, 0.98, f'MAE: {mae:.1f}\nR²: {r2:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 5: Scatter plot (24min ahead)
    ax = axes[1, 1]
    pred_24 = predictions[:, 11]  # 12th index for 24min ahead
    target_24 = targets[:, 11]
    
    # Sample points if too many
    if len(pred_24) > 1000:
        indices = np.random.choice(len(pred_24), 1000, replace=False)
        pred_24 = pred_24[indices]
        target_24 = target_24[indices]
    
    ax.scatter(target_24, pred_24, alpha=0.6, s=20)
    min_val = min(target_24.min(), pred_24.min())
    max_val = max(target_24.max(), pred_24.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Actual Events')
    ax.set_ylabel('Predicted Events')
    ax.set_title('24min Ahead: Predicted vs Actual')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Metrics comparison
    ax = axes[1, 2]
    metric_names = ['MAE', 'RMSE', 'R2']
    x_pos = np.arange(len(horizon_names))
    
    for i, metric in enumerate(metric_names):
        if metric == 'R2':
            values = [metrics[name][metric] for name in horizon_names]
        else:
            # Normalize for comparison
            all_values = [metrics[name][metric] for name in horizon_names]
            max_val = max(all_values) if max(all_values) > 0 else 1
            values = [v/max_val for v in all_values]
        
        ax.bar(x_pos + i*0.25, values, width=0.25, label=metric, alpha=0.8)
    
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('Normalized Metric')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x_pos + 0.25)
    ax.set_xticklabels(horizon_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('forecasting_demo_results.png', dpi=300, bbox_inches='tight')
    print("   ✅ Results saved as 'forecasting_demo_results.png'")
    plt.show()


def main():
    """Main function to run the forecasting demo."""
    
    print("🎯 ScaleIQ Forecasting Model - Quick Demo")
    print("=" * 50)
    
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
    X, y, feature_names = prepare_data(df, sequence_length=24)
    
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
    
    # Prepare small validation set
    val_size = 100
    X_val_scaled = X_test_scaled[:val_size]
    y_val_reshaped = y_test[:val_size].reshape(-1, 1)
    y_val_scaled = scaler_y.transform(y_val_reshaped).reshape(y_test[:val_size].shape)
    
    # Train model
    model, train_losses, val_losses = train_simple_model(
        X_train_scaled, y_train_scaled, 
        X_val_scaled, y_val_scaled,
        input_dim=X_train.shape[-1],
        epochs=25
    )
    
    # Evaluate model
    predictions_scaled, metrics = evaluate_model(model, X_test_scaled, y_test, scaler_y)
    
    # Create visualizations
    create_visualizations(predictions_scaled, y_test, train_losses, val_losses, metrics)
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"   ✅ Model trained and evaluated")
    print(f"   ✅ Visualizations saved")
    print(f"\n📊 Quick Summary:")
    print(f"   • Best 12min MAE: {metrics['12min']['MAE']:.1f} events")
    print(f"   • Best 24min R²: {metrics['24min']['R2']:.3f}")
    print(f"   • Model shows {metrics['12min']['R2']*100:.1f}% explained variance at 12min horizon")


if __name__ == "__main__":
    main()
