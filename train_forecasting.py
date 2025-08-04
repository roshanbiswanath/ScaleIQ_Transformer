"""
Focused forecasting model training for ScaleIQ.
State-of-the-art Transformer-based event traffic forecasting.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import math
import warnings
from rich.console import Console

warnings.filterwarnings('ignore')

# Rich console for beautiful printing
console = Console()

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiScaleConv1D(nn.Module):
    """Multi-scale convolutional feature extraction."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        
        # Ensure each conv outputs the same number of channels for proper concatenation
        channels_per_conv = out_channels // len(kernel_sizes)
        remainder = out_channels % len(kernel_sizes)
        
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # Add remainder to first conv to handle uneven division
            out_ch = channels_per_conv + (1 if i < remainder else 0)
            self.convs.append(
                nn.Conv1d(in_channels, out_ch, kernel_size=k, padding=k//2)
            )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(torch.relu(conv(x)))
        
        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)
        x = self.bn(x)
        x = self.dropout(x)
        
        # Back to (batch, seq_len, features)
        return x.transpose(1, 2)


class EventForecastingTransformer(pl.LightningModule):
    """
    State-of-the-art event traffic forecasting model using Transformer architecture.
    
    Features:
    - Multi-scale CNN for local pattern extraction
    - Transformer encoder for long-range dependencies
    - Multi-horizon prediction
    - Advanced loss functions
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 512,   # Increased for better capacity
                 nhead: int = 16,      # More attention heads
                 num_encoder_layers: int = 6,  # Deeper network
                 dim_feedforward: int = 1024,  # Larger feedforward network
                 dropout: float = 0.1,         # Keep dropout reasonable
                 sequence_length: int = 24,
                 forecast_horizons: List[int] = [3, 6, 12, 24, 48],
                 learning_rate: float = 5e-4,  # Higher learning rate
                 weight_decay: float = 1e-4,   # Keep weight decay
                 warmup_steps: int = 2000):    # More warmup for stability
        
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.forecast_horizons = forecast_horizons
        self.max_horizon = max(forecast_horizons)
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Simplified multi-scale convolutional feature extraction
        self.conv_extractor = MultiScaleConv1D(d_model, d_model, kernel_sizes=[3, 5])
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Feature fusion layer - remove this since we're simplifying
        # self.feature_fusion = nn.Linear(d_model * 2, d_model)
        
        # Output projection layers for different horizons
        self.horizon_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, h)
            ) for h in forecast_horizons
        })
        
        # Main prediction head
        self.main_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.max_horizon)
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(delta=1.0)
        
        # Metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simplified and stable forward pass."""
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Multi-scale convolution for local patterns
        conv_features = self.conv_extractor(x)
        
        # Add positional encoding
        pos_encoded = self.pos_encoder(conv_features.transpose(0, 1)).transpose(0, 1)
        
        # Apply layer normalization
        normalized = self.layer_norm(pos_encoded)
        
        # Transformer encoding
        transformer_output = self.transformer(normalized)
        
        # Simple global average pooling
        global_features = torch.mean(transformer_output, dim=1)
        
        # Multi-horizon predictions
        predictions = {}
        for horizon_name, head in self.horizon_heads.items():
            predictions[horizon_name] = head(global_features)
        
        # Main prediction
        predictions['main'] = self.main_head(global_features)
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Stabilized loss function with proper weighting and clipping."""
        
        losses = {}
        total_loss = 0.0
        
        # Main prediction loss (full horizon) - primary objective
        main_pred = predictions['main']
        target_main = targets[:, :self.max_horizon]
        
        # Robust losses with gradient clipping
        mse_loss = self.mse_loss(main_pred, target_main)
        mae_loss = self.mae_loss(main_pred, target_main)
        
        # Clamp losses to prevent explosion
        mse_loss = torch.clamp(mse_loss, max=10.0)
        mae_loss = torch.clamp(mae_loss, max=10.0)
        
        # Main loss with conservative weighting
        main_loss = 0.8 * mse_loss + 0.2 * mae_loss
        
        total_loss += main_loss
        losses['main_loss'] = main_loss
        
        # Multi-horizon losses with much lower weights and stabilization
        horizon_loss_weight = 0.1  # Reduced from 0.5 to prevent overwhelming main loss
        
        for horizon in self.forecast_horizons:
            if f'horizon_{horizon}' in predictions:
                horizon_pred = predictions[f'horizon_{horizon}']
                horizon_target = targets[:, :horizon]
                
                # Stabilized horizon losses
                horizon_mse = torch.clamp(self.mse_loss(horizon_pred, horizon_target), max=5.0)
                horizon_mae = torch.clamp(self.mae_loss(horizon_pred, horizon_target), max=5.0)
                
                horizon_loss = 0.8 * horizon_mse + 0.2 * horizon_mae
                
                # Much lower weight to prevent instability
                total_loss += horizon_loss_weight * horizon_loss
                losses[f'horizon_{horizon}_loss'] = horizon_loss
        
        # Gradient clipping at loss level
        total_loss = torch.clamp(total_loss, max=20.0)
        losses['total_loss'] = total_loss
        
        return total_loss, losses
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        predictions = self(x)
        
        loss, loss_components = self.compute_loss(predictions, y)
        
        # Log metrics
        for key, value in loss_components.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'predictions': predictions['main'].detach()[:5],  # Save first 5 for analysis
            'targets': y.detach()[:5, :self.max_horizon]
        })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        predictions = self(x)
        
        loss, loss_components = self.compute_loss(predictions, y)
        
        # Log metrics
        for key, value in loss_components.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate additional metrics for main prediction
        main_pred = predictions['main']
        main_target = y[:, :self.max_horizon]
        
        # MAPE
        mape = torch.mean(torch.abs((main_target - main_pred) / (main_target + 1e-8))) * 100
        self.log('val_mape', mape, on_epoch=True, prog_bar=True)
        
        # R2 Score approximation
        ss_res = torch.sum((main_target - main_pred) ** 2)
        ss_tot = torch.sum((main_target - main_target.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        self.log('val_r2', r2, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'predictions': main_pred.detach(),
            'targets': main_target.detach(),
            'mape': mape.detach()
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """Clean up training step outputs."""
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Clean up validation step outputs."""
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer with stabilized learning rate schedule."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),  # More conservative beta2 for stability
            eps=1e-6  # Smaller epsilon for numerical stability
        )
        
        # Gentler learning rate schedule
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,  # Less aggressive reduction
            patience=10,  # More patience before reducing
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_total_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True
            }
        }


class ForecastingDataModule(pl.LightningDataModule):
    """Data module for forecasting."""
    
    def __init__(self,
                 data_file: str,
                 sequence_length: int = 24,
                 forecast_horizons: List[int] = [3, 6, 12, 24, 48],
                 batch_size: int = 32,
                 test_size: float = 0.2,
                 num_workers: int = 4):
        
        super().__init__()
        self.data_file = data_file
        self.sequence_length = sequence_length
        self.forecast_horizons = forecast_horizons
        self.max_horizon = max(forecast_horizons)
        self.batch_size = batch_size
        self.test_size = test_size
        self.num_workers = num_workers
        
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None
        # Remove log transform flag since we're not using it
    
    def prepare_data(self):
        """Prepare the data (download, etc.)."""
        # This runs on only one GPU/process
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Setup the data for training/validation."""
        
        # Load data
        df = pd.read_csv(self.data_file)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        print(f"Loaded data: {df.shape}")
        print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        # Feature engineering
        df = self.create_features(df)
        
        # Select features
        feature_cols = [col for col in df.columns if col not in ['DateTime']]
        self.feature_names = feature_cols
        
        print(f"Features created: {len(feature_cols)}")
        
        # Remove NaN values
        df_clean = df[feature_cols].dropna()
        print(f"Clean data shape: {df_clean.shape}")
        
        # Create sequences
        X, y = self.create_sequences(df_clean, feature_cols)
        print(f"Sequences created: X={X.shape}, y={y.shape}")
        
        # STRATIFIED TEMPORAL SPLIT: Better approach for time series generalization
        # Divide data into time intervals and sample from each interval
        # This ensures both train and validation see all temporal patterns
        
        # Define interval size (e.g., 7 days worth of hourly data = 168 hours)
        interval_size = 168  # 7 days * 24 hours
        num_intervals = len(df_clean) // interval_size
        
        print(f"Creating stratified temporal split with {num_intervals} intervals of {interval_size} hours each")
        
        train_indices = []
        val_indices = []
        
        for interval_idx in range(num_intervals):
            start_idx = interval_idx * interval_size
            end_idx = min((interval_idx + 1) * interval_size, len(df_clean))
            interval_length = end_idx - start_idx
            
            if interval_length < 48:  # Skip intervals too small for sequences
                continue
            
            # Within each interval, take sequences for train/val
            # Leave buffer zones to prevent sequence overlap
            buffer = 12  # 12-hour buffer between train and val sequences
            
            # Calculate available sequence positions in this interval
            available_sequences = []
            for seq_start in range(start_idx, end_idx - 24):  # -24 for sequence length
                if seq_start + 24 <= end_idx:  # Ensure full sequence fits
                    # Convert to sequence index
                    seq_idx = seq_start - 24  # Adjust for sequence creation offset
                    if 0 <= seq_idx < len(X):
                        available_sequences.append(seq_idx)
            
            if len(available_sequences) < 20:  # Need minimum sequences per interval
                continue
            
            # Split sequences within interval: 80% train, 20% val with buffer
            n_sequences = len(available_sequences)
            n_train = int(n_sequences * 0.8)
            
            # Randomly sample train sequences from first 80% of interval
            train_candidates = available_sequences[:n_train]
            val_candidates = available_sequences[n_train:]
            
            # Add buffer constraint: ensure train and val sequences don't overlap temporally
            filtered_train = []
            filtered_val = []
            
            for seq_idx in train_candidates:
                filtered_train.append(seq_idx)
            
            for seq_idx in val_candidates:
                # Check if this val sequence has enough buffer from train sequences
                seq_start_time = seq_idx + 24  # Actual start time in df_clean
                too_close = False
                
                for train_seq_idx in filtered_train:
                    train_start_time = train_seq_idx + 24
                    if abs(seq_start_time - train_start_time) < buffer:
                        too_close = True
                        break
                
                if not too_close:
                    filtered_val.append(seq_idx)
            
            train_indices.extend(filtered_train)
            val_indices.extend(filtered_val)
        
        # Convert to boolean masks
        train_mask = np.zeros(len(X), dtype=bool)
        val_mask = np.zeros(len(X), dtype=bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        
        print(f"Stratified split: {len(train_indices)} train sequences, {len(val_indices)} val sequences")
        print(f"Train/Val ratio: {len(train_indices)/(len(train_indices)+len(val_indices)):.2f}/{len(val_indices)/(len(train_indices)+len(val_indices)):.2f}")
        
        # Validate temporal distribution
        if len(train_indices) > 0 and len(val_indices) > 0:
            train_times = [df_clean.index[idx + 24] for idx in train_indices[:10]]  # Sample for display
            val_times = [df_clean.index[idx + 24] for idx in val_indices[:10]]
            
            print("Sample train sequence start positions:", train_times[:5])
            print("Sample val sequence start positions:", val_times[:5])
            
            # Check temporal spread
            train_time_spread = max(train_indices) - min(train_indices)
            val_time_spread = max(val_indices) - min(val_indices)
            print(f"Train temporal spread: {train_time_spread} hours, Val temporal spread: {val_time_spread} hours")
        
        if stage == 'fit' or stage is None:
            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]
            
            print(f"Stratified temporal split complete:")
            print(f"  • Train sequences: {len(X_train)} (covers {len(X_train)} sequences across multiple time intervals)")
            print(f"  • Val sequences: {len(X_val)} (covers {len(X_val)} sequences across multiple time intervals)")
            print(f"  • This ensures both train and val see all temporal patterns for better generalization")
            
            # Use StandardScaler for both - simpler and more stable
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            # Reshape for scaling
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_train_scaled = self.scaler_X.fit_transform(X_train_reshaped)
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_val_scaled = self.scaler_X.transform(X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            
            # Simple scaling without log transformation
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))
            y_train_scaled = y_train_scaled.reshape(y_train.shape)
            
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1))
            y_val_scaled = y_val_scaled.reshape(y_val.shape)
            
            # Convert to tensors
            self.train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train_scaled),
                torch.FloatTensor(y_train_scaled)
            )
            
            self.val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val_scaled),
                torch.FloatTensor(y_val_scaled)
            )
            
            print(f"Training set: {len(self.train_dataset)}")
            print(f"Validation set: {len(self.val_dataset)}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for forecasting."""
        
        df = df.copy()
        
        # Temporal features
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        df['day_of_month'] = df['DateTime'].dt.day
        df['month'] = df['DateTime'].dt.month
        df['quarter'] = df['DateTime'].dt.quarter
        df['week_of_year'] = df['DateTime'].dt.isocalendar().week
        
        # Cyclical encoding
        for feature, period in [('hour', 24), ('day_of_week', 7), ('day_of_month', 31), 
                               ('month', 12), ('week_of_year', 52)]:
            df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / period)
            df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / period)
        
        # Business indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_peak_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Lag features
        target_col = 'avg_logged_events_in_interval'
        for lag in [3, 6, 12, 24, 72]:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [6, 12, 24, 72]:
            df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_std_{window}'] = df[target_col].rolling(window).std()
            df[f'{target_col}_min_{window}'] = df[target_col].rolling(window).min()
            df[f'{target_col}_max_{window}'] = df[target_col].rolling(window).max()
        
        # Exponential weighted features
        for span in [6, 12, 24]:
            df[f'{target_col}_ewm_{span}'] = df[target_col].ewm(span=span).mean()
        
        # Difference features
        for diff in [1, 6, 12]:
            df[f'{target_col}_diff_{diff}'] = df[target_col].diff(diff)
            df[f'{target_col}_pct_change_{diff}'] = df[target_col].pct_change(diff)
        
        # Interaction features
        df['queue_pressure'] = df['avg_unprocessed_events_count'] / (df['avg_processed_events_in_interval'] + 1)
        df['processing_efficiency'] = df['avg_processed_events_in_interval'] / (df['avg_average_processing_duration_ms'] + 1)
        df['input_output_ratio'] = df['avg_logged_events_in_interval'] / (df['avg_processed_events_in_interval'] + 1)
        df['queue_utilization'] = df['avg_queued_events_in_interval'] / (df['avg_logged_events_in_interval'] + 1)
        
        # System load indicators
        df['system_load'] = df['avg_unprocessed_events_count'] * df['avg_average_processing_duration_ms']
        df['throughput_capacity'] = df['avg_processed_events_in_interval'] / df['avg_average_processing_duration_ms']
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        
        features = df[feature_cols].values
        target = df['avg_logged_events_in_interval'].values
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(df) - self.max_horizon + 1):
            X.append(features[i-self.sequence_length:i])
            y.append(target[i:i+self.max_horizon])
        
        return np.array(X), np.array(y)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,  # Use more workers for GPU
            pin_memory=True if torch.cuda.is_available() else False,  # Pin memory for GPU
            persistent_workers=True if torch.cuda.is_available() else False
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,  # Use more workers for GPU
            pin_memory=True if torch.cuda.is_available() else False,  # Pin memory for GPU
            persistent_workers=True if torch.cuda.is_available() else False
        )


def train_forecasting_model(
    data_file: str = "EventsMetricsMarJul.csv",
    max_epochs: int = 50,
    batch_size: int = 32,
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    learning_rate: float = 1e-4,
    sequence_length: int = 24,
    forecast_horizons: List[int] = [3, 6, 12, 24, 48]
):
    """Train the forecasting model."""
    
    console.print("🚀 [bold green]Starting ScaleIQ Forecasting Model Training[/bold green]")
    console.print("=" * 60)
    
    # Data module
    data_module = ForecastingDataModule(
        data_file=data_file,
        sequence_length=sequence_length,
        forecast_horizons=forecast_horizons,
        batch_size=batch_size,
        test_size=0.2,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Setup data to get input dimensions
    data_module.setup('fit')
    input_dim = len(data_module.feature_names)
    
    console.print(f"📊 [bold blue]Model Configuration:[/bold blue]")
    console.print(f"   • Input dimension: [cyan]{input_dim}[/cyan]")
    console.print(f"   • Sequence length: [cyan]{sequence_length}[/cyan]")
    console.print(f"   • Forecast horizons: [cyan]{forecast_horizons}[/cyan]")
    console.print(f"   • Transformer dim: [cyan]{d_model}[/cyan]")
    console.print(f"   • Attention heads: [cyan]{nhead}[/cyan]")
    console.print(f"   • Encoder layers: [cyan]{num_encoder_layers}[/cyan]")
    
    # Model
    # Check if we should load from existing checkpoint
    checkpoint_dir = 'checkpoints'
    resume_from_checkpoint = None
    model = None
    
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoint_files:
            # Find the best checkpoint (lowest val_total_loss)
            best_checkpoint = None
            best_loss = float('inf')
            
            for ckpt_file in checkpoint_files:
                if 'val_total_loss=' in ckpt_file:
                    loss_str = ckpt_file.split('val_total_loss=')[1].split('.ckpt')[0]
                    try:
                        loss_val = float(loss_str)
                        if loss_val < best_loss:
                            best_loss = loss_val
                            best_checkpoint = os.path.join(checkpoint_dir, ckpt_file)
                    except ValueError:
                        continue
            
            if best_checkpoint:
                console.print(f"🔄 [bold yellow]Loading model from checkpoint:[/bold yellow] {best_checkpoint}")
                console.print(f"   [cyan]Previous best validation loss:[/cyan] {best_loss:.3f}")
                
                # Load the model directly from checkpoint
                model = EventForecastingTransformer.load_from_checkpoint(
                    best_checkpoint,
                    input_dim=input_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_encoder_layers,
                    sequence_length=sequence_length,
                    forecast_horizons=forecast_horizons,
                    learning_rate=learning_rate
                )
                resume_from_checkpoint = best_checkpoint
    
    # Create new model if no checkpoint found
    if model is None:
        console.print("🆕 [bold green]Creating new model[/bold green]")
        model = EventForecastingTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            sequence_length=sequence_length,
            forecast_horizons=forecast_horizons,
            learning_rate=learning_rate
        )
    
    # Enhanced callbacks for stable training
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        ModelCheckpoint(
            monitor='val_total_loss',
            mode='min',
            save_top_k=5,  # Save more checkpoints
            filename='forecaster-{epoch:02d}-{val_total_loss:.3f}',
            dirpath='checkpoints',
            save_last=True  # Always save the last checkpoint
        ),
        EarlyStopping(
            monitor='val_total_loss',
            patience=40,  # Much more patience for deeper convergence
            mode='min',
            min_delta=0.001  # Smaller threshold for finer convergence detection
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Logger
    logger = TensorBoardLogger('lightning_logs', name='forecasting_model')
    
    # If resuming from checkpoint, try to use the same logger version
    if resume_from_checkpoint:
        # Extract epoch from checkpoint filename to find corresponding logger version
        try:
            checkpoint_epoch = int(best_checkpoint.split('epoch=')[1].split('-')[0])
            logger_version = None
            
            # Try to find the logger version that corresponds to this checkpoint
            log_dir = 'lightning_logs/forecasting_model'
            if os.path.exists(log_dir):
                versions = [d for d in os.listdir(log_dir) if d.startswith('version_')]
                if versions:
                    # Use the latest version for resumed training
                    latest_version = max(versions, key=lambda x: int(x.split('_')[1]))
                    logger = TensorBoardLogger('lightning_logs', name='forecasting_model', version=int(latest_version.split('_')[1]))
                    console.print(f"   [cyan]Resuming with logger version:[/cyan] {latest_version}")
        except (ValueError, IndexError):
            pass
    
    # Trainer with stabilized settings for robust training
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='32',  # Use 32-bit precision for stability
        gradient_clip_val=0.5,  # More aggressive gradient clipping
        gradient_clip_algorithm='norm',
        accumulate_grad_batches=1,  # Remove gradient accumulation for stability
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=50,
        # Stability optimizations
        sync_batchnorm=False,
        deterministic=True,     # Enable deterministic training
        benchmark=False,        # Disable for consistent behavior
    )
    
    console.print(f"🎯 [bold yellow]Training on: {str(trainer.accelerator.__class__.__name__).replace('Accelerator', '').upper()}[/bold yellow]")
    if torch.cuda.is_available():
        console.print(f"   [green]GPU:[/green] {torch.cuda.get_device_name(0)}")
        console.print(f"   [green]GPU Memory:[/green] {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        console.print(f"   [green]Mixed Precision:[/green] {'16-mixed' if trainer.precision == '16-mixed' else 'No'}")
        console.print(f"   [green]Batch Size:[/green] {batch_size} (optimized for GPU)")
    
    console.print(f"\n⏱️  [bold magenta]Estimated training time:[/bold magenta] ~{max_epochs * 2:.0f}-{max_epochs * 4:.0f} minutes")
    console.print(f"📈 [bold cyan]Progress will be shown with rich progress bars below[/bold cyan]\n")
    
    # Train
    trainer.fit(model, data_module, ckpt_path=resume_from_checkpoint)
    
    # Load best model
    best_model = EventForecastingTransformer.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    
    print(f"✅ Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    
    return best_model, data_module, trainer


def evaluate_and_visualize(model, data_module, save_plots: bool = True):
    """Evaluate model and create visualizations."""
    
    print("\n📊 Evaluating Model Performance...")
    
    # Get validation data
    val_loader = data_module.val_dataloader()
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Ensure model is on correct device
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            # Move data to same device as model
            x = x.to(device)
            y = y.to(device)
            
            predictions = model(x)
            
            all_predictions.append(predictions['main'].cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"Evaluation data shape: predictions={predictions.shape}, targets={targets.shape}")
    
    # Inverse transform predictions and targets - simplified
    max_horizon = model.max_horizon
    
    # Reshape for inverse transform
    pred_reshaped = predictions.reshape(-1, 1)
    target_reshaped = targets[:, :max_horizon].reshape(-1, 1)
    
    # Direct inverse scaling
    pred_original = data_module.scaler_y.inverse_transform(pred_reshaped)
    target_original = data_module.scaler_y.inverse_transform(target_reshaped)
    
    pred_original = pred_original.reshape(predictions.shape)
    target_original = target_original.reshape(targets[:, :max_horizon].shape)
    
    # Calculate metrics for each horizon
    horizons = [3, 6, 12, 24, 48]  # 6min, 12min, 24min, 48min, 96min
    horizon_names = ['6min', '12min', '24min', '48min', '96min']
    
    print(f"\n📈 Performance Metrics:")
    
    metrics_results = {}
    
    for i, (horizon, name) in enumerate(zip(horizons, horizon_names)):
        if horizon <= max_horizon:
            # Single step prediction at horizon
            pred_h = pred_original[:, horizon-1]
            target_h = target_original[:, horizon-1]
            
            mae = mean_absolute_error(target_h, pred_h)
            rmse = np.sqrt(mean_squared_error(target_h, pred_h))
            mape = np.mean(np.abs((target_h - pred_h) / target_h)) * 100
            
            # R2 score
            ss_res = np.sum((target_h - pred_h) ** 2)
            ss_tot = np.sum((target_h - np.mean(target_h)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            metrics_results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            
            print(f"   {name} ahead forecast:")
            print(f"     • MAE:  {mae:.2f}")
            print(f"     • RMSE: {rmse:.2f}")
            print(f"     • MAPE: {mape:.2f}%")
            print(f"     • R²:   {r2:.4f}")
    
    if save_plots:
        create_forecast_visualizations(pred_original, target_original, metrics_results, horizons, horizon_names)
    
    return metrics_results


def create_forecast_visualizations(predictions, targets, metrics, horizons, horizon_names):
    """Create comprehensive forecast visualizations."""
    
    print("\n📊 Creating Visualizations...")
    
    # Create figure with subplots - adjusted for 5 horizons
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle('ScaleIQ Event Traffic Forecasting Results - Multi-Scale Horizons', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series comparison (last 500 points) - First 3 horizons
    n_plot = min(500, len(predictions))
    
    for i, (horizon, name) in enumerate(zip(horizons[:3], horizon_names[:3])):
        if horizon <= predictions.shape[1]:
            ax = axes[i, 0]
            
            pred_h = predictions[-n_plot:, horizon-1]
            target_h = targets[-n_plot:, horizon-1]
            
            ax.plot(range(n_plot), target_h, label='Actual', alpha=0.8, linewidth=2, color='darkblue')
            ax.plot(range(n_plot), pred_h, label='Predicted', alpha=0.8, linewidth=2, color='red')
            
            ax.set_title(f'{name} Ahead Forecast', fontsize=12, fontweight='bold')
            ax.set_ylabel('Events per Interval')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics text
            mae = metrics[name]['MAE']
            rmse = metrics[name]['RMSE']
            mape = metrics[name]['MAPE']
            r2 = metrics[name]['R2']
            
            textstr = f'MAE: {mae:.1f}\nRMSE: {rmse:.1f}\nMAPE: {mape:.1f}%\nR²: {r2:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    # Plot 2: Scatter plots (predicted vs actual) - First 3 horizons
    for i, (horizon, name) in enumerate(zip(horizons[:3], horizon_names[:3])):
        if horizon <= predictions.shape[1]:
            ax = axes[i, 1]
            
            pred_h = predictions[:, horizon-1]
            target_h = targets[:, horizon-1]
            
            # Sample for visualization if too many points
            if len(pred_h) > 5000:
                indices = np.random.choice(len(pred_h), 5000, replace=False)
                pred_h = pred_h[indices]
                target_h = target_h[indices]
            
            ax.scatter(target_h, pred_h, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(target_h.min(), pred_h.min())
            max_val = max(target_h.max(), pred_h.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Actual Events')
            ax.set_ylabel('Predicted Events')
            ax.set_title(f'{name} Ahead: Predicted vs Actual', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = np.corrcoef(target_h, pred_h)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('forecasting_results.png', dpi=300, bbox_inches='tight')
    print("   ✅ Main results saved as 'forecasting_results.png'")
    
    # Create separate figure for longer horizons and error analysis
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
    fig2.suptitle('Extended Horizon Analysis & Error Distribution', fontsize=16, fontweight='bold')
    
    # Plot longer horizons (48min and 96min)
    for i, (horizon, name) in enumerate(zip(horizons[3:], horizon_names[3:])):
        if horizon <= predictions.shape[1]:
            ax = axes2[0, i]
            
            pred_h = predictions[-n_plot:, horizon-1]
            target_h = targets[-n_plot:, horizon-1]
            
            ax.plot(range(n_plot), target_h, label='Actual', alpha=0.8, linewidth=2, color='darkblue')
            ax.plot(range(n_plot), pred_h, label='Predicted', alpha=0.8, linewidth=2, color='red')
            
            ax.set_title(f'{name} Ahead Forecast', fontsize=12, fontweight='bold')
            ax.set_ylabel('Events per Interval')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics text
            mae = metrics[name]['MAE']
            rmse = metrics[name]['RMSE']
            mape = metrics[name]['MAPE']
            r2 = metrics[name]['R2']
            
            textstr = f'MAE: {mae:.1f}\nRMSE: {rmse:.1f}\nMAPE: {mape:.1f}%\nR²: {r2:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    # Overall metrics comparison on remaining subplot
    ax = axes2[0, 2]
    metric_names = ['MAE', 'RMSE', 'MAPE', 'R2']
    metric_colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    x_pos = np.arange(len(horizon_names))
    
    for i, metric in enumerate(metric_names):
        if metric == 'R2':
            values = [metrics[name][metric] for name in horizon_names]
        else:
            # Normalize other metrics for comparison
            all_values = [metrics[name][metric] for name in horizon_names]
            max_val = max(all_values)
            values = [v/max_val for v in all_values]
        
        ax.bar(x_pos + i*0.15, values, width=0.15, label=metric, 
               color=metric_colors[i], alpha=0.8)
    
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('Normalized Metric Value')
    ax.set_title('Performance Metrics Comparison Across All Horizons')
    ax.set_xticks(x_pos + 0.225)
    ax.set_xticklabels(horizon_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error analysis for key horizons on bottom row
    for i, (horizon, name) in enumerate(zip([3, 12, 48], ['6min', '24min', '96min'])):
        if horizon <= predictions.shape[1] and i < 3:
            ax = axes2[1, i]
            
            pred_h = predictions[:, horizon-1]
            target_h = targets[:, horizon-1]
            errors = target_h - pred_h
            
            # Error histogram
            ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_title(f'{name} Prediction Errors Distribution')
            ax.set_xlabel('Error (Actual - Predicted)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax.text(0.02, 0.98, f'Mean: {mean_error:.1f}\nStd: {std_error:.1f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('forecasting_error_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Error analysis saved as 'forecasting_error_analysis.png'")
    
    plt.show()


if __name__ == "__main__":
    # Stabilized training configuration for robust convergence
    config = {
        'data_file': 'EventsMetricsMarJul.csv',
        'max_epochs': 100,     # Keep extended training
        'batch_size': 32,      # Stable batch size
        'd_model': 512,        # Keep high capacity
        'nhead': 16,          # Keep rich attention
        'num_encoder_layers': 6,  # Deep architecture
        'learning_rate': 1e-4,    # REDUCED from 5e-4 for stability
        'sequence_length': 24,    # Proven optimal length
        'forecast_horizons': [3, 6, 12, 24, 48]  # Multi-scale approach
    }
    
    try:
        # Train model
        model, data_module, trainer = train_forecasting_model(**config)
        
        # Evaluate and visualize
        metrics = evaluate_and_visualize(model, data_module)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Files generated:")
        print(f"   • Model checkpoint: checkpoints/")
        print(f"   • TensorBoard logs: lightning_logs/")
        print(f"   • Forecasting results: forecasting_results.png")
        print(f"   • Error analysis: forecasting_error_analysis.png")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
