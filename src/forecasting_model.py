"""
State-of-the-art event traffic forecasting model using Transformer architecture.
Combines temporal attention, seasonal patterns, and external features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import math
import warnings
warnings.filterwarnings('ignore')


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
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(kernel_sizes), 
                     kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(F.relu(conv(x)))
        
        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)
        x = self.bn(x)
        x = self.dropout(x)
        
        # Back to (batch, seq_len, features)
        return x.transpose(1, 2)


class EventTrafficForecaster(pl.LightningModule):
    """
    State-of-the-art event traffic forecasting model.
    
    Architecture combines:
    - Multi-scale CNN for local pattern extraction
    - Transformer encoder for long-range dependencies
    - Seasonal attention mechanism
    - Multi-head prediction for different horizons
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 sequence_length: int = 24,
                 forecast_horizons: List[int] = [6, 12, 24],
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 warmup_steps: int = 1000,
                 max_steps: int = 10000,
                 seasonal_periods: List[int] = [720, 10080],  # Daily, Weekly
                 **kwargs):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.forecast_horizons = forecast_horizons
        self.seasonal_periods = seasonal_periods
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Multi-scale convolutional feature extraction
        self.conv_extractor = MultiScaleConv1D(d_model, d_model)
        
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
        
        # Seasonal attention layers
        self.seasonal_attention = nn.ModuleDict({
            f'period_{period}': nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            ) for period in seasonal_periods
        })
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * (1 + len(seasonal_periods)), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        # Multi-horizon prediction heads
        self.prediction_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, h)
            ) for h in forecast_horizons
        })
        
        # Global prediction head (for main forecasting)
        self.global_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, max(forecast_horizons))
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(delta=1.0)
        
        # Metrics storage
        self.train_metrics = []
        self.val_metrics = []
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Dictionary with predictions for different horizons
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Multi-scale convolution
        conv_features = self.conv_extractor(x)
        
        # Add positional encoding
        x = self.pos_encoder(conv_features.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoding
        transformer_output = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        global_features = transformer_output.mean(dim=1)  # (batch, d_model)
        
        # Seasonal attention
        seasonal_features = []
        for period_name, attention_layer in self.seasonal_attention.items():
            # Use last few timesteps for seasonal patterns
            seasonal_input = transformer_output[:, -min(seq_len, 12):, :]
            seasonal_out, _ = attention_layer(
                seasonal_input, seasonal_input, seasonal_input
            )
            seasonal_features.append(seasonal_out.mean(dim=1))
        
        # Feature fusion
        if seasonal_features:
            all_features = torch.cat([global_features] + seasonal_features, dim=-1)
            fused_features = self.feature_fusion(all_features)
        else:
            fused_features = global_features
        
        # Multi-horizon predictions
        predictions = {}
        for horizon_name, pred_head in self.prediction_heads.items():
            predictions[horizon_name] = pred_head(fused_features)
        
        # Global prediction
        predictions['global'] = self.global_head(fused_features)
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-objective loss."""
        
        total_loss = 0.0
        loss_components = {}
        
        # Multi-horizon losses
        for horizon_name in self.prediction_heads.keys():
            if horizon_name in targets:
                pred = predictions[horizon_name]
                target = targets[horizon_name]
                
                # Combine different loss functions
                mse_loss = self.mse_loss(pred, target)
                mae_loss = self.mae_loss(pred, target)
                huber_loss = self.huber_loss(pred, target)
                
                horizon_loss = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * huber_loss
                total_loss += horizon_loss
                
                loss_components[f'{horizon_name}_loss'] = horizon_loss
        
        # Global loss
        if 'global' in targets:
            pred = predictions['global']
            target = targets['global']
            
            global_mse = self.mse_loss(pred, target)
            global_mae = self.mae_loss(pred, target)
            
            global_loss = 0.7 * global_mse + 0.3 * global_mae
            total_loss += global_loss
            
            loss_components['global_loss'] = global_loss
        
        loss_components['total_loss'] = total_loss
        return total_loss, loss_components
    
    def training_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], 
                     batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, targets = batch
        predictions = self(x)
        
        loss, loss_components = self.compute_loss(predictions, targets)
        
        # Log metrics
        for key, value in loss_components.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], 
                       batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, targets = batch
        predictions = self(x)
        
        loss, loss_components = self.compute_loss(predictions, targets)
        
        # Log metrics
        for key, value in loss_components.items():
            self.log(f'val_{key}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate additional metrics
        if 'global' in targets and 'global' in predictions:
            pred = predictions['global']
            target = targets['global']
            
            # MAPE
            mape = torch.mean(torch.abs((target - pred) / (target + 1e-8))) * 100
            self.log('val_mape', mape, on_epoch=True, prog_bar=True)
            
            # R2 Score approximation
            ss_res = torch.sum((target - pred) ** 2)
            ss_tot = torch.sum((target - target.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log('val_r2', r2, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.warmup_steps,
            T_mult=2,
            eta_min=self.hparams.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_total_loss',
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step."""
        return self(batch)


class EventDataModule(pl.LightningDataModule):
    """Data module for event forecasting."""
    
    def __init__(self,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str = 'avg_logged_events_in_interval',
                 sequence_length: int = 24,
                 forecast_horizons: List[int] = [6, 12, 24],
                 batch_size: int = 32,
                 num_workers: int = 4):
        
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.forecast_horizons = forecast_horizons
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def create_sequences(self, df: pd.DataFrame) -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Create sequences for training."""
        features = df[self.feature_cols].values
        targets = df[self.target_col].values
        
        sequences = []
        max_horizon = max(self.forecast_horizons)
        
        for i in range(self.sequence_length, len(df) - max_horizon + 1):
            # Input sequence
            x = torch.FloatTensor(features[i-self.sequence_length:i])
            
            # Multi-horizon targets
            target_dict = {}
            for horizon in self.forecast_horizons:
                target_dict[f'horizon_{horizon}'] = torch.FloatTensor(
                    targets[i:i+horizon]
                )
            
            # Global target (longest horizon)
            target_dict['global'] = torch.FloatTensor(targets[i:i+max_horizon])
            
            sequences.append((x, target_dict))
        
        return sequences
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = self.create_sequences(self.train_df)
            self.val_dataset = self.create_sequences(self.val_df)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def train_forecasting_model(train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          feature_cols: List[str],
                          target_col: str = 'avg_logged_events_in_interval',
                          max_epochs: int = 100,
                          gpus: int = 1 if torch.cuda.is_available() else 0,
                          **model_kwargs) -> EventTrafficForecaster:
    """Train the forecasting model."""
    
    # Data module
    data_module = EventDataModule(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col=target_col
    )
    
    # Model
    model = EventTrafficForecaster(
        input_dim=len(feature_cols),
        **model_kwargs
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_total_loss',
            mode='min',
            save_top_k=3,
            filename='forecaster-{epoch:02d}-{val_total_loss:.3f}'
        ),
        EarlyStopping(
            monitor='val_total_loss',
            patience=10,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Logger
    logger = TensorBoardLogger('lightning_logs', name='event_forecaster')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        precision=16 if gpus > 0 else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    trainer.fit(model, data_module)
    
    return model, trainer


if __name__ == "__main__":
    # Example usage will be in the main training script
    print("Event Traffic Forecasting Model - State of the Art Architecture")
    print("Features:")
    print("- Multi-scale CNN for local patterns")
    print("- Transformer encoder for long-range dependencies") 
    print("- Seasonal attention mechanism")
    print("- Multi-horizon prediction heads")
    print("- Advanced loss functions and metrics")
