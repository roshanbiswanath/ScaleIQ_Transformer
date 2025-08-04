"""
Main training orchestration script for ScaleIQ.
Trains both forecasting and auto-scaling models with state-of-the-art architectures.
"""

import os
import sys
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import yaml
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import EventDataProcessor, load_and_preprocess_data
from forecasting_model import EventTrafficForecaster, EventDataModule, train_forecasting_model
from autoscaling_model import AutoScalingAgent, ScalingActionSpace, train_scaling_agent
from visualization import ModelPerformanceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scaleiq_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScaleIQTrainer:
    """Main trainer class for ScaleIQ system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.data_processor = None
        self.forecasting_model = None
        self.scaling_agent = None
        self.analyzer = ModelPerformanceAnalyzer()
        
        logger.info(f"ScaleIQ Trainer initialized with device: {self.device}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        default_config = {
            'data': {
                'file_path': 'EventsMetricsMarJul.csv',
                'target_column': 'avg_logged_events_in_interval',
                'test_size': 0.2,
                'sequence_length': 24,
                'forecast_horizons': [6, 12, 24]
            },
            'forecasting': {
                'd_model': 256,
                'nhead': 8,
                'num_encoder_layers': 6,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'max_epochs': 100,
                'batch_size': 32,
                'warmup_steps': 1000
            },
            'scaling': {
                'min_jobs': 1,
                'max_jobs': 100,
                'scaling_steps': [0.5, 0.8, 1.0, 1.25, 1.5, 2.0],
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 10000,
                'buffer_size': 100000,
                'batch_size': 64,
                'num_episodes': 1000
            },
            'training': {
                'gpus': 1 if torch.cuda.is_available() else 0,
                'precision': 16 if torch.cuda.is_available() else 32,
                'save_checkpoints': True,
                'checkpoint_dir': 'checkpoints'
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Merge configurations
            def merge_configs(default: Dict, user: Dict) -> Dict:
                for key, value in user.items():
                    if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                        default[key] = merge_configs(default[key], value)
                    else:
                        default[key] = value
                return default
            
            return merge_configs(default_config, user_config)
        else:
            logger.info(f"Config file {config_path} not found, using default configuration")
            self.save_config(default_config, config_path)
            return default_config
    
    def save_config(self, config: Dict, config_path: str):
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare and preprocess data."""
        logger.info("Starting data preprocessing...")
        
        data_config = self.config['data']
        
        # Initialize data processor
        self.data_processor = EventDataProcessor(
            lookback_windows=[6, 12, 24, 72],
            forecast_horizons=data_config['forecast_horizons'],
            seasonal_periods=[720, 10080]  # Daily, Weekly patterns
        )
        
        # Load and preprocess data
        df = self.data_processor.load_data(data_config['file_path'])
        
        # Prepare for forecasting
        train_df, test_df = self.data_processor.prepare_forecasting_data(
            df, 
            target_col=data_config['target_column'],
            test_size=data_config['test_size']
        )
        
        # Scale data
        train_scaled, test_scaled = self.data_processor.prepare_scaling_data(
            train_df, test_df,
            target_cols=[data_config['target_column'], 'avg_unprocessed_events_count']
        )
        
        logger.info(f"Data preprocessing completed. Train: {train_scaled.shape}, Test: {test_scaled.shape}")
        
        return train_scaled, test_scaled
    
    def train_forecasting_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> EventTrafficForecaster:
        """Train the forecasting model."""
        logger.info("Starting forecasting model training...")
        
        forecast_config = self.config['forecasting']
        data_config = self.config['data']
        training_config = self.config['training']
        
        # Prepare feature columns
        feature_cols = self.data_processor.feature_names
        
        # Create data module
        data_module = EventDataModule(
            train_df=train_df,
            val_df=test_df,
            feature_cols=feature_cols,
            target_col=f"{data_config['target_column']}_scaled",
            sequence_length=data_config['sequence_length'],
            forecast_horizons=data_config['forecast_horizons'],
            batch_size=forecast_config['batch_size']
        )
        
        # Create model
        model = EventTrafficForecaster(
            input_dim=len(feature_cols),
            d_model=forecast_config['d_model'],
            nhead=forecast_config['nhead'],
            num_encoder_layers=forecast_config['num_encoder_layers'],
            dim_feedforward=forecast_config['dim_feedforward'],
            dropout=forecast_config['dropout'],
            sequence_length=data_config['sequence_length'],
            forecast_horizons=data_config['forecast_horizons'],
            learning_rate=forecast_config['learning_rate'],
            weight_decay=forecast_config['weight_decay'],
            warmup_steps=forecast_config['warmup_steps']
        )
        
        # Setup trainer
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor='val_total_loss',
                mode='min',
                save_top_k=3,
                filename='forecaster-{epoch:02d}-{val_total_loss:.3f}',
                dirpath=training_config['checkpoint_dir']
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_total_loss',
                patience=10,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        
        logger_tb = pl.loggers.TensorBoardLogger('lightning_logs', name='event_forecaster')
        
        trainer = pl.Trainer(
            max_epochs=forecast_config['max_epochs'],
            callbacks=callbacks,
            logger=logger_tb,
            accelerator='gpu' if training_config['gpus'] > 0 else 'cpu',
            devices=training_config['gpus'] if training_config['gpus'] > 0 else 1,
            precision=training_config['precision'],
            gradient_clip_val=1.0,
            enable_checkpointing=training_config['save_checkpoints'],
            enable_progress_bar=True
        )
        
        # Train
        trainer.fit(model, data_module)
        
        # Save model
        best_model_path = trainer.checkpoint_callback.best_model_path
        logger.info(f"Best forecasting model saved at: {best_model_path}")
        
        self.forecasting_model = model
        
        return model
    
    def evaluate_forecasting_model(self, model: EventTrafficForecaster, 
                                 test_df: pd.DataFrame) -> Dict:
        """Evaluate forecasting model and save results."""
        logger.info("Evaluating forecasting model...")
        
        data_config = self.config['data']
        feature_cols = self.data_processor.feature_names
        
        # Create sequences for evaluation
        X_test, y_test = self.data_processor.create_sequences(
            test_df,
            feature_cols=feature_cols,
            target_col=f"{data_config['target_column']}_scaled",
            sequence_length=data_config['sequence_length'],
            forecast_horizon=max(data_config['forecast_horizons'])
        )
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test)
            predictions_dict = model(X_tensor)
            predictions = predictions_dict['global'].numpy()
        
        # Inverse transform predictions and actuals
        target_scaler = self.data_processor.scalers[data_config['target_column']]
        
        predictions_original = target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        
        actuals_original = target_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        # Create timestamps for predictions
        test_start_idx = len(test_df) - len(predictions_original)
        timestamps = test_df.iloc[test_start_idx:]['DateTime'].reset_index(drop=True)
        
        # Save results
        self.analyzer.save_forecasting_results(
            model_name="TransformerForecaster",
            predictions=predictions_original,
            actuals=actuals_original,
            timestamps=pd.DatetimeIndex(timestamps),
            config=self.config['forecasting']
        )
        
        logger.info("Forecasting model evaluation completed")
        
        return {
            'predictions': predictions_original,
            'actuals': actuals_original,
            'timestamps': timestamps
        }
    
    def train_scaling_agent(self, forecasting_model: EventTrafficForecaster) -> AutoScalingAgent:
        """Train the auto-scaling agent."""
        logger.info("Starting auto-scaling agent training...")
        
        scaling_config = self.config['scaling']
        
        # Create action space
        action_space = ScalingActionSpace(
            min_jobs=scaling_config['min_jobs'],
            max_jobs=scaling_config['max_jobs'],
            scaling_steps=scaling_config['scaling_steps']
        )
        
        # State dimension (adjust based on your state representation)
        state_dim = 10  # This should match your actual state features
        
        # Train agent
        agent = train_scaling_agent(
            state_dim=state_dim,
            action_space=action_space,
            forecasting_model=forecasting_model,
            num_episodes=scaling_config['num_episodes'],
            learning_rate=scaling_config['learning_rate'],
            gamma=scaling_config['gamma'],
            epsilon_start=scaling_config['epsilon_start'],
            epsilon_end=scaling_config['epsilon_end'],
            epsilon_decay=scaling_config['epsilon_decay'],
            buffer_size=scaling_config['buffer_size'],
            batch_size=scaling_config['batch_size']
        )
        
        self.scaling_agent = agent
        
        logger.info("Auto-scaling agent training completed")
        
        return agent
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        logger.info("Generating visualizations...")
        
        # Create output directory
        viz_dir = "visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Forecasting comparison
        forecasting_fig = self.analyzer.plot_forecasting_comparison()
        if forecasting_fig:
            forecasting_fig.write_html(f"{viz_dir}/forecasting_comparison.html")
        
        # Metrics comparison
        metrics_fig = self.analyzer.plot_metrics_comparison()
        if metrics_fig:
            metrics_fig.write_html(f"{viz_dir}/metrics_comparison.html")
        
        # Scaling performance (if available)
        if self.analyzer.scaling_results:
            scaling_fig = self.analyzer.plot_scaling_performance()
            if scaling_fig:
                scaling_fig.write_html(f"{viz_dir}/scaling_performance.html")
        
        # Generate comprehensive report
        self.analyzer.generate_report("comprehensive_report.html")
        
        logger.info(f"Visualizations saved to {viz_dir}/")
    
    def run_full_training(self):
        """Run complete training pipeline."""
        logger.info("Starting ScaleIQ full training pipeline...")
        
        try:
            # 1. Data preparation
            train_df, test_df = self.prepare_data()
            
            # 2. Train forecasting model
            forecasting_model = self.train_forecasting_model(train_df, test_df)
            
            # 3. Evaluate forecasting model
            forecast_results = self.evaluate_forecasting_model(forecasting_model, test_df)
            
            # 4. Train scaling agent
            scaling_agent = self.train_scaling_agent(forecasting_model)
            
            # 5. Generate visualizations
            self.generate_visualizations()
            
            logger.info("ScaleIQ training pipeline completed successfully!")
            
            return {
                'forecasting_model': forecasting_model,
                'scaling_agent': scaling_agent,
                'forecast_results': forecast_results
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="ScaleIQ Training Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, default="EventsMetricsMarJul.csv",
                       help="Path to data file")
    parser.add_argument("--forecast-only", action="store_true",
                       help="Train only forecasting model")
    parser.add_argument("--scaling-only", action="store_true",
                       help="Train only scaling agent")
    parser.add_argument("--visualize-only", action="store_true",
                       help="Generate visualizations only")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ScaleIQTrainer(args.config)
    
    # Update data path if provided
    if args.data != "EventsMetricsMarJul.csv":
        trainer.config['data']['file_path'] = args.data
    
    try:
        if args.visualize_only:
            trainer.generate_visualizations()
        elif args.forecast_only:
            train_df, test_df = trainer.prepare_data()
            model = trainer.train_forecasting_model(train_df, test_df)
            trainer.evaluate_forecasting_model(model, test_df)
            trainer.generate_visualizations()
        elif args.scaling_only:
            # This would require loading a pre-trained forecasting model
            logger.error("Scaling-only training requires a pre-trained forecasting model")
        else:
            # Full training
            results = trainer.run_full_training()
            
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
