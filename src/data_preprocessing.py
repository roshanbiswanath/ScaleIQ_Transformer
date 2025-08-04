"""
Data preprocessing module for event processing forecasting system.
Handles feature engineering, temporal features, and data preparation.
"""

import pandas as pd
import numpy as np
import pendulum
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventDataProcessor:
    """Advanced data processor for event metrics with feature engineering."""
    
    def __init__(self, 
                 scaler_type: str = "robust",
                 lookback_windows: List[int] = [6, 12, 24, 72],  # 12min, 24min, 48min, 144min
                 forecast_horizons: List[int] = [6, 12, 24],     # 12min, 24min, 48min ahead
                 seasonal_periods: List[int] = [720, 10080]):    # Daily, Weekly patterns
        
        self.scaler_type = scaler_type
        self.lookback_windows = lookback_windows
        self.forecast_horizons = forecast_horizons
        self.seasonal_periods = seasonal_periods
        self.scalers = {}
        self.feature_names = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and initial preprocessing of event data."""
        logger.info(f"Loading data from {file_path}")
        
        df = pd.read_csv(file_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        # Basic data quality checks
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive temporal features."""
        logger.info("Creating temporal features")
        
        df = df.copy()
        dt = df['DateTime']
        
        # Basic time features
        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        df['day_of_month'] = dt.dt.day
        df['month'] = dt.dt.month
        df['quarter'] = dt.dt.quarter
        df['week_of_year'] = dt.dt.isocalendar().week
        
        # Cyclical encoding for temporal features
        for feature, period in [('hour', 24), ('day_of_week', 7), ('day_of_month', 31), 
                              ('month', 12), ('week_of_year', 52)]:
            df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / period)
            df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / period)
        
        # Business time indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_peak_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Time since start (for trend capture)
        df['time_idx'] = np.arange(len(df))
        df['days_since_start'] = (dt - dt.min()).dt.days
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                          target_cols: List[str]) -> pd.DataFrame:
        """Create lagged features for time series."""
        logger.info("Creating lag features")
        
        df = df.copy()
        
        for col in target_cols:
            for window in self.lookback_windows:
                # Lag features
                df[f'{col}_lag_{window}'] = df[col].shift(window)
                
                # Rolling statistics
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                
                # Exponential weighted features
                df[f'{col}_ewm_{window}'] = df[col].ewm(span=window).mean()
                
                # Difference features
                df[f'{col}_diff_{window}'] = df[col].diff(window)
                df[f'{col}_pct_change_{window}'] = df[col].pct_change(window)
        
        return df
    
    def create_seasonal_features(self, df: pd.DataFrame, 
                               target_cols: List[str]) -> pd.DataFrame:
        """Create seasonal decomposition features."""
        logger.info("Creating seasonal features")
        
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        df = df.copy()
        
        for col in target_cols:
            for period in self.seasonal_periods:
                if len(df) >= 2 * period:
                    # Seasonal decomposition
                    try:
                        decomposition = seasonal_decompose(
                            df[col].dropna(), 
                            model='additive', 
                            period=period,
                            extrapolate_trend='freq'
                        )
                        
                        df[f'{col}_trend_{period}'] = decomposition.trend
                        df[f'{col}_seasonal_{period}'] = decomposition.seasonal
                        df[f'{col}_residual_{period}'] = decomposition.resid
                        
                    except Exception as e:
                        logger.warning(f"Seasonal decomposition failed for {col} with period {period}: {e}")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific interaction features."""
        logger.info("Creating interaction features")
        
        df = df.copy()
        
        # Queue pressure indicators
        df['queue_pressure'] = df['avg_unprocessed_events_count'] / (df['avg_processed_events_in_interval'] + 1)
        df['processing_efficiency'] = df['avg_processed_events_in_interval'] / (df['avg_average_processing_duration_ms'] + 1)
        df['input_output_ratio'] = df['avg_logged_events_in_interval'] / (df['avg_processed_events_in_interval'] + 1)
        df['queue_utilization'] = df['avg_queued_events_in_interval'] / (df['avg_logged_events_in_interval'] + 1)
        
        # Load indicators
        df['system_load'] = df['avg_unprocessed_events_count'] * df['avg_average_processing_duration_ms']
        df['throughput_capacity'] = df['avg_processed_events_in_interval'] / df['avg_average_processing_duration_ms']
        
        # Volatility measures
        for window in [6, 12, 24]:
            df[f'input_volatility_{window}'] = df['avg_logged_events_in_interval'].rolling(window).std()
            df[f'processing_volatility_{window}'] = df['avg_average_processing_duration_ms'].rolling(window).std()
        
        return df
    
    def create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create anomaly detection features using statistical methods."""
        logger.info("Creating anomaly features")
        
        df = df.copy()
        
        # Z-score based anomalies
        for col in ['avg_logged_events_in_interval', 'avg_processed_events_in_interval', 
                   'avg_unprocessed_events_count', 'avg_average_processing_duration_ms']:
            
            rolling_mean = df[col].rolling(window=72).mean()  # 144 minutes
            rolling_std = df[col].rolling(window=72).std()
            
            df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            df[f'{col}_is_anomaly'] = (np.abs(df[f'{col}_zscore']) > 2.5).astype(int)
        
        return df
    
    def prepare_forecasting_data(self, df: pd.DataFrame, 
                               target_col: str = 'avg_logged_events_in_interval',
                               test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data specifically for forecasting models."""
        logger.info(f"Preparing forecasting data for target: {target_col}")
        
        # Create all features
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df, [target_col, 'avg_processed_events_in_interval', 
                                         'avg_unprocessed_events_count'])
        df = self.create_seasonal_features(df, [target_col])
        df = self.create_interaction_features(df)
        df = self.create_anomaly_features(df)
        
        # Remove initial NaN values due to lag features
        max_lag = max(self.lookback_windows)
        df = df.iloc[max_lag:].reset_index(drop=True)
        
        # Time-based split
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        return train_df, test_df
    
    def prepare_scaling_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           target_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features for training."""
        logger.info("Scaling features")
        
        # Select numeric columns for scaling
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['time_idx'] + target_cols]
        
        if self.scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # Fit on training data
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()
        
        train_scaled[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        test_scaled[numeric_cols] = scaler.transform(test_df[numeric_cols])
        
        self.scalers['features'] = scaler
        self.feature_names = numeric_cols
        
        # Scale targets separately
        for target_col in target_cols:
            target_scaler = StandardScaler()
            train_scaled[f'{target_col}_scaled'] = target_scaler.fit_transform(
                train_df[[target_col]]
            ).flatten()
            test_scaled[f'{target_col}_scaled'] = target_scaler.transform(
                test_df[[target_col]]
            ).flatten()
            self.scalers[target_col] = target_scaler
        
        return train_scaled, test_scaled
    
    def create_sequences(self, df: pd.DataFrame, 
                        feature_cols: List[str],
                        target_col: str,
                        sequence_length: int = 24,
                        forecast_horizon: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for deep learning models."""
        
        features = df[feature_cols].values
        targets = df[target_col].values
        
        X, y = [], []
        
        for i in range(sequence_length, len(df) - forecast_horizon + 1):
            X.append(features[i-sequence_length:i])
            y.append(targets[i:i+forecast_horizon])
        
        return np.array(X), np.array(y)


def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main function to load and preprocess data."""
    processor = EventDataProcessor()
    
    # Load data
    df = processor.load_data(file_path)
    
    # Prepare for forecasting
    train_df, test_df = processor.prepare_forecasting_data(df)
    
    # Scale data
    train_scaled, test_scaled = processor.prepare_scaling_data(
        train_df, test_df, 
        target_cols=['avg_logged_events_in_interval', 'avg_unprocessed_events_count']
    )
    
    return train_scaled, test_scaled, processor


if __name__ == "__main__":
    # Example usage
    file_path = "EventsMetricsMarJul.csv"
    train_df, test_df, processor = load_and_preprocess_data(file_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Features created: {len(processor.feature_names)}")
