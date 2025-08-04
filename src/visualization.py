"""
Comprehensive visualization and model comparison script.
Enables comparison of different forecasting and scaling models without retraining.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Any, Union
import os
import json
from datetime import datetime, timedelta
import pickle


class ModelPerformanceAnalyzer:
    """Comprehensive model performance analysis and visualization."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Color schemes for consistent plotting
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff6b6b',
            'info': '#17a2b8',
            'models': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        }
        
        # Metrics storage
        self.forecasting_results = {}
        self.scaling_results = {}
        self.model_configs = {}
    
    def save_forecasting_results(self, 
                                model_name: str,
                                predictions: np.ndarray,
                                actuals: np.ndarray,
                                timestamps: pd.DatetimeIndex,
                                config: Dict[str, Any],
                                training_metrics: Dict[str, List] = None):
        """Save forecasting model results."""
        
        results = {
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': timestamps,
            'config': config,
            'training_metrics': training_metrics or {},
            'saved_at': datetime.now()
        }
        
        self.forecasting_results[model_name] = results
        
        # Save to disk
        with open(f"{self.results_dir}/{model_name}_forecasting.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Saved forecasting results for {model_name}")
    
    def save_scaling_results(self,
                           model_name: str,
                           actions: List[int],
                           rewards: List[float],
                           states: List[np.ndarray],
                           job_counts: List[int],
                           queue_sizes: List[int],
                           config: Dict[str, Any]):
        """Save scaling model results."""
        
        results = {
            'actions': actions,
            'rewards': rewards,
            'states': states,
            'job_counts': job_counts,
            'queue_sizes': queue_sizes,
            'config': config,
            'saved_at': datetime.now()
        }
        
        self.scaling_results[model_name] = results
        
        # Save to disk
        with open(f"{self.results_dir}/{model_name}_scaling.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Saved scaling results for {model_name}")
    
    def load_results(self, model_name: str, model_type: str = 'forecasting'):
        """Load saved model results."""
        file_path = f"{self.results_dir}/{model_name}_{model_type}.pkl"
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
            
            if model_type == 'forecasting':
                self.forecasting_results[model_name] = results
            else:
                self.scaling_results[model_name] = results
            
            print(f"Loaded {model_type} results for {model_name}")
        else:
            print(f"Results file not found: {file_path}")
    
    def calculate_forecasting_metrics(self, 
                                    predictions: np.ndarray, 
                                    actuals: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive forecasting metrics."""
        
        # Basic metrics
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        
        # Percentage metrics
        mape = np.mean(np.abs((actuals - predictions) / np.maximum(actuals, 1))) * 100
        smape = np.mean(2 * np.abs(predictions - actuals) / 
                       (np.abs(predictions) + np.abs(actuals))) * 100
        
        # Statistical metrics
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Directional accuracy
        actual_direction = np.diff(actuals) > 0
        pred_direction = np.diff(predictions) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Peak detection metrics
        actual_peaks = self.detect_peaks(actuals)
        pred_peaks = self.detect_peaks(predictions)
        peak_precision = len(np.intersect1d(actual_peaks, pred_peaks)) / max(len(pred_peaks), 1)
        peak_recall = len(np.intersect1d(actual_peaks, pred_peaks)) / max(len(actual_peaks), 1)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape,
            'R2': r2,
            'Direction_Accuracy': direction_accuracy,
            'Peak_Precision': peak_precision,
            'Peak_Recall': peak_recall
        }
    
    def detect_peaks(self, data: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Detect peaks in time series data."""
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(data, height=np.percentile(data, threshold * 100))
        return peaks
    
    def plot_forecasting_comparison(self, 
                                  model_names: List[str] = None,
                                  time_range: Tuple[str, str] = None,
                                  save_path: str = None) -> go.Figure:
        """Create interactive forecasting comparison plot."""
        
        if model_names is None:
            model_names = list(self.forecasting_results.keys())
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Event Traffic Forecasting Comparison', 'Prediction Errors'),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        for i, model_name in enumerate(model_names):
            if model_name not in self.forecasting_results:
                continue
                
            results = self.forecasting_results[model_name]
            timestamps = results['timestamps']
            predictions = results['predictions']
            actuals = results['actuals']
            
            # Filter by time range if specified
            if time_range:
                start_date, end_date = pd.to_datetime(time_range)
                mask = (timestamps >= start_date) & (timestamps <= end_date)
                timestamps = timestamps[mask]
                predictions = predictions[mask]
                actuals = actuals[mask]
            
            color = self.colors['models'][i % len(self.colors['models'])]
            
            # Actual vs Predicted
            if i == 0:  # Only plot actuals once
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=actuals,
                        mode='lines',
                        name='Actual',
                        line=dict(color='black', width=2)
                    ),
                    row=1, col=1
                )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=predictions,
                    mode='lines',
                    name=f'{model_name} Prediction',
                    line=dict(color=color, width=1.5),
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            # Prediction errors
            errors = predictions - actuals
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=errors,
                    mode='lines',
                    name=f'{model_name} Error',
                    line=dict(color=color, width=1),
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Event Traffic Forecasting Model Comparison",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Event Count", row=1, col=1)
        fig.update_yaxes(title_text="Prediction Error", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_metrics_comparison(self, model_names: List[str] = None) -> go.Figure:
        """Create metrics comparison radar chart."""
        
        if model_names is None:
            model_names = list(self.forecasting_results.keys())
        
        # Calculate metrics for all models
        all_metrics = {}
        for model_name in model_names:
            if model_name not in self.forecasting_results:
                continue
                
            results = self.forecasting_results[model_name]
            metrics = self.calculate_forecasting_metrics(
                results['predictions'], 
                results['actuals']
            )
            all_metrics[model_name] = metrics
        
        if not all_metrics:
            print("No forecasting results found for metrics comparison")
            return None
        
        # Normalize metrics for radar plot (0-1 scale, higher is better)
        metric_names = list(next(iter(all_metrics.values())).keys())
        normalized_metrics = {}
        
        for metric in metric_names:
            values = [all_metrics[model][metric] for model in all_metrics.keys()]
            
            if metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE']:
                # Lower is better - invert and normalize
                max_val = max(values)
                normalized_values = [(max_val - v) / max_val for v in values]
            else:
                # Higher is better - normalize
                max_val = max(values) if max(values) > 0 else 1
                normalized_values = [v / max_val for v in values]
            
            for i, model in enumerate(all_metrics.keys()):
                if model not in normalized_metrics:
                    normalized_metrics[model] = {}
                normalized_metrics[model][metric] = normalized_values[i]
        
        # Create radar chart
        fig = go.Figure()
        
        for i, model_name in enumerate(normalized_metrics.keys()):
            values = list(normalized_metrics[model_name].values())
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names + [metric_names[0]],
                fill='toself',
                name=model_name,
                line_color=self.colors['models'][i % len(self.colors['models'])]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Metrics Comparison (Normalized)"
        )
        
        return fig
    
    def plot_scaling_performance(self, 
                               model_names: List[str] = None,
                               save_path: str = None) -> go.Figure:
        """Plot scaling model performance."""
        
        if model_names is None:
            model_names = list(self.scaling_results.keys())
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Cumulative Rewards', 'Job Count Over Time', 'Queue Size Over Time'),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        for i, model_name in enumerate(model_names):
            if model_name not in self.scaling_results:
                continue
                
            results = self.scaling_results[model_name]
            rewards = results['rewards']
            job_counts = results['job_counts']
            queue_sizes = results['queue_sizes']
            
            color = self.colors['models'][i % len(self.colors['models'])]
            time_steps = list(range(len(rewards)))
            
            # Cumulative rewards
            cumulative_rewards = np.cumsum(rewards)
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=cumulative_rewards,
                    mode='lines',
                    name=f'{model_name} Rewards',
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )
            
            # Job counts
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=job_counts,
                    mode='lines',
                    name=f'{model_name} Jobs',
                    line=dict(color=color, width=1.5),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Queue sizes
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=queue_sizes,
                    mode='lines',
                    name=f'{model_name} Queue',
                    line=dict(color=color, width=1.5),
                    showlegend=False
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title="Auto-Scaling Model Performance Comparison",
            height=900,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time Steps", row=3, col=1)
        fig.update_yaxes(title_text="Cumulative Reward", row=1, col=1)
        fig.update_yaxes(title_text="Job Count", row=2, col=1)
        fig.update_yaxes(title_text="Queue Size", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dashboard(self) -> go.Figure:
        """Create comprehensive dashboard with all visualizations."""
        
        # This would create a full dashboard with multiple subplots
        # For now, return the forecasting comparison
        return self.plot_forecasting_comparison()
    
    def generate_report(self, output_file: str = "model_comparison_report.html"):
        """Generate comprehensive HTML report."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ScaleIQ Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ScaleIQ Model Comparison Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Forecasting Models</h2>
                <p>Number of models evaluated: {len(self.forecasting_results)}</p>
                
                <h3>Performance Metrics</h3>
                {self._generate_metrics_table()}
            </div>
            
            <div class="section">
                <h2>Auto-Scaling Models</h2>
                <p>Number of models evaluated: {len(self.scaling_results)}</p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._generate_recommendations()}
            </div>
        </body>
        </html>
        """
        
        with open(f"{self.results_dir}/{output_file}", 'w') as f:
            f.write(html_content)
        
        print(f"Report generated: {self.results_dir}/{output_file}")
    
    def _generate_metrics_table(self) -> str:
        """Generate HTML table of metrics."""
        
        if not self.forecasting_results:
            return "<p>No forecasting results available.</p>"
        
        # Calculate metrics for all models
        all_metrics = {}
        for model_name, results in self.forecasting_results.items():
            metrics = self.calculate_forecasting_metrics(
                results['predictions'], 
                results['actuals']
            )
            all_metrics[model_name] = metrics
        
        # Create HTML table
        html = '<table class="metrics-table"><tr><th>Model</th>'
        
        # Header
        metric_names = list(next(iter(all_metrics.values())).keys())
        for metric in metric_names:
            html += f'<th>{metric}</th>'
        html += '</tr>'
        
        # Rows
        for model_name, metrics in all_metrics.items():
            html += f'<tr><td><b>{model_name}</b></td>'
            for metric in metric_names:
                value = metrics[metric]
                if isinstance(value, float):
                    html += f'<td>{value:.4f}</td>'
                else:
                    html += f'<td>{value}</td>'
            html += '</tr>'
        
        html += '</table>'
        return html
    
    def _generate_recommendations(self) -> str:
        """Generate model recommendations."""
        
        recommendations = []
        
        if self.forecasting_results:
            # Find best forecasting model
            best_model = None
            best_score = float('inf')
            
            for model_name, results in self.forecasting_results.items():
                metrics = self.calculate_forecasting_metrics(
                    results['predictions'], 
                    results['actuals']
                )
                # Use RMSE as primary metric
                if metrics['RMSE'] < best_score:
                    best_score = metrics['RMSE']
                    best_model = model_name
            
            if best_model:
                recommendations.append(
                    f"<p><strong>Best Forecasting Model:</strong> {best_model} "
                    f"(RMSE: {best_score:.4f})</p>"
                )
        
        if not recommendations:
            recommendations.append("<p>Insufficient data for recommendations.</p>")
        
        return ''.join(recommendations)


# Example usage and utility functions
def load_sample_data(file_path: str) -> pd.DataFrame:
    """Load and prepare sample data for visualization."""
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df


def simulate_model_results(analyzer: ModelPerformanceAnalyzer, 
                         data: pd.DataFrame,
                         n_models: int = 3):
    """Simulate results from multiple models for demonstration."""
    
    # Simulate forecasting results
    for i in range(n_models):
        model_name = f"Model_{i+1}"
        
        # Create synthetic predictions with different characteristics
        actuals = data['avg_logged_events_in_interval'].values[-1000:]
        
        if i == 0:  # Good model
            noise = np.random.normal(0, 50, len(actuals))
            predictions = actuals + noise
        elif i == 1:  # Biased model
            predictions = actuals * 1.1 + np.random.normal(0, 100, len(actuals))
        else:  # Poor model
            predictions = actuals + np.random.normal(0, 200, len(actuals))
        
        timestamps = data['DateTime'].iloc[-1000:].values
        
        config = {
            'model_type': 'Transformer' if i == 0 else 'LSTM' if i == 1 else 'Linear',
            'd_model': 256,
            'num_layers': 6
        }
        
        analyzer.save_forecasting_results(
            model_name=model_name,
            predictions=predictions,
            actuals=actuals,
            timestamps=pd.DatetimeIndex(timestamps),
            config=config
        )
    
    # Simulate scaling results
    for i in range(n_models):
        model_name = f"Scaling_Model_{i+1}"
        
        n_steps = 1000
        actions = np.random.randint(0, 6, n_steps)
        
        if i == 0:  # Good scaling policy
            rewards = np.random.normal(10, 5, n_steps)
        else:  # Poor scaling policy
            rewards = np.random.normal(0, 10, n_steps)
        
        states = [np.random.randn(4) for _ in range(n_steps)]
        job_counts = np.random.randint(5, 50, n_steps)
        queue_sizes = np.random.randint(0, 200, n_steps)
        
        config = {
            'algorithm': 'DQN' if i == 0 else 'DDPG' if i == 1 else 'Random',
            'network_size': 'Large',
            'learning_rate': 1e-4
        }
        
        analyzer.save_scaling_results(
            model_name=model_name,
            actions=actions.tolist(),
            rewards=rewards.tolist(),
            states=states,
            job_counts=job_counts.tolist(),
            queue_sizes=queue_sizes.tolist(),
            config=config
        )


if __name__ == "__main__":
    # Example usage
    analyzer = ModelPerformanceAnalyzer()
    
    # Load sample data
    data = load_sample_data("EventsMetricsMarJul.csv")
    
    # Simulate some model results
    simulate_model_results(analyzer, data)
    
    # Create visualizations
    forecasting_fig = analyzer.plot_forecasting_comparison()
    forecasting_fig.show()
    
    metrics_fig = analyzer.plot_metrics_comparison()
    if metrics_fig:
        metrics_fig.show()
    
    scaling_fig = analyzer.plot_scaling_performance()
    scaling_fig.show()
    
    # Generate report
    analyzer.generate_report()
    
    print("Visualization system ready!")
    print("Use the ModelPerformanceAnalyzer to compare your trained models.")
