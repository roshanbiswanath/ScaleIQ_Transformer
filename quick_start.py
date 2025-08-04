"""
Quick start script for ScaleIQ.
Demonstrates the forecasting and visualization capabilities with sample data.
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Core ML libraries
    packages = [
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "scipy",
        "statsmodels",
        "tqdm",
        "pyyaml",
        "pendulum"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def run_demo():
    """Run a quick demo of the system."""
    print("\n🚀 Running ScaleIQ Demo...")
    
    try:
        # Import required modules
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        
        print("✅ All imports successful")
        
        # Load and analyze the data
        print("\n📊 Loading and analyzing data...")
        
        if os.path.exists("EventsMetricsMarJul.csv"):
            df = pd.read_csv("EventsMetricsMarJul.csv")
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
            print(f"Target variable stats:")
            print(df['avg_logged_events_in_interval'].describe())
            
            # Create basic visualizations
            print("\n📈 Creating visualizations...")
            
            # Time series plot
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Event counts over time
            plt.subplot(3, 1, 1)
            sample_data = df.iloc[-2000:]  # Last 2000 points for visibility
            plt.plot(sample_data['DateTime'], sample_data['avg_logged_events_in_interval'], 
                    label='Logged Events', alpha=0.7)
            plt.plot(sample_data['DateTime'], sample_data['avg_processed_events_in_interval'], 
                    label='Processed Events', alpha=0.7)
            plt.title('Event Traffic Over Time (Last 2000 Data Points)')
            plt.legend()
            plt.xticks(rotation=45)
            
            # Plot 2: Queue status
            plt.subplot(3, 1, 2)
            plt.plot(sample_data['DateTime'], sample_data['avg_unprocessed_events_count'], 
                    label='Unprocessed Events', color='red', alpha=0.7)
            plt.plot(sample_data['DateTime'], sample_data['avg_queued_events_in_interval'], 
                    label='Queued Events', color='orange', alpha=0.7)
            plt.title('Queue Status Over Time')
            plt.legend()
            plt.xticks(rotation=45)
            
            # Plot 3: Processing performance
            plt.subplot(3, 1, 3)
            plt.plot(sample_data['DateTime'], sample_data['avg_average_processing_duration_ms'], 
                    label='Avg Processing Time (ms)', color='green', alpha=0.7)
            plt.title('Processing Performance Over Time')
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig('scaleiq_demo_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Basic analysis plot saved as 'scaleiq_demo_analysis.png'")
            
            # Demonstrate feature engineering
            print("\n🔧 Demonstrating feature engineering...")
            
            # Add temporal features
            df['hour'] = df['DateTime'].dt.hour
            df['day_of_week'] = df['DateTime'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            
            # Add lag features
            for lag in [6, 12, 24]:
                df[f'logged_events_lag_{lag}'] = df['avg_logged_events_in_interval'].shift(lag)
                df[f'logged_events_ma_{lag}'] = df['avg_logged_events_in_interval'].rolling(lag).mean()
            
            # Queue pressure metric
            df['queue_pressure'] = df['avg_unprocessed_events_count'] / (df['avg_processed_events_in_interval'] + 1)
            
            print(f"✅ Created {df.shape[1]} features (original + engineered)")
            
            # Show patterns
            print("\n📊 Pattern Analysis:")
            
            # Hourly patterns
            hourly_avg = df.groupby('hour')['avg_logged_events_in_interval'].mean()
            print("\nAverage events by hour:")
            for hour, avg_events in hourly_avg.items():
                print(f"  Hour {hour:2d}: {avg_events:8.1f} events")
            
            # Weekend vs weekday
            weekend_avg = df.groupby('is_weekend')['avg_logged_events_in_interval'].mean()
            print(f"\nWeekday average: {weekend_avg[0]:.1f} events")
            print(f"Weekend average: {weekend_avg[1]:.1f} events")
            
            # High traffic periods
            high_traffic = df[df['avg_logged_events_in_interval'] > df['avg_logged_events_in_interval'].quantile(0.95)]
            print(f"\nHigh traffic periods (top 5%): {len(high_traffic)} instances")
            print(f"Most common hours for high traffic: {high_traffic['hour'].mode().values}")
            
            # Demonstrate forecasting concept
            print("\n🔮 Forecasting Demonstration...")
            
            # Simple moving average forecast (baseline)
            window = 24  # 48 minutes lookback
            df['forecast_ma'] = df['avg_logged_events_in_interval'].rolling(window).mean().shift(1)
            
            # Calculate error for last 1000 points
            recent_data = df.iloc[-1000:].dropna()
            if len(recent_data) > 0:
                mae = np.mean(np.abs(recent_data['avg_logged_events_in_interval'] - recent_data['forecast_ma']))
                mape = np.mean(np.abs((recent_data['avg_logged_events_in_interval'] - recent_data['forecast_ma']) / 
                                    recent_data['avg_logged_events_in_interval'])) * 100
                
                print(f"Simple MA Baseline - MAE: {mae:.2f}, MAPE: {mape:.2f}%")
                print("(The AI model will significantly outperform this baseline)")
            
            # Show scaling decision logic
            print("\n⚖️ Auto-Scaling Decision Logic Demo...")
            
            # Simple rule-based scaling simulation
            current_queue = df['avg_unprocessed_events_count'].iloc[-1]
            current_processing_time = df['avg_average_processing_duration_ms'].iloc[-1]
            predicted_load = df['forecast_ma'].iloc[-1] if not pd.isna(df['forecast_ma'].iloc[-1]) else df['avg_logged_events_in_interval'].iloc[-1]
            
            print(f"Current queue size: {current_queue}")
            print(f"Current processing time: {current_processing_time:.1f}ms")
            print(f"Predicted load: {predicted_load:.1f} events")
            
            # Simple scaling logic
            if current_queue > 150 or predicted_load > 2000:
                scaling_decision = "Scale UP (1.5x)"
                reason = "High queue or predicted high load"
            elif current_queue < 50 and predicted_load < 800:
                scaling_decision = "Scale DOWN (0.8x)"
                reason = "Low queue and predicted low load"
            else:
                scaling_decision = "No change (1.0x)"
                reason = "Current capacity sufficient"
            
            print(f"Scaling decision: {scaling_decision}")
            print(f"Reason: {reason}")
            
            print("\n🎯 Demo completed successfully!")
            print("\nNext steps:")
            print("1. Install full dependencies: pip install -r requirements.txt")
            print("2. Run full training: python train.py")
            print("3. View results in the 'results' and 'visualizations' directories")
            
        else:
            print("❌ EventsMetricsMarJul.csv not found in current directory")
            print("Please ensure your data file is in the correct location")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install missing dependencies")
    except Exception as e:
        print(f"❌ Demo error: {e}")


def main():
    """Main function."""
    print("🚀 ScaleIQ Quick Start Demo")
    print("=" * 50)
    
    # Check if dependencies are installed
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ Core dependencies already installed")
        run_demo()
    except ImportError:
        print("📦 Installing core dependencies...")
        install_dependencies()
        print("\n🔄 Restarting to run demo...")
        run_demo()


if __name__ == "__main__":
    main()
