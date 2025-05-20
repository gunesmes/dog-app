import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def analyze_trend_with_current_results(historical_csv, current_csv, metrics=None):
    """
    Analyzes the trend in historical data and compares current test results.
    
    Args:
        historical_csv (str): Path to the historical data CSV file.
        current_csv (str): Path to the current test results CSV file.
        metrics (list): Metrics to analyze. If None, will analyze all common metrics.
        
    Returns:
        dict: Analysis results for each metric.
    """
    # Read historical data
    historical_df = pd.read_csv(historical_csv, parse_dates=['timestamp'])
    
    # Read current test results
    current_df = pd.read_csv(current_csv, parse_dates=['timestamp'])
    
    # If metrics not specified, find metrics common to both datasets
    if metrics is None:
        metrics = list(set(historical_df.columns) & set(current_df.columns))
        # Remove timestamp column
        metrics = [m for m in metrics if m != 'timestamp']
    
    results = {}
    
    # Sort historical data by timestamp
    historical_df = historical_df.sort_values('timestamp')
    
    for metric in metrics:
        # Skip if metric not in either dataset
        if metric not in historical_df.columns or metric not in current_df.columns:
            print(f"Metric {metric} not found in both datasets, skipping.")
            continue
        
        # Create sequential indices (equally spaced) for historical data
        X = np.arange(len(historical_df)).reshape(-1, 1)
        y = historical_df[metric].values
        
        # To this (with boundaries that allow smaller values):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e3))
        
        # Create and fit model
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp.fit(X, y)
        
        # Get the current actual value
        current_value = current_df[metric].iloc[0]
        
        # Predict value for next point in sequence (current test)
        current_point = np.array([[len(historical_df)]])
        prediction, std = gp.predict(current_point, return_std=True)
        
        # Calculate a realistic lower bound based on historical data
        historical_min = np.min(y)
        historical_mean = np.mean(y)
        historical_std = np.std(y)
        
        # Use 80% of historical minimum as an absolute floor
        absolute_floor = max(0, historical_min * 0.8)
        
        # For values near zero, scale the lower bound proportionally
        if prediction[0] < historical_mean * 0.5:
            # Use a proportional approach for lower bound
            lower_bound = max(absolute_floor, prediction[0] * 0.5)
        else:
            # Use standard approach but don't go below the floor
            lower_bound = max(absolute_floor, prediction[0] - 2*std[0])
        
        # Determine confidence interval with realistic lower bound
        confidence_interval = [lower_bound, prediction[0] + 2*std[0]]
        is_within_ci = confidence_interval[0] <= current_value <= confidence_interval[1]
        
        # Calculate percent difference
        if prediction[0] != 0:  # Avoid division by zero
            percent_diff = ((current_value - prediction[0]) / prediction[0]) * 100
        else:
            percent_diff = float('inf') if current_value != 0 else 0
            
        # Store results for this metric
        results[metric] = {
            "predicted_value": prediction[0],
            "actual_value": current_value,
            "uncertainty": std[0],
            "confidence_interval": confidence_interval,
            "is_within_confidence_interval": is_within_ci,
            "percent_difference": percent_diff,
            "historical_stats": {
                "min": historical_min,
                "mean": historical_mean,
                "std": historical_std
            }
        }
        
    return results

def print_analysis_results(results):
    """Prints the analysis results in a readable format."""
    print("\n=== Performance Trend Analysis Results ===\n")
    
    all_within_ci = True
    
    for metric, data in results.items():
        print(f"Metric: {metric}")
        print(f"  Predicted value: {data['predicted_value']:.2f}")
        print(f"  Actual value: {data['actual_value']:.2f}")
        print(f"  Difference: {data['actual_value'] - data['predicted_value']:.2f} ({data['percent_difference']:.2f}%)")
        print(f"  Confidence interval: [{data['confidence_interval'][0]:.2f}, {data['confidence_interval'][1]:.2f}]")
        print(f"  Historical min/mean/std: {data['historical_stats']['min']:.2f}/{data['historical_stats']['mean']:.2f}/{data['historical_stats']['std']:.2f}")
        print(f"  Within confidence interval: {'✅' if data['is_within_confidence_interval'] else '❌'}")
        print()
            
        if not data['is_within_confidence_interval']:
            all_within_ci = False
    
    # Overall assessment
    print("=== Overall Assessment ===")
    if all_within_ci:
        print("✅ All metrics are within confidence intervals - Current results align with historical trends.")
    else:
        print("❌ Some metrics are outside confidence intervals - Current results may indicate a performance change.")

def plot_trend_with_current(historical_csv, current_csv, metric):
    """Plots the historical trend with prediction and current value."""
    historical_df = pd.read_csv(historical_csv, parse_dates=['timestamp'])
    current_df = pd.read_csv(current_csv, parse_dates=['timestamp'])
    
    # Sort historical data by timestamp
    historical_df = historical_df.sort_values('timestamp')
    
    # Create sequential indices (equally spaced) for historical data
    X = np.arange(len(historical_df)).reshape(-1, 1)
    y = historical_df[metric].values
    
    # Calculate important historical statistics
    historical_min = np.min(y)
    historical_mean = np.mean(y)
    historical_std = np.std(y)
    absolute_floor = max(0, historical_min * 0.8)
    
    # Fit model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e3))    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, y)
    
    # Generate points for plotting the trend
    x_pred = np.linspace(0, len(historical_df) + 1, 100).reshape(-1, 1)
    y_pred, y_std = gp.predict(x_pred, return_std=True)
    
    # Calculate adaptive lower bounds
    lower_bounds = np.zeros_like(y_pred)
    for i, (pred, std_val) in enumerate(zip(y_pred, y_std)):
        if pred < historical_mean * 0.5:
            # Use proportional approach
            lower_bounds[i] = max(absolute_floor, pred * 0.5)
        else:
            # Use standard approach with floor
            lower_bounds[i] = max(absolute_floor, pred - 2*std_val)
    
    # Get current value
    current_value = current_df[metric].iloc[0]
    
    # Current point is the next point after historical data
    current_x = len(historical_df)
    
    # Predict for current point to get confidence interval
    current_pred, current_std = gp.predict(np.array([[current_x]]), return_std=True)
    
    # Determine if current value is within confidence interval
    if current_pred[0] < historical_mean * 0.5:
        current_lower_bound = max(absolute_floor, current_pred[0] * 0.5)
    else:
        current_lower_bound = max(absolute_floor, current_pred[0] - 2*current_std[0])
    
    current_ci = [current_lower_bound, current_pred[0] + 2*current_std[0]]
    is_within_ci = current_ci[0] <= current_value <= current_ci[1]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(np.arange(len(historical_df)), historical_df[metric], 'ko', label='Historical Data')
    
    # Plot current test with appropriate color
    if is_within_ci:
        plt.plot(current_x, current_value, 'go', markersize=10, label='Current Test (Within CI)')
    else:
        plt.plot(current_x, current_value, 'ro', markersize=10, label='Current Test (Outside CI)')
    
    # Plot prediction line
    plt.plot(x_pred, y_pred, 'b-', label='Predicted Trend')
    
    # Plot confidence intervals
    plt.plot(x_pred, y_pred + 2*y_std, 'b--', label='Upper CI')
    plt.plot(x_pred, lower_bounds, 'b--', alpha=0.5, label='Lower CI')
    
    # Fill between confidence intervals
    plt.fill_between(x_pred.ravel(), 
                    lower_bounds,
                    y_pred + 2*y_std, 
                    alpha=0.2, 
                    color='blue', 
                    label='95% Confidence Interval')
    
    # Format plot
    plt.title(f'Performance Trend Analysis: {metric}')
    plt.xlabel('Test Run Sequence')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations with actual dates at some points
    test_dates = historical_df['timestamp'].dt.strftime('%Y-%m-%d').tolist()
    
    # Add date annotations for first, last, and some intermediate points
    indices_to_annotate = [0, len(test_dates)//2, len(test_dates)-1]
    for idx in indices_to_annotate:
        plt.annotate(test_dates[idx], 
                    xy=(idx, historical_df[metric].iloc[idx]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8)
    
    # Annotate current test date
    current_date = current_df['timestamp'].iloc[0].strftime('%Y-%m-%d')
    plt.annotate(current_date, 
                xy=(current_x, current_value),
                xytext=(0, -15),
                textcoords='offset points',
                ha='center',
                fontsize=8)
    
    # Add confidence interval info box
    info_text = f"Predicted: {current_pred[0]:.2f}\nActual: {current_value:.2f}\nCI: [{current_ci[0]:.2f}, {current_ci[1]:.2f}]"
    plt.text(0.98, 0.85, info_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7, pad=5, boxstyle='round'))
    
    # Add status text box
    if is_within_ci:
        status_text = "✅ WITHIN CONFIDENCE INTERVAL"
        box_color = 'lightgreen'
        text_color = 'darkgreen'
    else:
        status_text = "❌ OUTSIDE CONFIDENCE INTERVAL"
        box_color = 'lightcoral'
        text_color = 'darkred'
    
    plt.text(0.98, 0.95, status_text, 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             weight='bold',
             color=text_color,
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor=box_color, alpha=0.5, pad=10, boxstyle='round'))
    
    # Make directory if it doesn't exist
    os.makedirs('plots/analyze_trend', exist_ok=True)
    
    plt.savefig(f'plots/analyze_trend/trend_analysis_{metric}.png')
    plt.close()
    
    print(f"Plot saved as plots/analyze_trend/trend_analysis_{metric}.png")

if __name__ == "__main__":
    historical_csv = "dog-registration/test/historical_performance_data.csv"
    current_csv = "dog-registration/test/http_req_duration.csv"
    
    # Analyze all common metrics
    results = analyze_trend_with_current_results(historical_csv, current_csv)
    print_analysis_results(results)
    
    # Plot a specific metric (e.g., 'avg')
    for metric in results.keys():
        plot_trend_with_current(historical_csv, current_csv, metric)

    # assert all metrics are within confidence intervals
    assert all(data['is_within_confidence_interval'] for data in results.values()), "Some metrics are outside confidence intervals!"