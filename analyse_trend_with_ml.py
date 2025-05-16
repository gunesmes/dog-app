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
        
        # Define kernel - RBF is good for smooth trends
        kernel = C(1.0) * RBF(length_scale=1.0)
        
        # Create and fit model
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp.fit(X, y)
        
        # Get the current actual value
        current_value = current_df[metric].iloc[0]
        
        # Predict value for next point in sequence (current test)
        current_point = np.array([[len(historical_df)]])
        prediction, std = gp.predict(current_point, return_std=True)
        
        # Determine if current value is within confidence interval
        confidence_interval = [prediction[0] - 2*std[0], prediction[0] + 2*std[0]]
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
            "percent_difference": percent_diff
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
    
    # Fit model
    kernel = C(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, y)
    
    # Generate points for plotting the trend
    x_pred = np.linspace(0, len(historical_df) + 1, 100).reshape(-1, 1)
    y_pred, y_std = gp.predict(x_pred, return_std=True)
    
    # Get current value
    current_value = current_df[metric].iloc[0]
    
    # Current point is the next point after historical data
    current_x = len(historical_df)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(historical_df)), historical_df[metric], 'ko', label='Historical Data')
    plt.plot(current_x, current_value, 'ro', markersize=10, label='Current Test')
    plt.plot(x_pred, y_pred, 'b-', label='Predicted Trend')
    plt.plot(x_pred, y_pred + 2*y_std, 'b--', label='Confidence Interval')
    plt.fill_between(x_pred.ravel(), 
                    y_pred - 2*y_std, 
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
    
    plt.savefig(f'performance_results/trend_analysis_{metric}.png')
    plt.show(block=False)
    plt.pause(2)  # Display the plot for 2 seconds
    plt.close()  # Close the plot and continue to the next metric    
    
    print(f"Plot saved as trend_analysis_{metric}.png")

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