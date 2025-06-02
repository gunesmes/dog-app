import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def analyze_performance_ml(historical_csv, new_data_csv, contamination='auto', generate_plots=True):
    """
    Analyzes performance test results using Isolation Forest for anomaly detection.

    Args:
        historical_csv (str): Path to the historical data CSV file.
        new_data_csv (str): Path to the new data CSV file.
        contamination (float or 'auto'): The proportion of outliers in the data.
        generate_plots (bool): Whether to generate and save plots.

    Returns:
        dict: A dictionary containing the analysis results.
    """

    try:
        # Read historical data
        historical_df = pd.read_csv(historical_csv)
        historical_metrics = historical_df.drop(columns=['timestamp'])

        # Read new data
        new_df = pd.read_csv(new_data_csv)
        new_metrics = new_df.drop(columns=['timestamp'])

        # Scale the data
        scaler = StandardScaler()
        historical_scaled = scaler.fit_transform(historical_metrics)
        new_scaled = scaler.transform(new_metrics)

        # Train Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(historical_scaled)

        # Get anomaly scores for historical data
        historical_scores = model.decision_function(historical_scaled)
        historical_predictions = model.predict(historical_scaled)
        
        # Predict anomaly score for the new data
        new_score = model.decision_function(new_scaled)[0]
        is_anomaly = model.predict(new_scaled)[0] == -1
        
        # Calculate feature correlations with anomaly scores
        feature_correlations = {}
        for i, col in enumerate(historical_metrics.columns):
            feature_correlations[col] = abs(np.corrcoef(historical_scaled[:, i], historical_scores)[0, 1])

        # Store results
        results = {
            "anomaly_score": new_score,
            "is_anomaly": is_anomaly,
            "historical_scores": historical_scores,
            "historical_predictions": historical_predictions,
            "historical_data": historical_metrics,
            "new_data": new_metrics,
            "scaler": scaler,
            "model": model,
            "feature_correlations": feature_correlations
        }

        # Generate plots if requested
        if generate_plots:
            plot_anomaly_detection(results, historical_csv, new_data_csv)

        return results

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_anomaly_detection(results, historical_csv, new_data_csv):
    """
    Generates and saves visualizations for anomaly detection results.
    
    Args:
        results (dict): Results from analyze_performance_ml.
        historical_csv (str): Path to the historical data CSV file.
        new_data_csv (str): Path to the new data CSV file.
    """
    # Create directory for plots
    os.makedirs('plots/analyze_anomaly', exist_ok=True)
    
    # Extract data from results
    historical_data = results["historical_data"]
    new_data = results["new_data"]
    historical_scores = results["historical_scores"]
    historical_predictions = results["historical_predictions"]
    new_score = results["anomaly_score"]
    is_anomaly = results["is_anomaly"]
    feature_correlations = results["feature_correlations"]

    # 1. Anomaly Score Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(historical_scores, kde=True)
    plt.axvline(x=new_score, color='r', linestyle='--', 
                label=f'New Data Score: {new_score:.3f} ({"Anomaly" if is_anomaly else "Normal"})')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score (higher = more normal)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('plots/analyze_anomaly/anomaly_score_distribution.png')
    plt.close()

    # 2. Principal Component Analysis for 2D visualization
    pca = PCA(n_components=2)
    historical_pca = pca.fit_transform(results["scaler"].transform(historical_data))
    new_pca = pca.transform(results["scaler"].transform(new_data))
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(historical_pca[:, 0], historical_pca[:, 1], 
               c=historical_predictions, cmap='coolwarm',
               label='Historical Data')
    plt.scatter(new_pca[0, 0], new_pca[0, 1], 
               marker='X', color='lime' if not is_anomaly else 'red', 
               s=200, edgecolors='black',
               label=f'New Data ({"Normal" if not is_anomaly else "Anomaly"})')
    
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", title="Prediction")
    plt.gca().add_artist(legend1)
    plt.legend(loc="upper left")
    
    plt.title('PCA Projection of Performance Data with Anomaly Detection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('plots/analyze_anomaly/pca_visualization.png')
    plt.close()

    # 3. Feature correlation with anomaly scores visualization
    plt.figure(figsize=(14, 6))
    features = list(feature_correlations.keys())
    importance = list(feature_correlations.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [importance[i] for i in sorted_idx]
    
    plt.barh(range(len(sorted_idx)), sorted_importance, align='center')
    plt.yticks(range(len(sorted_idx)), sorted_features)
    plt.title('Feature Correlation with Anomaly Scores')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.savefig('plots/analyze_anomaly/feature_correlation.png')
    plt.close()

    # 4. Parallel Coordinates Plot showing all metrics
    # Use all available metrics instead of selecting top features
    top_features = features.copy()  # Start with all features
    
    # Make sure metrics are in a logical order
    preferred_order = ['min', 'med', 'avg', 'p(90)', 'p(95)', 'max']
    ordered_features = [f for f in preferred_order if f in top_features]
    other_features = [f for f in top_features if f not in preferred_order]
    top_features = ordered_features + other_features
    
    # Determine which metrics are normal/abnormal based on Z-score
    metric_status = {}
    for feature in top_features:
        historical_mean = historical_data[feature].mean()
        historical_std = historical_data[feature].std()
        new_value = new_data[feature].iloc[0]
        z_score = abs((new_value - historical_mean) / historical_std) if historical_std > 0 else 0
        metric_status[feature] = z_score <= 2.0
    
    # Create DataFrame for parallel plot
    historical_subset = historical_data[top_features].copy()
    historical_subset['type'] = ['Historical (Normal)' if p == 1 else 'Historical (Anomaly)' 
                                for p in historical_predictions]
    
    new_subset = new_data[top_features].copy()
    new_subset['type'] = ['New Data (Normal)' if not is_anomaly else 'New Data (Anomaly)']
    
    combined = pd.concat([historical_subset, new_subset])
    
    # Create the parallel coordinates plot
    plt.figure(figsize=(15, 8))
    pd.plotting.parallel_coordinates(combined, 'type', colormap='viridis')
    
    # Add colored dots for each metric directly on the plot
    # Get axes to work with
    ax = plt.gca()
    
    # Get x-axis tick positions (these are the feature positions)
    x_positions = ax.get_xticks()
    
    # Calculate appropriate y positions based on the actual values
    for i, feature in enumerate(top_features):
        is_normal = metric_status[feature]
        color = 'green' if is_normal else 'red'
        
        # Get the metric value from new data
        value = new_data[feature].iloc[0]
        
        # Place dot at the correct position
        if i < len(x_positions):
            plt.scatter(x_positions[i], value, s=150, color=color, 
                      edgecolors='black', linewidths=1.5, zorder=10)
    
    # Add annotation explaining the dots
    plt.annotate('● Normal metric', 
                xy=(0.01, 0.15), 
                xycoords='axes fraction',
                fontsize=10,
                color='green',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.annotate('● Abnormal metric', 
                xy=(0.10, 0.15), 
                xycoords='axes fraction',
                fontsize=10,
                color='red',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('Parallel Coordinates Plot of Top Features')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('plots/analyze_anomaly/parallel_coordinates.png')
    plt.close()

    # 5. Anomaly metrics comparison
    plt.figure(figsize=(15, 8))

    # For each important metric, show historical distribution and new value
    for i, feature in enumerate(top_features):
        ax = plt.subplot(1, len(top_features), i+1)
        
        # Plot historical data distribution
        sns.kdeplot(historical_data[feature], fill=True, alpha=0.3)
        
        # Get status and color for this metric
        is_normal = metric_status[feature]
        color = 'green' if is_normal else 'red'
        
        # Add vertical line for new value with appropriate color
        new_value = new_data[feature].iloc[0]
        plt.axvline(x=new_value, color=color, linestyle='--', 
                    label=f'New: {new_value:.2f}')
        
        # Add colored dot on y-axis to indicate metric status
        ax.plot(0, 0.5, 'o', color=color, markersize=15, clip_on=False,
                transform=ax.get_yaxis_transform())
        
        # Mark anomalous historical points
        historical_anomalies = historical_data[feature][historical_predictions == -1]
        if len(historical_anomalies) > 0:
            sns.rugplot(historical_anomalies, color='red', height=0.1, label='Historical Anomalies')
        
        plt.title(f'{feature}')
        if i == 0:  # Only add legend to first subplot
            plt.legend()

    # Add annotation explaining the dots
    plt.figtext(0.45, 0.01, '● Normal metric', 
               fontsize=10, ha='right',
               color='green',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
               
    plt.figtext(0.55, 0.01, '● Abnormal metric', 
               fontsize=10, ha='left',
               color='red')

    plt.suptitle('Key Metrics Distribution with Current Value', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust for suptitle and annotation
    plt.savefig('plots/analyze_anomaly/metrics_comparison.png')
    plt.close()

    # 6. Time series plot (if timestamp information is available)
    try:
        historical_df = pd.read_csv(historical_csv, parse_dates=['timestamp'])
        new_df = pd.read_csv(new_data_csv, parse_dates=['timestamp'])
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_features):
            ax = plt.subplot(len(top_features), 1, i+1)
            
            # Plot historical data as line
            plt.plot(historical_df['timestamp'], historical_df[feature], 'b-', alpha=0.7)
            
            # Mark anomalies in historical data
            anomaly_mask = historical_predictions == -1
            if np.any(anomaly_mask):
                plt.scatter(historical_df['timestamp'][anomaly_mask], 
                           historical_df[feature][anomaly_mask],
                           color='red', label='Historical Anomalies', zorder=5)
            
            # Plot the new data point with appropriate color
            is_normal = metric_status[feature]
            color = 'green' if is_normal else 'red'
            label = 'New Data (Normal)' if is_normal else 'New Data (Anomaly)'
            plt.scatter(new_df['timestamp'], new_df[feature], 
                       color=color, s=100, zorder=10, label=label)
            
            plt.ylabel(feature)
            if i == 0:
                plt.legend()
                
            plt.grid(True, alpha=0.3)
            
        plt.xlabel('Timestamp')
        plt.suptitle('Time Series Analysis of Key Metrics', fontsize=16)
        plt.tight_layout()
        plt.savefig('plots/analyze_anomaly/time_series.png')
        plt.close()
            
    except Exception as e:
        print(f"Couldn't create time series plot: {e}")

    print("Anomaly detection plots saved to 'plots/analyze_anomaly/' directory")


def print_analysis_results_ml(results):
    """Prints the analysis results in a readable format."""
    if results is None:
        print("No results to print.")
        return

    print("\nPerformance Analysis Results (Isolation Forest):")
    print(f"  Anomaly Score: {results['anomaly_score']:.2f}")
    print(f"  Is Anomaly: {results['is_anomaly']}")
    
    # Count historical anomalies for context
    if 'historical_predictions' in results:
        anomaly_count = sum(1 for p in results['historical_predictions'] if p == -1)
        total_count = len(results['historical_predictions'])
        print(f"  Historical context: {anomaly_count} anomalies out of {total_count} samples ({anomaly_count/total_count:.1%})")

    # Print top correlated features
    if 'feature_correlations' in results:
        print("\nTop features correlated with anomalies:")
        sorted_features = sorted(results['feature_correlations'].items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, corr in sorted_features[:5]:
            print(f"  {feature}: {abs(corr):.3f}")

# Example usage
if __name__ == "__main__":
    historical_csv = "dog-registration/test/historical_performance_data.csv" 
    new_data_csv = "dog-registration/test/http_req_duration.csv" 

    results = analyze_performance_ml(historical_csv, new_data_csv, generate_plots=True)

    if results:
        print_analysis_results_ml(results)

    if results:
        assert results['is_anomaly'] == False, "ALERT: Anomaly detected in the new data!"