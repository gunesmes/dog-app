import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_threshold(historical_csv, target_metric='avg'):
    """
    Calculates the difference threshold for a target metric using historical data.

    Args:
        historical_csv (str): Path to the historical data CSV file.
        target_metric (str): The metric to predict (e.g., 'avg', 'max').

    Returns:
        float: The calculated difference threshold.
    """

    try:
        # Read historical data
        historical_df = pd.read_csv(historical_csv)
        historical_df = historical_df.drop(columns=['timestamp']) 
        
        # Prepare data for regression
        X = historical_df.drop(columns=[target_metric])
        y = historical_df[target_metric]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train a simple Linear Regression model to predict the target metric
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_scaled)

        # Calculate the absolute differences between predicted and actual values
        differences = np.abs(y_pred - y_test)

        # Calculate the threshold as the mean plus a multiple of the standard deviation of the differences
        threshold = np.mean(differences) + 2 * np.std(differences)

        return threshold

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def analyze_performance_regression(historical_csv, new_data_csv, target_metric='avg', difference_threshold=None, lower_threshold_factor=0.5):
    """
    Analyzes performance test results using Random Forest Regression to predict a target metric.
    Uses asymmetric thresholds to better detect suspiciously low values.

    Args:
        historical_csv (str): Path to the historical data CSV file.
        new_data_csv (str): Path to the new data CSV file.
        target_metric (str): The metric to predict (e.g., 'avg', 'max').
        difference_threshold (float): The maximum acceptable difference between predicted and actual values.
                                      If None, it will be calculated using historical data.
        lower_threshold_factor (float): Factor to apply to the threshold for the lower bound (0-1).
                                        Lower values create a tighter lower bound.

    Returns:
        dict: A dictionary containing the analysis results.
    """

    try:
        # Read historical data
        historical_df = pd.read_csv(historical_csv)
        historical_df = historical_df.drop(columns=['timestamp'])

        # Read new data
        new_df = pd.read_csv(new_data_csv)
        new_df = new_df.drop(columns=['timestamp'])  # Drop timestamp column

        # Prepare data for regression
        X = historical_df.drop(columns=[target_metric])
        y = historical_df[target_metric]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest Regression model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Scale the new data
        new_scaled = scaler.transform(new_df.drop(columns=[target_metric]))

        # Predict the target metric for the new data
        predicted_value = model.predict(new_scaled)[0]
        actual_value = new_df[target_metric].iloc[0]

        # Calculate the difference between the predicted and actual values
        difference = abs(predicted_value - actual_value)

        # If difference_threshold is None, calculate it using historical data
        if difference_threshold is None:
            difference_threshold = calculate_threshold(historical_csv, target_metric)

        # Calculate asymmetric thresholds
        upper_threshold = difference_threshold
        lower_threshold = difference_threshold * lower_threshold_factor

        # Determine if we're above or below the prediction
        is_above = actual_value >= predicted_value
        
        # Check if the result is satisfied using asymmetric thresholds
        if is_above:
            # For values above prediction, use the full threshold
            is_satisfied = difference <= upper_threshold
        else:
            # For values below prediction, use the tighter threshold
            is_satisfied = difference <= abs(lower_threshold)

        # Get feature importance
        feature_importance = {}
        feature_names = X.columns
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            feature_importance[name] = importance

        # Store results with asymmetric thresholds
        results = {
            "target_metric": target_metric,
            "predicted_value": predicted_value,
            "actual_value": actual_value,
            "difference": difference,
            "upper_threshold": upper_threshold,
            "lower_threshold": lower_threshold,
            "mse": mse,
            "r2": r2,
            "is_satisfied": is_satisfied,
            "is_above": is_above,
            "feature_importance": feature_importance
        }

        return results

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def print_analysis_results_regression(results):
    """Prints the analysis results in a readable format."""
    if results is None:
        print("No results to print.")
        return

    print(f"\nPerformance Analysis Results (Random Forest Regression - {results['target_metric']}):")
    print(f"  Predicted Value: {results['predicted_value']:.2f}")
    print(f"  Actual Value: {results['actual_value']:.2f}")
    print(f"  Difference: {results['difference']:.2f}")
    print(f"  Direction: {'Above prediction' if results['is_above'] else 'Below prediction'}")
    print(f"  Acceptable Range: +{results['upper_threshold']:.2f}/-{results['lower_threshold']:.2f}")
    print(f"  Mean Squared Error (MSE): {results['mse']:.2f}")
    print(f"  R-squared (R2): {results['r2']:.2f}")
    print(f"  Is Satisfied: {results['is_satisfied']}")

def plot_regression_analysis(results_all_metrics, historical_csv, new_data_csv):
    """
    Creates distribution and time series visualizations for regression analysis results.
    Uses equally spaced points for time series while maintaining original data order.
    Shows asymmetric thresholds.
    
    Args:
        results_all_metrics (dict): Dictionary with results for each metric
        historical_csv (str): Path to historical data
        new_data_csv (str): Path to new data
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os
    
    # Create directory for plots/analyze_regression
    os.makedirs('plots/analyze_regression', exist_ok=True)
    
    # Read data
    historical_df = pd.read_csv(historical_csv)
    new_df = pd.read_csv(new_data_csv)
    
    # For each metric, create the two requested plots
    for metric, results in results_all_metrics.items():
        if results is None:
            continue
            
        # 1. Distribution plot with prediction and actual value
        plt.figure(figsize=(10, 6))
        sns.histplot(historical_df[metric], kde=True, color='gray', alpha=0.5)
        
        plt.axvline(x=results['predicted_value'], color='blue', linestyle='--', 
                   label=f'Predicted: {results["predicted_value"]:.2f}')
        
        plt.axvline(x=results['actual_value'], color='green' if results['is_satisfied'] else 'red', 
                   linestyle='-', linewidth=2,
                   label=f'Actual: {results["actual_value"]:.2f}')
        
        # Display asymmetric threshold range
        plt.axvspan(
            results['predicted_value'] - results['lower_threshold'], 
            results['predicted_value'] + results['upper_threshold'], 
            alpha=0.2, color='green', 
            label=f'Acceptable Range (+{results["upper_threshold"]:.2f}/-{results["lower_threshold"]:.2f})'
        )
        
        plt.title(f'Distribution for {metric} Metric')
        plt.xlabel(f'{metric} Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'plots/analyze_regression/{metric}_distribution.png')
        plt.close()
        
        # 2. Time series plot with equally spaced points (maintaining original order)
        plt.figure(figsize=(12, 6))
        
        # Create equally spaced x-values for historical data (keeping original order)
        x_historical = np.arange(len(historical_df))
        
        # Plot historical values with equally spaced points
        plt.plot(x_historical, historical_df[metric], 'o-', 
                color='blue', alpha=0.7, label='Historical')
        
        # Position for the new data point (at the end)
        x_new = len(historical_df)
        
        # Plot the actual new value
        plt.scatter(x_new, results['actual_value'], 
                  color='green' if results['is_satisfied'] else 'red', 
                  s=100, label='Actual New Value')
        
        # Plot the predicted value
        plt.scatter(x_new, results['predicted_value'], 
                  color='blue', marker='X', s=100, label='Predicted Value')
        
        # Add asymmetric threshold range around the prediction at the new point
        plt.fill_between(
            [x_new - 0.5, x_new + 0.5],
            [results['predicted_value'] - results['lower_threshold']]*2,
            [results['predicted_value'] + results['upper_threshold']]*2,
            color='green', alpha=0.2, 
            label=f'Acceptable Range (+{results["upper_threshold"]:.2f}/-{results["lower_threshold"]:.2f})'
        )
        
        # Extend asymmetric acceptable range as a horizontal band across the plot
        plt.fill_between(
            plt.xlim(),
            [results['predicted_value'] - results['lower_threshold']]*2,
            [results['predicted_value'] + results['upper_threshold']]*2,
            alpha=0.1, color='green', label='_nolegend_'  # Don't duplicate in legend
        )
        
        plt.title(f'Time Series for {metric} Metric')
        plt.ylabel(f'{metric} Value')
        plt.xlabel('Test Run (Chronological Order)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plots/analyze_regression/{metric}_time_series.png')
        plt.close()
    
    print("Regression analysis plots saved to 'plots/analyze_regression/' directory")

# Example usage
if __name__ == "__main__":
    historical_csv = "dog-registration/test/historical_performance_data.csv" 
    new_data_csv = "dog-registration/test/http_req_duration.csv" 
    metrics = ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)']
    results_all_metrics = {}

    # You can adjust the lower_threshold_factor to control how strict the lower bound is
    # 2 means the lower bound is twice as strict as the upper bound
    lower_threshold_factor = 2

    for metric in metrics:
        results = analyze_performance_regression(
            historical_csv, 
            new_data_csv, 
            metric, 
            difference_threshold=None,
            lower_threshold_factor=lower_threshold_factor
        )
        results_all_metrics[metric] = results
        if results:
            print_analysis_results_regression(results)

    # Generate plots
    plot_regression_analysis(results_all_metrics, historical_csv, new_data_csv)

    # Final assertion to check if all metrics satisfy requirements
    assert all([result['is_satisfied'] for result in results_all_metrics.values()]), "Some metrics are not satisfied."