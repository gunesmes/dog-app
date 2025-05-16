import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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
        historical_df = historical_df.drop(columns=['timestamp'])  # Drop timestamp column

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

def analyze_performance_regression(historical_csv, new_data_csv, target_metric='avg', difference_threshold=None):
    """
    Analyzes performance test results using Random Forest Regression to predict a target metric.

    Args:
        historical_csv (str): Path to the historical data CSV file.
        new_data_csv (str): Path to the new data CSV file.
        target_metric (str): The metric to predict (e.g., 'avg', 'max').
        difference_threshold (float): The maximum acceptable difference between predicted and actual values.
                                      If None, it will be calculated using historical data.

    Returns:
        dict: A dictionary containing the analysis results.
    """

    try:
        # Read historical data
        historical_df = pd.read_csv(historical_csv)
        historical_df = historical_df.drop(columns=['timestamp'])  # Drop timestamp column

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
        model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune hyperparameters
        model.fit(X_train_scaled, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Read new data
        new_df = pd.read_csv(new_data_csv)
        new_df = new_df.drop(columns=['timestamp'])  # Drop timestamp column

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

        # Check if the result is satisfied
        is_satisfied = difference <= difference_threshold

        # Store results
        results = {
            "target_metric": target_metric,
            "predicted_value": predicted_value,
            "actual_value": actual_value,
            "difference": difference,
            "difference_threshold": difference_threshold,
            "mse": mse,
            "r2": r2,
            "is_satisfied": is_satisfied,
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
    print(f"   Difference: {results['difference']:.2f}")
    print(f"  Difference Threshold: {results['difference_threshold']:.2f}")
    print(f"  Mean Squared Error (MSE): {results['mse']:.2f}")
    print(f"  R-squared (R2): {results['r2']:.2f}")
    print(f"  Is Satisfied: {results['is_satisfied']}")

# Example usage
historical_csv = "dog-registration/test/historical_performance_data.csv" 
new_data_csv = "dog-registration/test/http_req_duration.csv" 
metrics = ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)']
results_all_metrics = {}

for metric in metrics:
    results = analyze_performance_regression(historical_csv, new_data_csv, metric, difference_threshold=None)
    results_all_metrics[metric] = results
    if results:
        print_analysis_results_regression(results)

assert all([result['is_satisfied'] for result in results_all_metrics.values()]), "Some metrics are not satisfied."