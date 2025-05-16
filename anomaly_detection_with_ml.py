import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def analyze_performance_ml(historical_csv, new_data_csv, contamination='auto'):
    """
    Analyzes performance test results using Isolation Forest for anomaly detection.

    Args:
        historical_csv (str): Path to the historical data CSV file.
        new_data_csv (str): Path to the new data CSV file.
        contamination (float or 'auto'): The proportion of outliers in the data.

    Returns:
        dict: A dictionary containing the analysis results.
    """

    try:
        # Read historical data
        historical_df = pd.read_csv(historical_csv)
        historical_df = historical_df.drop(columns=['timestamp'])  # Drop timestamp column

        # Scale the data
        scaler = StandardScaler()
        historical_scaled = scaler.fit_transform(historical_df)

        # Train Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=None)
        model.fit(historical_scaled)

        # Read new data
        new_df = pd.read_csv(new_data_csv)
        new_df = new_df.drop(columns=['timestamp'])  # Drop timestamp column

        # Scale the new data
        new_scaled = scaler.transform(new_df)

        # Predict anomaly score for the new data
        anomaly_score = model.decision_function(new_scaled)[0]

        # Predict if the new data is an anomaly
        is_anomaly = model.predict(new_scaled)[0] == -1

        # Store results
        results = {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
        }

        return results

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def print_analysis_results_ml(results):
    """Prints the analysis results in a readable format."""
    if results is None:
        print("No results to print.")
        return

    print("\nPerformance Analysis Results (Isolation Forest):")
    print(f"  Anomaly Score: {results['anomaly_score']:.2f}")
    print(f"  Is Anomaly: {results['is_anomaly']}")

# Example usage
historical_csv = "dog-registration/test/historical_performance_data.csv" 
new_data_csv = "dog-registration/test/http_req_duration.csv" 

results = analyze_performance_ml(historical_csv, new_data_csv)

if results:
    print_analysis_results_ml(results)

# This is a simple assertion to check if the new data is not an anomaly
assert results['is_anomaly'] == False, "Anomaly detected in the new data!"