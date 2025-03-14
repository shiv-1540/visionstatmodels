from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.signal import medfilt

app = Flask(__name__)

# def load_and_preprocess_data(file_path):
#     df = pd.read_csv(file_path, parse_dates=["date"])
#     df.set_index("date", inplace=True)
#     df = df.dropna()
#     return df

def load_and_preprocess_data(data):
    # Convert JSON list to a DataFrame
    df = pd.DataFrame(data)
    
    # Add your preprocessing steps here (if any)
    return df

def detect_anomalies(df):
    # Z-Score Anomaly Detection
    df["z_score"] = np.abs(zscore(df["meantemp"]))
    df["z_anomaly"] = df["z_score"] > 3
    
    # Moving Average Anomaly Detection
    window = 7
    df["rolling_mean"] = df["meantemp"].rolling(window).mean()
    df["rolling_std"] = df["meantemp"].rolling(window).std()
    df["ma_anomaly"] = (df["meantemp"] > df["rolling_mean"] + 2 * df["rolling_std"]) | \
                       (df["meantemp"] < df["rolling_mean"] - 2 * df["rolling_std"])
    
    # IQR Anomaly Detection
    Q1 = df["meantemp"].quantile(0.25)
    Q3 = df["meantemp"].quantile(0.75)
    IQR = Q3 - Q1
    df["iqr_anomaly"] = (df["meantemp"] < (Q1 - 1.5 * IQR)) | (df["meantemp"] > (Q3 + 1.5 * IQR))
    
    # Hampel Filter Anomaly Detection
    df["hampel_anomaly"] = hampel_filter(df["meantemp"])

    # Summary
    methods = ["z_anomaly", "ma_anomaly", "iqr_anomaly", "hampel_anomaly"]
    summary = {method: int(df[method].sum()) for method in methods}
    best_method = max(summary, key=summary.get)
    
    return df, summary, best_method


def hampel_filter(series, window_size=7, n_sigma=3):
    median_filtered = medfilt(series, kernel_size=window_size)
    std_dev = np.std(series - median_filtered)
    outliers = np.abs(series - median_filtered) > n_sigma * std_dev
    return outliers


@app.route('/detect-anomalies', methods=['POST'])
def detect_anomalies_api():
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({"error": "Invalid input format. Send data as JSON with a 'data' key."}), 400

        df = load_and_preprocess_data(data['data'])
        df, summary, best_method = detect_anomalies(df)

        return jsonify({
            "anomalies_detected": summary,
            "best_performing_method": best_method,
            "anomaly_records": df[df[list(summary.keys())].any(axis=1)].reset_index().to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/test', methods=['GET'])
def test_api():
    return jsonify({"message": "API is working!"})


if __name__ == '__main__':
    app.run(debug=True)

# Save this as app.py and run: python app.py
# Test with a tool like Postman or curl by uploading a CSV file with date and meantemp fields.
# Let me know if you want me to add visualization endpoints or any improvements! 🚀