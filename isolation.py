from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.stats import zscore
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained LSTM model
lstm_model = load_model("model/lstm_model.h5")

# Load Isolation Forest (if needed for ensemble)
iso_forest = IsolationForest(contamination=0.05, random_state=42)


def load_and_preprocess_data(data):
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df


def feature_engineering(df):
    df['rolling_mean'] = df['meantemp'].rolling(window=7).mean().fillna(df['meantemp'])
    df['z_score'] = zscore(df['meantemp'])
    return df


def detect_anomalies(df):
    # LSTM model prediction
    X = df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']].values
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM input
    df['lstm_anomaly'] = lstm_model.predict(X)
    df['lstm_anomaly'] = (df['lstm_anomaly'] > 0.5).astype(int)

    # Isolation Forest
    df['iso_anomaly'] = iso_forest.fit_predict(df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']])
    
    # DBSCAN
    df['dbscan_anomaly'] = DBSCAN(eps=1, min_samples=5).fit_predict(df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']])
    
    # Z-score anomaly
    df['z_anomaly'] = (np.abs(df['z_score']) > 2).astype(int)
    
    # Consolidate anomalies
    df['is_anomaly'] = (df['iso_anomaly'] == -1) | (df['dbscan_anomaly'] == -1) | (df['z_anomaly'] == 1) | (df['lstm_anomaly'] == 1)
    return df


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({"error": "Invalid input format. Send data as JSON."}), 400
        
        df = load_and_preprocess_data(data['data'])
        df = feature_engineering(df)
        df = detect_anomalies(df)

        anomalies = df[df['is_anomaly']]

        return jsonify({
            "total_records": len(df),
            "anomalies_detected": len(anomalies),
            "anomaly_percentage": (len(anomalies) / len(df)) * 100,
            "anomalies": anomalies.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/fetch-results', methods=['GET'])
def fetch_results():
    try:
        df = pd.read_csv("detected_anomalies_weather.csv")
        return df.to_json(orient='records')
    except FileNotFoundError:
        return jsonify({"error": "No results found, please upload and process data first."}), 404


if __name__ == '__main__':
    app.run(debug=True)
