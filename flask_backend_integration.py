"""
EpiRing - Flask backend integration guide
========================================================================
FOR: the app friend building the Flask website backend

INSTALL FIRST (in your Flask project's terminal):
    pip install joblib scipy numpy pandas flask
"""

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
import joblib
from flask import Flask, request, jsonify

WINDOW_SECONDS = 5

# Load the model ONCE when the server starts - not on every request
model = joblib.load("epiring_model_final.pkl")

FEATURE_COLUMNS = [
    "hr_mean", "hr_std", "hr_bpm", "hrv_ms",
    "movement_energy", "movement_mean", "movement_jerkiness",
    "rotation_energy", "rotation_mean", "rotation_jerkiness",
]


def ppg_features(window, fs):
    hr_mean = np.mean(window)
    hr_std = np.std(window)
    peaks, _ = sp_signal.find_peaks(window, distance=fs * 0.4)
    hr_bpm = len(peaks) * (60 / WINDOW_SECONDS)
    if len(peaks) >= 3:
        peak_intervals_ms = np.diff(peaks) / fs * 1000
        hrv_ms = np.std(peak_intervals_ms)
    else:
        hrv_ms = 0.0
    return [hr_mean, hr_std, hr_bpm, hrv_ms]


def movement_features(window_xyz):
    magnitude = np.linalg.norm(window_xyz, axis=1)
    return [np.std(magnitude), np.mean(magnitude), np.mean(np.abs(np.diff(magnitude)))]


def rotation_features(window_xyz_gyro):
    magnitude = np.linalg.norm(window_xyz_gyro, axis=1)
    return [np.std(magnitude), np.mean(magnitude), np.mean(np.abs(np.diff(magnitude)))]


def predict_risk(ppg_window, ppg_fs, accel_window, gyro_window):
    feats = []
    feats += ppg_features(ppg_window, ppg_fs)
    feats += movement_features(accel_window)
    feats += rotation_features(gyro_window)

    feats_df = pd.DataFrame([feats], columns=FEATURE_COLUMNS)
    prediction = model.predict(feats_df)[0]
    return int(prediction)


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON like:
    {
      "ppg": [list of ~1280 numbers, 5 sec at 256 Hz],
      "ppg_fs": 256,
      "accel": [[x,y,z], [x,y,z], ... ~125 rows, 5 sec at 25 Hz],
      "gyro":  [[x,y,z], [x,y,z], ... ~125 rows]
    }
    Returns: {"risk": 0/1/2, "label": "Normal"/"Elevated"/"High risk"}
    """
    data = request.get_json()

    ppg_window = np.array(data["ppg"])
    ppg_fs = data["ppg_fs"]
    accel_window = np.array(data["accel"])
    gyro_window = np.array(data["gyro"])

    risk = predict_risk(ppg_window, ppg_fs, accel_window, gyro_window)
    label = {0: "Normal", 1: "Elevated", 2: "High risk"}[risk]

    return jsonify({"risk": risk, "label": label})


if __name__ == "__main__":
    app.run(debug=True, port=5000)