from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
import joblib

app = Flask(__name__)
CORS(app)

WINDOW_SECONDS = 5
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
        hrv_ms = np.std(np.diff(peaks) / fs * 1000)
    else:
        hrv_ms = 0.0
    return [hr_mean, hr_std, hr_bpm, hrv_ms], hr_bpm, hrv_ms

def movement_features(window_xyz):
    magnitude = np.linalg.norm(window_xyz, axis=1)
    return [np.std(magnitude), np.mean(magnitude), np.mean(np.abs(np.diff(magnitude)))]

def rotation_features(window_xyz_gyro):
    magnitude = np.linalg.norm(window_xyz_gyro, axis=1)
    return [np.std(magnitude), np.mean(magnitude), np.mean(np.abs(np.diff(magnitude)))]

def make_fake_window():
    fs_ppg = 256
    ppg = np.sin(np.linspace(0, 15, fs_ppg * WINDOW_SECONDS)) + np.random.normal(0, 0.05, fs_ppg * WINDOW_SECONDS)
    accel = np.random.normal(0, 0.3, (125, 3))
    gyro = np.random.normal(0, 5, (125, 3))
    return ppg, fs_ppg, accel, gyro

@app.route("/")
def home():
    return "EpiRing Backend is Running!"

@app.route("/risk", methods=["GET"])
def get_risk():
    ppg, fs, accel, gyro = make_fake_window()

    feats, hr_bpm, hrv_ms = ppg_features(ppg, fs)
    feats += movement_features(accel)
    feats += rotation_features(gyro)
    feats_df = pd.DataFrame([feats], columns=FEATURE_COLUMNS)

    prediction = int(model.predict(feats_df)[0])
    label = {0: "Normal", 1: "Elevated", 2: "High"}[prediction]

    return jsonify({
        "status": label,
        "hr": round(hr_bpm, 1),
        "eda": round(hrv_ms, 1)
    })

if __name__ == "__main__":
    app.run(debug=True)