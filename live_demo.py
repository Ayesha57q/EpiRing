"""
EpiRing - LIVE DEMO: simulates the ring streaming real data in real time
========================================================================
"""

import time
import numpy as np
import pandas as pd
import pyedflib
import joblib
from scipy import signal as sp_signal

WINDOW_SECONDS = 5
PATIENT_ID = "sub-001"

LIVE_DELAY = 0.3
CONTEXT_WINDOWS = 6


def load_signal(file_path, signal_name):
    f = pyedflib.EdfReader(file_path)
    labels = f.getSignalLabels()
    idx = labels.index(signal_name)
    data = f.readSignal(idx)
    fs = f.getSampleFrequency(idx)
    f.close()
    return data, fs


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


def run_live_demo():
    print("Loading trained model and patient data...\n")
    model = joblib.load("epiring_model_final.pkl")

    ecg_file = f"data/{PATIENT_ID}_ses-01_task-szMonitoring_run-01_ecg.edf"
    mov_file = f"data/{PATIENT_ID}_ses-01_task-szMonitoring_run-01_mov.edf"
    events_file = f"data/{PATIENT_ID}_ses-01_task-szMonitoring_run-01_events.tsv"

    ecg, ecg_fs = load_signal(ecg_file, "ECG SD")
    acc_x, acc_fs = load_signal(mov_file, "ECGEMG SD ACC X")
    acc_y, _ = load_signal(mov_file, "ECGEMG SD ACC Y")
    acc_z, _ = load_signal(mov_file, "ECGEMG SD ACC Z")
    accel = np.stack([acc_x, acc_y, acc_z], axis=1)
    gyr_a, gyr_fs = load_signal(mov_file, "ECGEMG SD GYR A")
    gyr_b, _ = load_signal(mov_file, "ECGEMG SD GYR B")
    gyr_c, _ = load_signal(mov_file, "ECGEMG SD GYR C")
    gyro = np.stack([gyr_a, gyr_b, gyr_c], axis=1)
    events = pd.read_csv(events_file, sep="\t")

    ecg_window_size = int(ecg_fs * WINDOW_SECONDS)
    accel_window_size = int(acc_fs * WINDOW_SECONDS)
    n_windows = min(len(ecg) // ecg_window_size, len(accel) // accel_window_size)

    real_seizure_windows = set()
    for _, row in events.iterrows():
        event_type = str(row.get("eventType", "")).strip().lower()
        if event_type.startswith("sz") or event_type == "impd":
            onset_sec, duration_sec = row["onset"], row["duration"]
            start_w = int(onset_sec // WINDOW_SECONDS)
            end_w = int((onset_sec + duration_sec) // WINDOW_SECONDS) + 1
            for w in range(start_w, end_w):
                real_seizure_windows.add(w)

    live_windows = set()
    for w in real_seizure_windows:
        for offset in range(-CONTEXT_WINDOWS, CONTEXT_WINDOWS + 1):
            live_windows.add(w + offset)

    print(f"Total windows: {n_windows}  |  Real seizure windows: {sorted(real_seizure_windows)}")
    print("Fast-forwarding through normal periods, playing live around each seizure...\n")
    print("=" * 60)

    columns = [
        "hr_mean", "hr_std", "hr_bpm", "hrv_ms",
        "movement_energy", "movement_mean", "movement_jerkiness",
        "rotation_energy", "rotation_mean", "rotation_jerkiness",
    ]

    for i in range(n_windows):
        if i not in live_windows:
            continue

        e_s, e_e = i * ecg_window_size, (i + 1) * ecg_window_size
        a_s, a_e = i * accel_window_size, (i + 1) * accel_window_size

        feats = []
        feats += ppg_features(ecg[e_s:e_e], ecg_fs)
        feats += movement_features(accel[a_s:a_e])
        feats += rotation_features(gyro[a_s:a_e])
        feats_df = pd.DataFrame([feats], columns=columns)

        pred = model.predict(feats_df)[0]
        risk_label = {0: "Normal", 1: "Elevated", 2: "HIGH RISK - ALERT TRIGGERED"}[pred]

        minute = round(i * WINDOW_SECONDS / 60, 1)
        marker = " <-- REAL SEIZURE HAPPENING" if i in real_seizure_windows else ""
        print(f"[t = {minute:>6} min]  Ring status: {risk_label}{marker}")

        time.sleep(LIVE_DELAY)

    print("=" * 60)
    print("\nDemo complete.")


if __name__ == "__main__":
    run_live_demo()