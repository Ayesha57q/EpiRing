"""
EpiRing - DEMO: show the trained model working on real patient data
========================================================================
This loads your saved model, runs it across one patient's full
recording, and prints a clear timeline showing where it predicted
risk - compared against where the REAL seizures actually happened.

This is what you show your team/judges - not raw numbers, but a
clear "here's what really happened vs what the model predicted."
"""

import numpy as np
import pandas as pd
import pyedflib
import joblib
from scipy import signal as sp_signal

WINDOW_SECONDS = 5

# Pick a patient that had usable seizure examples (sub-001, sub-002, or sub-004)
PATIENT_ID = "sub-001"


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


def run_demo():
    print(f"Loading trained model...")
    model = joblib.load("epiring_model_final.pkl")

    print(f"Loading real data for {PATIENT_ID}...\n")
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

    # Figure out which windows are REAL seizures (ground truth), for comparison
    real_seizure_windows = set()
    for _, row in events.iterrows():
        event_type = str(row.get("eventType", "")).strip().lower()
        if event_type.startswith("sz") or event_type == "impd":
            onset_sec, duration_sec = row["onset"], row["duration"]
            start_w = int(onset_sec // WINDOW_SECONDS)
            end_w = int((onset_sec + duration_sec) // WINDOW_SECONDS) + 1
            for w in range(start_w, end_w):
                real_seizure_windows.add(w)

    print(f"Real seizures in this recording occur in windows: {sorted(real_seizure_windows)}\n")
    print("Running model across the full recording...\n")
    print(f"{'Window':>8} {'Time (min)':>12} {'Predicted':>12} {'Actually was':>14}")
    print("-" * 50)

    labels_cols = {0: "Normal", 1: "Elevated", 2: "HIGH RISK"}
    caught, missed, false_alarms = 0, 0, 0

    for i in range(n_windows):
        e_s, e_e = i * ecg_window_size, (i + 1) * ecg_window_size
        a_s, a_e = i * accel_window_size, (i + 1) * accel_window_size

        feats = []
        feats += ppg_features(ecg[e_s:e_e], ecg_fs)
        feats += movement_features(accel[a_s:a_e])
        feats += rotation_features(gyro[a_s:a_e])

        # Wrap in a DataFrame with the same column names used during
        # training - this avoids the repeated "no feature names" warning
        # and is also just more correct.
        feats_df = pd.DataFrame([feats], columns=[
            "hr_mean", "hr_std", "hr_bpm", "hrv_ms",
            "movement_energy", "movement_mean", "movement_jerkiness",
            "rotation_energy", "rotation_mean", "rotation_jerkiness",
        ])
        pred = model.predict(feats_df)[0]
        is_real_seizure = i in real_seizure_windows

        # Only print interesting moments (a real seizure, or a model alert) - not all 13000 rows
        if is_real_seizure or pred == 2:
            minute = round(i * WINDOW_SECONDS / 60, 1)
            actual = "SEIZURE" if is_real_seizure else "normal"
            print(f"{i:>8} {minute:>12} {labels_cols[pred]:>12} {actual:>14}")

            if is_real_seizure and pred == 2:
                caught += 1
            elif is_real_seizure and pred != 2:
                missed += 1
            elif not is_real_seizure and pred == 2:
                false_alarms += 1

    print("\n" + "=" * 50)
    print(f"SUMMARY for {PATIENT_ID}:")
    print(f"  Real seizure windows caught by model: {caught}")
    print(f"  Real seizure windows missed:          {missed}")
    print(f"  False alarms (normal flagged as High): {false_alarms}")


if __name__ == "__main__":
    run_demo()
