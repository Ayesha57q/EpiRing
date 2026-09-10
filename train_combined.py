"""
EpiRing - Training on ALL 5 real patients combined
========================================================================
Same logic as train_real_data.py, but now loops through 5 patients
and combines all their windows into one bigger, more trustworthy
training set.
"""

import numpy as np
import pandas as pd
import pyedflib
import joblib
from scipy import signal as sp_signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

WINDOW_SECONDS = 5

# List every patient you downloaded here
PATIENT_IDS = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"]


def load_signal(file_path, signal_name):
    f = pyedflib.EdfReader(file_path)
    labels = f.getSignalLabels()
    idx = labels.index(signal_name)
    data = f.readSignal(idx)
    fs = f.getSampleFrequency(idx)
    f.close()
    return data, fs


def load_patient_data(patient_id):
    ecg_file = f"data/{patient_id}_ses-01_task-szMonitoring_run-01_ecg.edf"
    mov_file = f"data/{patient_id}_ses-01_task-szMonitoring_run-01_mov.edf"
    events_file = f"data/{patient_id}_ses-01_task-szMonitoring_run-01_events.tsv"

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

    return {
        "ecg": ecg, "ecg_fs": ecg_fs,
        "accel": accel, "accel_fs": acc_fs,
        "gyro": gyro,
        "events": events,
    }


def build_labels(events_df, n_windows, window_seconds=WINDOW_SECONDS):
    """
    Marks each window as High-risk ONLY for rows that are real seizures.
    This dataset uses several detailed seizure type codes (e.g.
    'sz_foc_a_nm', 'sz_foc_ia_m_hyperkinetic', 'impd') - all seizure
    codes start with 'sz' or are known short codes below. 'bckg'
    (background/normal) rows must always be skipped, or the whole
    recording gets wrongly labeled as one giant seizure.
    """
    labels = np.zeros(n_windows, dtype=int)

    KNOWN_SEIZURE_SHORT_CODES = {"impd", "fa", "fia", "fbtc"}

    for _, row in events_df.iterrows():
        event_type = str(row.get("eventType", "")).strip().lower()

        is_seizure = event_type.startswith("sz") or event_type in KNOWN_SEIZURE_SHORT_CODES

        if not is_seizure:
            if event_type != "bckg":
                print(f"  (note: unrecognized eventType '{event_type}' - skipped, treated as non-seizure)")
            continue

        onset_sec = row["onset"]
        duration_sec = row["duration"]
        end_sec = onset_sec + duration_sec
        start_window = max(0, int(onset_sec // window_seconds))
        end_window = min(n_windows, int(end_sec // window_seconds) + 1)
        labels[start_window:end_window] = 2

    return labels


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


def build_features_for_patient(patient_id):
    """Returns a feature DataFrame + labels for ONE patient."""
    d = load_patient_data(patient_id)
    ecg, ecg_fs = d["ecg"], d["ecg_fs"]
    accel, accel_fs = d["accel"], d["accel_fs"]
    gyro = d["gyro"]

    ecg_window_size = int(ecg_fs * WINDOW_SECONDS)
    accel_window_size = int(accel_fs * WINDOW_SECONDS)

    n_windows = min(len(ecg) // ecg_window_size, len(accel) // accel_window_size)
    labels = build_labels(d["events"], n_windows)

    rows = []
    for i in range(n_windows):
        e_s, e_e = i * ecg_window_size, (i + 1) * ecg_window_size
        a_s, a_e = i * accel_window_size, (i + 1) * accel_window_size

        feats = []
        feats += ppg_features(ecg[e_s:e_e], ecg_fs)
        feats += movement_features(accel[a_s:a_e])
        feats += rotation_features(gyro[a_s:a_e])
        rows.append(feats)

    columns = [
        "hr_mean", "hr_std", "hr_bpm", "hrv_ms",
        "movement_energy", "movement_mean", "movement_jerkiness",
        "rotation_energy", "rotation_mean", "rotation_jerkiness",
    ]
    X = pd.DataFrame(rows, columns=columns)
    y = labels
    return X, y


def build_combined_dataset():
    """Loops through every patient, stacks all their windows into one big dataset."""
    all_X = []
    all_y = []

    for patient_id in PATIENT_IDS:
        try:
            X, y = build_features_for_patient(patient_id)
            all_X.append(X)
            all_y.append(y)
            n_high = (y == 2).sum()
            print(f"{patient_id}: {len(y)} windows, {n_high} High-risk windows")
        except Exception as e:
            print(f"Skipping {patient_id} due to error: {e}")

    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = np.concatenate(all_y)
    return X_combined, y_combined


def train_on_combined_data():
    X, y = build_combined_dataset()
    print(f"\nCombined dataset: {X.shape[0]} windows total, {X.shape[1]} features")
    print(f"Label counts -> Normal: {(y==0).sum()}, High: {(y==2).sum()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification report (0=Normal, 2=High):")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nFeature importance:")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances)

    joblib.dump(model, "epiring_model_final.pkl")
    print("\nSaved final trained model to epiring_model_final.pkl")

    return model


if __name__ == "__main__":
    train_on_combined_data()
