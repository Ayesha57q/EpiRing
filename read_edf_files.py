import pyedflib
import numpy as np

# ---------------------------------------------------------------
# UPDATE THIS PATH to wherever you moved your downloaded file
# ---------------------------------------------------------------
ECG_FILE_PATH = "data/sub-001_ses-01_task-szMonitoring_run-01_ecg.edf"
MOV_FILE_PATH = "data/sub-001_ses-01_task-szMonitoring_run-01_mov.edf"


def inspect_edf_file(file_path):
    """
    Opens an .edf file properly and prints what's inside it -
    run this first just to SEE what you actually downloaded.
    """
    f = pyedflib.EdfReader(file_path)

    n_signals = f.signals_in_file
    signal_labels = f.getSignalLabels()

    print(f"File: {file_path}")
    print(f"Number of signals in this file: {n_signals}")
    print(f"Signal names: {signal_labels}")
    print(f"Recording duration: {f.file_duration} seconds")

    for i in range(n_signals):
        sample_rate = f.getSampleFrequency(i)
        print(f"  - Signal '{signal_labels[i]}': sample rate = {sample_rate} Hz")

    f.close()
    return signal_labels


def load_edf_signal(file_path, signal_name):
    """
    Loads ONE specific signal (e.g. 'ECG' or 'ACC_X') from an .edf
    file as a plain numpy array of numbers.
    """
    f = pyedflib.EdfReader(file_path)
    signal_labels = f.getSignalLabels()

    if signal_name not in signal_labels:
        f.close()
        raise ValueError(f"'{signal_name}' not found. Available: {signal_labels}")

    idx = signal_labels.index(signal_name)
    data = f.readSignal(idx)
    sample_rate = f.getSampleFrequency(idx)

    f.close()
    return data, sample_rate


if __name__ == "__main__":
    print("=== Inspecting ECG file ===")
    inspect_edf_file(ECG_FILE_PATH)

    print("\n=== Inspecting MOV file ===")
    inspect_edf_file(MOV_FILE_PATH)