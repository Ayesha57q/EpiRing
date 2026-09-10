
EpiRing — AI-Powered Multimodal Wearable for Early Seizure-Risk Warning

**Smart India Hackathon 2026 | Team NeuralNests | Team ID 342**

A ring-form wearable that predicts seizure risk in real time using personalized, multimodal sensor data — giving people with epilepsy and their caregivers an early warning before a seizure happens.

---

## What it does

EpiRing continuously monitors heart activity, movement, and rotation through finger-worn sensors, and uses an AI model to classify risk as **Normal**, **Elevated**, or **High**. A High-risk prediction triggers a ring vibration and a connected app alert.

---

## Team & responsibilities

| Member | Role |
|---|---|
| AI/ML (this repo section: `ai-model/`) | Data collection, feature engineering, model training |
| Hardware (`hardware/`) | Ring circuit design, sensor integration, ESP32 firmware |
| App/Backend (`app/`) | Flask backend, live prediction serving, alert delivery |

---

## Parameters used

| Parameter | Sensor | Role |
|---|---|---|
| Heart Rate (HR) | MAX30102 | Heart-rate changes |
| Heart Rate Variability (HRV) | MAX30102 | Autonomic nervous system changes |
| Movement | MPU6050 accelerometer | Abnormal movement patterns |
| Rotation | MPU6050 gyroscope | Sudden/rhythmic movement |
| EDA *(planned, not yet trained)* | EDA sensor | Sympathetic activity |
| Temperature *(planned, not yet trained)* | Temp sensor | Deviation from personal baseline |

---

## How the model works

```
Sense -> Window (5s) -> Extract features -> Compare to baseline -> Classify -> Alert
```

1. Raw sensor data is grouped into 5-second windows.
2. Each window is converted into 10 numeric features (heart rate stats, movement energy, rotation energy, etc.) — not fed to the model as raw signal.
3. A Random Forest classifier predicts Normal / Elevated / High risk per window.
4. A High prediction triggers the alert pipeline.

---

## Data

Trained on real patient data from **SeizeIT2** (OpenNeuro, accession ds005873) — a multicenter clinical epilepsy monitoring dataset.

- 5 real patients used
- ECG signal → HR + HRV features
- Accelerometer + gyroscope → Movement + Rotation features
- Real seizure onset/duration timestamps used for labeling

---

## Results

- **53,796** total 5-second windows evaluated (5 patients combined)
- **130** real seizure windows identified
- **Precision: 71%** — when the model flags High risk, it's correct ~71% of the time
- **Recall: 15%** — the model currently catches a portion of real seizure windows; improving this is the top priority for future iterations
- Near-zero false alarm rate on normal activity

### Honest limitations
- 2 of 5 patients had a motion-sensor recording gap shorter than their ECG recording, reducing usable seizure examples from those patients
- EDA and Temperature are part of the hardware design but have no matching public dataset yet — currently untrained, planned as future work
- Small overall seizure sample size (130 windows) — more patients would strengthen reliability

---

## Repository structure

```
EpiRing/
├── ai-model/          # Training scripts, feature extraction, trained model
├── app/               # Flask backend, live prediction API
├── hardware/           # Firmware, circuit design
└── README.md
```

---

## Future work

- Add SpO2 (same PPG sensor, near-zero extra hardware cost)
- Train EDA and Temperature once a matching wearable dataset is available
- Expand training to more patients
- Add an "Elevated" pre-seizure buffer window, not just Normal/High
- On-device (edge) inference for lower latency

---

## Disclaimer

EpiRing is an assistive early-warning device. It does not replace clinical diagnosis or EEG monitoring.
 

