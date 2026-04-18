# Predictive Maintenance with NASA CMAPSS

Anomaly detection and Remaining Useful Life (RUL) estimation on NASA turbofan engine data, using Isolation Forest and an interactive Streamlit dashboard.

## Overview

This project applies unsupervised anomaly detection on the **NASA CMAPSS FD001** dataset — a benchmark dataset of turbofan engine sensor readings until failure. The model detects abnormal engine behavior and tracks RUL in real time.

## Demo

```bash
python train.py
streamlit run app.py
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| NASA CMAPSS | Turbofan engine sensor dataset |
| Isolation Forest | Unsupervised anomaly detection |
| Scikit-learn | ML pipeline & preprocessing |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | Interactive dashboard |

## Dataset

**NASA CMAPSS FD001** — 100 engines, 20,631 multivariate time series observations across 21 sensors. Each engine runs from healthy state to failure. Available on [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).

## Results

- **20,631** sensor readings across 100 engines
- **1,032 anomalies** detected (5% contamination rate)
- Anomalies consistently appear in the final cycles before engine failure

## Project Structure
├── train.py              # Training script (Isolation Forest)
├── app.py                # Streamlit dashboard
├── download_data.py      # Data download helper
└── README.md

## Installation

```bash
git clone https://github.com/Salma928/ml-predictive-maintenance.git
cd ml-predictive-maintenance
python3.11 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn streamlit torch joblib
python train.py
streamlit run app.py
```

## Author

**Salma Bentahar Alaoui** — M2 Mesure et Traitement de l'Information, Université de Lorraine  
AI/ML Engineer seeking CDI from October 2026