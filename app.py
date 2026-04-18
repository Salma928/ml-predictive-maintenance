import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_models():
    iso_forest = joblib.load('model_iso_forest.pkl')
    scaler = joblib.load('scaler.pkl')
    return iso_forest, scaler

iso_forest, scaler = load_models()

sensor_cols = [f's_{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]

st.title("Predictive Maintenance Dashboard")
st.write("Anomaly detection on NASA CMAPSS engine sensor data")

# Load processed data
@st.cache_data
def load_data():
    return pd.read_csv('data/train_processed.csv')

df = load_data()

# Sidebar
st.sidebar.header("Select Engine")
unit = st.sidebar.selectbox("Engine unit", sorted(df['unit'].unique()))

unit_df = df[df['unit'] == unit].copy()

# Plot sensor data
st.subheader(f"Engine {unit} — Sensor readings over time")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(unit_df['cycle'], unit_df['s_7'], label='Sensor 7', color='steelblue')
anomalies = unit_df[unit_df['anomaly'] == 1]
ax.scatter(anomalies['cycle'], anomalies['s_7'], color='red', label='Anomaly', zorder=5, s=40)
ax.set_xlabel("Cycle")
ax.set_ylabel("Normalized value")
ax.legend()
st.pyplot(fig)

# RUL over time
st.subheader(f"Engine {unit} — Remaining Useful Life")
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(unit_df['cycle'], unit_df['RUL'], color='orange')
ax2.set_xlabel("Cycle")
ax2.set_ylabel("RUL (cycles)")
ax2.axhline(y=30, color='red', linestyle='--', label='Critical threshold (30 cycles)')
ax2.legend()
st.pyplot(fig2)

# Stats
col1, col2, col3 = st.columns(3)
col1.metric("Total cycles", len(unit_df))
col2.metric("Anomalies detected", int(unit_df['anomaly'].sum()))
col3.metric("Final RUL", int(unit_df['RUL'].iloc[-1]))

st.subheader("Raw sensor data")
st.dataframe(unit_df[['cycle', 'RUL', 'anomaly'] + sensor_cols[:5]].tail(10))