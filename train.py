import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

cols = ['unit', 'cycle'] + [f'op_{i}' for i in range(1,4)] + [f's_{i}' for i in range(1,22)]

print("Loading data...")
train_df = pd.read_csv('data/CMaps/train_FD001.txt', sep='\s+', header=None, names=cols, engine='python')
train_df = train_df.dropna(axis=1)
train_df['unit'] = train_df['unit'].astype(int)
train_df['cycle'] = train_df['cycle'].astype(int)

print("Computing RUL...")
max_cycles = train_df.groupby('unit')['cycle'].max()
train_df['RUL'] = train_df.apply(lambda row: max_cycles[row['unit']] - row['cycle'], axis=1).astype(int)

sensor_cols = [f's_{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
sensor_cols = [c for c in sensor_cols if c in train_df.columns]

scaler = MinMaxScaler()
train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols].astype(float))

print("Training Isolation Forest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
train_df['anomaly'] = iso_forest.fit_predict(train_df[sensor_cols])
train_df['anomaly'] = train_df['anomaly'].map({1: 0, -1: 1})

print(f"Anomaly rate: {train_df['anomaly'].mean()*100:.1f}%")

train_df.to_csv('data/train_processed.csv', index=False)
joblib.dump(iso_forest, 'model_iso_forest.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Done!")
print(train_df[['unit','cycle','RUL']].head(5))