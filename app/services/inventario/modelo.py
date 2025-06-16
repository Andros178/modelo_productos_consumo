# modelo.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib

def modelar(df,seq_len,features, target):

    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)


    
    # Dataset


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    le = LabelEncoder()
    print(f"Tipo de objeto en df['Product ID']: {type(df['Product ID'])}")
    print(f"Dispositivo: {device}")

    df['Product_encoded'] = le.fit_transform(df['Product ID'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].map(lambda x: x.timestamp() if pd.notnull(x) else 0)
    

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[[target]])  # ← CORREGIDO

    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)

    
    
    train_size = int(len(df) * 0.82)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

    if len(X_train) <= seq_len or len(X_test) <= seq_len:
        raise ValueError(f"El tamaño de los datos de entrenamiento ({len(X_train)}) o test ({len(X_test)}) es menor o igual a seq_len ({seq_len}). Reduce seq_len o usa más datos.", print(df.head(30)))    
    
    # Crear secuencias manualmente
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)

    print("X_train_seq shape:", X_train_seq.shape)
    print("y_train_seq shape:", y_train_seq.shape)


    print("X_train_seq shape:", X_train_seq.shape)
    print("y_train_seq shape:", y_train_seq.shape)
    print("X_test_seq shape:", X_test_seq.shape)
    print("y_test_seq shape:", y_test_seq.shape)


    if X_train_seq.shape[0] != y_train_seq.shape[0]:
        raise ValueError(f"X_train_seq y y_train_seq tienen diferente número de muestras: {X_train_seq.shape[0]} vs {y_train_seq.shape[0]}")
    if X_test_seq.shape[0] != y_test_seq.shape[0]:
        raise ValueError(f"X_test_seq y y_test_seq tienen diferente número de muestras: {X_test_seq.shape[0]} vs {y_test_seq.shape[0]}")

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

    
    #print(f"[modelo.py] Modelado completado. Tensores creados.")

    # (Opcional) Función para graficar la serie temporal
    def plot_delta(data):
        plt.plot(data)
        plt.ylabel('Delta')
        plt.show()
    joblib.dump(le, "/home/usco/Documents/modelo_productos_consumo/modelo_inventario/scaler/label_encoder.pkl")

    # (Opcional) Función para crear secuencias manualmente (no necesaria si usas TimeseriesGenerator)
    return  scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device, df

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(500, d_model))  # hasta 500 pasos
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        seq_len = x.shape[1]
        x = x + self.positional_encoding[:seq_len]
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out
    

__all__ = [
    "TransformerModel", "seq_len", "features", "target",
    "scaler_x", "scaler_y", "X_train_tensor", "y_train_tensor", "X_test_tensor", "y_test_tensor", "device", "df"
]