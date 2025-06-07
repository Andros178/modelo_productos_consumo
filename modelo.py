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


# json_data = [
#       {"Date": "2022-01-01", "Product ID": "P0001", "Inventory Level": 231, "Units Sold": 127, "Units Ordered": 55, "Demand Forecast": 135.47, "Price": 33.5},
#       {"Date": "2022-01-02", "Product ID": "P0001", "Inventory Level": 220, "Units Sold": 130, "Units Ordered": 60, "Demand Forecast": 138.70, "Price": 34.0},
#      # ... más registros
#   ]

# json_data = "" #Se obtiene con la peticion a backend

# #Convertir a DataFrame
# df_json = pd.DataFrame(json_data)

# #Crear nuevo DataFrame con columnas seleccionadas
# df = pd.DataFrame({
#      'cantidad_salida': df_json['Units Sold'],  # o 'Demand Forecast' si prefieres usar el pronóstico
#      'producto': df_json['Product ID'],
#      'stock': df_json['Inventory Level'],
#      'temporada_inicio': df_json['fechaHora_Inicio'],
#      'temporada_fin':df_json['fechaHora_Fin'],
#      'tipo_salida': df_json['tipo_salida'], # debe usarse one-hot encodding
#      })

# print(df.head())



# df = df.sort_values(by='fecha_Inicio')



def create_sequences(X, y, seq_len=14):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)



# Dataset
df = pd.read_csv('Dataset/retail_store_inventory.csv')
producto_id = "P0001"
df = df[df['Product ID'] == producto_id].copy()

le = LabelEncoder()
df['Product_encoded'] = le.fit_transform(df['Product ID'])
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Date'] = df['Date'].map(lambda x: x.timestamp() if pd.notnull(x) else 0)
df['Holiday/Promotion'] = df['Holiday/Promotion'].astype(int)
df['Discount'] = df['Discount'] / 100.0  # Si está en 0-100

features = ['Product_encoded', 'Inventory Level', 'Units Sold', 'Date', 'Price', 'Discount', 'Holiday/Promotion']
target = ['Units Sold']

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[target])
if y_scaled.ndim == 1:
    y_scaled = y_scaled.reshape(-1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 14
train_size = int(len(df) * 0.82)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# Crear secuencias manualmente
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)

X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

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

# (Opcional) Función para graficar la serie temporal
def plot_delta(data):
    plt.plot(data)
    plt.ylabel('Delta')
    plt.show()

# (Opcional) Función para crear secuencias manualmente (no necesaria si usas TimeseriesGenerator)


__all__ = [
    "TransformerModel", "seq_len", "features", "target",
    "scaler_x", "scaler_y", "X_train_tensor", "y_train_tensor", "X_test_tensor", "y_test_tensor", "device", "df"
]