# modelo.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn

# Dataset
df = pd.read_csv('Dataset/retail_store_inventory.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Codificación
le = LabelEncoder()
df['Product_encoded'] = le.fit_transform(df['Product ID'])

# Variables
features = ['Product_encoded', 'Inventory Level', 'Units Sold', 'Price']
target = ['Inventory Level']
seq_len = 14

# Escaladores
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[target])

# Secuencias
def create_sequences(X, y, seq_len=14):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)

# Tensores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device)

# Modelo Transformer
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
