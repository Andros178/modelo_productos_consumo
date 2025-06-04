import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Simulamos entrada JSON (esto sería tu input real)
# json_data = [
#     {"Date": "2022-01-01", "Product ID": "P0001", "Inventory Level": 231, "Units Sold": 127, "Units Ordered": 55, "Demand Forecast": 135.47, "Price": 33.5},
#     {"Date": "2022-01-02", "Product ID": "P0001", "Inventory Level": 220, "Units Sold": 130, "Units Ordered": 60, "Demand Forecast": 138.70, "Price": 34.0},
#     # ... más registros
# ]

# # Convertir a DataFrame
# df_json = pd.DataFrame(json_data)

# # Crear nuevo DataFrame con columnas seleccionadas
# df = pd.DataFrame({
#     'demanda': df_json['Units Sold'],  # o 'Demand Forecast' si prefieres usar el pronóstico
#     'precio': df_json['Price']
# })

# print(df.head())



# df = df.sort_values(by='Date')

df = pd.read_csv('Dataset/retail_store_inventory.csv')
df['Date'] = pd.to_datetime(df['Date'])


features = ['Inventory Level', 'Units Sold', 'Price']

print(df.head(30))

target = ['Inventory Level']

scaler_x= MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[target])

def create_sequences(X, y, seq_len=14):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

seq_len = 14
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)



plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Units Sold'], label='Unidades vendidas')
plt.title(f"Demanda diaria - Producto {target}")
plt.xlabel("Fecha")
plt.ylabel("Unidades vendidas")
plt.legend()
plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device)


dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(500, d_model))  # Máximo 500 pasos
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


