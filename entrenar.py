import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from modelo import TransformerModel, seq_len, features, target, scaler_x, scaler_y, create_sequences, device, train_loader, X_tensor, y_tensor, df


model = TransformerModel(input_dim=len(features)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()



# epochs = 50
# model.train()
# for epoch in range(epochs):
#     total_loss = 0
#     for xb, yb in train_loader:
#         pred = model(xb)
#         loss = loss_fn(pred, yb)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
#     print("Saving model...")
#     torch.save(model.state_dict(), 'transformer_model.pth')

model.eval()
with torch.no_grad():
    predictions = model(X_tensor).cpu().numpy()
    true_vals = y_tensor.cpu().numpy()

# Invertir escalado
predicted_units = scaler_y.inverse_transform(predictions)
true_units = scaler_y.inverse_transform(true_vals)

# Graficar
plt.figure(figsize=(12, 5))
plt.plot(df['Date'][seq_len:], true_units, label='Real')
plt.plot(df['Date'][seq_len:], predicted_units, label='Predicción')
plt.title(f"Predicción de demanda - Producto {target}")
plt.xlabel("Fecha")
plt.ylabel("Unidades vendidas")
plt.legend()
plt.show()