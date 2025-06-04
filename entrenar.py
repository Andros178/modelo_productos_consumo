import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from modelo import TransformerModel, seq_len, features, target, scaler_x, scaler_y, create_sequences, device, X_tensor, y_tensor, df


# Parámetros ajustados
batch_size = 8  # puedes bajar a 8 si aún hay problemas
epochs = 2
learning_rate = 0.001

# Dataset y DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modelo
model = TransformerModel(input_dim=len(features)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Entrenamiento
model.train()
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        del xb, yb, pred  # liberar memoria intermedia
        torch.cuda.empty_cache()  # limpieza GPU si aplica

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Guardar modelo
print("Saving model...")
torch.save(model.state_dict(), 'transformer_model.pth')

# Evaluación
model.eval()
preds = []
trues = []

eval_loader = DataLoader(dataset, batch_size=batch_size)
with torch.no_grad():
    for xb, yb in eval_loader:
        xb = xb.to(device)
        output = model(xb).cpu().numpy()
        preds.append(output)
        trues.append(yb.cpu().numpy())
        del xb, yb
        torch.cuda.empty_cache()

# Unir resultados
predictions = np.vstack(preds)
true_vals = np.vstack(trues)

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
plt.tight_layout()
plt.show()
