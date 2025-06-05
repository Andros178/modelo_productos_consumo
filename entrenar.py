# entrenamiento.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from modelo import(TransformerModel,seq_len,features,target,scaler_x,scaler_y,X_tensor,y_tensor,device,train_data_gen,test_data_gen,df)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder



xb, yb = train_data_gen[0]
print("Shape de xb:", xb.shape)
print("Features:", features)
print("Primeras filas de df[features]:\n", df[features].head())

X_train_batches = []
y_train_batches = []
for xb, yb in train_data_gen:
    X_train_batches.append(xb)
    y_train_batches.append(yb)
X_train_tensor = torch.tensor(np.concatenate(X_train_batches), dtype=torch.float32)
y_train_tensor = torch.tensor(np.concatenate(y_train_batches), dtype=torch.float32)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Inicializar el modelo
input_dim = X_train_tensor.shape[2]
model = TransformerModel(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 2


for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        output = model(xb)
        loss = loss_fn(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluación en test
model.eval()

preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_data_gen:
        xb = torch.tensor(xb, dtype=torch.float32).to(device)
        yb = torch.tensor(yb, dtype=torch.float32).to(device)
        output = model(xb)
        preds.append(output.cpu().numpy())
        trues.append(yb.cpu().numpy())

preds = np.vstack(preds)
trues = np.vstack(trues)

# Invertir el escalado para obtener valores reales
preds_inv = scaler_y.inverse_transform(preds)
trues_inv = scaler_y.inverse_transform(trues)

rmse = np.sqrt(mean_squared_error(trues_inv, preds_inv))
print(f"Test RMSE: {rmse:.2f}")

torch.save(model.state_dict(), 'transformer_model.pth')


def predecir_futuro_producto(
    producto_id: str,
    fecha_inicio: str,
    pasos: int,
    frecuencia: str = "D"  # "D"=día, "H"=hora, "W"=semana, "M"=mes
):
    
    from modelo import df, features, scaler_x, scaler_y, seq_len
    import pandas as pd

    # Filtrar el historial del producto
    df_producto = df[df['Product_encoded'] == df[df['Product ID'] == producto_id]['Product_encoded'].iloc[0]].copy()
    df_producto = df_producto.sort_values('Date')

    # Tomar la última secuencia conocida
    X_hist = df_producto[features].values
    X_hist_scaled = scaler_x.transform(X_hist)
    secuencia = X_hist_scaled[-seq_len:]  # [seq_len, features]

    # Generar fechas futuras
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fechas_futuras = pd.date_range(start=fecha_inicio_dt, periods=pasos, freq=frecuencia)

    predicciones = []
    for fecha in fechas_futuras:
        # Crear input para el modelo
        nueva_fila = df_producto.iloc[-1].copy()
        nueva_fila['Date'] = fecha.timestamp()
        # Puedes ajustar aquí otras features si tienes lógica para ello (ej: precio futuro)
        nueva_fila['Demand Forecast'] = 0  # Placeholder, no se usa como input real

        # Codificar y escalar la nueva fila
        X_nueva = nueva_fila[features].values.reshape(1, -1)
        X_nueva_scaled = scaler_x.transform(X_nueva)

        # Construir la secuencia para el modelo
        secuencia = np.vstack([secuencia[1:], X_nueva_scaled])  # Desplazar ventana
        input_modelo = torch.tensor(secuencia[np.newaxis, :, :], dtype=torch.float32).to(device)  # [1, seq_len, features]

        # Predecir
        with torch.no_grad():
            pred_scaled = model(input_modelo).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
        predicciones.append((fecha, pred))

        # Actualizar la secuencia con la predicción (si quieres usar la predicción como input)
        secuencia[-1, features.index('Demand Forecast')] = scaler_y.transform([[pred]])[0, 0]

    # Mostrar resultados
    for fecha, pred in predicciones:
        print(f"{fecha.strftime('%Y-%m-%d')}: Predicción demanda = {pred:.2f}")

    return predicciones