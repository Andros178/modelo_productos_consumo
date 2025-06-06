import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from modelo import TransformerModel, seq_len, features, target, scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device, df
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

# Mostrar info de features [Depuracion]

print("Features:", features)
print("Primeras filas de df[features]:\n", df[features].head())

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instanciar Modelo
input_dim = X_train_tensor.shape[2]
model = TransformerModel(input_dim=input_dim, d_model=64, nhead=4, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 2
train_losses, train_rmses, test_rmses = [], [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    preds_all, trues_all = [], []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds_all.append(out.detach().cpu().numpy())
        trues_all.append(yb.detach().cpu().numpy())
    avg_loss = total_loss / len(train_loader)
    preds_all = np.vstack(preds_all)
    trues_all = np.vstack(trues_all)
    train_preds_inv = scaler_y.inverse_transform(preds_all)
    train_trues_inv = scaler_y.inverse_transform(trues_all)
    train_rmse = np.sqrt(mean_squared_error(train_trues_inv, train_preds_inv))

    # Evaluación en test
    model.eval()
    preds_test, trues_test = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            preds_test.append(output.cpu().numpy())
            trues_test.append(yb.cpu().numpy())
    preds_test = np.vstack(preds_test)
    trues_test = np.vstack(trues_test)
    preds_test_inv = scaler_y.inverse_transform(preds_test)
    trues_test_inv = scaler_y.inverse_transform(trues_test)
    test_rmse = np.sqrt(mean_squared_error(trues_test_inv, preds_test_inv))

    train_losses.append(avg_loss)
    train_rmses.append(train_rmse)
    test_rmses.append(test_rmse)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")

# Graficar métricas
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, marker='o', color='blue')
plt.title('Train Loss por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 3, 2)
plt.plot(epochs_range, train_rmses, marker='o', color='green')
plt.title('Train RMSE por Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.subplot(1, 3, 3)
plt.plot(epochs_range, test_rmses, marker='o', color='red')
plt.title('Test RMSE por Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()

torch.save(model.state_dict(), 'transformer_model.pth')

# Evaluación final en test
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        if xb.ndim < 3 or xb.shape[0] == 0 or xb.shape[1] == 0 or xb.shape[2] == 0:
            continue
        xb, yb = xb.to(device), yb.to(device)
        output = model(xb)
        preds.append(output.cpu().numpy())
        trues.append(yb.cpu().numpy())

preds = np.vstack(preds)
trues = np.vstack(trues)
preds_inv = scaler_y.inverse_transform(preds)
trues_inv = scaler_y.inverse_transform(trues)
rmse = np.sqrt(mean_squared_error(trues_inv, preds_inv))
print(f"Test RMSE: {rmse:.2f}")

def predecir_futuro_producto(producto_id, fecha_inicio, pasos, frecuencia):
    producto_mask = df['Product ID'] == producto_id
    if not producto_mask.any():
        print(f"Producto {producto_id} no encontrado.")
        return np.array([]), None
    prod_encoded = df.loc[producto_mask, 'Product_encoded'].iloc[0]
    df_producto = df[df['Product_encoded'] == prod_encoded].copy()
    df_producto = df_producto.sort_values('Date')

    X_hist = df_producto[features].values
    X_hist_scaled = scaler_x.transform(X_hist)
    secuencia = X_hist_scaled[-seq_len:] 

    import pandas as pd
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fechas_futuras = pd.date_range(start=fecha_inicio_dt, periods=pasos, freq=frecuencia)

    predicciones = []
    for fecha in fechas_futuras:
        nueva_fila = df_producto.iloc[-1].copy()
        nueva_fila['Date'] = fecha.timestamp()
        nueva_fila['Units Sold'] = 0
        X_nueva = pd.DataFrame([nueva_fila[features].values], columns=features)
        X_nueva_scaled = scaler_x.transform(X_nueva)
        secuencia = np.vstack([secuencia[1:], X_nueva_scaled])
        input_modelo = torch.tensor(secuencia[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(input_modelo).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
        predicciones.append(pred)

    return np.array(predicciones), None

from modelo import producto_id
if __name__ == "__main__":
    
    producto_id = producto_id
    fecha_inicio = "2023-12-25"
    paso = 7
    frecuencia = "D"
    predicciones, _ = predecir_futuro_producto(producto_id, fecha_inicio, paso, frecuencia)
    if predicciones is not None:
        print("Pred:", predicciones.flatten()[:5])

        # Obtener el último Inventory Level real del producto
        df_producto = df[df['Product ID'] == producto_id].copy()
        df_producto = df_producto.sort_values('Date')
        ultimo_inventory = df_producto['Inventory Level'].iloc[-1]

        fechas_futuras = pd.date_range(start=fecha_inicio, periods=paso, freq=frecuencia)

        # Calcular la evolución del inventario restando las predicciones
        inventario = [ultimo_inventory]
        necesidad = []
        umbral = 10
        notificado = False
        alerta_fecha = None  # Guardar la fecha de la alerta

        inventario = [ultimo_inventory]
        necesidad = []
        for fecha, pred in zip(fechas_futuras, predicciones):
            nuevo_inv = inventario[-1] - pred
            inventario.append(nuevo_inv)
            necesidad.append(max(0, -nuevo_inv))
            if not notificado and nuevo_inv <= umbral:
                print(f"¡ALERTA! El {fecha.strftime('%Y-%m-%d')} el inventario proyectado baja al umbral ({nuevo_inv:.0f} unidades). ¡Debe reabastecer!")
                alerta_fecha = fecha
                notificado = True

        inventario = inventario[1:]  # Quitar el inventario inicial para alinear con fechas

        # Graficar
        plt.figure(figsize=(12, 6))
        bars = plt.bar(fechas_futuras, inventario, color='orange', alpha=0.6, label='Inventory Level proyectado')
        line, = plt.plot(fechas_futuras, predicciones, marker='o', color='blue', label='Predicción Units Sold')
        bars_nec = plt.bar(fechas_futuras, necesidad, color='red', alpha=0.4, label='Necesidad (faltante)', bottom=[min(0, inv) for inv in inventario])
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title(f'Predicción de ventas futuras y trazabilidad de inventario\nProducto: {producto_id}')
        plt.xlabel('Fecha')
        plt.ylabel('Unidades')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Añadir valores numéricos sobre las barras de inventario
        for bar, valor in zip(bars, inventario):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{valor:.0f}', 
                    ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

        # Añadir valores numéricos sobre las barras de necesidad (si hay necesidad)
        for bar, valor in zip(bars_nec, necesidad):
            if valor > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_y(), f'{valor:.0f}',
                        ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

        # Añadir valores numéricos sobre los puntos de predicción de ventas
        for x, y in zip(fechas_futuras, predicciones):
            plt.text(x, y, f'{y:.0f}', ha='center', va='bottom', fontsize=9, color='blue', fontweight='bold')

        # Resaltar la fecha de alerta en la gráfica
        if alerta_fecha is not None:
            plt.axvline(alerta_fecha, color='red', linestyle='--', linewidth=2, label='Fecha de alerta')
            plt.legend()

        plt.show()