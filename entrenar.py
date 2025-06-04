# entrenamiento.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from modelo import TransformerModel, seq_len, features, target, scaler_x, scaler_y, create_sequences, device, X_tensor, y_tensor, df
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Parámetros
batch_size = 8
epochs = 2
learning_rate = 0.001

# Dataset y DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modelo
model = TransformerModel(input_dim=len(features)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

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
        del xb, yb, pred
        torch.cuda.empty_cache()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Guardar modelo
torch.save(model.state_dict(), 'transformer_model.pth')

# Función de predicción para un producto específico
def predecir_producto(producto_id: str):
    from modelo import TransformerModel, seq_len, features, target, scaler_x, scaler_y, create_sequences, device, df

    # Codificación de productos
    le = LabelEncoder()
    df['Product_encoded'] = le.fit_transform(df['Product ID'])

    if producto_id not in df['Product ID'].unique():
        print(f"Producto '{producto_id}' no encontrado en el dataset.")
        return None, None

    # Filtrar y preparar
    df_producto = df[df['Product ID'] == producto_id].copy()
    df_producto['Product_encoded'] = le.transform(df_producto['Product ID'])

    X_scaled = scaler_x.transform(df_producto[features])
    y_scaled = scaler_y.transform(df_producto[target])
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)

    if len(X_seq) == 0:
        print(f"No hay suficientes datos para el producto '{producto_id}'.")
        return None, None

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device)

    # Cargar modelo
    model = TransformerModel(input_dim=len(features)).to(device)
    model.load_state_dict(torch.load('transformer_model.pth', map_location=device))
    model.eval()

    # Predicción
    preds, trues = [], []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i+batch_size]
            yb = y_tensor[i:i+batch_size]
            output = model(xb).cpu().numpy()
            preds.append(output)
            trues.append(yb.cpu().numpy())
            del xb, yb
            torch.cuda.empty_cache()

    predictions = np.vstack(preds)
    true_vals = np.vstack(trues)

    predicted_units = scaler_y.inverse_transform(predictions)
    true_units = scaler_y.inverse_transform(true_vals)

    # Gráfico
    plt.figure(figsize=(12, 5))
    plt.plot(df_producto['Date'].values[seq_len:], true_units, label='Real')
    plt.plot(df_producto['Date'].values[seq_len:], predicted_units, label='Predicción')
    plt.title(f"Predicción de demanda - Producto {producto_id}")
    plt.xlabel("Fecha")
    plt.ylabel("Inventario estimado")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return predicted_units, true_units

# Ejemplo de uso
if __name__ == "__main__":
    predicciones, reales = predecir_producto("1")
    if predicciones is not None and reales is not None:
        print("Ejemplo de predicción vs real:")
        print("Pred:", predicciones.flatten()[:5])
        print("Real:", reales.flatten()[:5])
