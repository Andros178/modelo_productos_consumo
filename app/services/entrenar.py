import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from app.services.modelo import TransformerModel
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def entrenar(scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device):
    print(f"Shape de X_train_tensor: {X_train_tensor.shape}")
    print(f"Shape de y_train_tensor: {y_train_tensor.shape}")

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instanciar Modelo
    input_dim = X_train_tensor.shape[2]
    model = TransformerModel(input_dim=input_dim, d_model=640, nhead=40, num_layers=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    epochs = 2000

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

    torch.save(model.state_dict(), '/home/usco/Documents/modelo_productos_consumo/modelo_inventario/modelo/Modelo.pth')
    joblib.dump(scaler_x, '/home/usco/Documents/modelo_productos_consumo/modelo_inventario/scaler/scaler_x.pkl')  # Guardar scaler_x
    joblib.dump(scaler_y, '/home/usco/Documents/modelo_productos_consumo/modelo_inventario/scaler/scaler_y.pkl')  # Guardar scaler_y

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

    # Convertir las predicciones finales a una lista serializable
    preds_inv = preds_inv.tolist()  # Convertir a lista
    trues_inv = trues_inv.tolist()  # Convertir a lista

    return {"predicciones": preds_inv, "valores_reales": trues_inv, "rmse": rmse}
