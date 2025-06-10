import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modelo import TransformerModel, features, scaler_x, scaler_y, seq_len, df, device

# Carga del modelo
input_dim = len(features)
model = TransformerModel(input_dim=input_dim, d_model=64, nhead=4, num_layers=2).to(device)
model.load_state_dict(torch.load('Modelo.pth', map_location=device))
model.eval()

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
    fechas_futuras = pd.date_range(start=fecha_inicio_dt+pd.Timedelta(days=1), periods=pasos, freq=frecuencia)

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
    from datetime import datetime, timedelta
    producto_id = "P0001"
    fecha_inicio = "2022-01-01"
    fecha_inicio = pd.to_datetime(fecha_inicio)
    paso = 15
    frecuencia = "D"
    
    
    
    if frecuencia == "D":
        predicciones, _ = predecir_futuro_producto(producto_id, fecha_inicio+timedelta(days=1), paso, frecuencia)
    if frecuencia == "M":
        predicciones, _ = predecir_futuro_producto(producto_id, fecha_inicio+timedelta(months=1), paso, frecuencia)
    if frecuencia == "Y":
        predicciones, _ = predecir_futuro_producto(producto_id, fecha_inicio+timedelta(years=1), paso, frecuencia)
    if predicciones is not None:
        print("Pred:", predicciones.flatten()[:5])

        # Obtener el último Inventory Level real del producto
        df_producto = df[df['Product ID'] == producto_id].copy()
        df_producto = df_producto.sort_values('Date')
        tolerancia = 1
        fecha_ts = fecha_inicio.timestamp()
        
        mask = ( (df['Product ID'] == producto_id) & (df['Date'] >= fecha_ts - tolerancia) & (df['Date'] <= fecha_ts + tolerancia)
        )
        
        ultimo_inventory = df.loc[mask, 'Inventory Level']

        if not ultimo_inventory.empty:
            ultimo_inventory = ultimo_inventory.iloc[0]

        fechas_futuras = pd.date_range(start=fecha_inicio, periods=paso, freq=frecuencia)

        # Calcular la evolución del inventario restando las predicciones
        inventario = [ultimo_inventory]
        necesidad = []
        umbral = 10
        notificado = False
        alerta_fecha = None  

        inventario = [ultimo_inventory]
        necesidad = []
        for fecha, pred in zip(fechas_futuras, predicciones):
            nuevo_inv = inventario[-1] - pred
            inventario.append(nuevo_inv)
            sobrante = nuevo_inv - umbral
            if sobrante > 0:
                necesidad.append(0)
            else:
                necesidad.append(sobrante)

            
            if not notificado and nuevo_inv <= umbral:
                print(f"¡ALERTA! El {fecha.strftime('%Y-%m-%d')} el inventario proyectado baja al umbral ({nuevo_inv:.0f} unidades). ¡Debe reabastecer!")
                alerta_fecha = fecha
                notificado = True
        import time
        import datetime
        
        

        print("inventario nuevo",nuevo_inv.shape)

        print("necesidad len", len(necesidad))

        print("inventario", len(inventario))
        


        plt.figure(figsize=(12, 6))
        plt.bar(fechas_futuras.insert(0, fechas_futuras[0] - pd.Timedelta(days=1)), inventario, color='orange', alpha=0.6, label='Inventory Level proyectado')
        plt.title(f'Inventario proyectado - Producto: {producto_id}')
        plt.xlabel('Fecha')
        plt.ylabel('Unidades')
        plt.grid(True)
        plt.legend()

        
        for fecha, valor in zip(fechas_futuras.insert(0, fechas_futuras[0] - pd.Timedelta(days=1)), inventario):
            plt.text(fecha, valor, f'{valor:.0f}', ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

        plt.show()




        plt.figure(figsize=(12, 6))

        # Barras de predicciones (Units Sold)
        plt.bar(fechas_futuras, predicciones, color='blue', alpha=0.6, label='Predicción Units Sold')

        # Barras de necesidad (faltante)
        plt.bar(fechas_futuras, necesidad, color='red', alpha=0.4, label='Necesidad (faltante)', bottom=0)

        plt.title(f'Predicción de ventas y necesidad - Producto: {producto_id}')
        plt.xlabel('Fecha')
        plt.ylabel('Unidades')
        plt.grid(True)
        plt.legend()

        # Añadir valores numéricos sobre las barras de predicción y necesidad
        for fecha, val_pred, val_nec in zip(fechas_futuras, predicciones, necesidad):
            plt.text(fecha, val_pred, f'{val_pred:.0f}', ha='center', va='bottom', fontsize=9, color='blue', fontweight='bold')
            if val_nec != 0:  # Mostrar solo si necesidad es negativa (déficit)
                plt.text(fecha, val_nec, f'{val_nec:.0f}', ha='center', va='top', fontsize=9, color='red', fontweight='bold')
        
        
        if alerta_fecha is not None:
            plt.axvline(alerta_fecha, color='red', linestyle='--', linewidth=2, label='Fecha de alerta')
            plt.legend()

        plt.show()
