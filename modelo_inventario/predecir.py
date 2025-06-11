import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from modelo import TransformerModel
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def predecir_futuro_producto(frecuencia, paso, producto_id, df, fecha_inicio, seq_len, features,target):
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(features)
    model = TransformerModel(input_dim=input_dim, d_model=64, nhead=4, num_layers=2).to(device)
    # Cargar pesos entrenados
    model.load_state_dict(torch.load("/home/usco/Documents/modelo_productos_consumo/modelo_inventario/modelo/Modelo.pth", map_location=device))
    model.eval()

    scaler_x = joblib.load('/home/usco/Documents/modelo_productos_consumo/modelo_inventario/scaler/scaler_x.pkl')
    scaler_y = joblib.load('/home/usco/Documents/modelo_productos_consumo/modelo_inventario/scaler/scaler_y.pkl')

    # Codificar producto
    df['Product_encoded'] = le.fit_transform(df['Product ID'])
    producto_mask = df['Product ID'] == producto_id
    if not producto_mask.any():
        print(f"Producto {producto_id} no encontrado.")
        return np.array([])

    prod_encoded = df.loc[producto_mask, 'Product_encoded'].iloc[0]
    df_producto = df[df['Product_encoded'] == prod_encoded].copy()
    df_producto = df_producto.sort_values('Date')

    # Convertir Date a timestamp si es string
    if df_producto['Date'].dtype == object or str(df_producto['Date'].dtype).startswith('datetime'):
        df_producto['Date'] = pd.to_datetime(df_producto['Date'], errors='coerce')
        df_producto['Date'] = df_producto['Date'].map(lambda x: x.timestamp() if pd.notnull(x) else 0)

    # Preparar secuencia inicial
    X_hist = df_producto[features].values
    X_hist_scaled = scaler_x.transform(X_hist)
    secuencia = X_hist_scaled[-seq_len:]  # Última secuencia

    # Fechas futuras
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fechas_futuras = pd.date_range(start=fecha_inicio_dt + pd.Timedelta(days=1), periods=paso, freq=frecuencia)

    predicciones = []
    for fecha in fechas_futuras:
        nueva_fila = df_producto.iloc[-1].copy()
        nueva_fila['Date'] = fecha.timestamp()
        nueva_fila[target] = 0  # Si es feature
        X_nueva = pd.DataFrame([nueva_fila[features].values], columns=features)
        X_nueva_scaled = scaler_x.transform(X_nueva)
        secuencia = np.vstack([secuencia[1:], X_nueva_scaled])
        input_modelo = torch.tensor(secuencia[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(input_modelo).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
        predicciones.append(pred)

    return np.array(predicciones), fechas_futuras

def graficar_prediccion(frecuencia, paso, producto_id, df, predicciones, fecha_inicio, seq_len, features,target):
    import pandas as pd
    from datetime import timedelta

    if predicciones is None or len(predicciones) == 0:
        print("No hay predicciones para graficar.")
        return

    # Obtener el último Inventory Level real del producto
    df_producto = df[df['Product ID'] == producto_id].copy()
    df_producto = df_producto.sort_values('Date')

    # Convertir Date a timestamp si es string
    if df_producto['Date'].dtype == object or str(df_producto['Date'].dtype).startswith('datetime'):
        df_producto['Date'] = pd.to_datetime(df_producto['Date'], errors='coerce')
        df_producto['Date'] = df_producto['Date'].map(lambda x: x.timestamp() if pd.notnull(x) else 0)

    tolerancia = 1
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fecha_ts = fecha_inicio_dt.timestamp()
    if df['Date'].dtype == object or str(df['Date'].dtype).startswith('datetime'):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Date'] = df['Date'].map(lambda x: x.timestamp() if pd.notnull(x) else 0)
    mask = (
        (df['Product ID'] == producto_id) &
        (df['Date'] >= fecha_ts - tolerancia) &
        (df['Date'] <= fecha_ts + tolerancia)
    )
    ultimo_stock = df.loc[mask, 'stock']
    if not ultimo_stock.empty:
        ultimo_stock = float(ultimo_stock.iloc[0])
    else:
        ultimo_stock = 0.0

    # Fechas futuras para graficar
    fechas_futuras = pd.date_range(start=fecha_inicio_dt, periods=paso, freq=frecuencia)

    # Calcular la evolución del inventario restando las predicciones
    inventario = [ultimo_stock]
    necesidad = []
    umbral = 10
    notificado = False
    alerta_fecha = None
    print("Tipo de predicciones:", type(predicciones))
    print("Ejemplo de predicciones:", predicciones[:5])
    for fecha, pred in zip(fechas_futuras, predicciones):
    # Solo procesa si pred es un número
        if isinstance(pred, (np.ndarray, list, pd.Series)):
            pred_val = float(np.ravel(pred)[0])
        elif isinstance(pred, (int, float, np.integer, np.floating)):
            pred_val = float(pred)
        else:
            print(f"Predicción inválida para la fecha {fecha}: {pred} (tipo {type(pred)})")
            continue  # Salta este ciclo si no es válido
        nuevo_inv = float(inventario[-1]) - pred_val
        sobrante = nuevo_inv - umbral
        inventario.append(nuevo_inv)
        if sobrante > 0:
            necesidad.append(0)
        else:
            necesidad.append(sobrante)
        if not notificado and nuevo_inv <= umbral:
            print(f"¡ALERTA! El {fecha.strftime('%Y-%m-%d')} el inventario proyectado baja al umbral ({nuevo_inv:.0f} unidades). ¡Debe reabastecer!")
            alerta_fecha = fecha
            notificado = True

    # Para graficar, las fechas deben tener la misma longitud que inventario
    fechas_graf = [fechas_futuras[0] - pd.Timedelta(days=1)] + list(fechas_futuras)

    plt.figure(figsize=(12, 6))
    plt.bar(fechas_graf, inventario, color='orange', alpha=0.6, label=f'{target} Level proyectado')
    plt.title(f'Inventario proyectado - Producto: {producto_id}')
    plt.xlabel('Fecha')
    plt.ylabel('Unidades')
    plt.grid(True)
    plt.legend()
    for fecha, valor in zip(fechas_graf, inventario):
        plt.text(fecha, valor, f'{valor:.0f}', ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')
    if alerta_fecha is not None:
        plt.axvline(alerta_fecha, color='red', linestyle='--', linewidth=2, label='Fecha de alerta')
        plt.legend()
    plt.show()

    # Gráfica de predicción de ventas y necesidad
    plt.figure(figsize=(12, 6))
    plt.bar(fechas_futuras, predicciones, color='blue', alpha=0.6, label=f'Predicción {target}')
    plt.bar(fechas_futuras, necesidad, color='red', alpha=0.4, label='Necesidad (faltante)', bottom=0)
    plt.title(f'Predicción de ventas y necesidad - Producto: {producto_id}')
    plt.xlabel('Fecha')
    plt.ylabel('Unidades')
    plt.grid(True)
    plt.legend()
    for fecha, val_pred, val_nec in zip(fechas_futuras, predicciones, necesidad):
        plt.text(fecha, val_pred, f'{val_pred:.0f}', ha='center', va='bottom', fontsize=9, color='blue', fontweight='bold')
        if val_nec != 0:
            plt.text(fecha, val_nec, f'{val_nec:.0f}', ha='center', va='top', fontsize=9, color='red', fontweight='bold')
    if alerta_fecha is not None:
        plt.axvline(alerta_fecha, color='red', linestyle='--', linewidth=2, label='Fecha de alerta')
        plt.legend()
    plt.show()