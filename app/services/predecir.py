import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from app.services.modelo import TransformerModel
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def predecir_futuro_producto(frecuencia, paso, producto_id, df, fecha_inicio, seq_len, features,target):
    
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(features)
    model = TransformerModel(input_dim=input_dim, d_model=64, nhead=4, num_layers=2).to(device)
    # Cargar pesos entrenados
    model.load_state_dict(torch.load("/home/ubuntuandros/Documents/modelo_productos_consumo/modelo_inventario/modelo/Modelo.pth", map_location=device))
    model.eval()

    scaler_x = joblib.load('/home/ubuntuandros/Documents/modelo_productos_consumo/modelo_inventario/scaler/scaler_x.pkl')
    scaler_y = joblib.load('/home/ubuntuandros/Documents/modelo_productos_consumo/modelo_inventario/scaler/scaler_y.pkl')

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

def calcular_proyeccion_inventario(predicciones, df, producto_id, fecha_inicio, paso, frecuencia, umbral=10):
    import pandas as pd
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fechas_futuras = pd.date_range(start=fecha_inicio_dt, periods=paso, freq=frecuencia)

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    ultimo_stock = df.loc[
        (df['Product ID'] == producto_id) & (df['Date'] == fecha_inicio_dt),
        'stock'
    ]
    ultimo_stock = float(ultimo_stock.iloc[0]) if not ultimo_stock.empty else 0.0

    inventario = [ultimo_stock]
    necesidad = []
    alerta_fecha = None
    notificado = False

    for fecha, pred in zip(fechas_futuras, predicciones):
        pred_val = float(np.ravel(pred)[0]) if isinstance(pred, (np.ndarray, list, pd.Series)) else float(pred)
        nuevo_inv = float(inventario[-1]) - pred_val
        sobrante = nuevo_inv - umbral
        inventario.append(nuevo_inv)
        necesidad.append(sobrante if sobrante < 0 else 0)
        if not notificado and nuevo_inv <= umbral:
            alerta_fecha = fecha.strftime("%Y-%m-%d")
            notificado = True

    return {
        "fechas": [d.strftime("%Y-%m-%d") for d in fechas_futuras],
        "predicciones": [float(p) for p in predicciones],
        "inventario": inventario[1:],  # eliminar el stock inicial
        "necesidad": necesidad,
        "umbral": umbral,
        "alerta_fecha": alerta_fecha
    }
