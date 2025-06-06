from entrenar import model, scaler_x, scaler_y, seq_len, features, df, device, np, df

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
    secuencia = X_hist_scaled[-seq_len:]  # [seq_len, features]

    import pandas as pd
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fechas_futuras = pd.date_range(start=fecha_inicio_dt, periods=pasos, freq=frecuencia)

    predicciones = []
    for fecha in fechas_futuras:
        nueva_fila = df_producto.iloc[-1].copy()
        nueva_fila['Date'] = fecha.timestamp()
        nueva_fila['Units Sold'] = 0  # Placeholder
        X_nueva = nueva_fila[features].values.reshape(1, -1)
        X_nueva_scaled = scaler_x.transform(X_nueva)
        secuencia = np.vstack([secuencia[1:], X_nueva_scaled])
        input_modelo = torch.tensor(secuencia[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(input_modelo).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
        predicciones.append(pred)
        # secuencia[-1, features.index('Units Sold')] = scaler_y.transform([[pred]])[0, 0]  # Si quieres usar la predicción como input

    return np.array(predicciones), None  # No hay reales para el futuro

# Uso en main:
if __name__ == "__main__":
    producto_id = "P0011"
    fecha_inicio = "2023-12-25"
    paso = 7
    frecuencia = "D"
    predicciones, _ = predecir_futuro_producto(producto_id, fecha_inicio, paso, frecuencia)
    if predicciones is not None:
        print("Pred:", predicciones.flatten()[:5])