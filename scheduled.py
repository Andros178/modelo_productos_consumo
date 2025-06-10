from connect_db import conexion
from dataframe import dataset
from modelo import modelar
from entrenar import entrenar
from predecir import predecir_futuro_producto

print("Iniciando scheduler")

import sys
# Leer parámetro desde línea de comandos
seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 14  # valor por defecto: 14

print(f"Usando seq_len = {seq_len}")

result_json = conexion()
if result_json:
    df=  dataset(result_json)
    features, scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device, df, seq_len, producto_id= modelar(df,seq_len)

    entrenar(features, scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device, df,seq_len,producto_id )

    predecir_futuro_producto(producto_id, fecha_inicio, pasos, frecuencia)

else:
    print("No se pudo obtener datos desde la base de datos.")