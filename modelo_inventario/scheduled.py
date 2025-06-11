from connect_db import conexion
from dataframe import dataset
from modelo import modelar
from entrenar import entrenar
from predecir import predecir_futuro_producto, graficar_prediccion
import torch
import joblib
print("Iniciando scheduler")

import sys
# Leer parámetro desde línea de comandos


# Leer parámetros desde la línea de comandos
seq_len = int(sys.argv[1])  # Primer argumento: seq_len
frecuencia = sys.argv[2]    # Segundo argumento: frecuencia ("D", "M", "Y")
producto_id = int(sys.argv[3])  # Tercer argumento: producto_id
paso = int(sys.argv[4])      # Cuarto argumento: paso (debe ser un número)
features_str = sys.argv[5]   # Quinto argumento: características (como una cadena)
target_str = sys.argv[6]     # Sexto argumento: objetivo (como una cadena)
fecha_inicio = sys.argv[7]

# Convertir los argumentos de características y objetivo de cadena a listas
features = features_str.split(",")  # Convertir a lista de características
target = target_str.split(",")  # Convertir a lista de objetivos

# Comprobar si la frecuencia es válida
if frecuencia not in ["D", "M", "Y"]:
    print(f"Error: Frecuencia inválida. Debe ser 'D', 'M' o 'Y'.")
    sys.exit(1)

# Comprobar que seq_len sea positivo
if seq_len <= 0:
    print("Error: seq_len debe ser un número positivo.")
    sys.exit(1)


# Mostrar los valores recibidos
print(f"Usando seq_len = {seq_len}")
print(f"Usando frecuencia = {frecuencia}")
print(f"Usando producto_id = {producto_id}")
print(f"Usando paso = {paso}")
print(f"Usando features = {features}")
print(f"Usando target = {target}")
print(f"Usando fecha inicio = {fecha_inicio}")




try: 
    # Inicializar dispositivo (CPU o GPU)
    

    # Cargar los scalers previamente guardados
    scaler_x = joblib.load('scaler/scaler_x.pkl')  # Cargar scaler_x
    scaler_y = joblib.load('scaler/scaler_y.pkl')  # Cargar scaler_y
except:
    print(f"No se encontraron scalers o modelo para predicción sin entrenamiento")


result_json = conexion()
if result_json:
    df =  dataset(result_json)

    # scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device, df = modelar(df,seq_len,features,target)

    # pasos = entrenar(scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device )
    
    predicciones, fechas_futuras = predecir_futuro_producto(frecuencia,paso, producto_id, df, fecha_inicio, seq_len, features, target)

    graficar_prediccion(frecuencia, paso, producto_id, df, predicciones, fecha_inicio, seq_len, features,target)


else:
    print("No se pudo obtener datos desde la base de datos.")