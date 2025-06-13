from fastapi import HTTPException

from app.services.connect_db import conexion
from app.services.dataframe import dataset
from app.services.modelo import modelar
from app.services.entrenar import entrenar
from app.services.predecir import predecir_futuro_producto, calcular_proyeccion_inventario

# routes.py

# Endpoint para entrenar el modelo
async def train_model(seq_len: int, frecuencia: str, producto_id: int, paso: int, features: str, target: str, fecha_inicio: str):
    features_list = features.split(",")
    target_list = target.split(",")

    result_json = conexion()
    if result_json:
        df = dataset(result_json)

        scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device, df = modelar(
            df, seq_len, features_list, target_list
        )

        pasos = entrenar(scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device)

        umbral = 10  # Valor fijo o calculado

        return {
            "message": "Modelo entrenado con éxito",
            "steps": pasos,
            "umbral": umbral  # <-- Agrega este campo
        }
    else:
        raise HTTPException(status_code=400, detail="No se pudo obtener datos desde la base de datos.")



# Endpoint para predecir con el modelo entrenado
async def predict_model(frecuencia: str, paso: int, producto_id: int, fecha_inicio: str, seq_len: int, features: str, target: str):
    # Convertir características y objetivo a listas
    features_list = features.split(",")
    target_list = target.split(",")

    # Conectar a la base de datos y obtener el dataset
    result_json = conexion()
    if result_json:
        df = dataset(result_json)  # Aquí convertimos el JSON a DataFrame

        # Realizar predicciones
        predicciones, fechas_futuras = predecir_futuro_producto(frecuencia, paso, producto_id, df, fecha_inicio, seq_len, features_list, target_list)

        # Graficar la predicción
        # routes.py
        proyeccion = calcular_proyeccion_inventario(
            predicciones, df, producto_id, fecha_inicio, paso, frecuencia, umbral=10
        )

        return {
            "message": "Predicciones realizadas con éxito",
            "proyeccion": proyeccion  # incluye fechas, inventario, necesidad, etc.
        }

    else:
        raise HTTPException(status_code=400, detail="No se pudo obtener datos desde la base de datos.")
