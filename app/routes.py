from fastapi import HTTPException
from app.state import history_cache  # <- Global cache

#Inventario
from app.services.inventario.connect_db import conexion
from app.services.inventario.dataframe import dataset
from app.services.inventario.modelo import modelar
from app.services.inventario.entrenar import entrenar
from app.services.inventario.predecir import predecir_futuro_producto, calcular_proyeccion_inventario

#regresion
from app.services.regresion.dataframe import dataset_regression
from app.services.regresion.model import modelar_regression
from app.services.regresion.predecir import predecir_regresion



async def train_model(seq_len: int, frecuencia: str, producto_id: int, paso: int, features: str, target: str, fecha_inicio: str):
    result_json = conexion()
    if not result_json:
        raise HTTPException(status_code=400, detail="No se pudo obtener datos desde la base de datos.")

    df = dataset(result_json)
    features_list = features.split(",")
    target_list = target.split(",")

    # Modelado y entrenamiento
    scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device, df = modelar(
        df, seq_len, features_list, target_list[0]
    )
    resultado = entrenar(scaler_x, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device)

    # Guardar en cache
    history_cache["history"] = resultado["history"]

    return {
        "message": "Modelo entrenado con éxito",
        "rmse": resultado["rmse"],
        "predicciones": resultado["predicciones"],
        "valores_reales": resultado["valores_reales"],
        "history": resultado["history"],
        "umbral": 10
    }


async def predict_model(seq_len: int, frecuencia: str, producto_id: int, paso: int, features: str, target: str, fecha_inicio: str):
    result_json = conexion()
    if not result_json:
        raise HTTPException(status_code=400, detail="No se pudo obtener datos desde la base de datos.")

    df = dataset(result_json)
    features_list = features.split(",")
    target_list = target.split(",")

    # Predicción sin reentrenar
    predicciones, fechas_futuras = predecir_futuro_producto(
        frecuencia, paso, producto_id, df, fecha_inicio, seq_len, features_list, target_list
    )
    proyeccion = calcular_proyeccion_inventario(predicciones, df, producto_id, fecha_inicio, paso, frecuencia, umbral=10)

    return {
        "message": "Predicciones realizadas con éxito",
        "proyeccion": proyeccion,
        "history": history_cache.get("history", {})  # <- Devuelve historial si existe
    }



async def regression_train(features, target):

    datos = dataset_regression()
    features_list = features.split(",")
    target_list = target.split(",")

    modelar_regression(datos,features_list, target_list)

    return{
        "message": "Modelo entrenado con éxito"
    }
    

async def regression_predict():   

    resultado = predecir_regresion()
    return {
        "message": "prediccion exitosa",
        "valores_reales": resultado["valores_reales"],
        "predicciones": resultado["predicciones"],
        "coeficientes": resultado["Coeficientes"],
        "intercepto": resultado['Intercepto'],"MSE": resultado['MSE'],
        "R²": resultado['R²'],
        
        
    }