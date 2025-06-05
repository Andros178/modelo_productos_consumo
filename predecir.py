from entrenar import predecir_futuro_producto

if __name__ == "__main__":
    producto_id = "P0011"
    fecha_inicio = "2023-12-25"  # Fecha que quieres predecir
    paso=7
    frecuencia="D"
    predicciones, reales = predecir_futuro_producto(producto_id, fecha_inicio, paso, frecuencia)
    if predicciones is not None:
        print("Ejemplo de predicción vs real:")
        print("Pred:", predicciones.flatten()[:5])
        print("Real:", reales.flatten()[:5])