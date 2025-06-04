from entrenar import predecir_producto
# Ejemplo de uso

if __name__ == "__main__":
    predicciones, reales = predecir_producto("1")
    if predicciones is not None and reales is not None:
        print("Ejemplo de predicción vs real:")
        print("Pred:", predicciones.flatten()[:5])
        print("Real:", reales.flatten()[:5])