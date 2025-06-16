from sklearn.metrics import mean_squared_error, r2_score
import joblib

def predecir_regresion():
    # Predecir
    modelo = joblib.load("modeloRegression.pkl")
    y_test = joblib.load("pesosRegression_y_test.pkl")
    X_test = joblib.load("pesosRegression_X_test.pkl")
    y_pred = modelo.predict(X_test)

    # Evaluar
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    coeficientes = modelo.coef_
    intercepto = modelo.intercept_

    # Convertir tipos para serializar en JSON
    y_test_list = y_test.iloc[:, 0].tolist()  # ← Corrige aquí
    y_pred_list = y_pred.tolist()

    return {
        "message": "prediccion exitosa",
        "valores_reales": y_test_list,
        "predicciones": y_pred_list,
        "Coeficientes": coeficientes.tolist(),
        "Intercepto": float(intercepto),
        "MSE": float(mse),
        "R²": float(r2)
    }
