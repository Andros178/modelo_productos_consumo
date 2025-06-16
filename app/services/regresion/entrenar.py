from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def entrenar_forecasting_modelo(datos):
    datos_lags = preparar_lags(datos, max_lag=1)

    X = datos_lags[['lag_1', 'year']]  # puedes incluir más features como county/industry dummies
    y = datos_lags['total_gallons_all_sources']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("MSE:", mean_squared_error(y_test, y_pred))
    return modelo
