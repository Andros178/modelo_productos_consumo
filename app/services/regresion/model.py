import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


def modelar_regression(datos,features,target):    
    print("caracteristicas", features)

    print("valor a predecir:", target)

    X = datos[features]
    y= datos[target]

    print("dataframe:", datos)

    # Codificación One-Hot con pandas (más simple y suficiente)
    X_encoded = pd.get_dummies(X, columns=['county', 'industry'], drop_first=True)

    print("X_encoded", X_encoded)


    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


    # Entrenar modelo

    modelo = LinearRegression()

    

    modelo.fit(X_train, y_train)

    joblib.dump(modelo, "modeloRegression.pkl")
    joblib.dump(y_test, "pesosRegression_y_test.pkl")
    joblib.dump(X_test, "pesosRegression_X_test.pkl")

