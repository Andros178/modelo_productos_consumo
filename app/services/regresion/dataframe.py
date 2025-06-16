import pandas as pd

# Simulamos un DataFrame de ejemplo

def dataset_regression():
    datos = pd.read_csv('/home/usco/Documents/modelo_productos_consumo/app/services/regresion/water_use_data_2013_to_2022.csv')
    datos.drop('Unnamed: 0', axis=1, inplace=True)
    datos.to_csv('datos.csv', index=False)

    datos = pd.read_csv('datos.csv')
    datos.info()

    
    return datos
