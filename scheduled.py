from connect_db import conexion
from dataframe import dataset
from modelo import modelar

print("Iniciando scheduler")

result_json = conexion()
if result_json:
    df=  dataset(result_json)
    modelar(df)

else:
    print("No se pudo obtener datos desde la base de datos.")