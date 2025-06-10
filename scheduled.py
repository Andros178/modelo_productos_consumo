from connect_db import conexion
from dataframe import dataset
from modelo import modelar

print("Iniciando scheduler")

import sys
# Leer parámetro desde línea de comandos
seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 14  # valor por defecto: 14

print(f"Usando seq_len = {seq_len}")

result_json = conexion()
if result_json:
    df=  dataset(result_json)
    modelar(df,seq_len)

else:
    print("No se pudo obtener datos desde la base de datos.")