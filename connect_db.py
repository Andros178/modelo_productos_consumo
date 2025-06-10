import pg8000
from dotenv import load_dotenv
import os
import json


def conexion():
    # Cargar variables de entorno desde .env
    load_dotenv()

    # Detalles de conexión a la base de datos
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = int(os.getenv("DB_PORT"))

    try:
        # Conectarse a la base de datos
        connection = pg8000.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        print("¡Conexión exitosa!")

        # Crear cursor
        cursor = connection.cursor()

        # Ejecutar consulta
        query = "SELECT * FROM producto;"
        cursor.execute(query)
        records = cursor.fetchall()

        # Obtener nombres de columnas
        column_names = [desc[0] for desc in cursor.description]

        from datetime import date, datetime

        def serialize_record(record):
            return {
                key: (value.isoformat() if isinstance(value, (date, datetime)) else value)
                for key, value in zip(column_names, record)
            }

        result_json = [serialize_record(record) for record in records]


        # Imprimir resultado como JSON
        print(json.dumps(result_json, indent=4, ensure_ascii=False))

        # Cerrar cursor y conexión
        cursor.close()
        connection.close()
        return result_json

    except Exception as e:
        print("Ocurrió un error:", e)
        return None

