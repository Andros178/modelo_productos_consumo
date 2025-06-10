

# Download latest version


# path = kagglehub.dataset_download("anirudhchauhan/retail-store-inventory-forecasting-dataset")

# print("Path to dataset files:", path)

# for filename in os.listdir(path):
#     shutil.move(os.path.join(path, filename), os.getcwd())

# print("Files moved to workspace.")

# df = pd.read_csv("retail_store_inventory.csv")


# df_clean = df[['Product ID', 'Inventory Level', 'Units Sold', 'Date', 'Price','Discount', 'Holiday/Promotion']].dropna()

# df_clean.to_csv('Dataset/retail_store_inventory.csv', index=False)


def dataset(result_json):
    #import kagglehub
    import shutil
    import os
    import pandas as pd

    

    df_json = pd.DataFrame(result_json)

    df = pd.DataFrame({
        #'cantidad_salida': df_json['Units Sold'],  # o 'Demand Forecast' si prefieres usar el pronóstico
        'Product ID': df_json['id'],
        'stock': df_json['cantidad'],
        #'temporada_inicio': df_json['fechaHora_Inicio'],
        #'temporada_fin':df_json['fechaHora_Fin'],
        #'tipo_salida': df_json['tipo_salida'], # debe usarse one-hot encodding
        'Date':df_json['fecha_registro']
    })

    print(df.head())
    df = df.sort_values(by='Date')
    

    # Guardar CSV
    df.to_csv('/home/usco/Documents/modelo_productos_consumo/Dataset/123.csv', index=False)

    df_test= pd.read_csv('/home/usco/Documents/modelo_productos_consumo/Dataset/123.csv')
    df_test.head()
    return df
