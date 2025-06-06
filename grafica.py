import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset limpio
df = pd.read_csv('Dataset/retail_store_inventory.csv')


df['Date'] = pd.to_datetime(df['Date'])


productos = df['Product ID'].unique()[:1] #<- :1 significa solamente 1 producto (demasiadas graficas)

# Crear una gráfica por producto
for producto in productos:
    df_prod = df[df['Product ID'] == producto]
    
    plt.figure(figsize=(14, 6))
    plt.title(f'Producto {producto} - Evolución de Variables en el Tiempo')


    plt.plot(df_prod['Date'], df_prod['Inventory Level'], label='Inventario')
    plt.plot(df_prod['Date'], df_prod['Units Sold'], label='Unidades Vendidas')
    plt.plot(df_prod['Date'], df_prod['Price'], label='Precio')

    plt.xlabel('Fecha')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
