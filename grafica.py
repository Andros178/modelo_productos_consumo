import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset limpio
df = pd.read_csv('Dataset/retail_store_inventory.csv')

# Convertir la columna 'Date' a tipo fecha
df['Date'] = pd.to_datetime(df['Date'])

# Obtener lista de productos únicos
productos = df['Product ID'].unique()[:1]

# Crear una gráfica por producto
for producto in productos:
    df_prod = df[df['Product ID'] == producto]
    
    plt.figure(figsize=(14, 6))
    plt.title(f'Producto {producto} - Evolución de Variables en el Tiempo')

    # Graficar cada variable en el mismo eje
    plt.plot(df_prod['Date'], df_prod['Inventory Level'], label='Inventario')
    plt.plot(df_prod['Date'], df_prod['Units Sold'], label='Unidades Vendidas')
    plt.plot(df_prod['Date'], df_prod['Price'], label='Precio')

    plt.xlabel('Fecha')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
