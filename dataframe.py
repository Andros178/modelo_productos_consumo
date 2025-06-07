import kagglehub
import shutil
import os
import pandas as pd

# Download latest version


# path = kagglehub.dataset_download("anirudhchauhan/retail-store-inventory-forecasting-dataset")

# print("Path to dataset files:", path)

# for filename in os.listdir(path):
#     shutil.move(os.path.join(path, filename), os.getcwd())

# print("Files moved to workspace.")

df = pd.read_csv("retail_store_inventory.csv")


df_clean = df[['Product ID', 'Inventory Level', 'Units Sold', 'Date', 'Price','Discount', 'Holiday/Promotion']].dropna()

df_clean.to_csv('Dataset/retail_store_inventory.csv', index=False)

