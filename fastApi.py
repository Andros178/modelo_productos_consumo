from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API funcionando"}

@app.get("/resultado/{product_id}")
def get_by_product(product_id: str):
    try:
        df = pd.read_csv("Dataset/retail_store_inventory11.csv")
        df = df[df["producto ID"] == product_id]

        if df.empty:
            return JSONResponse(status_code=404, content={"error": "Producto no encontrado"})

        return df.to_dict(orient="records")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

