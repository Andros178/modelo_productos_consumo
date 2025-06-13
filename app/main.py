# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import train_model, predict_model
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    seq_len: int
    frecuencia: str
    producto_id: int
    paso: int
    features: str
    target: str
    fecha_inicio: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Ajusta al host de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/train_model")
async def train(request: Request):
    # Extraer los datos del objeto
    return await train_model(
        request.seq_len, request.frecuencia, request.producto_id,
        request.paso, request.features, request.target, request.fecha_inicio
    )
@app.post("/predict_model")
async def predict(request: Request):
    print("Datos recibidos:", request.dict())
    return await predict_model(
        frecuencia=request.frecuencia,
        paso=request.paso,
        producto_id=request.producto_id,
        fecha_inicio=request.fecha_inicio,
        seq_len=request.seq_len,
        features=request.features,
        target=request.target
)



