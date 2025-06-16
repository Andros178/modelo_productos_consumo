from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.routes import train_model, predict_model, regression_predict, regression_train

app = FastAPI()

# CORS para permitir conexión desde Spring Boot en localhost:8080
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train_model")
async def train(request: Request):
    data = await request.json()
    return await train_model(**data)

@app.post("/predict_model")
async def predict(request: Request):
    data = await request.json()
    return await predict_model(**data)


@app.post("/train_regression_model")
async def train_regression(request:Request):
    data = await request.json()
    return await regression_train(**data)

@app.post("/predict_regression_model")
async def predict_regression():
    
    return await regression_predict()
