import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from prediction import PredictionInput, load_models, clear_models, process_prediction

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield 
    clear_models()

app = FastAPI(title="Disease Prediction API", lifespan=lifespan)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/diabetes")
async def predict_diabetes(data: PredictionInput):
    REQUIRED_FEATURES_DIABETES = 5
    THRESHOLD_DIABETES = 0.31
    return process_prediction(data, "diabetes", REQUIRED_FEATURES_DIABETES, THRESHOLD_DIABETES)

@app.post("/predict/typhoid")
async def predict_typhoid(data: PredictionInput):
    REQUIRED_FEATURES_TYPHOID = 5
    THRESHOLD_TYPHOID = 0.31
    return process_prediction(data, "typhoid", REQUIRED_FEATURES_TYPHOID, THRESHOLD_TYPHOID)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)