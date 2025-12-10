import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import prediction

@asynccontextmanager
async def lifespan(app: FastAPI):
    prediction.load_models()
    yield
    prediction.clear_models()

app = FastAPI(lifespan=lifespan)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_all")
def predict_endpoint(data: prediction.PredictionInput):
    return prediction.predict_all_diseases(data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)