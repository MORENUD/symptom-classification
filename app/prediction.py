import pickle
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel

ml_models = {}

class PredictionInput(BaseModel):
    features: list[float]

def load_models():
    model_configs = {
        "diabetes": '../models/lgb_model_diabetes.pkl',
        "typhoid": '../models/lgb_model_typhoid.pkl' 
    }
    
    for model_name, path in model_configs.items():
        try:
            with open(path, 'rb') as file:
                ml_models[model_name] = pickle.load(file)
            print(f"Model '{model_name}' loaded successfully from {path}")
        except FileNotFoundError:
            print(f"Error: Model file for '{model_name}' not found at {path}")
            ml_models[model_name] = None
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            ml_models[model_name] = None

def clear_models():
    ml_models.clear()

def process_prediction(data: PredictionInput, model_name: str, required_features: int, threshold: float):
    
    model = ml_models.get(model_name)
    
    if not model:
        raise HTTPException(status_code=500, detail=f"Model '{model_name}' is not loaded.")

    try:
        input_array = np.array(data.features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")

    if input_array.shape[1] != required_features:
          raise HTTPException(
              status_code=400, 
              detail=f"Model '{model_name}' expects {required_features} features, got {input_array.shape[1]}"
          )

    try:
        probs = model.predict_proba(input_array)
        prob_positive = probs[0][1] 
    except AttributeError:
        raise HTTPException(status_code=500, detail="Loaded model does not support probability prediction.")

    prediction = 1 if prob_positive >= threshold else 0
    label = f"Positive" if prediction == 1 else f"Negative"

    return {
        "prediction_class": prediction,
        "prediction_label": label,
        "probability_positive": float(prob_positive),
        "threshold_used": threshold,
        "is_above_threshold": bool(prob_positive >= threshold)
    }