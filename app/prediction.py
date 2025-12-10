import os
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi import HTTPException

# --- Configuration ---
REQUIRED_NUM_FEATURES = 5
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_CONFIGS = {
    "alert": {
        "path": os.path.join(BASE_DIR, 'models', 'model_alert_xgb.pkl'),
        "threshold": 0.690
    },
    "cardiovascular": {
        "path": os.path.join(BASE_DIR, 'models', 'model_side_cardiovascular_diseases_xgb.pkl'),
        "threshold": 0.590
    },
    "gastrointestinal_liver": {
        "path": os.path.join(BASE_DIR, 'models', 'model_side_gastrointestinal_liver_diseases_xgb.pkl'),
        "threshold": 0.940
    },
    "infectious": {
        "path": os.path.join(BASE_DIR, 'models', 'model_side_infectious_diseases_xgb.pkl'),
        "threshold": 0.890
    }
}

ml_models = {}

class PredictionInput(BaseModel):
    features: list[float]

def load_models():
    for name, config in MODEL_CONFIGS.items():
        path = config["path"]
        try:
            with open(path, 'rb') as file:
                ml_models[name] = pickle.load(file)
            print(f"[{name}] Loaded successfully from {path}")
        except Exception as e:
            print(f"[{name}] Error loading: {e}")
            ml_models[name] = None

def clear_models():
    ml_models.clear()

def _process_single_prediction(features: list[float], model_name: str, threshold: float):
    
    model = ml_models.get(model_name)
    
    if not model:
        return {
            "prediction_class": -1,
            "prediction_label": "Error: Model Not Loaded",
            "probability_positive": 0.0,
            "threshold_used": threshold,
            "is_above_threshold": False
        }

    try:
        input_array = np.array(features).reshape(1, -1)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid feature format")

    if input_array.shape[1] != REQUIRED_NUM_FEATURES:
          raise HTTPException(
              status_code=400, 
              detail=f"Model expects {REQUIRED_NUM_FEATURES} features, got {input_array.shape[1]}"
          )

    # Predict
    probs = model.predict_proba(input_array)
    prob_positive = float(probs[0][1])
    
    prediction = 1 if prob_positive >= threshold else 0
    label = "Positive" if prediction == 1 else "Negative"

    return {
        "prediction_class": prediction,
        "prediction_label": label,
        "probability_positive": prob_positive,
        "threshold_used": threshold,
        "is_above_threshold": prediction == 1
    }

def predict_all_diseases(data: PredictionInput):
    
    results = {}
    
    for name, config in MODEL_CONFIGS.items():
        results[name] = _process_single_prediction(
            features=data.features, 
            model_name=name, 
            threshold=config["threshold"]
        )
        
    return {
        "status": "success",
        "input_features_count": len(data.features),
        "results": results
    }