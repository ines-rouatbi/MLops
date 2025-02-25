from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List, Optional
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
MODEL_PATH = "trained_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading the model: {e}")
    raise RuntimeError(f"Error loading the model: {e}")

# Initialize FastAPI app
app = FastAPI()

# Define request schemas
class InputData(BaseModel):
    features: List[float]

class RetrainParams(BaseModel):
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = None
    random_state: Optional[int] = 42

@app.get("/")
def root():
    return {"message": "API is running 🚀"}

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    try:
        logger.info(f"🔍 Received input: {data.features}")
        
        expected_features = 68
        if len(data.features) != expected_features:
            raise HTTPException(status_code=400, detail=f"Invalid input: Expected {expected_features} features, but got {len(data.features)}.")
        
        input_array = np.array(data.features).reshape(1, -1)
        logger.info(f"📊 Reshaped input: {input_array}")
        
        prediction = model.predict(input_array)
        logger.info(f"✅ Prediction: {prediction}")
        
        return {
            "message": "Prediction successful.",
            "prediction": prediction.tolist()
        }
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Retraining route
@app.post("/retrain")
def retrain(params: RetrainParams):
    try:
        logger.info(f"♻️ Retraining model with parameters: {params.dict()}")
        
        X, y = make_classification(n_samples=5000, n_features=68, random_state=params.random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=params.random_state)
        
        new_model = RandomForestClassifier(
            n_estimators=params.n_estimators, 
            max_depth=params.max_depth, 
            random_state=params.random_state
        )
        new_model.fit(X_train, y_train)
        
        joblib.dump(new_model, MODEL_PATH)
        global model
        model = new_model
        
        return {
            "message": "Model retrained successfully.",
            "n_estimators": params.n_estimators,
            "max_depth": params.max_depth,
            "random_state": params.random_state
        }
    except Exception as e:
        logger.error(f"❌ Retraining error: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining error: {e}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
