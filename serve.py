from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, validator, ValidationError
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from .training_script import MODEL_DV_PATH

app = FastAPI()

# Load the trained model and the DictVectorizer
model: xgb.XGBClassifier
dv: DictVectorizer
model, dv = load(MODEL_DV_PATH)

# Define a request model
class PredictRequest(BaseModel):
    # SIZE is categorical, int values from 1 to 7
    SIZE: float  # setting int here leads to internal pre-validator conversion
    # FUEL is categorical, str values from ['gasoline', 'kerosene', 'thinner', 'lpg']
    FUEL: str
    DISTANCE: float
    DECIBEL: float
    AIRFLOW: float
    FREQUENCY: float

    # FUEL is categorical, str values from ['gasoline', 'kerosene', 'thinner', 'lpg']
    @validator('FUEL')
    def check_fuel(cls, v) -> str:
        if v.lower() not in ['gasoline', 'kerosene', 'thinner', 'lpg']:
            raise ValueError('Invalid fuel type, allowed types: ["gasoline", "kerosene", "thinner", "lpg"]')
        return v.lower()  # Optionally return the lowercase version directly
    
    # SIZE is categorical, int values from 1 to 7
    @validator('SIZE')
    def size_must_be_within_range(cls, value) -> int:
        if value not in range(1, 8):
            raise ValueError('Invalid size, allowed sizes: [1, 2, 3, 4, 5, 6, 7]')
        return int(value)

# Define a response model
class PredictResponse(BaseModel):
    prediction: bool
    comment: str

@app.post('/predict', response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        # Create a dictionary from the request to transform via DictVectorizer
        input_data = {
            'SIZE': request.SIZE,
            'FUEL': request.FUEL,
            'DISTANCE': request.DISTANCE,
            'DESIBEL': request.DECIBEL,  # Intial dataset had deSibel as feature name
            'AIRFLOW': request.AIRFLOW,
            'FREQUENCY': request.FREQUENCY
        }
        # Transform the input data using the DictVectorizer
        transformed_input = dv.transform(input_data)
        # Make the prediction
        prediction = model.predict(transformed_input)
        comment = "Flame {} be extinguished.".format('WILL' if prediction else 'WILL NOT')

        response = PredictResponse(prediction=bool(prediction), comment=comment)
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# for serving/testing outside of docker, via `python serve.py`
import uvicorn

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, log_level="info")
