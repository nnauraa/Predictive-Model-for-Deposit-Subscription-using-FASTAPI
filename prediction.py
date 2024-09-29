from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

with open('BestModel.pkl', 'rb') as f:
    model = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('labelEncoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

app = FastAPI()

class PredictionRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str

def predict(data):
    try:
        data_df = pd.DataFrame(data, columns=[
            'age', 'duration', 'campaign', 'pdays', 'previous', 'job', 'marital', 'education',
            'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'
        ])
        data_preprocessed = preprocessor.transform(data_df)
        predictions = model.predict(data_preprocessed)
        predictions_int = [int(pred) for pred in predictions]  
        return predictions_int
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
def get_prediction(data: PredictionRequest):
    input_data = [[
        data.age, data.duration, data.campaign, data.pdays, data.previous,
        data.job, data.marital, data.education, data.default, data.housing,
        data.loan, data.contact, data.month, data.day_of_week, data.poutcome
    ]]
    prediction = predict(input_data)
    return {"prediction": prediction[0]}
