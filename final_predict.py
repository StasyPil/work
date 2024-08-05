import joblib
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI
model = joblib.load('model/client_pipe.pkl')

class Form(BaseModel):
    session_id: float
    client_id: float
    visit_date: int
    visit_time: int
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    geo_country: str
    geo_city: str

class Prediction(BaseModel):
    session_id: float
    Result: int

@app.get('/status')
def status():
    return "i'm OK"

@app.get('/version')
def version():
    return model['metadata']

@app.post('/predict')
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'session_id':form.session_id,
        'Result': y[0]
    }

