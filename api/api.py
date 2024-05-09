import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


class HouseInfo(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float


app = FastAPI()

# Obtiene la ruta del directorio actual
dir_actual = os.path.dirname(__file__)

# Construye la ruta del archivo en un directorio específico
ruta_archivo_folder = os.path.join(dir_actual, '../modelo', 'logistic_regression_model_V0.2.pkl')
model = joblib.load(ruta_archivo_folder )


@app.post('/predict')
def predict_price(house: HouseInfo):
    data = pd.DataFrame([dict(house)])
    prediction = model.predict(data)
    return {"predicted_price": prediction[0]}

@app.get('/ampliacion')
def prueba():
    return {'Trabajo para curso de ampliacion':'Hola Jorge!'}
