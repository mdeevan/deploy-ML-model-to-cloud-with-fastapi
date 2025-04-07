"""
Author: Muhammad Naveed
Date: April 4th, 2025

This app is FastAPI interface used to run Random Forest Classifier on census data on Heroku'
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib
import uvicorn
import logging
import os
import dvc.api

import census_class as cls

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()



# Alias Generator funtion for class CensusData
def replace_underscore(string: str) -> str:
    return string.replace('_','-')

# Class definition of the data that will be provided as POST request
class CensusData(BaseModel):
    # Using the first row of census.csv as sample
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')
    salary: Optional[str]


    class Config:
        alias_generator = replace_underscore


if "render" in os.environ['PATH'] and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

# Load models on startup to speed-up POST request step
@app.on_event("startup")
async def startup_event():
    global model, encoder, lb, cat_features

    params = dvc.api.params_show()
    model_path = params['model']['model_path']
    model_name = params['model']['model_name']
    encoder_name = params['model']['encoder_name']
    lb_name = params['model']['lb_name']
    cat_features = params['cat_features']

    # census_obj = cls.Census()
    model = joblib.load(os.path.join(model_path, model_name))
    encoder = joblib.load(os.path.join(model_path, encoder_name))
    lb = joblib.load(os.path.join(model_path, lb_name))



# Home site with welcome message - GET request
@app.get("/", tags=["home"])
async def get_root() -> dict:
    """
    Home page, returned as GET request
    """
    return {
        "message": "Welcome to FastAPI interface to Random Forest Classifier on census data"
    }

# POST request to /predict site. Used to validate model with sample census data
@app.post('/predict')
async def predict(input: CensusData):
    """
    POST request that will provide sample census data and expect a prediction

    Output:
        0 or 1
    """

    # Read data sent as POST
    print(f"input  = {input}, input type = {type(input)}")
    input_data = input.dict(by_alias=True)
    print(f"input data = {input_data}")
    print (f"model type {type(model)}")

    input_df = pd.DataFrame(input_data, index=[0])
    logger.info(f"Input data: {input_df}")


    census_obj = cls.Census()
    pred = census_obj.execute_inference(model=model, encoder=encoder, lb=lb, df=input_df)
    # Process the data
    pred = lb.inverse_transform(pred)[0]
    response = {"Salary prediction": pred}

    logger.info(f"Prediction: {response}")

    return response


#if __name__ == "__main__":

#    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)