"""
Author: Muhammad Naveed
Date: April 4th, 2025

This app is FastAPI interface used to run Random Forest Classifier on census data on Heroku'
"""

import os
import sys

import logging
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# import uvicorn
import dvc.api

# import boto3

import census_class as cls

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()


# Alias Generator funtion for class CensusData
def replace_underscore(string: str) -> str:
    """
    function to modify the attribute name in pydantic class
    by apply to all attributes
    where _ will be replaced with -
    """
    return string.replace("_", "-")


class CensusData(BaseModel):
    """
    Class definition of the data that will be provided as POST request
    and the attributes name will get _ replaced with - via function
    """

    # Using the first row of census.csv as sample
    age: int = Field(None, example=39)
    workclass: str = Field(None, example="State-gov")
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example="Bachelors")
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example="Never-married")
    occupation: str = Field(None, example="Adm-clerical")
    relationship: str = Field(None, example="Not-in-family")
    race: str = Field(None, example="White")
    sex: str = Field(None, example="Female")
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example="United-States")
    salary: Optional[str]

    class Config:
        """
        generate alias for the class attribute names
        """

        alias_generator = replace_underscore


if "render" in os.environ["PATH"] and os.path.isdir(".dvc"):

    print("loading files from dvc")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        print("dvc load failed")
        sys.exit("dvc pull failed")
    # os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


# Load models on startup to speed-up POST request step
@app.on_event("startup")
async def startup_event():
    """
    capture parameters and setup the environment variables
    expensive function, to be done only at startup
    """
    global model, encoder, lb, cat_features

    params = dvc.api.params_show()
    model_path = params["model"]["model_path"]
    model_name = params["model"]["model_name"]
    encoder_name = params["model"]["encoder_name"]
    lb_name = params["model"]["lb_name"]
    cat_features = params["cat_features"]

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
@app.post("/predict")
async def predict(input_params: CensusData) -> str:
    """
    POST request that will provide sample census data and expect a prediction

    Output:
        Salary value as,  >50K or <=50K
    """

    # Read data sent as POST
    print(f"input  = {input_params}, input type = {type(input_params)}")
    input_data = input_params.dict(by_alias=True)
    # print(f"input data \n {input_data}")
    # print (f"model type \n{type(model)}")

    input_df = pd.DataFrame(input_data, index=[0])
    print(f"input df \n {input_df}")
    # logger.info(f"Input data: {input_df}")

    census_obj = cls.Census()
    pred = census_obj.execute_inference(
        model=model, encoder=encoder, lb=lb, df=input_df
    )
    # Process the data
    pred = str(lb.inverse_transform(pred)[0])
    response = {"Salary prediction": pred}
    logger.info("Pred %s and its type %s", pred, type(pred))
    # response = pred

    logger.info("Prediction: %s", response)

    # return response
    try:
        return response
    except Exception as e:
        # raise HTTPException(status_code=422, detail=e.errors())
        logger.info("Exception : %s", e)


# if __name__ == "__main__":

#    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
