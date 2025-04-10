import json
from fastapi.testclient import TestClient
import dvc.api
import joblib
import os

from main import app


# Test data
VALID_CENSUS_DATA = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

def test_root():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200


def test_predict_positive():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_CENSUS_DATA)
        assert response.status_code == 200
        assert response.json() == {"Salary prediction": "<=50K"}

def test_predict_negative():
    with TestClient(app) as client:
        invalid_data = VALID_CENSUS_DATA.copy()
        invalid_data["age"] = "not-an-integer"

        response = client.post("/predict", json=invalid_data)
        print(f"reponse json: {response.json()}")

        assert response.status_code == 422