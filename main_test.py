"""
main_test.py

Test the main.py using pytest

"""

# import os
# import json

from fastapi.testclient import TestClient

# import dvc.api
# import joblib

from main import app


# Test data
FOR_UNDER_50K_PREDICT = {
    """
        define a test data as default

    """
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
    "native-country": "United-States",
}

FOR_OVER_50K_PREDICT = {
    'age':39,
    'workclass':'Private',
    'fnlgt':45781,
    'education':'Bachelors',
    'education_num':14,
    'marital_status':'Never-married',
    'occupation':'Prof-specialty',
    'relationship':'Not-in-family',
    'race':'White',
    'sex':'Female',
    'capital_gain':14084,
    'capital_loss':0,
    'hours_per_week':50,
    'native_country':'United-States'
    }


def test_root():
    """'
    test the funciton is callable and responding
    the app is getting created as part of the test execution
    """
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200


def test_predict_positive():
    """
    test a positive case of predict using the test data setup above
    """
    with TestClient(app) as client:
        response = client.post("/predict", json=FOR_UNDER_50K_PREDICT)
        assert response.status_code == 200
        assert response.json() == {"Salary prediction": "<=50K"}

def test_predict_positive_over_50K():
    """
    test a positive case of predict using the test data setup above
    """
    with TestClient(app) as client:
        response = client.post("/predict", json=FOR_OVER_50K_PREDICT)
        assert response.status_code == 200
        assert response.json() == {"Salary prediction": "<=50K"}


def test_predict_negative():
    """
    test negative predict API by modifying one attribute with invalid data
    """
    with TestClient(app) as client:
        invalid_data = FOR_UNDER_50K_PREDICT.copy()
        invalid_data["age"] = "not-an-integer"

        response = client.post("/predict", json=invalid_data)
        print(f"reponse json: {response.json()}")

        assert response.status_code == 422
