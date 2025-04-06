import json
from fastapi.testclient import TestClient
import dvc.api
import joblib
import os

from main import app

# client = TestClient(app)


# @app.on_event("startup")
# async def startup_event():
#     global model, encoder, lb, cat_features

#     params = dvc.api.params_show()
#     model_path = params['model']['model_path']
#     model_name = params['model']['model_name']
#     encoder_name = params['model']['encoder_name']
#     lb_name = params['model']['lb_name']
#     cat_features = params['cat_features']

#     # census_obj = cls.Census()
#     model = joblib.load(os.path.join(model_path, model_name))
#     encoder = joblib.load(os.path.join(model_path, encoder_name))
#     lb = joblib.load(os.path.join(model_path, lb_name))

def test_root():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200


def test_predict_positive():
    data = {"age": 52,
            "workclass": "Self-emp-inc",
            "fnlgt": 287927,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital_gain": 15024,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    with TestClient(app) as client:
        response = client.post("/predict", data=json.dumps(data))
        assert response.status_code == 200
        assert response.json() == {"Salary prediction": "<=50K"}