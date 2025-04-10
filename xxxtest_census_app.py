# tests/test_fastapi.py
import pytest
from fastapi.testclient import TestClient
from main import app, CensusData
# , data_store
# from unittest.mock import patch, MagicMock

client = TestClient(app)

# @pytest.fixture(autouse=True)
# def clear_data_store():
#     data_store.clear()
#     yield
#     data_store.clear()

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

def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_CENSUS_DATA)
        print(f"reponse json: {response.json()}")
        assert response.status_code == 200
        assert "prediction" in response.json()
    # assert len(data_store) == 1

def test_predict_invalid_data():
    invalid_data = VALID_CENSUS_DATA.copy()
    invalid_data["age"] = "not-an-integer"
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_get_data_empty():
    response = client.get("/data")
    assert response.status_code == 200
    assert response.json() == []

def test_get_data_with_items():
    # First add data
    client.post("/predict", json=VALID_CENSUS_DATA)
    response = client.get("/data")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["age"] == 39

# # Authentication tests
# MOCK_TOKEN = jwt.encode(
#     {"sub": "123", "email": "test@example.com"},
#     key="secret",
#     algorithm="HS256"
# )

# @patch('main.get_current_user')
# def test_protected_predict(mock_user):
#     mock_user.return_value = MagicMock(email="test@example.com")
#     response = client.post(
#         "/predict",
#         json=VALID_CENSUS_DATA,
#         headers={"Authorization": f"Bearer {MOCK_TOKEN}"}
#     )
#     assert response.status_code == 200

# def test_unauthenticated_predict():
#     response = client.post("/predict", json=VALID_CENSUS_DATA)
#     assert response.status_code == 401

# @patch('main.oauth.oidc.authorize_access_token')
# def test_login_flow(mock_token):
#     mock_token.return_value = {"access_token": "test_token"}
#     # Simulate OAuth callback
#     response = client.get("/auth?code=testcode&state=teststate")
#     assert response.status_code == 200
#     assert "access_token" in response.json()

# @pytest.mark.parametrize("field,value,expected_status", [
#     ("age", -1, 422),  # Negative age
#     ("hours-per-week", 100, 422),  # Too many hours
#     ("capital-gain", "not-a-number", 422),
#     ("education", "", 422)  # Empty string
# ])
# def test_data_validation(field, value, expected_status):
#     test_data = VALID_CENSUS_DATA.copy()
#     test_data[field] = value
#     response = client.post("/predict", json=test_data)
#     assert response.status_code == expected_status