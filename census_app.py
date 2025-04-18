"""
census_app.py

streamlit based frontend application to make Salary prediction
basd on the model trained

"""

# census_app.py - Streamlit Application
import os
from typing import Optional
import pandas as pd
import streamlit as st
import requests
from pydantic import BaseModel, Field
import dvc.api


class CensusData(BaseModel):
    """
    defining a pydantic model for fastAPI
    """

    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")
    salary: Optional[str] = Field(None)


# API configuration
FASTAPI_BASE_URL = "https://deploy-ml-model-to-cloud-with-fastapi.onrender.com/"  #"http://localhost:8000"

# URL Is configured in params.yml


PREDICT_ENDPOINT = f"{FASTAPI_BASE_URL}/predict"
DATA_ENDPOINT = f"{FASTAPI_BASE_URL}/data"


def show_prediction_form():
    """
    show the streamlit form to accept the parameters
    to predict the salary
    """
    st.header("Salary Prediction (POST)")

    with st.form(key="census_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=17, max_value=90, value=39)
            workclass = st.selectbox(
                "Workclass", st.session_state.values_list["workclass"]
            )
            education = st.selectbox(
                "Education", st.session_state.values_list["education"]
            )
            marital_status = st.selectbox(
                "Marital Status", st.session_state.values_list["marital-status"]
            )
            occupation = st.selectbox(
                "Occupation", st.session_state.values_list["occupation"]
            )
            relationship = st.selectbox(
                "Relationship", st.session_state.values_list["relationship"]
            )

        with col2:
            race = st.selectbox("Race", st.session_state.values_list["race"])
            sex = st.selectbox("Sex", st.session_state.values_list["sex"])

            capital_gain = st.number_input("Capital Gain", min_value=0, value=2174)
            capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
            hours_per_week = st.number_input(
                "Hours/Week", min_value=1, max_value=99, value=40
            )
            native_country = st.selectbox(
                "Native Country", st.session_state.values_list["native-country"]
            )
            fnlgt = st.number_input("FNLGT", min_value=0, value=77516)
            education_num = st.number_input(
                "Education Years",
                min_value=1,
                max_value=16,
                value=13,
                help="Number of years of education",
            )

        submitted = st.form_submit_button("Predict Salary")

        if submitted:
            census_data = {
                "age": age,
                "workclass": workclass,
                "fnlgt": fnlgt,
                "education": education,
                "education-num": education_num,
                "marital-status": marital_status,
                "occupation": occupation,
                "relationship": relationship,
                "race": race,
                "sex": sex,
                "capital-gain": capital_gain,
                "capital-loss": capital_loss,
                "hours-per-week": hours_per_week,
                "native-country": native_country,
            }

            try:
                response = requests.post(
                    st.session_state.fastapi_predict_url, json=census_data, timeout=10
                )
                if response.status_code == 200:
                    prediction = response.json().get("Salary prediction", "Unknown")
                    st.success(f"Prediction Result: {prediction}")
                else:
                    st.error(f"API Error: {response.text}")
            except requests.ConnectionError:
                st.error("Could not connect to API server")


def show_data_view():
    """
    display about page with static text
    """
    st.header("About")
    st.text("Udacity Machine Learning DevOps Engineer - Nanodegree")
    st.text("Project 3 - Deploying scalable ML pipeline in production")
    st.text("Use the Make prediction tab to run prediction")


def main():
    """
    main function
    it initialized the parameters based on the values in dvc.params file

    displays the main page with available options
    """
    if "initialized" not in st.session_state or not st.session_state.initialized:
        # print("inside initialization")

        params = dvc.api.params_show()
        categories = params["cat_features"]
        clean_data_path = params["data"]["path"]
        clean_data_file = params["data"]["clean_file"]

        fastapi_base_url = params["fastapi_base_url"]

        # global FASTAPI_BASE_URL, PREDICT_ENDPOINT, DATA_ENDPOINT
        # # FASTAPI_BASE_URL = "http://localhost:8000"

        # FASTAPI_BASE_URL = params['fastapi_base_url']
        # PREDICT_ENDPOINT = f"{FASTAPI_BASE_URL}/predict"
        # DATA_ENDPOINT = f"{FASTAPI_BASE_URL}/data"

        filepath = os.path.join(clean_data_path, clean_data_file)
        df = pd.read_csv(filepath)

        values_list = {}
        for category in categories:
            values_list[category] = df[category].unique().tolist()

        st.session_state.values_list = values_list
        st.session_state.fastapi_predict_url = f"{fastapi_base_url}/predict"

        print(f"url: {fastapi_base_url}, sessoin_url {st.session_state.fastapi_predict_url}")

        df = None

        # st.write("Initializing category values")
        st.session_state.initialized = (
            True  # Mark initialization as done to avoid re-running
        )

    # else:
    #     # print("intialization failed")
    #     st.write("Initialization already done.")

    st.sidebar.title("Census Data API")
    action = st.sidebar.radio("Select Operation", ["About", "Make Prediction"])

    if action == "Make Prediction":
        show_prediction_form()
    else:
        show_data_view()


if __name__ == "__main__":
    main()
