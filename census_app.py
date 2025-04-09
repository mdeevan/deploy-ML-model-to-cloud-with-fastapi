# census_app.py - Streamlit Application
import streamlit as st
import requests
from pydantic import BaseModel, Field
from typing import Optional
import dvc.api
import pandas as pd
import os

# Define Pydantic model matching FastAPI
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example='State-gov')
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example='Bachelors')
    education_num: int = Field(..., alias='education-num', example=13)
    marital_status: str = Field(..., alias='marital-status', example='Never-married')
    occupation: str = Field(..., example='Adm-clerical')
    relationship: str = Field(..., example='Not-in-family')
    race: str = Field(..., example='White')
    sex: str = Field(..., example='Female')
    capital_gain: int = Field(..., alias='capital-gain', example=2174)
    capital_loss: int = Field(..., alias='capital-loss', example=0)
    hours_per_week: int = Field(..., alias='hours-per-week', example=40)
    native_country: str = Field(..., alias='native-country', example='United-States')
    salary: Optional[str] = Field(None)

# API configuration
FASTAPI_BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{FASTAPI_BASE_URL}/predict"
DATA_ENDPOINT = f"{FASTAPI_BASE_URL}/data"

class selections():
    '''
        static class to hold the values of the category variables
    '''

    values_list = {}


    @staticmethod
    def get_category_values():
        params = dvc.api.params_show()
        categories = params['cat_features']
        clean_data_path = params['data']['path']
        clean_data_file = params['data']['clean_file']
        filepath = os.path.join(clean_data_path, clean_data_file)
        df = pd.read_csv(filepath)

        for category in categories:
            selections.values_list[category] = df[category].unique().tolist()


def show_prediction_form():
    st.header("Salary Prediction (POST)")
    
    with st.form(key='census_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=17, max_value=90, value=39)
            workclass = st.selectbox(
                "Workclass",
                selections.values_list['workclass']
                # ["State-gov", "Private", "Self-emp", "Federal-gov", "Local-gov"]
            )
            education = st.selectbox(
                "Education",
                selections.values_list['education']
                # ["Bachelors", "Masters", "HS-grad", "Some-college", "Doctorate"]
            )
            marital_status = st.selectbox(
                "Marital Status",
                selections.values_list['marital-status']
                # ["Never-married", "Married", "Divorced", "Separated", "Widowed"]
            )
            occupation = st.selectbox(
                "Occupation",
                selections.values_list['occupation']
                # ["Adm-clerical", "Exec-managerial", "Prof-specialty", "Tech-support", "Other"]
            )
            relationship = st.selectbox(
                "Relationship",
                selections.values_list['relationship']
                # ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried"]
            )
            
        with col2:
            race = st.selectbox(
                "Race",
                selections.values_list['race']
                # ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
            )
            sex = st.selectbox("Sex", 
                            #    ["Female", "Male"]
                               selections.values_list['sex']
                               )
                
            capital_gain = st.number_input("Capital Gain", min_value=0, value=2174)
            capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
            hours_per_week = st.number_input("Hours/Week", min_value=1, max_value=99, value=40)
            native_country = st.selectbox(
                "Native Country",
                selections.values_list['native-country']
                # ["United-States", "Mexico", "Canada", "Germany", "Philippines"]
            )
            fnlgt = st.number_input("FNLGT", min_value=0, value=77516)
            education_num = st.number_input(
                "Education Years", 
                min_value=1, 
                max_value=16, 
                value=13,
                help="Number of years of education"
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
                "native-country": native_country
            }
            
            try:
                response = requests.post(PREDICT_ENDPOINT, json=census_data)
                if response.status_code == 200:
                    prediction = response.json().get("Salary prediction", "Unknown")
                    st.success(f"Prediction Result: {prediction}")
                else:
                    st.error(f"API Error: {response.text}")
            except requests.ConnectionError:
                st.error("Could not connect to API server")

def show_data_view():
    st.header("Historical Data (GET)")
    if st.button("Fetch Data"):
        try:
            response = requests.get(DATA_ENDPOINT)
            if response.status_code == 200:
                data = response.json()
                if data:
                    st.dataframe(data)
                else:
                    st.info("No data available")
            else:
                st.error(f"API Error: {response.text}")
        except requests.ConnectionError:
            st.error("Could not connect to API server")

def main():
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        
        ### TODO: make the dictionary initilaization here
        params = dvc.api.params_show()
        categories = params['cat_features']
        clean_data_path = params['data']['path']
        clean_data_file = params['data']['clean_file']
        filepath = os.path.join(clean_data_path, clean_data_file)
        df = pd.read_csv(filepath)

        for category in categories:

        selections.get_category_values()
        print("inside initialization")

        st.write("Initializing category values")
        st.session_state.initialized = True  # Mark initialization as done to avoid re-running
        
    else:
        print("intialization failed")
        st.write("Initialization already done.")

    st.sidebar.title("Census Data API")
    action = st.sidebar.radio(
        "Select Operation",
        ["Make Prediction", "View Historical Data"]
    )
    
    if action == "Make Prediction":
        show_prediction_form()
    else:
        show_data_view()

if __name__ == "__main__":
    main()