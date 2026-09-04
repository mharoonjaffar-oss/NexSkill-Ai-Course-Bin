import pandas as pd
import streamlit as st
import joblib

model = joblib.load("xgb_model.jb")

st.title("🏠 House Price Prediction")

inputs = [
    'OverallQual',
    'GrLivArea',
    'GarageArea',
    '1stFlrSF',
    'FullBath',
    'YearBuilt',
    'YearRemodAdd',
    'MasVnrArea',
    'Fireplaces',
    'BsmtFinSF1',
    'LotFrontage',
    'WoodDeckSF',
    'OpenPorchSF',
    'LotArea',
    'CentralAir'
]

input_data = {}

for feature in inputs:

    if feature == "CentralAir":
        input_data[feature] = st.selectbox(
            "Central Air",
            ["Yes", "No"]
        )
    else:
        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            step=1.0
        )

if st.button("Predict Price"):

    input_data["CentralAir"] = (
        1 if input_data["CentralAir"] == "Yes" else 0
    )

    input_df = pd.DataFrame(
        [input_data],
        columns=inputs
    )

    prediction = model.predict(input_df)[0]

    st.success(
        f"Predicted House Price: ${prediction:,.2f}"
    )