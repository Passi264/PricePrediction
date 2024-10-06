import streamlit as st
import pickle
import json
import numpy as np
import sklearn

# Load the model and columns
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    with open("columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 are sqft, bath, bhk

    with open("banglore_home_prices_model.pickle", "rb") as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")

def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

# Load the model and columns when the app starts
load_saved_artifacts()

# Streamlit app
st.title("Bangalore House Price Prediction")

# Input fields
st.header("Enter the details:")

sqft = st.number_input("Total Square Feet", min_value=300.0, max_value=10000.0, step=100.0, value=1000.0)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1, value=2)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1, value=3)

# Location select box from pre-loaded locations
location = st.selectbox("Location", __locations)

# When the user clicks the "Predict" button
if st.button("Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"The estimated price for the house is ₹ {price} Lakh.")

