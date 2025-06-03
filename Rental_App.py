import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Auckland Rent Estimator", page_icon="🏡")

st.title("🏡 Auckland Weekly Rent Estimator")
st.write("Enter property details to get a weekly rent estimate.")

@st.cache_resource
def load_model_and_features():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("xgb_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, feature_names

model, model_features = load_model_and_features()

suburb_list = sorted([f.replace("Suburb_", "") for f in model_features if f.startswith("Suburb_")])

suburb = st.selectbox("Suburb", suburb_list)
property_type = st.selectbox("Property Type", ["Apartment", "House"])
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=1)
parking = st.number_input("Parking Spots", min_value=0, max_value=5, value=1)

avg_median = 650
avg_upper = 780
avg_lower = 540
avg_bonds = 5000

features = {
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "HasParking": 1 if parking > 0 else 0,
    "BathsPerBedroom": bathrooms / bedrooms if bedrooms else 0,
    "Avg_MedianRent": avg_median,
    "Avg_UpperQuartileRent": avg_upper,
    "Avg_LowerQuartileRent": avg_lower,
    "Avg_LodgedBonds": avg_bonds,
    "Property Type_House": 1 if property_type == "House" else 0
}

for s in suburb_list:
    features[f"Suburb_{s}"] = 1 if suburb == s else 0

X_input = pd.DataFrame([features])
X_input = X_input.reindex(columns=model_features, fill_value=0)

if st.button("Predict Weekly Rent"):
    try:
        prediction = model.predict(X_input)[0]
        st.subheader(f"💰 Estimated Weekly Rent: ${prediction:.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
