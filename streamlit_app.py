import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from postcode_dict import postcode_dict
import json

# Load your trained model pipeline (with preprocessing included)
#model = joblib.load('models/xgb_model.pkl')  # or 'rf_model.pkl'

model_path = os.path.join(os.getcwd(), 'models', 'xgb_model.pkl')
model = joblib.load(model_path)

# Load precomputed postcode_avg_price_per_m2 dictionary
with open('postcode_avg_price.json', 'r') as f:
    postcode_avg_price = json.load(f)

# Title of the app
#st.title(" 🏡 Belgian Real Estate price prediction ")
st.markdown(
    "<h1 style='text-align: center;'>🏡 Belgian Real Estate price prediction</h1>",
    unsafe_allow_html=True
)

st.markdown("<h3 style='text-align: center;'>---  Get the price of your house  ---</h3>", unsafe_allow_html=True)


# Grouping user input for a sample real estate model
st.header("Enter Property Details")

# User input for property details
# Categorical features as dropdowns
col1, col2 = st.columns(2)
with col1:
    property_type = st.selectbox(
    "Property Type",
    [
        "Apartment",
        "House",
    ]
)
with col2:
    subtype = st.selectbox("Subtype", ['APARTMENT', 'APARTMENT_BLOCK', 'APARTMENT_GROUP', 'BUNGALOW', 'CASTLE', 'CHALET', 'COUNTRY_COTTAGE', 'DUPLEX', 'EXCEPTIONAL_PROPERTY',
                                 'FARMHOUSE', 'FLAT_STUDIO', 'GROUND_FLOOR', 'HOUSE', 'HOUSE_GROUP', 'KOT', 'LOFT', 'MANOR_HOUSE', 'MANSION', 'MIXED_USE_BUILDING',
                                 'OTHER_PROPERTY', 'PAVILION', 'PENTHOUSE', 'SERVICE_FLAT', 'TOWN_HOUSE', 'TRIPLEX', 'VILLA',])

col1, col2 = st.columns(2)
with col1:
    province = st.selectbox("Province", ['Antwerp', 'Brussels', 'East Flanders', 'Flemish Brabant', 'Hainaut', 'Limburg', 'Liège', 'Luxembourg', 'Namur', 'Walloon Brabant', 'West Flanders'])
with col2:
    postcode_options = [f"{code} - {locality}" for code, locality in postcode_dict.items()]

    # Streamlit selectbox supports typing to filter
    selected = st.selectbox("Postcode", sorted(postcode_options))

    # Extract and display
    postcode, locality = selected.split(" - ")

col1, col2 = st.columns(2)
with col1:
    bedroom_count = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
with col2:
    bathroom_count = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=1)

col1, col2 = st.columns(2)
with col1:
    facadeCount = st.number_input("Facade Count", min_value=1, max_value=4, value=2)
with col2:
    # EPC mapping: letter -> numeric score (based on province)
    epc_maps = {
    "Wallonia": {'A++': 0, 'A+': 30, 'A': 65, 'B': 125, 'C': 200, 'D': 300, 'E': 375, 'F': 450, 'G': 510},
    "Flanders": {'A++': 0, 'A+': 0, 'A': 50, 'B': 150, 'C': 250, 'D': 350, 'E': 450, 'F': 500, 'G': 510},
    "Brussels": {'A++': 0, 'A+': 0, 'A': 45, 'B': 75, 'C': 125, 'D': 175, 'E': 250, 'F': 300, 'G': 350}
}


    # User selects EPC letter
    epc_letter = st.selectbox("EPC Type", ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])

    # Map province to region for mapping
    wallonia_provinces = ['Liège', 'Walloon Brabant', 'Namur', 'Hainaut', 'Luxembourg']
    flanders_provinces = ['Antwerp', 'Flemish Brabant', 'East Flanders', 'West Flanders', 'Limburg']

    if province in wallonia_provinces:
        epc_score = epc_maps['Wallonia'][epc_letter]
    elif province in flanders_provinces:
        epc_score = epc_maps['Flanders'][epc_letter]
    else:
        epc_score = epc_maps['Brussels'][epc_letter]


landSurface = st.number_input("Land Surface (m²)", min_value=0, max_value=10000, value=200)
#landSurface = st.slider("Land Surface (m²)", min_value=0, max_value=10000, value=200, step=10)
habitableSurface = st.number_input("Habitable Surface (m²)", min_value=0, max_value=1000, value=150)
#habitableSurface = st.slider("Habitable Surface (m²)", min_value=0, max_value=2000, value=150, step=10)
buildingConstructionYear = st.slider("Building Construction Year", min_value=1800, max_value=2025, value=1990)


# Building Condition Mapping
buildingCondition_mapping = {
    "New": 0,
    "Good": 1,
    "To renovate": 2,
    "To restore": 3,
    "Just renovated": 4,
    "To be done up": 5
}

building_condition_label = st.selectbox(
    "Select the building condition",
    list(buildingCondition_mapping.keys())
)

# Get numeric value for prediction
building_condition_value = buildingCondition_mapping[building_condition_label]


# Flood Zone Type Mapping
flood_mapping = {
    "NON_FLOOD_ZONE": 0,
    "POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE": 1,
    "CIRCUMSCRIBED_WATERSIDE_ZONE": 2,
    "POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE": 3,
    "POSSIBLE_FLOOD_ZONE": 4,
    "CIRCUMSCRIBED_FLOOD_ZONE": 5,
    "RECOGNIZED_FLOOD_ZONE": 6,
    "RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE": 7,
    "RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE": 8
}

flood_label = st.selectbox(
    "Select the Flood Zone Type",
    list(flood_mapping.keys())
)

# Get the numeric value
flood_value = flood_mapping[flood_label]

# Boolean features
col1, col2 = st.columns(2)

with col1:
    hasLift = st.checkbox("Lift")
    hasHeatPump = st.checkbox("Heat Pump")
    hasPhotovoltaicPanels = st.checkbox("Photovoltaic Panels")
    hasAirConditioning = st.checkbox("Air Conditioning")
    hasVisiophone = st.checkbox("Visiophone")
    hasOffice = st.checkbox("Office")

with col2:
    hasSwimmingPool = st.checkbox("Swimming Pool")
    hasFireplace = st.checkbox("Fireplace")
    hasAttic = st.checkbox("Attic")
    parkingCountIndoor = st.checkbox("Indoor Parking")
    parkingCountOutdoor = st.checkbox("Outdoor Parking")

# Lookup postcode average price per m2
default_avg_price = np.mean(list(postcode_avg_price.values()))
postcode_avg = postcode_avg_price.get(postcode, default_avg_price)

# Prepare input features for the model
features = {
    "type": property_type,                        
    "subtype": subtype,
    "province": province,
    "postCode": postcode,                          
    "bedroomCount": bedroom_count,                 
    "bathroomCount": bathroom_count,
    "facadeCount": facadeCount,
    "epc_enum": epc_score,                         
    "landSurface": landSurface,
    "habitableSurface": habitableSurface,
    "buildingConstructionYear": buildingConstructionYear,
    "buildingCondition_mapping": building_condition_value,  
    "floodZoneType_mapping": flood_value,
    "hasLift": int(hasLift),
    "hasHeatPump": int(hasHeatPump),
    "hasPhotovoltaicPanels": int(hasPhotovoltaicPanels),
    "hasAirConditioning": int(hasAirConditioning),
    "hasVisiophone": int(hasVisiophone),
    "hasOffice": int(hasOffice),
    "hasSwimmingPool": int(hasSwimmingPool),
    "hasFireplace": int(hasFireplace),
    "hasAttic": int(hasAttic),
    "parkingCountIndoor": int(parkingCountIndoor),
    "parkingCountOutdoor": int(parkingCountOutdoor),
    "postcode_avg_price_per_m2": postcode_avg
}


# Convert to DataFrame for model input (one row)
input_df = pd.DataFrame([features])

# Button to predict
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🏠 Estimated Price: €{prediction:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")