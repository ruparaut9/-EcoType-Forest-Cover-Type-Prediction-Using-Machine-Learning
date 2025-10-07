
# 📦 Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import os

# ---------------------------------------------
# 🔧 Load model and label encoder
# ---------------------------------------------
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "..", "models", "best_rf_model.pkl")
encoder_path = os.path.join(current_dir, "..", "models", "label_encoder.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

try:
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
except:
    label_encoder = None

# ---------------------------------------------
# 📊 Load training dataset to extract slider limits
# ---------------------------------------------
# Replace with your actual dataset filename
data_path = os.path.join(current_dir, "..", "data", "cleaned_cover_type.csv")
df = pd.read_csv(data_path)

# ---------------------------------------------
# 🌲 Forest Cover Type Mapping
# ---------------------------------------------
cover_type_map = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# 🏞️ Wilderness Area Mapping
wilderness_map = {
    "Wilderness_Area_1": "Rawah",
    "Wilderness_Area_4": "Cache la Poudre"
}

# ---------------------------------------------
# 🖼️ Set up the Streamlit page
# ---------------------------------------------
st.set_page_config(page_title="Forest Cover Type Predictor", layout="centered")
st.title("🌲 Forest Cover Type Prediction")
st.markdown("Enter forest features below to predict the forest cover type.")

# ---------------------------------------------
# 🧾 Use a form to prevent reload on slider change
# ---------------------------------------------
with st.form("prediction_form"):
    # 🌄 Terrain features
    elevation = st.slider("Elevation (meters above sea level)", 
                          int(df["Elevation"].min()), 
                          int(df["Elevation"].max()), 
                          int(df["Elevation"].mean()))

    aspect = st.slider("Aspect (degrees)", 
                       int(df["Aspect"].min()), 
                       int(df["Aspect"].max()), 
                       int(df["Aspect"].mean()))

    hillshade_9am = st.slider("Hillshade at 9am (0–255)", 
                              int(df["Hillshade_9am"].min()), 
                              int(df["Hillshade_9am"].max()), 
                              int(df["Hillshade_9am"].mean()))

    hillshade_noon = st.slider("Hillshade at Noon (0–255)", 
                               int(df["Hillshade_Noon"].min()), 
                               int(df["Hillshade_Noon"].max()), 
                               int(df["Hillshade_Noon"].mean()))

    hillshade_3pm = st.slider("Hillshade at 3pm (0–255)", 
                              int(df["Hillshade_3pm"].min()), 
                              int(df["Hillshade_3pm"].max()), 
                              int(df["Hillshade_3pm"].mean()))

    # 🛣️ Distance features
    distance_to_roadways = st.slider("Distance to Roadways (meters)", 
                                     int(df["Horizontal_Distance_To_Roadways"].min()), 
                                     int(df["Horizontal_Distance_To_Roadways"].max()), 
                                     int(df["Horizontal_Distance_To_Roadways"].mean()))

    distance_to_fire_points = st.slider("Distance to Fire Points (meters)", 
                                        int(df["Horizontal_Distance_To_Fire_Points"].min()), 
                                        int(df["Horizontal_Distance_To_Fire_Points"].max()), 
                                        int(df["Horizontal_Distance_To_Fire_Points"].mean()))

    distance_to_hydrology = st.slider("Distance to Hydrology (meters)", 
                                      int(df["Horizontal_Distance_To_Hydrology"].min()), 
                                      int(df["Horizontal_Distance_To_Hydrology"].max()), 
                                      int(df["Horizontal_Distance_To_Hydrology"].mean()))

    vertical_distance_to_hydrology = st.slider("Vertical Distance to Hydrology (meters)", 
                                               int(df["Vertical_Distance_To_Hydrology"].min()), 
                                               int(df["Vertical_Distance_To_Hydrology"].max()), 
                                               int(df["Vertical_Distance_To_Hydrology"].mean()))

    # 🏞️ Wilderness areas (binary indicators)
    wilderness_area_1 = st.selectbox("Is Wilderness Area Rawah present?", options=[0, 1])
    wilderness_area_4 = st.selectbox("Is Wilderness Area Cache la Poudre present?", options=[0, 1])

    # 🔘 Submit button
    submitted = st.form_submit_button("Predict Forest Cover Type")

# ---------------------------------------------
# 🧮 Compute engineered features and predict
# ---------------------------------------------
if submitted:
    # Compute engineered features from raw inputs
    fire_road_ratio = distance_to_fire_points / distance_to_roadways if distance_to_roadways != 0 else 0
    hydrology_road_ratio = distance_to_hydrology / distance_to_roadways if distance_to_roadways != 0 else 0
    morning_vs_noon_shade = hillshade_9am - hillshade_noon
    noon_vs_evening_shade = hillshade_noon - hillshade_3pm

    # Prepare input DataFrame with model's expected features
    input_data = pd.DataFrame([{
        "Elevation": elevation,
        "Horizontal_Distance_To_Roadways": distance_to_roadways,
        "Horizontal_Distance_To_Fire_Points": distance_to_fire_points,
        "Fire_Road_Ratio": fire_road_ratio,
        "Hydrology_Road_Ratio": hydrology_road_ratio,
        "Vertical_Distance_To_Hydrology": vertical_distance_to_hydrology,
        "Wilderness_Area_1": wilderness_area_1,
        "Horizontal_Distance_To_Hydrology": distance_to_hydrology,
        "Noon_vs_Evening_Shade": noon_vs_evening_shade,
        "Aspect": aspect,
        "Morning_vs_Noon_Shade": morning_vs_noon_shade,
        "Hillshade_Noon": hillshade_noon,
        "Hillshade_3pm": hillshade_3pm,
        "Hillshade_9am": hillshade_9am,
        "Wilderness_Area_4": wilderness_area_4
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]
    if label_encoder:
        prediction = label_encoder.inverse_transform([prediction])[0]

    # Map prediction to readable forest type
    readable_prediction = cover_type_map.get(prediction, f"Type {prediction}")

    # Map wilderness areas to readable names
    selected_areas = []
    if wilderness_area_1 == 1:
        selected_areas.append(wilderness_map["Wilderness_Area_1"])
    if wilderness_area_4 == 1:
        selected_areas.append(wilderness_map["Wilderness_Area_4"])
    area_display = ", ".join(selected_areas) if selected_areas else "None"

    # 🎯 Display results
    st.markdown(f"🏞️ Selected Wilderness Area(s): **{area_display}**")
    st.success(f"🌳 Predicted Forest Cover Type: **{readable_prediction}**")
