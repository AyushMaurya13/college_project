import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# --- Page Config & CSS ---
st.set_page_config(
    page_title="Crop Yield Predictor 🌾",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the UI
st.markdown("""
<style>
    body { background-color: #0f111a; color: #ffffff; }
    .section-box {
        background: #1f2233;
        padding: 25px;
        border-radius: 20px;
        border: 2px solid #6c5ce7;
        box-shadow: 6px 6px 20px rgba(0,0,0,0.4);
        margin-bottom: 25px;
    }
    .section-box h3 { color: #00cec9; }
    .section-box .stNumberInput, .section-box .stSelectbox { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Indian Crop Yield Prediction System")

# --- Season & Crop Data ---
# Consolidate season data into a single, clean dictionary
SEASONS_DATA = {
    "Kharif (Jun–Oct)": [
        "Rice", "Maize", "Cotton", "Soybean", "Sugarcane",
        "Groundnut", "Millets", "Sorghum", "Jowar", "Pigeon Pea"
    ],
    "Rabi (Nov–Apr)": [
        "Wheat", "Barley", "Gram", "Mustard", "Lentils",
        "Peas", "Sunflower", "Rapeseed", "Oats", "Chickpea"
    ],
    "Zaid (Mar–Jun)": [
        "Watermelon", "Cucumber", "Muskmelon", "Vegetables", "Tomato",
        "Pumpkin", "Cabbage", "Okra", "Brinjal", "Bitter Gourd"
    ]
}

# --- Generate Dataset & Train Model ---
@st.cache_data
def generate_and_train_model():
    """Generates a dummy dataset and trains the ML model."""
    np.random.seed(42)
    n_samples = 3000
    df = pd.DataFrame({
        "Rainfall (mm)": np.random.uniform(100, 2000, n_samples),
        "Temperature (°C)": np.random.uniform(10, 40, n_samples),
        "Pesticide Usage (kg/l)": np.random.uniform(0, 20, n_samples),
        "Fertilizer (kg)": np.random.uniform(0, 200, n_samples),
    })
    df["Yield (tons/ha)"] = (
        0.002 * df["Rainfall (mm)"] +
        0.05 * df["Temperature (°C)"] -
        0.01 * df["Pesticide Usage (kg/l)"] +
        0.03 * df["Fertilizer (kg)"] +
        np.random.normal(0, 2, n_samples)
    )

    X_train = df[["Rainfall (mm)", "Temperature (°C)", "Pesticide Usage (kg/l)", "Fertilizer (kg)"]]
    y_train = df["Yield (tons/ha)"]
    model = LinearRegression().fit(X_train, y_train)
    return model, df

model, df = generate_and_train_model()
st.success("✅ Dataset and ML model loaded successfully!")

# Display a sample of the data
if st.checkbox('Show sample data'):
    st.dataframe(df.sample(5))

# --- Prediction Function ---
def predict_yield_ml(rainfall, temp, pesticide, fertilizer):
    """Predicts crop yield using the trained ML model."""
    input_df = pd.DataFrame([[rainfall, temp, pesticide, fertilizer]],
                            columns=["Rainfall (mm)", "Temperature (°C)", "Pesticide Usage (kg/l)", "Fertilizer (kg)"])
    return round(model.predict(input_df)[0], 2)

# --- Main App Logic (UI) ---
for season, crops in SEASONS_DATA.items():
    with st.container():
        # Removed: st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown(f"### {season} 🌱")

        # Environmental Inputs in two columns
        cols = st.columns(2)
        with cols[0]:
            rainfall = st.number_input("Rainfall (mm)", 0.0, 2000.0, 500.0, key=f"rainfall_{season}")
            temperature = st.number_input("Average Temperature (°C)", 0.0, 50.0, 25.0, key=f"temp_{season}")
        with cols[1]:
            pesticide = st.number_input("Pesticide Usage (kg/l)", 0.0, 100.0, 5.0, key=f"pest_{season}")
            fertilizer = st.number_input("Fertilizer Application (kg)", 0.0, 500.0, 50.0, key=f"fert_{season}")

        # Crop Selection and Prediction Button
        crop = st.selectbox("Select Crop", crops, key=f"crop_{season}")
        if st.button(f"Predict Yield for {season}", key=f"predict_{season}"):
            yield_pred = predict_yield_ml(rainfall, temperature, pesticide, fertilizer)
            st.success(f"🌾 Estimated Yield for {crop}: **{yield_pred} tons/ha**")

            # Data for Plotly chart
            df_plot = pd.DataFrame({
                "Factor": ["Rainfall (mm)", "Temperature (°C)", "Pesticide Usage (kg/l)", "Fertilizer (kg)"],
                "Value": [rainfall, temperature, pesticide, fertilizer]
            })
            fig = px.bar(df_plot, x="Factor", y="Value", color="Value", text="Value",
                         title=f"🌱 {crop} Input Values")
            fig.update_layout(plot_bgcolor='#1f2233', paper_bgcolor='#1f2233', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        # Removed: st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("<hr style='border:1px solid #444;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Developed with ❤️ for Indian Farmers | Crop Yield Predictor 🌱</p>", unsafe_allow_html=True)