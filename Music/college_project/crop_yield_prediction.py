import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Animated Crop Yield Predictor 🌾",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CSS Animations --------------------
st.markdown("""
<style>
body { background-color: #0f111a; color: #ffffff; }
h1 { animation: fadeIn 2s ease-in-out; }
.section-box {
    background: #1f2233;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 5px 5px 15px rgba(0,0,0,0.3);
    margin-bottom: 20px;
    animation: slideIn 1s ease-in-out;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(-20px);}
    100% { opacity: 1; transform: translateY(0);}
}
@keyframes slideIn {
    0% { opacity: 0; transform: translateX(-50px);}
    100% { opacity: 1; transform: translateX(0);}
}
</style>
""", unsafe_allow_html=True)

st.title("🌾 Indian Crop Yield Prediction System")

# -------------------- Seasons --------------------
seasons = ["Kharif (Jun–Oct)", "Rabi (Nov–Apr)", "Zaid (Mar–Jun)"]
season_dict = {
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


# -------------------- Generate Random Dataset with Units --------------------
np.random.seed(42)
n_samples = 3000

df = pd.DataFrame({
    "Rainfall (mm)": np.random.uniform(100, 2000, n_samples),
    "Temperature (°C)": np.random.uniform(10, 40, n_samples),
    "Pesticide Usage (kg/l)": np.random.uniform(0, 20, n_samples),
    "Fertilizer (kg)": np.random.uniform(0, 200, n_samples),
})

# Simulate Yield
df["Yield (tons/ha)"] = (0.002 * df["Rainfall (mm)"] +
                          0.05 * df["Temperature (°C)"] -
                          0.01 * df["Pesticide Usage (kg/l)"] +
                          0.03 * df["Fertilizer (kg)"] +
                          np.random.normal(0, 2, n_samples))

st.success("✅ Random dataset with units generated successfully!")
st.dataframe(df.sample(5))

# -------------------- ML Model Setup --------------------
X_train = df[["Rainfall (mm)", "Temperature (°C)", "Pesticide Usage (kg/l)", "Fertilizer (kg)"]]
y_train = df["Yield (tons/ha)"]

model = LinearRegression()
model.fit(X_train, y_train)

def predict_yield_ml(rainfall, temp, pesticide, fertilizer):
    input_df = pd.DataFrame({
        "Rainfall (mm)": [rainfall],
        "Temperature (°C)": [temp],
        "Pesticide Usage (kg/l)": [pesticide],
        "Fertilizer (kg)": [fertilizer]
    })
    prediction = model.predict(input_df)
    return round(prediction[0], 2)

# -------------------- Layout Sections --------------------
for season in seasons:
    with st.container():
        st.markdown(f'<div class="section-box"><h2>{season}</h2></div>', unsafe_allow_html=True)
        cols = st.columns(2)
        
        with cols[0]:
            st.subheader("Environmental Inputs 🌤️")
            rainfall = st.number_input("Rainfall (mm)", 0.0, 2000.0, 500.0, step=10.0, key=f"rainfall_{season}")
            temperature = st.number_input("Average Temperature (°C)", 0.0, 50.0, 25.0, step=0.5, key=f"temp_{season}")
            pesticide = st.number_input("Pesticide Usage (kg/l)", 0.0, 100.0, 5.0, step=0.5, key=f"pest_{season}")
            fertilizer = st.number_input("Fertilizer Application (kg)", 0.0, 500.0, 50.0, step=1.0, key=f"fert_{season}")
        
        with cols[1]:
            st.subheader("Crop Selection & Prediction 🌱")
            crop = st.selectbox("Select Crop", season_dict[season], key=f"crop_{season}")
            if st.button(f"Predict Yield for {season}", key=f"predict_{season}"):
                yield_pred = predict_yield_ml(rainfall, temperature, pesticide, fertilizer)
                st.success(f"🌾 Estimated Yield for {crop}: **{yield_pred} tons/ha**")
                
                # -------------------- Plotly Animated Bar Chart --------------------
                df_plot = pd.DataFrame({
                    "Factor": ["Rainfall (mm)", "Temperature (°C)", "Pesticide Usage (kg/l)", "Fertilizer (kg)"],
                    "Value": [rainfall, temperature, pesticide, fertilizer],
                    "Step": [1, 1, 1, 1]  # Placeholder for animation (can extend for multiple frames)
                })

                fig = px.bar(
                    df_plot,
                    x="Factor",
                    y="Value",
                    color="Value",
                    text="Value",
                    range_y=[0, max(df_plot["Value"])*1.2],
                    labels={"Value": "Input Value"},
                    title=f"🌱 {crop} Input Values"
                )
                fig.update_layout(
                    plot_bgcolor='#1f2233',
                    paper_bgcolor='#1f2233',
                    font_color='white'
                )

                st.plotly_chart(fig, use_container_width=True)

# -------------------- Footer --------------------
st.markdown("<hr style='border:1px solid #444;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Developed with ❤️ for Indian Farmers | Animated Crop Yield Predictor 🌱</p>", unsafe_allow_html=True)
