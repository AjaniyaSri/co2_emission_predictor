# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import r2_score

# =========================
# Load trained model & data
# =========================
model = joblib.load("final_model.pkl")
df = pd.read_csv("cleaned_vehicle_dataset.csv")

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="CO₂ Emissions Calculator",
    page_icon="🚗",
    layout="wide"
)

# =========================
# Title
# =========================
st.title("🚗 Vehicle CO₂ Emissions Calculator")
st.markdown("Find out how much CO₂ your vehicle produces and compare it with similar vehicles.")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🔧 Enter Vehicle Specifications")

engine_size = st.sidebar.number_input("Engine size (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
cylinders = st.sidebar.number_input("Number of Cylinders", min_value=3, max_value=12, step=1, value=4)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Ethanol", "Natural Gas", "Premium Petrol"]
)

combined_l_100km = st.sidebar.number_input("Combined (L/100 km)", min_value=2.0, max_value=25.0, step=0.1, value=8.5)

# =========================
# Prediction
# =========================
if st.sidebar.button("🔍 Calculate Emissions", type="primary"):

    # ---- Single input prediction ----
    input_data = pd.DataFrame({
        "Engine size (L)": [engine_size],
        "Cylinders": [cylinders],
        "Fuel type": [fuel_type],
        "Combined (L/100 km)": [combined_l_100km]
    })

    prediction = model.predict(input_data)[0]

    avg_emission = df["CO2 emissions (g/km)"].mean()
    difference = prediction - avg_emission
    percent_diff = (difference / avg_emission) * 100

    # =========================
    # Rating logic
    # =========================
    def get_emission_rating(value):
        if value < 150:
            return "✅ Excellent", "Very low emissions", "success"
        elif value < 200:
            return "✅ Good", "Below average emissions", "success"
        elif value < 250:
            return "⚠️ Average", "Average emissions", "warning"
        else:
            return "❌ High", "Above average emissions", "error"

    rating, rating_msg, rating_color = get_emission_rating(prediction)

    # =========================
    # Metrics
    # =========================
    st.markdown("---")
    st.subheader("📊 CO₂ Emission Results")

    m1, m2 = st.columns(2)
    with m1:
        st.metric(
            "Your Vehicle",
            f"{prediction:.0f} g/km",
            f"{difference:.0f} g/km",
            delta_color="inverse"
        )
    with m2:
        st.metric("Average Vehicle", f"{avg_emission:.0f} g/km")

    if difference < 0:
        st.success(
            f"Your vehicle produces **{abs(difference):.0f} g/km less CO₂** "
            f"({abs(percent_diff):.1f}% better than average)"
        )
    else:
        st.error(
            f"Your vehicle produces **{difference:.0f} g/km more CO₂** "
            f"({percent_diff:.1f}% higher than average)"
        )

    # =========================
    # Environmental Impact
    # =========================
    st.markdown("---")
    st.subheader("🌍 Annual Environmental Impact")

    annual_km = 15000
    annual_co2 = (prediction * annual_km) / 1000

    trees_needed = int(annual_co2 / 21)
    flight_hours = int(annual_co2 / 90)

    st.markdown(f"""
    - **Annual CO₂ emissions:** {annual_co2:.0f} kg  
    - 🌳 Trees required to offset: **{trees_needed}**  
    - ✈️ Equivalent to a **{flight_hours}-hour flight**
    """)

    # =========================
    # Tips
    # =========================
    st.markdown("---")
    st.subheader("💡 Tips to Reduce Emissions")

    tips = []
    if combined_l_100km > 8:
        tips.append("🚗 Reduce fuel consumption with smooth driving.")
    if engine_size > 2.5:
        tips.append("🔧 Smaller engines generally emit less CO₂.")
    if cylinders > 6:
        tips.append("⚙️ Fewer cylinders = better efficiency.")

    tips.extend([
        "🔋 Consider hybrid or electric vehicles.",
        "🚴 Use public transport for short trips.",
        "🛠️ Keep your vehicle well-maintained."
    ])

    for tip in tips[:4]:
        st.markdown(f"- {tip}")

    # =========================
    # Charts
    # =========================
    st.markdown("---")
    st.subheader("📈 Visual Comparison")

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots()
        ax.bar(["Your Vehicle", "Average Vehicle"], [prediction, avg_emission])
        ax.set_ylabel("CO₂ Emissions (g/km)")
        ax.set_title("Your Vehicle vs Average")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots()
        ax.hist(df["CO2 emissions (g/km)"], bins=40, alpha=0.6)
        ax.axvline(prediction, linestyle="--", label="Your Vehicle")
        ax.axvline(avg_emission, linestyle=":", label="Average")
        ax.legend()
        ax.set_title("Emission Distribution")
        st.pyplot(fig)

    # =========================
    # Model Evaluation Plots
    # =========================
    st.markdown("---")
    st.subheader("🧪 Model Evaluation (on dataset)")

    # Prepare X, y from dataset (must match training columns!)
    feature_cols = ["Engine size (L)", "Cylinders", "Fuel type", "Combined (L/100 km)"]
    target_col = "CO2 emissions (g/km)"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Predict for whole dataset (only works if model can handle Fuel type properly)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    col1, col2 = st.columns(2)

    # 1. Actual vs Predicted Scatter Plot
    with col1:
        st.write("### 📈 Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x=y, y=y_pred, alpha=0.6, ax=ax)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        ax.set_xlabel("Actual CO₂ emissions (g/km)")
        ax.set_ylabel("Predicted CO₂ emissions (g/km)")
        ax.set_title(f"R² = {r2:.3f}")
        st.pyplot(fig)

    # 2. Residuals Plot
    with col2:
        st.write("### 📊 Residuals Distribution")
        residuals = y - y_pred
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(residuals, bins=30, kde=True, ax=ax, color="purple")
        ax.set_xlabel("Prediction Error (g/km)")
        ax.set_title("Residuals Distribution")
        st.pyplot(fig)

    # =========================
    # Rating Box
    # =========================
    st.markdown("---")
    if rating_color == "success":
        st.success(f"**{rating}** — {rating_msg}")
    elif rating_color == "warning":
        st.warning(f"**{rating}** — {rating_msg}")
    else:
        st.error(f"**{rating}** — {rating_msg}")

else:
    st.info("👈 Enter vehicle details in the sidebar and click **Calculate Emissions**")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("Built with ❤️ using **Python, Streamlit & Machine Learning**")
