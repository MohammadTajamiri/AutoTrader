"""
app.py

Streamlit dashboard: type in a car's details, get the predicted price
plus a SHAP breakdown of how much each input is expected to be
dragging the model's error up or down for a car shaped like this one.

Run from the `src` directory with: streamlit run app.py
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from AutoDealer.config import CONFIG
from AutoDealer.inference import preprocess_single_input
from AutoDealer.predict import predict
from AutoDealer.error_model import explain_row

st.set_page_config(page_title="AutoDealer Price Check", layout="centered")
st.title("Used Car Price Check")


@st.cache_data
def load_known_values():
    """
    Pulls Make/Model/transmission options from the already-built
    training dataset, so the dropdowns offer values that actually
    match the project's own Make/Model spelling conventions - typing
    a Model that doesn't match EngineType_enriched.xlsx's spelling is
    exactly what causes Cylinders/Body Type to fall back to "Unknown".
    """
    df = pd.read_csv(CONFIG.raw_data_path)
    makes = sorted(df["Make"].dropna().unique().tolist())
    make_to_models = {
        make: sorted(df.loc[df["Make"] == make, "Model"].dropna().unique().tolist())
        for make in makes
    }
    transmissions = sorted(df["transmission"].dropna().unique().tolist())
    return makes, make_to_models, transmissions


makes, make_to_models, transmissions = load_known_values()

with st.form("car_form"):
    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox("Make", makes)
        model = st.selectbox("Model", make_to_models.get(make, []))
        trim = st.text_input("Trim (optional)", "")
        year = st.number_input("Year", min_value=1990, max_value=CONFIG.current_year, value=2022)
        mileage = st.number_input("Mileage (km)", min_value=0, value=15000, step=1000)
    with col2:
        transmission = st.selectbox("Transmission", transmissions)
        is_hybrid = st.checkbox("Hybrid")
        no_accidents = st.checkbox("No accidents reported")
        has_carfax = st.checkbox("Carfax report available")
        one_owner = st.checkbox("One owner")
        service_records = st.checkbox("Service records available")
        certified = st.checkbox("Certified pre-owned")

    submitted = st.form_submit_button("Predict")

if submitted:
    raw = {
        "Make": make,
        "Model": model,
        "trim": trim,
        "Year": year,
        "mileage": float(mileage),
        "transmission": transmission,
        "fuel_type": "Hybrid" if is_hybrid else "Gas",
        "Is_Hybrid": int(is_hybrid),
        "no_accidents": int(no_accidents),
        "has_carfax": int(has_carfax),
        "one_owner": int(one_owner),
        "service_records": int(service_records),
        "certified": int(certified),
    }

    X = preprocess_single_input(raw)

    st.subheader("Detected attributes")
    st.write(X[["Cylinders", "Body Type", "Brand_Segment", "std_tier", "Age"]])
    if (X["Cylinders"] == "Unknown").any() or (X["Body Type"] == "Unknown").any():
        st.warning(
            "Cylinders/Body Type came back 'Unknown' - this Make/Model combination "
            "wasn't found in EngineType_enriched.xlsx. The prediction will still run, "
            "but without this signal it's less reliable."
        )

    predicted_price = predict(X)[0]
    st.metric("Predicted price", f"${predicted_price:,.0f}")

    explanation = explain_row(X)
    base_value = explanation.pop("base_value")
    expected_error = explanation.pop("expected_error")

    st.subheader("Expected error for cars like this")
    st.write(
        f"Based on the price model's own historical errors on similar cars, "
        f"this model's predictions for cars shaped like this one typically run "
        f"**${expected_error:,.0f}** off (baseline ${base_value:,.0f})."
    )

    contributions = pd.Series(explanation).sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(contributions) + 1))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in contributions.values]
    ax.barh(contributions.index, contributions.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to expected error ($)")
    ax.set_title("Which inputs are driving the expected error")
    st.pyplot(fig)