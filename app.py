import streamlit as st
import pandas as pd
import pickle

st.title("📊 Telco Customer Churn Prediction")

st.write("Upload a **test CSV file** to predict customer churn.")

# Load model
@st.cache_resource
def load_model():
    with open("models/churn_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)

    if "Churn" in test_df.columns:
        test_df = test_df.drop(columns=["Churn"])

    # Fix TotalCharges
    test_df["TotalCharges"] = pd.to_numeric(
        test_df["TotalCharges"], errors="coerce"
    )
    test_df.dropna(inplace=True)

    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)[:, 1]

    test_df["Churn_Prediction"] = predictions
    test_df["Churn_Probability"] = probabilities

    test_df["Churn_Prediction"] = test_df["Churn_Prediction"].map({
        0: "No",
        1: "Yes"
    })

    st.success("Prediction completed 🎉")
    st.dataframe(test_df.head(20))

    st.download_button(
        "Download Results",
        test_df.to_csv(index=False),
        file_name="churn_predictions.csv"
    )
