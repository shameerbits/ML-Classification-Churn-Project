import streamlit as st
import pandas as pd
import joblib
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import blank_to_nan, normalize_strings, coerce_numeric
import sklearn


# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("Telco Customer Churn Prediction Application")


# Load results.json
@st.cache_resource
def load_results():
    with open("models/results.json", "r") as f:
        return json.load(f)

results = load_results()

# Load Model
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "LogisticRegression": "models/logistic.pkl",
        "DecisionTree": "models/decision_tree.pkl",
        "KNN": "models/knn.pkl",
        "NaiveBayes": "models/naive_bayes.pkl",
        "RandomForest": "models/random_forest.pkl",
        "XGBoost": "models/xgboost.pkl"
    }
    return joblib.load(model_paths[model_name])


# Sidebar
st.sidebar.header("App Configuration")
model_name = st.sidebar.selectbox(
    "Select Model",
    list(results.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV only)",
    type=["csv"]
)


# Sample Dataset Download
st.sidebar.divider()
st.sidebar.subheader("Sample Dataset")

sample_df = pd.read_csv("dataset/telco_test.csv")
sample_csv = sample_df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="Download Sample telco_test.csv",
    data=sample_csv,
    file_name="telco_test.csv",
    mime="text/csv"
)


# Load Selected Model
model = load_model(model_name)

# CSV Uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    TARGET_COL = "Churn"
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].map({"Yes": 1, "No": 0})


    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    st.subheader(f"{model_name} - Evaluation Metrics & Confusion Matrix (Test Data)")

    left, right = st.columns([2, 1])
    with left:
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
            st.metric("Precision", f"{precision_score(y, y_pred):.4f}")

        with m2:
            st.metric("Recall", f"{recall_score(y, y_pred):.4f}")
            st.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")

        with m3:
            st.metric(
                "ROC-AUC",
                f"{roc_auc_score(y, y_prob):.4f}" if y_prob is not None else "N/A"
            )
            st.metric(
                "MCC",
                f"{matthews_corrcoef(y, y_pred):.4f}"
            )
    with right:
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(2.3, 2), dpi=150)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 8},
            ax=ax
        )
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

        fig.tight_layout()
        st.pyplot(fig, width='content')


    # Classification Report
    #st.subheader("Classification Report (Test Data)")
    #report = classification_report(y, y_pred, output_dict=True)
    #report_df = pd.DataFrame(report).transpose()
    #st.dataframe(report_df)


    # Download Predictions CSV
    st.subheader("Download Test Prediction Results")

    output_df = df.copy()
    output_df["Predicted_Churn"] = y_pred

    if y_prob is not None:
        output_df["Churn_Probability"] = y_prob

    result_csv = output_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=result_csv,
        file_name="telco_test_predictions.csv",
        mime="text/csv"
    )


# Training vs Evaluation Comparison
st.divider()
st.subheader(f"{model_name} - Training v/s Evaluation Performance")

train = results[model_name]["train"]
eval_ = results[model_name]["evaluate"]

comparison_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "MCC"],
    "Training": [
        train["accuracy"],
        train["precision"],
        train["recall"],
        train["f1"],
        train["roc_auc"],
        train["mcc"]
    ],
    "Evaluation": [
        eval_["accuracy"],
        eval_["precision"],
        eval_["recall"],
        eval_["f1"],
        eval_["roc_auc"],
        eval_["mcc"]
    ]
})

st.dataframe(comparison_df)


# Confusion Matrix Comparison
st.subheader(f"{model_name} - Confusion Matrices")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Training")
    fig1, ax1 = plt.subplots(figsize=(2.5, 2))
    sns.heatmap(
        train["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax1,
        cbar=False
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1, width='content')

with col2:
    st.markdown("### Evaluation")
    fig2, ax2 = plt.subplots(figsize=(2.5, 2))
    sns.heatmap(
        eval_["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Oranges",
        ax=ax2,
        cbar=False
    )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2, width='content')
