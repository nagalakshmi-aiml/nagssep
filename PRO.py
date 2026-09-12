import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier

# =========================================================
# TRAIN + SAVE MODEL
# =========================================================
def train_and_save_model(df, target_col):
    df = df.copy()

    # ----------- CLEAN & MAP TARGET -----------
    if df[target_col].dtype == "object":
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"approved": 1, "rejected": 0})
        )

    # Drop rows with invalid target
    df = df.dropna(subset=[target_col])

    # ----------- FILL FEATURE MISSING VALUES -----------
    for col in df.columns:
        if col == target_col:
            continue

        if df[col].dtype == "object":
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
            else:
                df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    # Split features & target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # ----------- ONE-HOT ENCODING -----------
    X = pd.get_dummies(X)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ----------- TRAIN MODEL -----------
    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ----------- EVALUATION -----------
    y_pred = model.predict(X_test)

    st.success(f"✅ Model trained successfully! Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    st.text("📊 Classification Report:\n" + classification_report(y_test, y_pred))

    # ----------- SAVE MODEL + COLUMNS -----------
    joblib.dump(model, "loan_lightgbm_model.pkl")
    joblib.dump(X.columns.tolist(), "loan_feature_columns.pkl")

    st.info("💾 Model saved as loan_lightgbm_model.pkl")


# =========================================================
# PREDICT CUSTOMER
# =========================================================
def predict_customer(customer_data):
    model = joblib.load("loan_lightgbm_model.pkl")
    columns = joblib.load("loan_feature_columns.pkl")

    df_new = pd.DataFrame([customer_data])

    # One-hot encode
    df_new = pd.get_dummies(df_new)

    # Add missing columns
    for col in columns:
        if col not in df_new.columns:
            df_new[col] = 0

    # Remove extra columns (safety)
    df_new = df_new[columns]

    # Predict
    pred = int(model.predict(df_new)[0])
    prob = model.predict_proba(df_new)[0][1] * 100

    if pred == 1:
        st.success(f"✅ Customer is ELIGIBLE (Approved) | Probability: {prob:.2f}%")
    else:
        st.error(f"❌ Customer is NOT ELIGIBLE (Rejected) | Probability: {prob:.2f}%")


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Loan Eligibility Prediction", layout="centered")
st.title("📊 Loan Eligibility Prediction (LightGBM)")

uploaded_file = st.file_uploader("Upload your loan dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Preview of uploaded data:", df.head())

    target_col = st.selectbox("🎯 Select target column", df.columns)

    if st.button("🚀 Train Model"):
        train_and_save_model(df, target_col)

    st.divider()
    st.subheader("🔍 Test Loan Eligibility Prediction")

    customer = {}
    for col in df.drop(columns=[target_col]).columns:
        if df[col].dtype == "object":
            customer[col] = st.text_input(f"{col}")
        else:
            customer[col] = st.number_input(
                f"{col}",
                value=float(df[col].median())
            )

    if st.button("🔮 Predict Eligibility"):
        predict_customer(customer)
