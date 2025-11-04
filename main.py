# main.py
# Streamlit hypertension / cardiovascular risk prediction
# Classes: Low / Medium / High

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Hypertension Prediction App", page_icon="🧠")

@st.cache_data(show_spinner=True)
def load_and_train_model(csv_path: str):
    df = pd.read_csv(csv_path)

    # --- New holistic risk labeling ---
    def categorize_risk(row):
        score = 0
        # Blood pressure
        if row["SystolicBP"] >= 140 or row["DiastolicBP"] >= 90:
            score += 2
        elif row["SystolicBP"] >= 120 or row["DiastolicBP"] >= 80:
            score += 1
        # BMI
        if "BMI" in row and row["BMI"] >= 30:
            score += 1
        # Cholesterol
        if "Cholesterol" in row and row["Cholesterol"] >= 240:
            score += 1
        # Glucose
        if "Glucose" in row and row["Glucose"] >= 126:
            score += 1
        # Lifestyle
        if "Smoking" in row and str(row["Smoking"]).lower() == "yes":
            score += 1
        if "Alcohol" in row and str(row["Alcohol"]).lower() == "yes":
            score += 1

        if score <= 1:
            return 0  # Low
        elif score <= 3:
            return 1  # Medium
        else:
            return 2  # High

    df["RiskLevel"] = df.apply(categorize_risk, axis=1)

    # Feature engineering
    df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]
    df["MAP"] = (2 * df["DiastolicBP"] + df["SystolicBP"]) / 3

    # Drop ID and raw BP (to avoid trivial leakage)
    drop_cols = [c for c in ["ID", "Hypertension"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    y = df["RiskLevel"]
    X = df.drop(columns=["RiskLevel"])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", rf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 15],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    clf_report_df = pd.DataFrame(clf_report).transpose()
    for col in ["precision", "recall", "f1-score"]:
        if col in clf_report_df.columns:
            clf_report_df[col] = (clf_report_df[col] * 100).round(2)

    # Permutation importance
    perm = permutation_importance(best_model, X_test, y_test,
                                  n_repeats=10, random_state=42, scoring="balanced_accuracy")

    preprocessor = best_model.named_steps["preprocessor"]
    feature_out = preprocessor.get_feature_names_out()
    n_features = len(perm.importances_mean)
    feature_out = feature_out[:n_features]

    raw_importance = {}
    for idx, f_name in enumerate(feature_out):
        imp = perm.importances_mean[idx]
        raw_col = f_name.split("__")[1] if "__" in f_name else f_name
        raw_importance[raw_col] = raw_importance.get(raw_col, 0.0) + imp

    importance_df = pd.DataFrame(
        sorted(raw_importance.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "importance"]
    )

    schema = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dropped_cols": drop_cols
    }

    return best_model, acc, clf_report_df, importance_df, schema


# -----------------------------
# App UI
# -----------------------------
st.title("🧠 Hypertension Prediction System")

model, accuracy, classification_report_df, importance_df, schema = load_and_train_model("hyperpred.csv")

st.markdown(f"**Model accuracy:** {accuracy * 100:.2f}%")
st.header("Classification report (percent)")
st.dataframe(classification_report_df)

st.header("Top features driving predictions")
st.dataframe(importance_df.head(10))

st.header("Enter patient data")

with st.form("patient_form", clear_on_submit=False):
    user_numeric = {}
    user_categorical = {}

    if "Gender" in schema["categorical_cols"]:
        user_categorical["Gender"] = st.selectbox("Sex", ["Female", "Male"])

    if "Age" in schema["numeric_cols"]:
        user_numeric["Age"] = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)

    if "Smoking" in schema["categorical_cols"]:
        user_categorical["Smoking"] = st.selectbox("Smoking", ["No", "Yes"])

    if "Alcohol" in schema["categorical_cols"]:
        user_categorical["Alcohol"] = st.selectbox("Alcohol", ["No", "Yes"])

    if "BMI" in schema["numeric_cols"]:
        user_numeric["BMI"] = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0, step=0.1)

    for col in schema["numeric_cols"]:
        if col in ["Age", "BMI", "PulsePressure", "MAP", "SystolicBP", "DiastolicBP"]:
            continue
        if col.lower() == "cholesterol":
            user_numeric[col] = st.number_input(col, min_value=100.0, max_value=400.0, value=200.0, step=0.1)
        elif col.lower() == "glucose":
            user_numeric[col] = st.number_input(col, min_value=60.0, max_value=300.0, value=100.0, step=0.1)
        else:
            user_numeric[col] = st.number_input(col, value=0.0)

    systolic = st.number_input("SystolicBP", min_value=80.0, max_value=250.0, value=120.0, step=0.5)
    diastolic = st.number_input("DiastolicBP", min_value=40.0, max_value=150.0, value=80.0, step=0.5)
    user_numeric["PulsePressure"] = systolic - diastolic
    user_numeric["MAP"] = (2 * diastolic + systolic) / 3
    user_numeric["SystolicBP"] = systolic
    user_numeric["DiastolicBP"] = diastolic

    for col in schema["categorical_cols"]:
        lowcol = col.lower()
        if col in ["Gender", "Smoking", "Alcohol"]:
            continue
        elif lowcol in ["familyhistory"]:
            user_categorical[col] = st.selectbox(col, ["No", "Yes"])
        elif "physical" in lowcol:
            user_categorical[col] = st.selectbox(col, ["Low", "Medium", "High"])
        else:
            user_categorical[col] = st.text_input(col, "")

    submitted = st.form_submit_button("Predict Risk Level")
    # end of form
if submitted:
    input_row = {**user_numeric, **user_categorical}
    input_df = pd.DataFrame([input_row])
    try:
        pred = int(model.predict(input_df)[0])
        proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
        mapping = {0: "🟩 Low risk", 1: "🟨 Medium risk", 2: "🟥 High risk"}
        result = mapping.get(pred, "Unknown")
        st.subheader(f"Prediction: {result}")
        if proba is not None:
            st.write("Confidence by class:")
            for cls, p in enumerate(proba):
                st.write(f"{mapping[cls]}: {p*100:.1f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
