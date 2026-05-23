from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# =========================
# Load dataset
# =========================
data = pd.read_csv("data.csv")

# =========================
# BALANCE THE DATASET
# =========================
diabetes = data[data['Diabetes_012'] == 1]
no_diabetes = data[data['Diabetes_012'] == 0]

# Take equal samples
no_diabetes = no_diabetes.sample(len(diabetes), random_state=42)

balanced_data = pd.concat([diabetes, no_diabetes])

# =========================
# Features & Target
# =========================
features = ['BMI', 'Age', 'HighBP', 'HighChol', 'Smoker', 'PhysHlth']
X = balanced_data[features]
y = balanced_data['Diabetes_012']

# =========================
# Train model
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# =========================
# Convert age years → group
# =========================
def age_to_group(age):
    if age <= 24: return 1
    elif age <= 29: return 2
    elif age <= 34: return 3
    elif age <= 39: return 4
    elif age <= 44: return 5
    elif age <= 49: return 6
    elif age <= 54: return 7
    elif age <= 59: return 8
    elif age <= 64: return 9
    elif age <= 69: return 10
    elif age <= 74: return 11
    elif age <= 79: return 12
    else: return 13

# =========================
# SIMPLE & CORRECT RISK LOGIC
# =========================
def get_risk(prob):
    if prob >= 0.70:
        return "🔴 High Risk"
    elif prob >= 0.40:
        return "🟠 Moderate Risk"
    else:
        return "🟢 Low Risk"

# =========================
# Web Route
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    probability = None
    risk = None

    if request.method == "POST":
        BMI = float(request.form["BMI"])
        age_years = int(request.form["Age"])
        HighBP = int(request.form["HighBP"])
        HighChol = int(request.form["HighChol"])
        Smoker = int(request.form["Smoker"])
        PhysHlth = int(request.form["PhysHlth"])

        Age = age_to_group(age_years)

        user_df = pd.DataFrame(
            [[BMI, Age, HighBP, HighChol, Smoker, PhysHlth]],
            columns=features
        )

        probability = model.predict_proba(user_df)[0][1]
        risk = get_risk(probability)

    return render_template(
        "index.html",
        probability=probability,
        risk=risk
    )

if __name__ == "__main__":
    app.run(debug=True)