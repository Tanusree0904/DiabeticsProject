import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Load dataset
# ----------------------------
data = pd.read_csv("data.csv")

features = ['BMI', 'Age', 'HighBP', 'HighChol', 'Smoker', 'PhysHlth']
X = data[features]
y = data['Diabetes_012']

# ----------------------------
# Train model
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
# Convert REAL age → age group
# ----------------------------
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

# ----------------------------
# Risk level
# ----------------------------
def get_risk(prob):
    if prob >= 0.7:
        return "🔴 High Risk"
    elif prob >= 0.4:
        return "🟠 Moderate Risk"
    else:
        return "🟢 Low Risk"

# ----------------------------
# User Input
# ----------------------------
print("\nEnter patient details (REAL VALUES):")

BMI = float(input("BMI (15–50): "))
age_years = int(input("Age in years (18–90): "))
HighBP = int(input("High BP? (0=No, 1=Yes): "))
HighChol = int(input("High Cholesterol? (0=No, 1=Yes): "))
Smoker = int(input("Smoker? (0=No, 1=Yes): "))
PhysHlth = int(input("Unhealthy days in last 30 days (0–30): "))

# Convert age automatically
Age = age_to_group(age_years)

# ----------------------------
# Prediction (NO WARNING)
# ----------------------------
user_df = pd.DataFrame(
    [[BMI, Age, HighBP, HighChol, Smoker, PhysHlth]],
    columns=features
)

prob = model.predict_proba(user_df)[0][1]
risk = get_risk(prob)

print("\n==============================")
print(f"Predicted Probability : {prob*100:.2f}%")
print(f"Risk Level            : {risk}")
print("==============================")