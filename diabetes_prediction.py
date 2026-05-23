# diabetes_prediction.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1️⃣ Load the dataset
try:
    data = pd.read_csv("data.csv")
    print("File loaded successfully!")
except Exception as e:
    print("Error loading file:", e)
    exit()

# Optional: show first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# 2️⃣ Prepare features (X) and target (y)
target_column = 'Diabetes_012'
X = data.drop(columns=[target_column])
y = data[target_column]

# 3️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5️⃣ Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# 6️⃣ Example prediction
example_input = [X.iloc[0].tolist()]  # first row from dataset as example
pred = model.predict(example_input)
print("\nPrediction for example input (first row of dataset):", 
      "Diabetes" if pred[0]==1 else "No Diabetes")

# 7️⃣ Interactive input by feature names
print("\nEnter new values for prediction:")

# Get feature names from dataset
feature_names = X.columns.tolist()
user_values = []

for feature in feature_names:
    while True:
        val = input(f"{feature}: ")
        try:
            val = float(val)
            user_values.append(val)
            break
        except:
            print("Please enter a valid number.")

# Make prediction
prediction = model.predict([user_values])
print("\nPrediction for your input:", "Diabetes" if prediction[0]==1 else "No Diabetes")