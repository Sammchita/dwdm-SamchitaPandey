import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 1: Create sample medical dataset
data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 48],
    'BP': [120, 140, 130, 150, 110, 135, 160, 145],
    'Glucose': [85, 130, 120, 150, 90, 140, 160, 155],
    'Cholesterol': [180, 220, 200, 240, 170, 210, 260, 230],
    'Disease': [0, 1, 0, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Step 2: Split features and target
X = df[['Age', 'BP', 'Glucose', 'Cholesterol']]
y = df['Disease']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Step 4: Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Prediction
y_pred = model.predict(X_test)

# Step 6: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Predict new patient
import pandas as pd

new_patient = pd.DataFrame([[45, 140, 135, 220]],
                           columns=['Age', 'BP', 'Glucose', 'Cholesterol'])

result = model.predict(new_patient)

if result[0] == 1:
    print("Patient is likely to have disease")
else:
    print("Patient is likely healthy")