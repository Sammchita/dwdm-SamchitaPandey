# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
# Features: [Size (sq ft), Number of Bedrooms]
X = np.array([
    [1000, 2],
    [1500, 3],
    [2000, 4],
    [2500, 4],
    [3000, 5]
])

# Target: House Price
y = np.array([200000, 300000, 400000, 450000, 500000])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output
print("Actual values:", y_test)
print("Predicted values:", y_pred)

# Show coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)