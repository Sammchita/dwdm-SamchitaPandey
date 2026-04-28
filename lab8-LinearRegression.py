# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset (Experience vs Salary)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # feature
y = np.array([30000, 35000, 50000, 60000, 70000])  # target

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

# Output results
print("Actual value:", y_test)
print("Predicted value:", y_pred)

# Plot graph
plt.scatter(X, y, color='blue')        # actual data
plt.plot(X, model.predict(X), color='red')  # regression line
plt.title("Linear Regression Example")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()