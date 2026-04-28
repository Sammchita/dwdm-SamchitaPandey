from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Sample dataset (1 = word present, 0 = not present)
# Features: [Free, Win, Offer, Click]
X = [
    [1, 1, 1, 1],  # Spam
    [1, 0, 1, 1],  # Spam
    [0, 1, 0, 1],  # Spam
    [0, 0, 0, 0],  # Not Spam
    [0, 0, 1, 0],  # Not Spam
    [0, 1, 0, 0]   # Not Spam
]

# Labels: 1 = Spam, 0 = Not Spam
y = [1, 1, 1, 0, 0, 0]

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Step 3: Train ID3 model (using entropy)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Accuracy
print("Predictions:", y_pred)
print("Actual:", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Test with new message
# Example: "Free Offer Click"
new_msg = [[1, 0, 1, 1]]

result = model.predict(new_msg)

if result[0] == 1:
    print("Message is SPAM")
else:
    print("Message is NOT SPAM")