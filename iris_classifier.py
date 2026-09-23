from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import matplotlib.pyplot as plt
# ----------------------------------------
# 1. Load the Iris dataset
# ----------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("============================================")
print("       IRIS FLOWER CLASSIFICATION")
print("============================================")

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Classes:", iris.target_names)


# ----------------------------------------
# 2. Scale the features
# ----------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ----------------------------------------
# 3. Split data into training and testing
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("--------------------------------------------")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ----------------------------------------
# 4. Create the KNN model
# ----------------------------------------

model = KNeighborsClassifier(n_neighbors=5)


# ----------------------------------------
# 5. Train the model
# ----------------------------------------

model.fit(X_train, y_train)

print("--------------------------------------------")
print("Model trained successfully!")


# ----------------------------------------
# 6. Make predictions
# ----------------------------------------

predictions = model.predict(X_test)


# ----------------------------------------
# 7. Evaluate the model
# ----------------------------------------

accuracy = accuracy_score(y_test, predictions)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

cm = confusion_matrix(
    y_test,
    predictions
)


# ----------------------------------------
# 8. Display results
# ----------------------------------------

print("--------------------------------------------")
print("MODEL RESULTS")
print("--------------------------------------------")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

print("--------------------------------------------")
print("Confusion Matrix:")
print(cm)

print("--------------------------------------------")
print("Classification completed successfully!")

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Iris Classification Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(3),
    iris.target_names
)

plt.yticks(
    range(3),
    iris.target_names
)

plt.colorbar()

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()