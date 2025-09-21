# Step 1: Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from itertools import cycle

# --- Data Preparation ---
print("1. Loading and preparing data...")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode the labels (e.g., class '1' becomes [0, 1, 0])
# This is required for categorical_crossentropy loss function
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)

# Scale the feature data for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- Model Building ---
print("2. Building the deep learning model...")

# Create a simple sequential model
model = keras.Sequential([
    # Input layer with 4 features (sepal length/width, petal length/width)
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # Hidden layer
    keras.layers.Dense(32, activation='relu'),
    # Output layer with 3 units (one for each Iris class) and softmax activation
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- Model Training ---
print("\n3. Training the model...")

history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=0)
print("Training complete.")


# --- Model Evaluation & Prediction ---
print("\n4. Evaluating the model and making predictions...")

# Make predictions on the test data
# model.predict returns probabilities for each class
y_pred_prob = model.predict(X_test)

# Convert probabilities to class labels (0, 1, or 2)
y_pred_labels = np.argmax(y_pred_prob, axis=1)
y_test_labels = np.argmax(y_test, axis=1) # Convert one-hot encoded test labels back

# --- Calculate and Display Metrics ---
print("\n--- Performance Metrics ---")

# Accuracy
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print(f"Accuracy: {accuracy:.4f}")

# Precision (macro-average treats each class equally)
precision = precision_score(y_test_labels, y_pred_labels, average='macro')
print(f"Precision (Macro Avg): {precision:.4f}")

# Recall (macro-average)
recall = recall_score(y_test_labels, y_pred_labels, average='macro')
print(f"Recall (Macro Avg): {recall:.4f}")

# AUC Score (One-vs-Rest for multi-class)
# Note: roc_auc_score for multi-class needs class probabilities
auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print(f"AUC (One-vs-Rest): {auc:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_labels, y_pred_labels)
print(cm)