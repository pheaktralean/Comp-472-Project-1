# COMP 472 - Artificial Intelligence
# Project 1 - MNIST Classifier
# Students: 
    # - Sopheaktra Lean 40225014
    # - An-Khiem Le 40280775
    # -
# Deadline: June 1, 2025

# Import necessary libraries
import numpy as np                     
import matplotlib.pyplot as plt       
from sklearn.datasets import load_digits  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
digits = load_digits()  
X = digits.data  
y = digits.target  

# Visualize the first 10 digits images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Normalize the data
X_normalized = X / 16.0

# Split the normalized data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

# Print the number of images in each set
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])


# Model Training

# Set max_iter to 1000. This sets the max number of iterations the modeil will go through to converge.
model = LogisticRegression(max_iter=1000) 
model.fit(X_train, y_train)

# Making predictions on test data
y_predict = model.predict(X_test)

# Compariung the predicted labels with the actual label
score = accuracy_score(y_predict, y_test)
print('Logistic Regression MNIST dataset: {}%'.format(score * 100))


# Model Evaluation
