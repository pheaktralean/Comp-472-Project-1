# COMP 472 - Artificial Intelligence
# Project 1 - MNIST Classifier
# Students: 
    # - Sopheaktra Lean 40225014
    # -
    # -
# Deadline: June 1, 2025

# Import necessary libraries
import numpy as np                     
import matplotlib.pyplot as plt       
from sklearn.datasets import load_digits  
from sklearn.model_selection import train_test_split  

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





# Model Evaluation
