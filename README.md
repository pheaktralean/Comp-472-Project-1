# Comp-472-Project-1

## Project Description
This project is a Machine Learning application that classifies handwritten digits (0-9) using the **digits dataset** from **scikit-learn**. We want to build a digit recognizer using **Logistic Regression**.

We focus on:
- Loading and exploring the dataset 
- Training a Logistic Regression Model
- Evaluating the model performance using metrics like accuracy, precision, recall, and confusion matrix.


## Project Structure
Comp-472-Project-1/ \
|---- mnist_classifier.py \
|---- README.md \
|---- requirements.txt

## Setup Instruction
1. Clone the repository
2. Create and activate a virtual environment
3. Install the dependencies
`pip install -r requirements.txt`
4. How to run the project
`python3 mnist_classifier.py`

## Dataset Used
We use the **digits** dataset from **scikit-learn**:
- 1797 grayscale images
- 8x8 pixels each (flattened to 64 features)
- Labels from o to 9

## Evaluation:
Generate a **confusion matrix** to see how well each digit is classified.
Generate a **classification report** for precision, recall, and F1-score.
Display the confusion matrix and classification report using **print()**.


