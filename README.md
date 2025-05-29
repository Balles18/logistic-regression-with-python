# Logistic Regression with PyTorch for MNIST Dataset

This project implements a simple logistic regression model using PyTorch to classify handwritten digits from the MNIST dataset. It trains a model on images flattened into vectors and evaluates the accuracy of the model.

## 🚀 Project Overview

- **Data**: MNIST dataset (as CSV)
- **Model**: Logistic Regression (Linear Layer + CrossEntropyLoss)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Evaluation**: Accuracy on test set

## 📂 Files

- `main.py`: Main script containing data preprocessing, model definition, training, and evaluation.
- `data/train.csv`: CSV file with the MNIST dataset. It should contain a `label` column and 784 pixel columns.
🧮 Model Architecture
Input Layer: 784 features (28x28 pixels flattened)

Output Layer: 10 classes (digits 0-9)

📊 Example Output
The model trains for 10 epochs and prints the loss per epoch. After training, it reports the test accuracy and displays a sample test image with its predicted label.

✨ Sample Visualization

📋 Results
Accuracy on test set: ~90% (depending on training)

📌 Notes
Ensure the train.csv file is in the correct format: one column named label for targets and the rest for pixel values (normalized between 0 and 255).

